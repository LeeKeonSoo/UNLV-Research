#!/usr/bin/env python3
"""Reference-trained quality classifier for continuous quality scoring."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from data_eval_common import QUALITY_REFERENCE_MODEL_PATH, QUALITY_REFERENCE_META_PATH, clamp01, safe_float


MODEL_VERSION = "reference-quality-classifier-v1"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


BOILERPLATE_SNIPPETS = (
    "click here to continue",
    "terms of service privacy policy",
    "all rights reserved",
    "sign up now",
    "cookie policy",
    "buy now free shipping",
)


@dataclass
class ReferenceQualityModel:
    classifier: Any
    n_features: int
    ngram_min: int
    ngram_max: int
    decision_center: float
    decision_scale: float
    metadata: Dict[str, Any]

    def _vectorizer(self) -> HashingVectorizer:
        return HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(self.ngram_min, self.ngram_max),
        )

    def score_texts(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        X = self._vectorizer().transform(texts)
        raw = np.asarray(self.classifier.decision_function(X), dtype=np.float32)
        scaled = (raw - self.decision_center) / max(self.decision_scale, 1e-6)
        probs = 1.0 / (1.0 + np.exp(-scaled))
        results: List[Dict[str, Any]] = []
        for score, raw_value, text in zip(probs, raw, texts):
            token_count = len(TOKEN_RE.findall(text))
            results.append(
                {
                    "score": round(clamp01(float(score)), 6),
                    "details": {
                        "raw_decision": round(float(raw_value), 6),
                        "token_count": int(token_count),
                        "decision_center": round(self.decision_center, 6),
                        "decision_scale": round(self.decision_scale, 6),
                        "model_version": MODEL_VERSION,
                        "reference_source": self.metadata.get("reference_source"),
                    },
                }
            )
        return results

    def score_text(self, text: str) -> Dict[str, Any]:
        return self.score_texts([text])[0]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "classifier": self.classifier,
            "n_features": self.n_features,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "decision_center": self.decision_center,
            "decision_scale": self.decision_scale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ReferenceQualityModel":
        return cls(
            classifier=payload["classifier"],
            n_features=int(payload.get("n_features") or 2**18),
            ngram_min=int(payload.get("ngram_min") or 1),
            ngram_max=int(payload.get("ngram_max") or 2),
            decision_center=safe_float(payload.get("decision_center"), 0.0),
            decision_scale=max(safe_float(payload.get("decision_scale"), 1.0), 1e-6),
            metadata=dict(payload.get("metadata") or {}),
        )


def tokenize_quality_text(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p.strip() for p in parts if p.strip()]


def _shuffle_sentences(text: str, rng: random.Random) -> str:
    sents = _sentences(text)
    if len(sents) < 2:
        return text
    rng.shuffle(sents)
    return " ".join(sents)


def _repeat_segment(text: str, rng: random.Random) -> str:
    tokens = tokenize_quality_text(text)
    if len(tokens) < 12:
        return text
    start = rng.randint(0, max(0, len(tokens) - 8))
    seg = tokens[start : start + 6]
    repeated = tokens[:start] + seg * 4 + tokens[start + 6 :]
    return " ".join(repeated)


def _shuffle_tokens(text: str, rng: random.Random) -> str:
    tokens = tokenize_quality_text(text)
    if len(tokens) < 12:
        return text
    rng.shuffle(tokens)
    return " ".join(tokens)


def _boilerplate_wrap(text: str, rng: random.Random) -> str:
    snippet = rng.choice(BOILERPLATE_SNIPPETS)
    return f"{snippet} {text[: max(60, len(text)//3)]} {snippet}"


def corrupt_reference_text(text: str, rng: random.Random) -> str:
    ops = (_shuffle_sentences, _repeat_segment, _shuffle_tokens, _boilerplate_wrap)
    op = rng.choice(ops)
    out = op(text, rng)
    return " ".join(out.split())


def build_reference_quality_model(
    texts: Iterable[str],
    *,
    n_features: int = 2**18,
    ngram_range: tuple[int, int] = (1, 2),
    reference_source: str,
    metadata: Dict[str, Any] | None = None,
    seed: int = 42,
) -> ReferenceQualityModel:
    positives = [" ".join(str(text).split()) for text in texts if len(tokenize_quality_text(str(text))) >= 8]
    if len(positives) < 200:
        raise ValueError("Need at least 200 reference texts to build a stable quality classifier.")

    rng = random.Random(seed)
    negatives = [corrupt_reference_text(text, rng) for text in positives]

    indices = list(range(len(positives)))
    rng.shuffle(indices)
    split = max(100, int(len(indices) * 0.85))
    train_idx = indices[:split]
    val_idx = indices[split:]
    if len(val_idx) < 50:
        val_idx = indices[-50:]
        train_idx = indices[:-50]

    train_texts = [positives[i] for i in train_idx] + [negatives[i] for i in train_idx]
    train_y = np.array([1] * len(train_idx) + [0] * len(train_idx), dtype=np.int32)
    val_texts = [positives[i] for i in val_idx] + [negatives[i] for i in val_idx]
    val_y = np.array([1] * len(val_idx) + [0] * len(val_idx), dtype=np.int32)

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=ngram_range,
    )
    X_train = vectorizer.transform(train_texts)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-6,
        max_iter=100,
        tol=1e-4,
        random_state=seed,
    )
    clf.fit(X_train, train_y)

    X_val = vectorizer.transform(val_texts)
    raw_val = np.asarray(clf.decision_function(X_val), dtype=np.float32)
    raw_pos = raw_val[val_y == 1]
    raw_neg = raw_val[val_y == 0]
    pos_med = float(np.median(raw_pos))
    neg_med = float(np.median(raw_neg))
    center = 0.5 * (pos_med + neg_med)
    scale = max(abs(pos_med - neg_med) / 4.0, 0.25)
    auc = float(roc_auc_score(val_y, raw_val))

    model = ReferenceQualityModel(
        classifier=clf,
        n_features=n_features,
        ngram_min=ngram_range[0],
        ngram_max=ngram_range[1],
        decision_center=center,
        decision_scale=scale,
        metadata={
            "reference_source": reference_source,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "train_rows": int(len(train_texts)),
            "validation_rows": int(len(val_texts)),
            "auc": round(auc, 6),
            "seed": seed,
            **(metadata or {}),
        },
    )
    return model


def save_reference_quality_model(model: ReferenceQualityModel) -> None:
    QUALITY_REFERENCE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.to_payload(), QUALITY_REFERENCE_MODEL_PATH)
    QUALITY_REFERENCE_META_PATH.write_text(
        json.dumps(
            {
                "version": MODEL_VERSION,
                "reference_source": model.metadata.get("reference_source"),
                "positive_count": model.metadata.get("positive_count"),
                "negative_count": model.metadata.get("negative_count"),
                "train_rows": model.metadata.get("train_rows"),
                "validation_rows": model.metadata.get("validation_rows"),
                "auc": model.metadata.get("auc"),
                "decision_center": model.decision_center,
                "decision_scale": model.decision_scale,
                "n_features": model.n_features,
                "ngram_range": [model.ngram_min, model.ngram_max],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_reference_quality_model(path=QUALITY_REFERENCE_MODEL_PATH) -> ReferenceQualityModel:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "classifier" not in payload:
        raise ValueError(
            "Reference quality model payload is outdated or invalid. "
            "Run prepare_reference_quality_model.py to rebuild it."
        )
    return ReferenceQualityModel.from_payload(payload)
