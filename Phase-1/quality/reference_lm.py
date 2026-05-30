#!/usr/bin/env python3
"""Lightweight reference-corpus LM quality scorer."""

from __future__ import annotations

import math
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import joblib

from data_eval_common import QUALITY_REFERENCE_MODEL_PATH, QUALITY_REFERENCE_META_PATH, clamp01, safe_float


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
BOS = "<s>"
UNK = "<unk>"
MODEL_VERSION = "reference-lm-v1"


def tokenize_quality_text(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class ReferenceLMQualityModel:
    token_to_id: Dict[str, int]
    unigram_counts: List[int]
    prev_token_counts: List[int]
    bigram_counts: Dict[int, Dict[int, int]]
    smoothing_alpha: float
    calibration_low: float
    calibration_mid: float
    calibration_high: float
    metadata: Dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return len(self.unigram_counts)

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def total_unigrams(self) -> int:
        return int(sum(self.unigram_counts))

    def token_ids(self, text: str) -> List[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokenize_quality_text(text)]

    def _unigram_prob(self, token_id: int) -> float:
        return (self.unigram_counts[token_id] + 1.0) / (self.total_unigrams + self.vocab_size)

    def _conditional_prob(self, prev_id: int, curr_id: int) -> float:
        prev_count = self.prev_token_counts[prev_id]
        transitions = self.bigram_counts.get(prev_id) or {}
        bigram_count = transitions.get(curr_id, 0)
        unigram_prob = self._unigram_prob(curr_id)
        alpha = self.smoothing_alpha
        return (bigram_count + alpha * unigram_prob) / (prev_count + alpha)

    def average_nll(self, text: str) -> Dict[str, float]:
        ids = self.token_ids(text)
        return self.average_nll_ids(ids)

    def average_nll_ids(self, ids: List[int]) -> Dict[str, float]:
        if not ids:
            return {"avg_nll": 0.0, "token_count": 0.0, "oov_ratio": 1.0}
        nll_sum = 0.0
        prev_id = self.bos_id
        oov = 0
        for curr_id in ids:
            if curr_id == self.unk_id:
                oov += 1
            prob = max(self._conditional_prob(prev_id, curr_id), 1e-12)
            nll_sum += -math.log(prob)
            prev_id = curr_id
        token_count = len(ids)
        return {
            "avg_nll": nll_sum / token_count,
            "token_count": float(token_count),
            "oov_ratio": oov / token_count,
        }

    def score_text(self, text: str) -> Dict[str, Any]:
        stats = self.average_nll(text)
        avg_nll = float(stats["avg_nll"])
        low = self.calibration_low
        high = max(self.calibration_high, low + 1e-6)
        normalized = clamp01((high - avg_nll) / (high - low))
        return {
            "score": round(normalized, 6),
            "details": {
                "avg_nll": round(avg_nll, 6),
                "token_count": int(stats["token_count"]),
                "oov_ratio": round(float(stats["oov_ratio"]), 6),
                "calibration_low": round(low, 6),
                "calibration_mid": round(self.calibration_mid, 6),
                "calibration_high": round(high, 6),
                "model_version": MODEL_VERSION,
                "reference_source": self.metadata.get("reference_source"),
            },
        }

    def to_payload(self) -> Dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "token_to_id": self.token_to_id,
            "unigram_counts": self.unigram_counts,
            "prev_token_counts": self.prev_token_counts,
            "bigram_counts": self.bigram_counts,
            "smoothing_alpha": self.smoothing_alpha,
            "calibration_low": self.calibration_low,
            "calibration_mid": self.calibration_mid,
            "calibration_high": self.calibration_high,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ReferenceLMQualityModel":
        return cls(
            token_to_id={str(k): int(v) for k, v in (payload.get("token_to_id") or {}).items()},
            unigram_counts=[int(x) for x in payload.get("unigram_counts") or []],
            prev_token_counts=[int(x) for x in payload.get("prev_token_counts") or []],
            bigram_counts={
                int(prev): {int(curr): int(count) for curr, count in transitions.items()}
                for prev, transitions in (payload.get("bigram_counts") or {}).items()
            },
            smoothing_alpha=float(payload.get("smoothing_alpha") or 0.2),
            calibration_low=safe_float(payload.get("calibration_low"), default=0.0),
            calibration_mid=safe_float(payload.get("calibration_mid"), default=0.0),
            calibration_high=safe_float(payload.get("calibration_high"), default=1.0),
            metadata=dict(payload.get("metadata") or {}),
        )


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def build_reference_lm_model(
    texts: Iterable[str],
    *,
    max_vocab: int = 50000,
    smoothing_alpha: float = 0.2,
    reference_source: str,
    metadata: Dict[str, Any] | None = None,
) -> ReferenceLMQualityModel:
    raw_sequences: List[List[str]] = []
    unigram_counter: Counter[str] = Counter()
    total_tokens = 0
    for text in texts:
        tokens = tokenize_quality_text(text)
        if len(tokens) < 8:
            continue
        raw_sequences.append(tokens)
        unigram_counter.update(tokens)
        total_tokens += len(tokens)

    vocab_tokens = [token for token, _ in unigram_counter.most_common(max(0, max_vocab - 2))]
    token_to_id = {BOS: 0, UNK: 1}
    for token in vocab_tokens:
        token_to_id[token] = len(token_to_id)

    unigram_counts = [0 for _ in range(len(token_to_id))]
    prev_token_counts = [0 for _ in range(len(token_to_id))]
    bigram_counts: Dict[int, Dict[int, int]] = defaultdict(dict)
    mapped_sequences: List[List[int]] = []

    for tokens in raw_sequences:
        ids = [token_to_id.get(token, token_to_id[UNK]) for token in tokens]
        mapped_sequences.append(ids)
        for curr_id in ids:
            unigram_counts[curr_id] += 1
        prev_id = token_to_id[BOS]
        for curr_id in ids:
            prev_token_counts[prev_id] += 1
            transitions = bigram_counts[prev_id]
            transitions[curr_id] = transitions.get(curr_id, 0) + 1
            prev_id = curr_id

    model = ReferenceLMQualityModel(
        token_to_id=token_to_id,
        unigram_counts=unigram_counts,
        prev_token_counts=prev_token_counts,
        bigram_counts={int(k): v for k, v in bigram_counts.items()},
        smoothing_alpha=float(smoothing_alpha),
        calibration_low=0.0,
        calibration_mid=0.0,
        calibration_high=1.0,
        metadata={
            "reference_source": reference_source,
            "sequence_count": len(mapped_sequences),
            "reference_token_count": total_tokens,
            "max_vocab": max_vocab,
            **(metadata or {}),
        },
    )

    nll_values = [model.average_nll_ids(ids)["avg_nll"] for ids in mapped_sequences]
    model.calibration_low = _quantile(nll_values, 0.05)
    model.calibration_mid = _quantile(nll_values, 0.50)
    model.calibration_high = _quantile(nll_values, 0.95)
    return model


def save_reference_lm_model(model: ReferenceLMQualityModel) -> None:
    QUALITY_REFERENCE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.to_payload(), QUALITY_REFERENCE_MODEL_PATH)
    QUALITY_REFERENCE_META_PATH.write_text(
        json.dumps(
            {
                "version": MODEL_VERSION,
                "reference_source": model.metadata.get("reference_source"),
                "reference_token_count": model.metadata.get("reference_token_count"),
                "sequence_count": model.metadata.get("sequence_count"),
                "max_vocab": model.metadata.get("max_vocab"),
                "calibration_low": model.calibration_low,
                "calibration_mid": model.calibration_mid,
                "calibration_high": model.calibration_high,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_reference_lm_model(path=QUALITY_REFERENCE_MODEL_PATH) -> ReferenceLMQualityModel:
    payload = joblib.load(path)
    return ReferenceLMQualityModel.from_payload(payload)
