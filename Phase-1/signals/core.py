#!/usr/bin/env python3
"""Core metric scorers for the generic data evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer

from data_eval_common import (
    INDEX_DIR,
    QUALITY_REFERENCE_MODEL_PATH,
    alpha_ratio,
    clamp01,
    repeated_token_ratio,
    safe_float,
    sentence_count,
    sigmoid,
)
from quality.reference_quality import load_reference_quality_model
from utility.features import utility_feature_vector


INDEX_DB_PATH = INDEX_DIR / "index.sqlite"
UTILITY_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "utility_predictor.joblib"
UTILITY_META_PATH = Path(__file__).resolve().parents[1] / "models" / "utility_predictor.meta.json"
VECTORIZER_FEATURES = 2**16
NEAR_DUP_REFINED_PREFIX_HEX = 6
NEAR_DUP_FALLBACK_PREFIX_HEX = 8
NEAR_DUP_MAX_SUBGROUP_SIZE = 256
NEAR_DUP_HAMMING_THRESHOLD = 10
NEAR_DUP_JACCARD_THRESHOLD = 0.75
NEAR_DUP_CONTAINMENT_THRESHOLD = 0.88
NEAR_DUP_PREFIX_PRESSURE_LOG_BASE = 64.0
NEAR_DUP_VERIFIED_BLEND_WEIGHT = 0.72
NEAR_DUP_PREFIX_BLEND_WEIGHT = 0.18
NEAR_DUP_REPEAT_BLEND_WEIGHT = 0.10
NEAR_DUP_UNVERIFIED_MAX = 0.35

UTILITY_MIN_TRAIN_ROWS = 120
UTILITY_MIN_TEST_ROWS = 24
UTILITY_MIN_SPEARMAN = 0.35
UTILITY_MIN_R2 = 0.05
UTILITY_MIN_LABEL_SPREAD = 0.30

QUALITY_POSITIVE_PROTOTYPES = [
    "This passage explains a topic clearly with coherent sentences, meaningful detail, and useful structure.",
    "A well-formed article that contains substantial information, logical flow, and readable language.",
]
QUALITY_NEGATIVE_PROTOTYPES = [
    "Click here subscribe buy now advertisement terms and conditions repeated many times.",
    "Random broken text symbols placeholder filler low content fragment.",
    "Glossary term list with bullet points and shallow reference material but little real explanation.",
    "Procedural click-by-click instructions with low conceptual learning value.",
]
UTILITY_PROTOTYPES = [
    "A step-by-step explanation that teaches a concept with examples and reasoning.",
    "A concise factual explanation that helps a language model learn useful knowledge patterns.",
    "Structured content that supports reasoning, instruction following, or concept understanding.",
]
UTILITY_NEGATIVE_PROTOTYPES = [
    "Glossary term definition bullet list repeated with little explanation or context.",
    "Click here tap here scroll to settings and follow the app instructions with shallow learning value.",
    "Generic conclusion summary boilerplate with little new information.",
]

_DEFINITION_PATTERNS = (
    "is defined as",
    "refers to",
    "means that",
    "is the process of",
    "can be described as",
)
_EXPLANATION_MARKERS = (
    "because",
    "therefore",
    "for example",
    "for instance",
    "as a result",
    "this means",
    "in order to",
)
_QUESTION_MARKERS = ("question", "answer", "why", "how", "what is", "what are")
_PROCEDURAL_MARKERS = (
    "click",
    "tap",
    "scroll",
    "sign up",
    "open the app",
    "go to the",
    "select the",
    "press the",
)
_COHESION_MARKERS = (
    "however",
    "therefore",
    "in contrast",
    "on the other hand",
    "first,",
    "second,",
    "finally,",
)


def _count_contains(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def _bullet_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    bullet_lines = sum(1 for line in lines if line.startswith(("-", "*")) or re.match(r"^\d+\.", line))
    return bullet_lines / len(lines)


def _style_bucket_from_text(text: str) -> str:
    lowered = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_ratio = _bullet_ratio(text)
    if bullet_ratio >= 0.40:
        return "structured_list"
    instructional_markers = ("step ", "steps", "instructions", "click", "tap", "select", "open the", "go to ")
    if any(marker in lowered for marker in instructional_markers):
        return "instructional"
    technical_markers = ("http", "api", "parameter", "returns", "arguments", "syntax", "config", "option")
    colon_lines = sum(1 for line in lines if ":" in line)
    if any(marker in lowered for marker in technical_markers) or (lines and colon_lines / len(lines) >= 0.35):
        return "technical_reference"
    conversational_markers = ("q:", "a:", "question:", "answer:", "you", "your")
    if any(marker in lowered for marker in conversational_markers):
        return "conversational"
    return "general_prose"


def _max_repeated_token_run(tokens: List[str]) -> int:
    max_run = 0
    current_run = 0
    previous = None
    for token in tokens:
        if token == previous:
            current_run += 1
        else:
            current_run = 1
            previous = token
        max_run = max(max_run, current_run)
    return max_run


def _max_repeated_char_run(text: str) -> int:
    max_run = 0
    current_run = 0
    previous = None
    for char in text:
        if char == previous:
            current_run += 1
        else:
            current_run = 1
            previous = char
        max_run = max(max_run, current_run)
    return max_run


def _normalize_similarity_margin(pos: float, neg: float) -> float:
    # A small margin proxy that is easier to interpret than the previous raw
    # prototype-weighted sigmoid recipe.
    return clamp01((pos - neg + 0.20) / 0.80)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize_dense(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v)
    if denom == 0:
        return v
    return v / denom


def _token_shingle_set(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 2:
        return set()
    if len(tokens) < 10:
        n = 1
    elif len(tokens) < 24:
        n = 2
    else:
        n = 3
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _overlap_scores(a: set[str], b: set[str]) -> Tuple[float, float, float]:
    if not a or not b:
        return 0.0, 0.0, 0.0
    inter = len(a & b)
    union = len(a | b)
    min_size = min(len(a), len(b))
    jaccard = (inter / union) if union else 0.0
    containment = (inter / min_size) if min_size else 0.0
    overlap = max(jaccard, containment)
    return jaccard, containment, overlap


def _hamming_distance_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


class CoreMetricScorer:
    def __init__(self, index_db_path: Path = INDEX_DB_PATH):
        self.index_db_path = index_db_path
        self.conn = sqlite3.connect(str(index_db_path))
        self.vectorizer = HashingVectorizer(
            n_features=VECTORIZER_FEATURES,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self.quality_pos = self._mean_vector(QUALITY_POSITIVE_PROTOTYPES)
        self.quality_neg = self._mean_vector(QUALITY_NEGATIVE_PROTOTYPES)
        self.utility_ref = self._mean_vector(UTILITY_PROTOTYPES)
        self.utility_neg = self._mean_vector(UTILITY_NEGATIVE_PROTOTYPES)
        self.quality_pos_unit = _normalize_dense(self.quality_pos)
        self.quality_neg_unit = _normalize_dense(self.quality_neg)
        self.utility_ref_unit = _normalize_dense(self.utility_ref)
        self.utility_neg_unit = _normalize_dense(self.utility_neg)
        self.utility_predictor = None
        self.utility_feature_names: List[str] = []
        self.utility_predictor_mode = "heuristic_only"
        self.utility_predictor_gate: Dict[str, Any] = {"enabled": False, "reasons": ["no_model"]}
        self._load_utility_predictor()
        self.reference_quality_model = self._load_reference_quality_model()
        self._simhash_prefix_counts = self._load_simhash_prefix_counts()
        self._active_verified_prefix: str | None = None
        self._active_verified_counts: Dict[str, int] = {}
        self._active_risk_scores: Dict[str, float] = {}
        self._active_risk_jaccard: Dict[str, float] = {}
        self._active_risk_containment: Dict[str, float] = {}
        self._active_verified_meta: Dict[str, Any] = {}
        self._dataset_cluster_size_rarity = self._load_dataset_cluster_size_rarity()
        row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        self.total_chunks = int(row[0]) if row else 0

    def close(self) -> None:
        self.conn.close()

    def _mean_vector(self, texts: List[str]) -> np.ndarray:
        mat = self.vectorizer.transform(texts)
        dense = mat.toarray().astype(np.float32)
        return dense.mean(axis=0)

    def _vector(self, text: str) -> np.ndarray:
        return self.vectorizer.transform([text]).toarray()[0].astype(np.float32)

    def _vector_matrix(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def _utility_model_gate(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        train_rows = int(meta.get("train_rows") or 0)
        test_rows = int(meta.get("test_rows") or 0)
        spearman = safe_float(meta.get("spearman"), default=-1.0)
        r2 = safe_float(meta.get("r2"), default=-1.0)
        label_min = safe_float(meta.get("label_min"), default=0.0)
        label_max = safe_float(meta.get("label_max"), default=0.0)
        label_spread = max(0.0, label_max - label_min)
        sklearn_meta = str(meta.get("sklearn_version") or "").strip()
        checks = {
            "train_rows": train_rows >= UTILITY_MIN_TRAIN_ROWS,
            "test_rows": test_rows >= UTILITY_MIN_TEST_ROWS,
            "spearman": spearman >= UTILITY_MIN_SPEARMAN,
            "r2": r2 >= UTILITY_MIN_R2,
            "label_spread": label_spread >= UTILITY_MIN_LABEL_SPREAD,
            "sklearn_version": sklearn_meta == sklearn.__version__,
        }
        reasons = [name for name, ok in checks.items() if not ok]
        return {
            "enabled": not reasons,
            "reasons": reasons,
            "checks": checks,
            "observed": {
                "train_rows": train_rows,
                "test_rows": test_rows,
                "spearman": round(spearman, 6),
                "r2": round(r2, 6),
                "label_spread": round(label_spread, 6),
                "sklearn_version": sklearn_meta or None,
                "runtime_sklearn_version": sklearn.__version__,
            },
            "thresholds": {
                "train_rows": UTILITY_MIN_TRAIN_ROWS,
                "test_rows": UTILITY_MIN_TEST_ROWS,
                "spearman": UTILITY_MIN_SPEARMAN,
                "r2": UTILITY_MIN_R2,
                "label_spread": UTILITY_MIN_LABEL_SPREAD,
            },
        }

    def _load_utility_predictor(self) -> None:
        meta: Dict[str, Any] = {}
        if UTILITY_META_PATH.exists():
            try:
                meta = json.loads(UTILITY_META_PATH.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        gate = self._utility_model_gate(meta)
        self.utility_predictor_gate = gate
        if not gate["enabled"]:
            return

        if not UTILITY_MODEL_PATH.exists():
            return
        try:
            payload = joblib.load(UTILITY_MODEL_PATH)
        except Exception:
            return
        model = payload.get("model")
        feature_names = list(payload.get("feature_names") or [])
        if model is None or not feature_names:
            return

        self.utility_predictor = model
        self.utility_feature_names = feature_names
        self.utility_predictor_mode = "probe_calibrated"

    def _load_simhash_prefix_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for prefix, count in self.conn.execute(
            "SELECT simhash_prefix, COUNT(*) FROM chunks GROUP BY simhash_prefix"
        ):
            out[str(prefix)] = int(count)
        return out

    def _load_dataset_cluster_size_rarity(self) -> Dict[str, Dict[int, float]]:
        """
        Build a dataset-relative rarity lookup for cluster_size.
        rarity ~= 1 - CDF(cluster_size), so smaller clusters get higher rarity.
        """
        grouped: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for dataset, cluster_size, count in self.conn.execute(
            "SELECT dataset, cluster_size, COUNT(*) FROM chunks GROUP BY dataset, cluster_size ORDER BY dataset, cluster_size"
        ):
            grouped[str(dataset)].append((int(cluster_size), int(count)))

        out: Dict[str, Dict[int, float]] = {}
        for dataset, rows in grouped.items():
            total = sum(count for _, count in rows)
            if total <= 0:
                continue
            csum = 0
            rarity_map: Dict[int, float] = {}
            for cluster_size, count in rows:
                csum += count
                cdf = csum / total
                rarity_map[int(cluster_size)] = clamp01(1.0 - cdf)
            out[dataset] = rarity_map
        return out

    def _calibrate_reference_quality(
        self,
        quality_payload: Dict[str, Any],
        validity: Dict[str, Any],
        text: str | None = None,
    ) -> Dict[str, Any]:
        raw_score = clamp01(float(quality_payload.get("score") or 0.0))
        details = dict(quality_payload.get("details") or {})
        token_count = int(details.get("token_count") or 0)
        validity_score = clamp01(float(validity.get("score") or 0.0))
        validity_details = validity.get("details") or {}
        repeated_ratio = clamp01(float(validity_details.get("repeated_token_ratio") or 0.0))
        word_count = int(validity_details.get("word_count") or token_count or 0)
        alpha = clamp01(float(validity_details.get("alpha_ratio") or 0.0))
        lexical_diversity = clamp01(float(validity_details.get("lexical_diversity") or 0.0))
        style_bucket = str(validity_details.get("style_bucket") or "unknown")
        raw_text = str(text or "")
        lowered = raw_text.lower()
        valid = bool(validity.get("valid"))

        short_penalty = clamp01((24.0 - token_count) / 24.0) * 0.06
        low_validity_penalty = clamp01((0.78 - validity_score) / 0.78) * 0.20
        repeat_penalty = clamp01((repeated_ratio - 0.22) / 0.45) * 0.15
        adjusted = clamp01(raw_score - short_penalty - low_validity_penalty - repeat_penalty)
        if not bool(validity.get("valid")):
            adjusted = min(adjusted, 0.55)

        boilerplate_hits = _count_contains(
            lowered,
            (
                "click here",
                "subscribe",
                "sign up",
                "buy now",
                "cookie policy",
                "privacy policy",
                "terms of service",
                "follow us",
                "like us on",
                "facebook page",
                "twitter",
            ),
        )
        explanatory_hits = _count_contains(lowered, _EXPLANATION_MARKERS)
        definition_hits = _count_contains(lowered, _DEFINITION_PATTERNS)
        technical_hits = _count_contains(
            lowered,
            ("http", "api", "parameter", "returns", "syntax", "config", "function", "class", "definition"),
        )
        structure_hits = int(":" in raw_text) + int("- " in raw_text or "\n-" in raw_text) + int(sentence_count(raw_text) >= 2)
        info_support = clamp01(
            0.35 * min(explanatory_hits, 2)
            + 0.25 * min(definition_hits, 2)
            + 0.20 * min(technical_hits, 2)
            + 0.20 * min(structure_hits, 2)
        )
        if word_count <= 64:
            length_bucket = "short"
        elif word_count <= 128:
            length_bucket = "medium"
        elif word_count <= 384:
            length_bucket = "long"
        else:
            length_bucket = "very_long"
        low_noise_valid = bool(
            valid
            and word_count >= 16
            and alpha >= 0.58
            and lexical_diversity >= 0.16
            and repeated_ratio <= 0.42
            and boilerplate_hits == 0
        )
        calibration_floor = 0.0
        if low_noise_valid:
            if word_count >= 40:
                calibration_floor = 0.36
            elif word_count >= 24:
                calibration_floor = 0.33
            else:
                calibration_floor = 0.29
            if style_bucket in {"technical_reference", "structured_list", "instructional"}:
                calibration_floor += 0.04
            calibration_floor += 0.08 * info_support
            calibration_floor = min(calibration_floor, 0.50)
        adjusted = max(adjusted, calibration_floor)

        # The reference classifier is intentionally general, but raw scores still
        # carry length/prose-style bias. The selector needs a quality signal that
        # rewards coherent information density without turning "long general
        # prose" into a hidden proxy for quality.
        style_length_correction = 0.0
        if low_noise_valid and style_bucket in {"instructional", "structured_list", "technical_reference"}:
            style_length_correction += 0.03 + (0.04 * info_support)
        if low_noise_valid and length_bucket == "short" and info_support >= 0.35:
            style_length_correction += 0.04
        if length_bucket == "very_long" and info_support < 0.20:
            style_length_correction -= 0.035
        if boilerplate_hits:
            style_length_correction -= min(0.08, 0.025 * boilerplate_hits)

        evidence_quality = clamp01(
            0.50 * adjusted
            + 0.18 * info_support
            + 0.16 * clamp01(lexical_diversity / 0.65)
            + 0.16 * validity_score
        )
        calibrated_quality = clamp01(0.72 * adjusted + 0.28 * evidence_quality + style_length_correction)

        details.update(
            {
                "raw_score_before_calibration": round(raw_score, 6),
                "short_text_penalty": round(short_penalty, 6),
                "low_validity_penalty": round(low_validity_penalty, 6),
                "repetition_penalty": round(repeat_penalty, 6),
                "boilerplate_hits": int(boilerplate_hits),
                "quality_info_support": round(float(info_support), 6),
                "style_bucket": style_bucket,
                "length_bucket": length_bucket,
                "word_count": int(word_count),
                "lexical_diversity": round(float(lexical_diversity), 6),
                "quality_calibration_floor": round(float(calibration_floor), 6),
                "style_length_quality_correction": round(float(style_length_correction), 6),
                "style_length_normalized_quality": round(float(calibrated_quality), 6),
                "quality_evidence_score": round(float(evidence_quality), 6),
                "quality_calibration_policy": "style_length_normalized_quality_v2",
                "calibrated_with_validity": True,
            }
        )
        return {
            "score": round(calibrated_quality, 6),
            "details": details,
        }

    def _load_reference_quality_model(self):
        if not QUALITY_REFERENCE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing reference quality model: {QUALITY_REFERENCE_MODEL_PATH}. "
                "Run prepare_reference_quality_model.py before scoring."
            )
        return load_reference_quality_model(QUALITY_REFERENCE_MODEL_PATH)

    def _load_verified_near_dup_prefix(
        self,
        simhash_prefix_value: str,
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT chunk_uid, text, simhash
            FROM chunks
            WHERE simhash_prefix = ?
            ORDER BY chunk_uid
            """,
            (simhash_prefix_value,),
        ).fetchall()
        bucket_count = len(rows)
        if bucket_count <= 1:
            return {}, {}, {}, {}, {
                "simhash_prefix_bucket_count": bucket_count,
                "verified_pair_count": 0,
                "subgroup_count": 0,
                "truncated_subgroup_count": 0,
            }

        subgroups: List[List[Tuple[str, str, str]]] = []
        refined: Dict[str, List[Tuple[str, str, str]]] = {}
        for uid, text, simhash_hex in rows:
            refined.setdefault(str(simhash_hex)[:NEAR_DUP_REFINED_PREFIX_HEX], []).append((str(uid), str(text), str(simhash_hex)))

        truncated_subgroup_count = 0
        for group_rows in refined.values():
            if len(group_rows) <= NEAR_DUP_MAX_SUBGROUP_SIZE:
                subgroups.append(group_rows)
                continue
            fallback: Dict[str, List[Tuple[str, str, str]]] = {}
            for item in group_rows:
                fallback.setdefault(item[2][:NEAR_DUP_FALLBACK_PREFIX_HEX], []).append(item)
            for fallback_rows in fallback.values():
                if len(fallback_rows) > NEAR_DUP_MAX_SUBGROUP_SIZE:
                    truncated_subgroup_count += 1
                    fallback_rows = fallback_rows[:NEAR_DUP_MAX_SUBGROUP_SIZE]
                subgroups.append(fallback_rows)

        counts: Counter[str] = Counter()
        max_overlap_by_uid: Dict[str, float] = {}
        max_jaccard_by_uid: Dict[str, float] = {}
        max_containment_by_uid: Dict[str, float] = {}
        verified_pair_count = 0
        for subgroup in subgroups:
            cached = [
                (uid, _token_shingle_set(text), simhash_hex)
                for uid, text, simhash_hex in subgroup
            ]
            for i in range(len(cached)):
                uid_i, shingles_i, simhash_i = cached[i]
                if not shingles_i:
                    continue
                for j in range(i + 1, len(cached)):
                    uid_j, shingles_j, simhash_j = cached[j]
                    if not shingles_j:
                        continue
                    jaccard, containment, overlap = _overlap_scores(shingles_i, shingles_j)
                    if jaccard > max_jaccard_by_uid.get(uid_i, 0.0):
                        max_jaccard_by_uid[uid_i] = jaccard
                    if jaccard > max_jaccard_by_uid.get(uid_j, 0.0):
                        max_jaccard_by_uid[uid_j] = jaccard
                    if containment > max_containment_by_uid.get(uid_i, 0.0):
                        max_containment_by_uid[uid_i] = containment
                    if containment > max_containment_by_uid.get(uid_j, 0.0):
                        max_containment_by_uid[uid_j] = containment
                    if overlap > max_overlap_by_uid.get(uid_i, 0.0):
                        max_overlap_by_uid[uid_i] = overlap
                    if overlap > max_overlap_by_uid.get(uid_j, 0.0):
                        max_overlap_by_uid[uid_j] = overlap
                    if _hamming_distance_hex(simhash_i, simhash_j) > NEAR_DUP_HAMMING_THRESHOLD:
                        continue
                    if jaccard >= NEAR_DUP_JACCARD_THRESHOLD or containment >= NEAR_DUP_CONTAINMENT_THRESHOLD:
                        counts[uid_i] += 1
                        counts[uid_j] += 1
                        verified_pair_count += 1

        return dict(counts), max_overlap_by_uid, max_jaccard_by_uid, max_containment_by_uid, {
            "simhash_prefix_bucket_count": bucket_count,
            "verified_pair_count": int(verified_pair_count),
            "subgroup_count": len(subgroups),
            "truncated_subgroup_count": int(truncated_subgroup_count),
        }

    def structural_validity_score(self, text: str) -> Dict[str, Any]:
        text = str(text or "")
        stripped = text.strip()
        word_count = max(len(stripped.split()), 1)
        sent_count = sentence_count(text)
        alpha = alpha_ratio(text)
        repeated = repeated_token_ratio(text)
        line_break_ratio = text.count("\n") / max(len(text), 1)
        tokens = re.findall(r"\w+", text.lower())
        lexical_diversity = (len(set(tokens)) / max(len(tokens), 1)) if tokens else 0.0
        symbol_count = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'-\"()[]{}"))
        symbol_ratio = symbol_count / max(len(text), 1)
        markup_hits = len(re.findall(r"</?[a-z][^>]*>|&[a-z]+;|\|", text.lower()))
        extraction_residue_hits = len(re.findall(r"<\s*(script|style)\b|javascript:|function\s*\(|\{[\"']", text.lower()))
        markup_residue_ratio = clamp01(markup_hits / max(word_count, 1))
        control_char_count = sum(
            1 for ch in text if ((ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 127)
        )
        control_char_ratio = control_char_count / max(len(text), 1)
        replacement_char_ratio = text.count("\ufffd") / max(len(text), 1)
        max_char_run = _max_repeated_char_run(text)
        max_token_run = _max_repeated_token_run(tokens)
        style_bucket = _style_bucket_from_text(text)

        violated_rules: List[str] = []
        warning_rules: List[str] = []

        if not stripped or word_count < 8:
            violated_rules.append("empty_or_too_short")
        elif word_count < 20:
            warning_rules.append("short_text_unit")
        learnable_unit_pass = bool(stripped and word_count >= 8)

        if sent_count < 1 and word_count < 20:
            violated_rules.append("non_language_fragment")
        elif sent_count < 1:
            warning_rules.append("missing_sentence_boundary")

        if replacement_char_ratio > 0.01:
            violated_rules.append("encoding_corruption")
        if control_char_ratio > 0.01:
            violated_rules.append("control_character_noise")

        if alpha < 0.45:
            violated_rules.append("non_language_fragment")
        elif alpha < 0.65:
            warning_rules.append("low_alpha_ratio")

        if lexical_diversity < 0.08 and word_count >= 20:
            violated_rules.append("non_language_fragment")
        elif lexical_diversity < 0.20:
            warning_rules.append("low_lexical_diversity")

        if symbol_ratio > 0.35:
            violated_rules.append("symbol_noise")
        elif symbol_ratio > 0.20:
            warning_rules.append("elevated_symbol_noise")

        if markup_residue_ratio > 0.18 or (extraction_residue_hits > 0 and markup_residue_ratio > 0.04):
            violated_rules.append("markup_or_extraction_residue")
        elif markup_residue_ratio > 0.06:
            warning_rules.append("possible_markup_residue")

        hard_repetition = (
            max_char_run >= 40
            or max_token_run >= 8
            or repeated > 0.72
            or (repeated > 0.58 and lexical_diversity < 0.12)
        )
        if hard_repetition:
            violated_rules.append("hard_broken_repetition")
        elif repeated > 0.45:
            if style_bucket in {"structured_list", "technical_reference", "instructional", "conversational"}:
                warning_rules.append("style_repetition_pattern")
            else:
                warning_rules.append("soft_repetition_warning")
        elif repeated > 0.30:
            warning_rules.append("mild_repetition_warning")

        score = (
            0.28 * clamp01(alpha)
            + 0.22 * clamp01(min(word_count / 120.0, 1.0))
            + 0.18 * clamp01(min(sent_count / 4.0, 1.0))
            + 0.12 * clamp01(lexical_diversity / 0.60)
            + 0.10 * clamp01(1.0 - repeated)
            + 0.10 * clamp01(1.0 - line_break_ratio * 10.0)
            - 0.08 * clamp01(symbol_ratio / 0.25)
            - 0.05 * markup_residue_ratio
            - 0.10 * clamp01(replacement_char_ratio / 0.02)
            - 0.06 * clamp01(control_char_ratio / 0.02)
            - 0.12 * clamp01(len(violated_rules) / 3.0)
            - 0.03 * clamp01(len(warning_rules) / 4.0)
        )
        valid = not violated_rules
        return {
            "score": round(clamp01(score), 6),
            "valid": valid,
            "details": {
                "decision_scope": "structural_usability_only",
                "policy": "hard_usability_filter_v3",
                "allowed_signal_groups": [
                    "decodability",
                    "minimum_learnable_unit_length",
                    "language_fragment_hygiene",
                    "character_symbol_hygiene",
                    "markup_extraction_residue",
                    "hard_broken_repetition",
                ],
                "excluded_signal_groups": [
                    "semantic_quality",
                    "duplicate_detection",
                    "coverage_balance",
                    "utility_outcome",
                ],
                "word_count": word_count,
                "sentence_count": sent_count,
                "learnable_unit_pass": learnable_unit_pass,
                "alpha_ratio": round(alpha, 6),
                "repeated_token_ratio": round(repeated, 6),
                "lexical_diversity": round(lexical_diversity, 6),
                "symbol_ratio": round(symbol_ratio, 6),
                "line_break_ratio": round(line_break_ratio, 6),
                "markup_residue_ratio": round(markup_residue_ratio, 6),
                "control_char_ratio": round(control_char_ratio, 6),
                "replacement_char_ratio": round(replacement_char_ratio, 6),
                "max_repeated_char_run": int(max_char_run),
                "max_repeated_token_run": int(max_token_run),
                "style_bucket": style_bucket,
                "violated_rules": list(violated_rules),
                "warning_rules": list(warning_rules),
                "hard_rule_count": int(len(violated_rules)),
                "warning_rule_count": int(len(warning_rules)),
            },
        }

    def structural_validity_gate(self, text: str, validity: Dict[str, Any] | None = None) -> Dict[str, Any]:
        validity_payload = validity or self.structural_validity_score(text)
        passed = bool(validity_payload.get("valid"))
        details = dict(validity_payload.get("details") or {})
        details["soft_score"] = round(float(validity_payload.get("score") or 0.0), 6)
        details["passed"] = passed
        return {
            "score": 1.0 if passed else 0.0,
            "valid": passed,
            "details": details,
        }

    def explanatory_quality_proxy(self, text: str, validity: Dict[str, Any]) -> Dict[str, Any]:
        lowered = text.lower()
        vec = self._vector(text)
        pos_sim = _cosine(vec, self.quality_pos_unit)
        neg_sim = _cosine(vec, self.quality_neg_unit)
        return self._explanatory_quality_proxy_from_parts(text, lowered, validity, pos_sim, neg_sim)

    def _explanatory_quality_proxy_from_parts(
        self,
        text: str,
        lowered: str,
        validity: Dict[str, Any],
        pos_sim: float,
        neg_sim: float,
    ) -> Dict[str, Any]:
        words = len(text.split())
        sentences = max(sentence_count(text), 1)
        avg_sentence = words / sentences
        explanatory_hits = _count_contains(lowered, _EXPLANATION_MARKERS)
        explanatory_signal = clamp01(explanatory_hits / 4.0)
        definition_hits = _count_contains(lowered, _DEFINITION_PATTERNS)
        definition_signal = clamp01(definition_hits / 3.0)
        structure_signal = clamp01(
            (1.0 if "\n\n" in text else 0.0)
            + (1.0 if ":" in text else 0.0)
            + (1.0 if sentences >= 3 else 0.0)
        ) / 3.0
        cohesion_signal = clamp01(_count_contains(lowered, _COHESION_MARKERS) / 3.0)
        lexical_diversity = len(set(re.findall(r"\w+", lowered))) / max(words, 1)
        info_density = clamp01((1.0 - repeated_token_ratio(text)) * min(words / 180.0, 1.0))
        procedural_hits = _count_contains(lowered, _PROCEDURAL_MARKERS)
        procedural_penalty = clamp01(procedural_hits / 6.0)
        glossary_penalty = 1.0 if lowered.strip().startswith("glossary") or "glossary:" in lowered else 0.0
        conclusion_penalty = 1.0 if lowered.strip().startswith("conclusion") or lowered.strip().startswith("in conclusion") else 0.0
        bullet_penalty = clamp01((_bullet_ratio(text) - 0.35) * 2.0) if _bullet_ratio(text) > 0.35 else 0.0
        sentence_shape_penalty = clamp01(abs(avg_sentence - 18.0) / 18.0)
        raw = (
            1.8 * (pos_sim - neg_sim)
            + 0.8 * info_density
            + 0.35 * validity["score"]
            + 0.25 * explanatory_signal
            + 0.20 * definition_signal
            + 0.15 * structure_signal
            + 0.15 * cohesion_signal
            + 0.10 * clamp01(lexical_diversity / 0.65)
            - 0.40 * glossary_penalty
            - 0.30 * procedural_penalty
            - 0.20 * conclusion_penalty
            - 0.20 * bullet_penalty
            - 0.15 * sentence_shape_penalty
        )
        return {
            "score": round(clamp01(sigmoid(raw)), 6),
            "details": {
                "positive_similarity": round(pos_sim, 6),
                "negative_similarity": round(neg_sim, 6),
                "info_density": round(info_density, 6),
                "explanatory_signal": round(explanatory_signal, 6),
                "definition_signal": round(definition_signal, 6),
                "structure_signal": round(structure_signal, 6),
                "cohesion_signal": round(cohesion_signal, 6),
                "lexical_diversity": round(lexical_diversity, 6),
                "procedural_penalty": round(procedural_penalty, 6),
                "glossary_penalty": glossary_penalty,
                "conclusion_penalty": conclusion_penalty,
                "bullet_penalty": round(bullet_penalty, 6),
            },
        }

    def reference_quality_score(self, text: str) -> Dict[str, Any]:
        return self.reference_quality_model.score_text(text)

    def reference_quality_scores(self, texts: List[str]) -> List[Dict[str, Any]]:
        return self.reference_quality_model.score_texts(texts)

    def predictive_utility_proxy(
        self,
        text: str,
        quality: Dict[str, Any],
        validity: Dict[str, Any],
        *,
        exact_duplicate: Dict[str, Any] | None = None,
        near_duplicate_risk: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        lowered = text.lower()
        vec = self._vector(text)
        utility_sim = _cosine(vec, self.utility_ref_unit)
        negative_sim = _cosine(vec, self.utility_neg_unit)
        return self._predictive_utility_proxy_from_parts(
            text=text,
            lowered=lowered,
            quality=quality,
            validity=validity,
            utility_sim=utility_sim,
            negative_sim=negative_sim,
            exact_duplicate=exact_duplicate,
            near_duplicate_risk=near_duplicate_risk,
        )

    def _predictive_utility_proxy_from_parts(
        self,
        *,
        text: str,
        lowered: str,
        quality: Dict[str, Any],
        validity: Dict[str, Any],
        utility_sim: float,
        negative_sim: float,
        exact_duplicate: Dict[str, Any] | None,
        near_duplicate_risk: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        words = len(text.split())
        sentences = max(sentence_count(text), 1)
        concept_ratio = sum(1 for w in text.split() if len(w) > 6 and w.isalpha()) / max(words, 1)
        explanatory_hits = _count_contains(lowered, _EXPLANATION_MARKERS)
        explanatory_signal = clamp01(explanatory_hits / 4.0)
        definition_hits = _count_contains(lowered, _DEFINITION_PATTERNS)
        definition_signal = clamp01(definition_hits / 3.0)
        qa_hits = _count_contains(lowered, _QUESTION_MARKERS)
        qa_signal = clamp01(qa_hits / 4.0)
        procedural_hits = _count_contains(lowered, _PROCEDURAL_MARKERS)
        procedural_penalty = clamp01(procedural_hits / 6.0)
        glossary_penalty = 1.0 if lowered.strip().startswith("glossary") or "glossary:" in lowered else 0.0
        conclusion_penalty = 1.0 if lowered.strip().startswith("conclusion") or lowered.strip().startswith("in conclusion") else 0.0
        bullet_penalty = clamp01((_bullet_ratio(text) - 0.35) * 2.0) if _bullet_ratio(text) > 0.35 else 0.0
        list_density_penalty = clamp01((text.count(" - ") + text.count("\n- ")) / 8.0)
        exact_dup_penalty = clamp01(float((exact_duplicate or {}).get("score") or 0.0))
        near_dup_penalty = clamp01(float((near_duplicate_risk or {}).get("score") or 0.0))
        quality_support = clamp01(0.6 * quality["score"] + 0.4 * validity["score"])
        prototype_support = _normalize_similarity_margin(utility_sim, negative_sim)
        concept_signal = clamp01(concept_ratio * 4.0)
        features = utility_feature_vector(text, quality["score"], validity["score"])
        positive_support = (
            0.28 * explanatory_signal
            + 0.20 * definition_signal
            + 0.12 * qa_signal
            + 0.12 * concept_signal
            + 0.18 * quality_support
            + 0.10 * prototype_support
        )
        negative_pressure = (
            0.35 * procedural_penalty
            + 0.25 * glossary_penalty
            + 0.15 * conclusion_penalty
            + 0.15 * bullet_penalty
            + 0.10 * list_density_penalty
            + 0.30 * exact_dup_penalty
            + 0.20 * near_dup_penalty
        )
        heuristic_score = clamp01(positive_support - negative_pressure + 0.35)
        mode = self.utility_predictor_mode
        score = heuristic_score
        if self.utility_predictor is not None and self.utility_feature_names:
            x = np.array([[features[name] for name in self.utility_feature_names]], dtype=np.float32)
            score = clamp01(float(self.utility_predictor.predict(x)[0]))
            mode = "probe_calibrated"
        return {
            "score": round(score, 6),
            "details": {
                "mode": mode,
                "heuristic_score": round(heuristic_score, 6),
                "utility_similarity": round(utility_sim, 6),
                "negative_similarity": round(negative_sim, 6),
                "prototype_support": round(prototype_support, 6),
                "quality_support": round(quality_support, 6),
                "explanatory_signal": round(explanatory_signal, 6),
                "definition_signal": round(definition_signal, 6),
                "qa_signal": round(qa_signal, 6),
                "concept_signal": round(concept_signal, 6),
                "procedural_penalty": round(procedural_penalty, 6),
                "glossary_penalty": glossary_penalty,
                "conclusion_penalty": conclusion_penalty,
                "bullet_penalty": round(bullet_penalty, 6),
                "list_density_penalty": round(list_density_penalty, 6),
                "exact_duplicate_penalty": round(exact_dup_penalty, 6),
                "near_duplicate_penalty": round(near_dup_penalty, 6),
                "positive_support": round(positive_support, 6),
                "negative_pressure": round(negative_pressure, 6),
                "concept_ratio": round(concept_ratio, 6),
                "predictor_gate_enabled": bool(self.utility_predictor_gate.get("enabled")),
                "predictor_gate_reasons": list(self.utility_predictor_gate.get("reasons") or []),
                "feature_vector": features,
            },
        }

    def exact_duplicate_indicator(self, text_hash: str) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT count FROM hash_counts WHERE text_hash = ?",
            (text_hash,),
        ).fetchone()
        count = int(row[0]) if row else 1
        burden = clamp01(math.log1p(max(count - 1, 0)) / math.log(9.0))
        return {
            "score": 1.0 if count > 1 else 0.0,
            "details": {"duplicate_count": count, "duplicate_burden": round(burden, 6)},
        }

    def shingle_near_duplicate_indicator(
        self,
        chunk_uid: str,
        simhash_prefix_value: str,
    ) -> Dict[str, Any]:
        if self._active_verified_prefix != simhash_prefix_value:
            counts, risk_scores, jaccard_scores, containment_scores, meta = self._load_verified_near_dup_prefix(
                simhash_prefix_value
            )
            self._active_verified_prefix = simhash_prefix_value
            self._active_verified_counts = counts
            self._active_risk_scores = risk_scores
            self._active_risk_jaccard = jaccard_scores
            self._active_risk_containment = containment_scores
            self._active_verified_meta = meta
        verified_neighbors = int(self._active_verified_counts.get(chunk_uid, 0))
        near_score = 1.0 if verified_neighbors > 0 else 0.0
        return {
            "score": round(near_score, 6),
            "details": {
                "verified_neighbor_count": verified_neighbors,
                "simhash_prefix_bucket_count": int(self._active_verified_meta.get("simhash_prefix_bucket_count") or self._simhash_prefix_counts.get(simhash_prefix_value, 1)),
                "verified_pair_count": int(self._active_verified_meta.get("verified_pair_count") or 0),
                "subgroup_count": int(self._active_verified_meta.get("subgroup_count") or 0),
                "truncated_subgroup_count": int(self._active_verified_meta.get("truncated_subgroup_count") or 0),
                "shingle_size_policy": "adaptive_1_to_3",
                "jaccard_threshold": NEAR_DUP_JACCARD_THRESHOLD,
                "containment_threshold": NEAR_DUP_CONTAINMENT_THRESHOLD,
                "hamming_threshold": NEAR_DUP_HAMMING_THRESHOLD,
            },
        }

    def shingle_near_duplicate_risk_score(
        self,
        chunk_uid: str,
        simhash_prefix_value: str,
        text: str | None = None,
    ) -> Dict[str, Any]:
        if self._active_verified_prefix != simhash_prefix_value:
            counts, risk_scores, jaccard_scores, containment_scores, meta = self._load_verified_near_dup_prefix(
                simhash_prefix_value
            )
            self._active_verified_prefix = simhash_prefix_value
            self._active_verified_counts = counts
            self._active_risk_scores = risk_scores
            self._active_risk_jaccard = jaccard_scores
            self._active_risk_containment = containment_scores
            self._active_verified_meta = meta
        overlap_only_score = float(self._active_risk_scores.get(chunk_uid, 0.0))
        max_jaccard = float(self._active_risk_jaccard.get(chunk_uid, 0.0))
        max_containment = float(self._active_risk_containment.get(chunk_uid, 0.0))
        prefix_bucket_count = int(
            self._active_verified_meta.get("simhash_prefix_bucket_count")
            or self._simhash_prefix_counts.get(simhash_prefix_value, 1)
        )
        prefix_pressure = clamp01(
            math.log1p(max(prefix_bucket_count - 1, 0))
            / math.log(NEAR_DUP_PREFIX_PRESSURE_LOG_BASE)
        )
        style_bucket = _style_bucket_from_text(text or "") if text is not None else "general_prose"
        repeat_divisor = 0.55
        prefix_pressure_factor = 1.0
        unverified_cap = NEAR_DUP_UNVERIFIED_MAX
        if overlap_only_score <= 0.0 and text is not None:
            if style_bucket == "instructional":
                prefix_pressure_factor = 0.72
                repeat_divisor = 0.75
                unverified_cap = 0.24
            elif style_bucket == "structured_list":
                prefix_pressure_factor = 0.68
                repeat_divisor = 0.80
                unverified_cap = 0.22
            elif style_bucket == "technical_reference":
                prefix_pressure_factor = 0.65
                repeat_divisor = 0.85
                unverified_cap = 0.20
        prefix_pressure = clamp01(prefix_pressure * prefix_pressure_factor)
        raw_repeated_ratio = repeated_token_ratio(text or "") if text is not None else 0.0
        repeat_pressure = clamp01(raw_repeated_ratio / repeat_divisor) if text is not None else 0.0
        lowered = str(text or "").lower()
        useful_marker_hits = _count_contains(
            lowered,
            _EXPLANATION_MARKERS
            + _DEFINITION_PATTERNS
            + (
                "example",
                "exercise",
                "definition",
                "theorem",
                "proof",
                "syntax",
                "parameter",
                "returns",
            ),
        )
        useful_style_prior = 1.0 if style_bucket in {"instructional", "technical_reference", "structured_list"} else 0.35
        useful_repeat_window = 1.0 if 0.08 <= raw_repeated_ratio <= 0.55 else 0.0
        useful_recurrence_score = clamp01(
            useful_style_prior
            * useful_repeat_window
            * (0.35 + 0.18 * min(useful_marker_hits, 3))
            * (1.0 - min(overlap_only_score, 0.85))
        )
        blended_risk = clamp01(
            NEAR_DUP_VERIFIED_BLEND_WEIGHT * overlap_only_score
            + NEAR_DUP_PREFIX_BLEND_WEIGHT * prefix_pressure
            + NEAR_DUP_REPEAT_BLEND_WEIGHT * repeat_pressure
        )
        harmful_redundancy_risk = clamp01(blended_risk - (0.16 * useful_recurrence_score))
        if overlap_only_score <= 0.0:
            harmful_redundancy_risk = min(harmful_redundancy_risk, unverified_cap)
        final_risk = clamp01(max(overlap_only_score, harmful_redundancy_risk))
        return {
            "score": round(final_risk, 6),
            "details": {
                "max_shingle_overlap": round(overlap_only_score, 6),
                "max_shingle_jaccard": round(max_jaccard, 6),
                "max_shingle_containment": round(max_containment, 6),
                "simhash_prefix_bucket_count": prefix_bucket_count,
                "verified_pair_count": int(self._active_verified_meta.get("verified_pair_count") or 0),
                "subgroup_count": int(self._active_verified_meta.get("subgroup_count") or 0),
                "truncated_subgroup_count": int(self._active_verified_meta.get("truncated_subgroup_count") or 0),
                "prefix_collision_pressure": round(prefix_pressure, 6),
                "intra_chunk_repeat_pressure": round(repeat_pressure, 6),
                "blended_risk_score": round(blended_risk, 6),
                "harmful_redundancy_risk": round(harmful_redundancy_risk, 6),
                "useful_recurrence_score": round(useful_recurrence_score, 6),
                "useful_recurrence_marker_hits": int(useful_marker_hits),
                "redundancy_policy": "harmful_redundancy_minus_useful_recurrence_v1",
                "overlap_only_score": round(overlap_only_score, 6),
                "style_bucket": style_bucket,
                "prefix_pressure_factor": round(prefix_pressure_factor, 6),
                "repeat_pressure_divisor": round(repeat_divisor, 6),
                "unverified_risk_cap": round(unverified_cap, 6),
                "shingle_size_policy": "adaptive_1_to_3",
            },
        }

    def tail_cluster_rarity_proxy(self, cluster_size: int, dataset: str | None = None) -> Dict[str, Any]:
        rarity = 1.0 / math.sqrt(max(cluster_size, 1))
        scaled = clamp01(rarity * math.sqrt(max(self.total_chunks, 1)) / 10.0)
        dataset_rarity = None
        if dataset:
            dataset_rarity = (self._dataset_cluster_size_rarity.get(str(dataset)) or {}).get(int(cluster_size))
        if dataset_rarity is None:
            score = scaled
        else:
            score = clamp01(0.7 * float(dataset_rarity) + 0.3 * scaled)
        return {
            "score": round(score, 6),
            "details": {
                "cluster_size": int(cluster_size),
                "dataset": str(dataset) if dataset is not None else None,
                "dataset_relative_rarity": round(float(dataset_rarity), 6) if dataset_rarity is not None else None,
                "global_inverse_sqrt_rarity": round(scaled, 6),
            },
        }

    def score_chunk(self, chunk_meta: Dict[str, Any]) -> Dict[str, Any]:
        validity = self.structural_validity_score(chunk_meta["text"])
        reference_quality = self._calibrate_reference_quality(
            self.reference_quality_score(chunk_meta["text"]),
            validity,
            text=chunk_meta["text"],
        )
        quality = self.explanatory_quality_proxy(chunk_meta["text"], validity)
        exact = self.exact_duplicate_indicator(chunk_meta["text_hash"])
        near = self.shingle_near_duplicate_indicator(
            chunk_meta["chunk_uid"],
            chunk_meta["simhash_prefix"],
        )
        near_risk = self.shingle_near_duplicate_risk_score(
            chunk_meta["chunk_uid"],
            chunk_meta["simhash_prefix"],
            text=chunk_meta["text"],
        )
        utility = self.predictive_utility_proxy(
            chunk_meta["text"],
            quality,
            validity,
            exact_duplicate=exact,
            near_duplicate_risk=near_risk,
        )
        coverage = self.tail_cluster_rarity_proxy(int(chunk_meta["cluster_size"]), dataset=str(chunk_meta.get("dataset") or ""))
        return {
            "structural_validity_gate": self.structural_validity_gate(chunk_meta["text"], validity),
            "structural_validity_score": validity,
            "reference_quality_score": reference_quality,
            "exact_duplicate_indicator": exact,
            "shingle_near_duplicate_indicator": near,
            "shingle_near_duplicate_risk_score": near_risk,
            "explanatory_quality_proxy": quality,
            "tail_cluster_rarity_proxy": coverage,
            "predictive_utility_proxy": utility,
        }

    def score_chunks(self, chunk_metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunk_metas:
            return []
        texts = [chunk_meta["text"] for chunk_meta in chunk_metas]
        reference_quality_scores = self.reference_quality_scores(texts)
        mat = self._vector_matrix(texts)
        quality_pos_sims = np.asarray(mat @ self.quality_pos_unit).ravel()
        quality_neg_sims = np.asarray(mat @ self.quality_neg_unit).ravel()
        utility_pos_sims = np.asarray(mat @ self.utility_ref_unit).ravel()
        utility_neg_sims = np.asarray(mat @ self.utility_neg_unit).ravel()
        results: List[Dict[str, Any]] = []
        for idx, (chunk_meta, reference_quality) in enumerate(zip(chunk_metas, reference_quality_scores)):
            validity = self.structural_validity_score(chunk_meta["text"])
            validity_gate = self.structural_validity_gate(chunk_meta["text"], validity)
            calibrated_reference_quality = self._calibrate_reference_quality(
                reference_quality,
                validity,
                text=chunk_meta["text"],
            )
            lowered = chunk_meta["text"].lower()
            quality = self._explanatory_quality_proxy_from_parts(
                chunk_meta["text"],
                lowered,
                validity,
                float(quality_pos_sims[idx]),
                float(quality_neg_sims[idx]),
            )
            exact = self.exact_duplicate_indicator(chunk_meta["text_hash"])
            near = self.shingle_near_duplicate_indicator(
                chunk_meta["chunk_uid"],
                chunk_meta["simhash_prefix"],
            )
            near_risk = self.shingle_near_duplicate_risk_score(
                chunk_meta["chunk_uid"],
                chunk_meta["simhash_prefix"],
                text=chunk_meta["text"],
            )
            utility = self._predictive_utility_proxy_from_parts(
                text=chunk_meta["text"],
                lowered=lowered,
                quality=quality,
                validity=validity,
                utility_sim=float(utility_pos_sims[idx]),
                negative_sim=float(utility_neg_sims[idx]),
                exact_duplicate=exact,
                near_duplicate_risk=near_risk,
            )
            coverage = self.tail_cluster_rarity_proxy(
                int(chunk_meta["cluster_size"]),
                dataset=str(chunk_meta.get("dataset") or ""),
            )
            results.append(
                {
                    "structural_validity_gate": validity_gate,
                    "structural_validity_score": validity,
                    "reference_quality_score": calibrated_reference_quality,
                    "exact_duplicate_indicator": exact,
                    "shingle_near_duplicate_indicator": near,
                    "shingle_near_duplicate_risk_score": near_risk,
                    "explanatory_quality_proxy": quality,
                    "tail_cluster_rarity_proxy": coverage,
                    "predictive_utility_proxy": utility,
                }
            )
        return results
