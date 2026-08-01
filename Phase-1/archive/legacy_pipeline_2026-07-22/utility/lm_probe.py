#!/usr/bin/env python3
"""Small LM probe based on fixed-budget held-out causal LM loss.

The probe uses order-invariant hash sampling for train/eval selection and a
very small causal LM (`sshleifer/tiny-gpt2`) for short controlled finetuning.
"""

from __future__ import annotations

import copy
import heapq
import hashlib
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

from data_eval_common import clamp01


DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
_DEFAULT_MAX_LENGTH = 128
_DEFAULT_TRAIN_BATCH_SIZE = 4
_DEFAULT_LEARNING_RATE = 5e-5
_DEFAULT_MAX_TRAIN_STEPS = 48

_TOKENIZER_CACHE: dict[str, Any] = {}
_BASE_MODEL_CACHE: dict[str, Any] = {}
_BASELINE_SEQUENCE_CACHE: dict[Tuple[Any, ...], tuple[List[List[int]], int, int, float]] = {}
_EVAL_SEQUENCE_CACHE: dict[Tuple[Any, ...], tuple[List[List[int]], int]] = {}
_SELECTED_EVAL_CACHE: dict[Tuple[Any, ...], Tuple[float, float, np.ndarray, Dict[str, Any]]] = {}

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


@dataclass
class SmallLMProbeContext:
    baseline_variant: str
    dataset: str
    eval_dataset: str
    model_name: str
    train_token_budget: int
    eval_token_budget: int
    holdout_modulo: int
    holdout_bucket: int
    sampling_hash_seed: int
    baseline_sampling_ratio: float
    baseline_sequences: List[List[int]]
    baseline_train_tokens: int
    baseline_non_holdout_tokens: int
    eval_sequences: List[List[int]]
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    max_train_steps: int
    train_epochs: float
    train_audit_token_budget: int


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:12], 16) % modulo


def _stable_unit(value: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _get_tokenizer(model_name: str):
    tokenizer = _TOKENIZER_CACHE.get(model_name)
    if tokenizer is not None:
        return tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def _get_base_model(model_name: str):
    model = _BASE_MODEL_CACHE.get(model_name)
    if model is not None:
        return model
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.eval()
    model.cpu()
    _BASE_MODEL_CACHE[model_name] = model
    return model


def _encode_text(text: str, *, tokenizer, max_length: int) -> List[int]:
    encoded = tokenizer(
        str(text or ""),
        truncation=True,
        max_length=int(max_length),
        padding=False,
        add_special_tokens=True,
        return_attention_mask=False,
        return_tensors=None,
    )
    return list(encoded["input_ids"])


def _eligible_sequences_from_query(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    baseline_allowed_uids: set[str] | None,
    holdout_modulo: int,
    holdout_bucket: int,
    tokenizer,
    max_length: int,
    mode: str,
) -> List[tuple[str, List[int]]]:
    rows = conn.execute(
        "SELECT chunk_uid, text FROM chunks WHERE dataset = ? ORDER BY chunk_uid",
        (dataset,),
    )
    out: List[tuple[str, List[int]]] = []
    want_holdout = mode == "holdout"
    for chunk_uid, text in rows:
        uid = str(chunk_uid)
        if baseline_allowed_uids is not None and uid not in baseline_allowed_uids:
            continue
        in_holdout = _stable_bucket(uid, holdout_modulo) == holdout_bucket
        if want_holdout != in_holdout:
            continue
        token_ids = _encode_text(str(text), tokenizer=tokenizer, max_length=max_length)
        if len(token_ids) < 2:
            continue
        out.append((uid, token_ids))
    return out


def _sample_by_ratio(
    rows: Sequence[tuple[str, List[int]]],
    *,
    ratio: float,
    seed: int,
) -> tuple[List[List[int]], int]:
    selected: List[List[int]] = []
    total_tokens = 0
    for uid, token_ids in rows:
        if _stable_unit(uid, seed) > ratio:
            continue
        selected.append(token_ids)
        total_tokens += len(token_ids)
    return selected, total_tokens


def _sample_by_token_budget(
    rows: Sequence[tuple[str, List[int]]],
    *,
    token_budget: int,
    seed: int,
) -> tuple[List[List[int]], int, float]:
    if not rows:
        return [], 0, 0.0
    budget = max(1, int(token_budget))
    total_available = sum(len(token_ids) for _, token_ids in rows)
    ordered = sorted(rows, key=lambda item: (_stable_unit(str(item[0]), seed), str(item[0])))
    selected: List[List[int]] = []
    total_tokens = 0
    for _, token_ids in ordered:
        token_len = len(token_ids)
        if token_len < 2:
            continue
        if total_tokens > 0 and total_tokens + token_len > budget:
            continue
        selected.append(token_ids)
        total_tokens += token_len
        if total_tokens >= budget:
            break
    if not selected:
        uid, token_ids = ordered[0]
        selected = [token_ids]
        total_tokens = len(token_ids)
    sampling_ratio = min(1.0, float(total_tokens) / float(max(1, total_available)))
    return selected, total_tokens, sampling_ratio


def _add_budget_candidate(
    heap: List[tuple[float, str, List[int]]],
    *,
    current_tokens: int,
    token_budget: int,
    uid: str,
    token_ids: List[int],
    seed: int,
) -> int:
    score = _stable_unit(uid, seed)
    heapq.heappush(heap, (-score, uid, token_ids))
    current_tokens += len(token_ids)
    budget = max(1, int(token_budget))
    while current_tokens > budget and len(heap) > 1:
        _, _, removed = heapq.heappop(heap)
        current_tokens -= len(removed)
    return current_tokens


def _add_proxy_candidate(
    heap: List[tuple[float, str, str, int]],
    *,
    current_proxy_tokens: int,
    proxy_budget: int,
    uid: str,
    text: str,
    proxy_tokens: int,
    seed: int,
) -> int:
    score = _stable_unit(uid, seed)
    heapq.heappush(heap, (-score, uid, text, int(proxy_tokens)))
    current_proxy_tokens += int(proxy_tokens)
    budget = max(1, int(proxy_budget))
    while current_proxy_tokens > budget and len(heap) > 1:
        _, _, _, removed_proxy_tokens = heapq.heappop(heap)
        current_proxy_tokens -= int(removed_proxy_tokens)
    return current_proxy_tokens


def _proxy_token_count_from_text(text: str, word_count: int | None = None) -> int:
    if word_count is not None and int(word_count) > 0:
        return max(2, int(round(float(word_count) * 1.35)) + 4)
    raw = str(text or "")
    return max(2, int(round(len(raw) / 4.0)) + 4)


def _finalize_budget_sample(
    heap: List[tuple[float, str, List[int]]],
    *,
    sampled_tokens: int,
    total_available_tokens: int,
) -> tuple[List[List[int]], int, float]:
    ordered = sorted(heap, key=lambda item: (-item[0], item[1]))
    sequences = [token_ids for _, _, token_ids in ordered]
    sampling_ratio = min(1.0, float(sampled_tokens) / float(max(1, total_available_tokens)))
    return sequences, int(sampled_tokens), float(sampling_ratio)


def _tokenize_budget_candidates(
    candidates: Sequence[tuple[float, str, str, int]],
    *,
    tokenizer,
    max_length: int,
    token_budget: int,
    sampling_hash_seed: int,
) -> tuple[List[List[int]], int, int, float]:
    heap: List[tuple[float, str, List[int]]] = []
    sampled_tokens = 0
    total_candidate_tokens = 0
    fallback: tuple[str, List[int]] | None = None
    ordered_candidates = sorted(candidates, key=lambda item: (-item[0], item[1]))
    for _, uid, text, _ in ordered_candidates:
        token_ids = _encode_text(str(text), tokenizer=tokenizer, max_length=max_length)
        if len(token_ids) < 2:
            continue
        if fallback is None:
            fallback = (uid, token_ids)
        total_candidate_tokens += len(token_ids)
        sampled_tokens = _add_budget_candidate(
            heap,
            current_tokens=sampled_tokens,
            token_budget=int(token_budget),
            uid=str(uid),
            token_ids=token_ids,
            seed=int(sampling_hash_seed),
        )
    if not heap and fallback is not None:
        _, token_ids = fallback
        heap.append((-1.0, fallback[0], token_ids))
        sampled_tokens = len(token_ids)
    return _finalize_budget_sample(
        heap,
        sampled_tokens=sampled_tokens,
        total_available_tokens=max(total_candidate_tokens, sampled_tokens),
    )


def _collect_eval_sequences_from_query(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    holdout_modulo: int,
    holdout_bucket: int,
    tokenizer,
    max_length: int,
    eval_token_budget: int,
    sampling_hash_seed: int,
) -> tuple[List[List[int]], int]:
    rows = conn.execute(
        "SELECT chunk_uid, text, word_count FROM chunks WHERE dataset = ? ORDER BY chunk_uid",
        (dataset,),
    )
    proxy_heap: List[tuple[float, str, str, int]] = []
    proxy_tokens = 0
    proxy_budget = max(int(eval_token_budget) * 4, int(eval_token_budget) + int(max_length) * 16)
    for chunk_uid, text, word_count in rows:
        uid = str(chunk_uid)
        if _stable_bucket(uid, holdout_modulo) != holdout_bucket:
            continue
        row_proxy_tokens = _proxy_token_count_from_text(str(text), int(word_count or 0))
        proxy_tokens = _add_proxy_candidate(
            proxy_heap,
            current_proxy_tokens=proxy_tokens,
            proxy_budget=proxy_budget,
            uid=uid,
            text=str(text),
            proxy_tokens=row_proxy_tokens,
            seed=int(sampling_hash_seed),
        )
    sequences, sampled_tokens, _ = _tokenize_budget_candidates(
        proxy_heap,
        tokenizer=tokenizer,
        max_length=int(max_length),
        token_budget=int(eval_token_budget),
        sampling_hash_seed=int(sampling_hash_seed),
    )
    return sequences, int(sampled_tokens)


def _sample_train_sequences_from_query(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    baseline_allowed_uids: set[str] | None,
    holdout_modulo: int,
    holdout_bucket: int,
    tokenizer,
    max_length: int,
    token_budget: int,
    sampling_hash_seed: int,
) -> tuple[List[List[int]], int, int, float]:
    rows = conn.execute(
        "SELECT chunk_uid, text, word_count FROM chunks WHERE dataset = ? ORDER BY chunk_uid",
        (dataset,),
    )
    proxy_heap: List[tuple[float, str, str, int]] = []
    proxy_tokens = 0
    total_available_proxy_tokens = 0
    proxy_budget = max(int(token_budget) * 4, int(token_budget) + int(max_length) * 16)
    for chunk_uid, text, word_count in rows:
        uid = str(chunk_uid)
        if baseline_allowed_uids is not None and uid not in baseline_allowed_uids:
            continue
        if _stable_bucket(uid, holdout_modulo) == holdout_bucket:
            continue
        row_proxy_tokens = _proxy_token_count_from_text(str(text), int(word_count or 0))
        total_available_proxy_tokens += row_proxy_tokens
        proxy_tokens = _add_proxy_candidate(
            proxy_heap,
            current_proxy_tokens=proxy_tokens,
            proxy_budget=proxy_budget,
            uid=uid,
            text=str(text),
            proxy_tokens=row_proxy_tokens,
            seed=int(sampling_hash_seed),
        )
    sequences, sampled_tokens, candidate_sampling_ratio = _tokenize_budget_candidates(
        proxy_heap,
        tokenizer=tokenizer,
        max_length=int(max_length),
        token_budget=int(token_budget),
        sampling_hash_seed=int(sampling_hash_seed),
    )
    corpus_sampling_ratio = min(1.0, float(sampled_tokens) / float(max(1, total_available_proxy_tokens)))
    return sequences, int(sampled_tokens), int(total_available_proxy_tokens), float(corpus_sampling_ratio or candidate_sampling_ratio)


def _collect_selected_sequences(
    selected_records: Iterable[tuple[str, str]],
    *,
    tokenizer,
    max_length: int,
    holdout_modulo: int,
    holdout_bucket: int,
    train_token_budget: int,
    sampling_hash_seed: int,
) -> tuple[List[List[int]], int, int, float]:
    materialized = list(selected_records) if not isinstance(selected_records, list) else selected_records
    proxy_heap: List[tuple[float, str, str, int]] = []
    proxy_tokens = 0
    selected_non_holdout_proxy_tokens = 0
    proxy_budget = max(int(train_token_budget) * 4, int(train_token_budget) + int(max_length) * 16)
    for chunk_uid, text in materialized:
        uid = str(chunk_uid)
        if _stable_bucket(uid, holdout_modulo) == holdout_bucket:
            continue
        row_proxy_tokens = _proxy_token_count_from_text(str(text))
        selected_non_holdout_proxy_tokens += row_proxy_tokens
        proxy_tokens = _add_proxy_candidate(
            proxy_heap,
            current_proxy_tokens=proxy_tokens,
            proxy_budget=proxy_budget,
            uid=uid,
            text=str(text),
            proxy_tokens=row_proxy_tokens,
            seed=int(sampling_hash_seed),
        )
    sequences, sampled_tokens, candidate_ratio = _tokenize_budget_candidates(
        proxy_heap,
        tokenizer=tokenizer,
        max_length=int(max_length),
        token_budget=int(train_token_budget),
        sampling_hash_seed=int(sampling_hash_seed),
    )
    ratio = min(1.0, float(sampled_tokens) / float(max(1, selected_non_holdout_proxy_tokens)))
    return sequences, sampled_tokens, selected_non_holdout_proxy_tokens, float(ratio or candidate_ratio)


def _selected_sequence_cache_key(
    *,
    selected_fingerprint: str,
    model_name: str,
    max_length: int,
    holdout_modulo: int,
    holdout_bucket: int,
    train_token_budget: int,
    sampling_hash_seed: int,
) -> Tuple[Any, ...]:
    return (
        str(selected_fingerprint),
        str(model_name),
        int(max_length),
        int(holdout_modulo),
        int(holdout_bucket),
        int(train_token_budget),
        int(sampling_hash_seed),
    )


def _select_trainable_params(model) -> List[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False
    n_layer = getattr(getattr(model, "config", None), "n_layer", None)
    if n_layer is None:
        selected = [p for _, p in list(model.named_parameters())[-8:]]
        for param in selected:
            param.requires_grad = True
        return selected
    last_block = f"transformer.h.{n_layer - 1}."
    trainable: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if name.startswith(last_block) or name.startswith("transformer.ln_f") or name.startswith("lm_head"):
            param.requires_grad = True
            trainable.append(param)
    if not trainable:
        trainable = [p for _, p in list(model.named_parameters())[-8:]]
        for param in trainable:
            param.requires_grad = True
    return trainable


def _pad_batch(sequences: Sequence[Sequence[int]], pad_token_id: int, device: str) -> dict[str, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    input_ids = []
    attention_mask = []
    for seq in sequences:
        padded = list(seq) + [pad_token_id] * (max_len - len(seq))
        input_ids.append(padded)
        attention_mask.append([1] * len(seq) + [0] * (max_len - len(seq)))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def _train_model(
    sequences: Sequence[Sequence[int]],
    *,
    model_name: str,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    max_train_steps: int,
    train_epochs: float,
    seed: int,
    audit_sequences: Sequence[Sequence[int]] | None = None,
) -> tuple[Any, str, Any, Dict[str, Any]]:
    if not sequences:
        raise RuntimeError("Probe training set is empty.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device()
    tokenizer = _get_tokenizer(model_name)
    model = copy.deepcopy(_get_base_model(model_name))
    model.to(device)
    audit_pre_nll = None
    audit_pre_ppl = None
    audit_post_nll = None
    audit_post_ppl = None
    audit_token_count = 0
    audit_sequence_count = 0
    if audit_sequences:
        model.eval()
        audit_pre_nll, audit_pre_ppl, _, audit_token_counts = _eval_model(
            model,
            device,
            tokenizer,
            audit_sequences,
            eval_batch_size=eval_batch_size,
        )
        audit_token_count = int(np.sum(audit_token_counts))
        audit_sequence_count = int(len(audit_sequences))
    model.train()
    trainable_params = _select_trainable_params(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=float(learning_rate))

    order = list(range(len(sequences)))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    cursor = 0
    batch_size = max(1, int(train_batch_size))
    total_tokens = sum(len(seq) for seq in sequences)
    average_sequence_tokens = float(total_tokens) / float(max(len(sequences), 1))
    estimated_tokens_per_step = max(1.0, float(batch_size) * average_sequence_tokens)
    one_epoch_steps = max(1, math.ceil(float(total_tokens) / estimated_tokens_per_step))
    target_exposure = max(1.0, float(train_epochs))
    target_steps = max(1, math.ceil(float(one_epoch_steps) * target_exposure))
    planned_steps = max(1, min(int(max_train_steps), int(target_steps)))
    seen_tokens = 0

    for _ in range(planned_steps):
        batch_indices = []
        for _ in range(batch_size):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            batch_indices.append(order[cursor])
            cursor += 1
        batch_sequences = [sequences[idx] for idx in batch_indices]
        seen_tokens += sum(len(seq) for seq in batch_sequences)
        batch = _pad_batch(batch_sequences, tokenizer.pad_token_id, device)
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    if audit_sequences:
        audit_post_nll, audit_post_ppl, _, _ = _eval_model(
            model,
            device,
            tokenizer,
            audit_sequences,
            eval_batch_size=eval_batch_size,
        )
    train_stats = {
        "effective_train_steps": int(planned_steps),
        "max_train_steps": int(max_train_steps),
        "target_train_steps": int(target_steps),
        "one_epoch_train_steps": int(one_epoch_steps),
        "train_epochs": round(float(target_exposure), 6),
        "target_train_exposure_ratio": round(float(target_exposure), 6),
        "train_sequence_count": int(len(sequences)),
        "train_tokens": int(total_tokens),
        "estimated_seen_train_tokens": int(seen_tokens),
        "train_token_exposure_ratio": round(float(seen_tokens) / float(max(total_tokens, 1)), 6),
        "average_sequence_tokens": round(float(average_sequence_tokens), 6),
        "estimated_tokens_per_step": round(float(estimated_tokens_per_step), 6),
        "step_cap_reached": bool(int(planned_steps) < int(target_steps)),
        "train_audit_sequence_count": int(audit_sequence_count),
        "train_audit_tokens": int(audit_token_count),
    }
    if audit_pre_nll is not None and audit_post_nll is not None:
        audit_delta = float(audit_pre_nll) - float(audit_post_nll)
        train_stats.update(
            {
                "train_audit_pre_nll": round(float(audit_pre_nll), 6),
                "train_audit_post_nll": round(float(audit_post_nll), 6),
                "train_audit_delta_nll": round(float(audit_delta), 6),
                "train_audit_relative_gain": round(float(audit_delta) / float(audit_pre_nll), 6)
                if float(audit_pre_nll) > 0.0
                else 0.0,
                "train_audit_pre_perplexity": round(float(audit_pre_ppl), 6),
                "train_audit_post_perplexity": round(float(audit_post_ppl), 6),
            }
        )
    return model, device, tokenizer, train_stats


def _eval_model(
    model,
    device: str,
    tokenizer,
    eval_sequences: Sequence[Sequence[int]],
    *,
    eval_batch_size: int = 1,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    loss_sums: List[float] = []
    token_lengths: List[int] = []
    total_loss = 0.0
    total_tokens = 0
    batch_size = max(1, int(eval_batch_size))
    with torch.inference_mode():
        filtered = [list(token_ids) for token_ids in eval_sequences if len(token_ids) >= 2]
        for i in range(0, len(filtered), batch_size):
            batch_sequences = filtered[i : i + batch_size]
            batch = _pad_batch(batch_sequences, tokenizer.pad_token_id, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()
            shift_mask = batch["attention_mask"][:, 1:].to(dtype=torch.float32).contiguous()
            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)
            per_doc_loss = (token_loss * shift_mask).sum(dim=1)
            per_doc_tokens = shift_mask.sum(dim=1).clamp_min(1.0)
            for doc_loss_tensor, token_count_tensor in zip(per_doc_loss, per_doc_tokens):
                doc_loss = float(doc_loss_tensor.detach().cpu())
                token_count = float(token_count_tensor.detach().cpu())
                loss_sums.append(doc_loss)
                token_lengths.append(int(token_count))
                total_loss += doc_loss
                total_tokens += int(token_count)
    if total_tokens <= 0:
        raise RuntimeError("No held-out evaluation tokens available for the small LM probe.")
    mean_nll = total_loss / total_tokens
    perplexity = math.exp(mean_nll)
    return mean_nll, perplexity, np.asarray(loss_sums, dtype=np.float64), np.asarray(token_lengths, dtype=np.float64)


def _sample_audit_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    token_budget: int,
    seed: int,
) -> tuple[List[List[int]], int]:
    """Deterministically sample a small train slice for before/after learning diagnostics."""
    budget = max(0, int(token_budget))
    if budget <= 0 or not sequences:
        return [], 0
    keyed: List[tuple[float, int, Sequence[int]]] = []
    for idx, seq in enumerate(sequences):
        if len(seq) < 2:
            continue
        keyed.append((_stable_unit(str(idx), seed), idx, seq))
    keyed.sort(key=lambda item: (item[0], item[1]))
    sampled: List[List[int]] = []
    total_tokens = 0
    for _, _, seq in keyed:
        token_len = len(seq)
        if sampled and total_tokens + token_len > budget:
            continue
        sampled.append(list(seq))
        total_tokens += token_len
        if total_tokens >= budget:
            break
    if not sampled and keyed:
        sampled = [list(keyed[0][2])]
        total_tokens = len(sampled[0])
    return sampled, int(total_tokens)


def _utility_causal_failure_mode(
    *,
    delta_nll: float,
    selected_train_delta: float,
    baseline_train_delta: float,
    mde: float,
) -> str:
    """Classify the local causal-style explanation for one probe run."""
    tolerance = max(1e-5, float(mde) * 0.25)
    train_gap = float(selected_train_delta) - float(baseline_train_delta)
    both_weak = max(float(selected_train_delta), float(baseline_train_delta)) <= tolerance
    if abs(float(delta_nll)) <= max(float(mde), tolerance):
        return "inconclusive_near_noise_floor"
    if both_weak:
        return "probe_or_training_insensitive"
    if float(delta_nll) < -tolerance and train_gap < -tolerance:
        return "weaker_selected_training_signal"
    if float(delta_nll) < -tolerance and train_gap >= -tolerance:
        return "overfit_or_distribution_shift"
    if float(delta_nll) > tolerance:
        return "positive_learning_signal"
    return "unresolved"


def build_probe_context(
    conn: sqlite3.Connection,
    *,
    baseline_variant: str = "baseline_full_random",
    baseline_allowed_uids: set[str] | None = None,
    baseline_uid_fingerprint: str = "all",
    dataset: str,
    eval_dataset: str | None = None,
    token_budget: int,
    eval_token_budget: int,
    holdout_modulo: int = 17,
    holdout_bucket: int = 0,
    sampling_hash_seed: int = 42,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = _DEFAULT_MAX_LENGTH,
    train_batch_size: int = _DEFAULT_TRAIN_BATCH_SIZE,
    eval_batch_size: int = _DEFAULT_TRAIN_BATCH_SIZE,
    learning_rate: float = _DEFAULT_LEARNING_RATE,
    max_train_steps: int = _DEFAULT_MAX_TRAIN_STEPS,
    train_epochs: float = 1.0,
    train_audit_token_budget: int = 4096,
    **_: Any,
) -> SmallLMProbeContext:
    tokenizer = _get_tokenizer(model_name)
    eval_dataset_name = str(eval_dataset or dataset)

    eval_cache_key = (
        str(eval_dataset_name),
        int(eval_token_budget),
        int(holdout_modulo),
        int(holdout_bucket),
        int(sampling_hash_seed),
        str(model_name),
        int(max_length),
    )
    cached_eval = _EVAL_SEQUENCE_CACHE.get(eval_cache_key)
    if cached_eval is not None:
        eval_sequences, eval_tokens = cached_eval
    else:
        eval_sequences, eval_tokens = _collect_eval_sequences_from_query(
            conn,
            dataset=eval_dataset_name,
            holdout_modulo=int(holdout_modulo),
            holdout_bucket=int(holdout_bucket),
            tokenizer=tokenizer,
            max_length=int(max_length),
            eval_token_budget=int(eval_token_budget),
            sampling_hash_seed=int(sampling_hash_seed),
        )
        _EVAL_SEQUENCE_CACHE[eval_cache_key] = (eval_sequences, int(eval_tokens))

    baseline_cache_key = (
        str(baseline_uid_fingerprint or "all"),
        str(dataset),
        int(token_budget),
        int(holdout_modulo),
        int(holdout_bucket),
        int(sampling_hash_seed),
        str(model_name),
        int(max_length),
    )
    cached_baseline = _BASELINE_SEQUENCE_CACHE.get(baseline_cache_key)
    if cached_baseline is not None:
        baseline_sequences, baseline_token_count, baseline_non_holdout_tokens, sampling_ratio = cached_baseline
    else:
        (
            baseline_sequences,
            baseline_token_count,
            baseline_non_holdout_tokens,
            sampling_ratio,
        ) = _sample_train_sequences_from_query(
            conn,
            dataset=str(dataset),
            baseline_allowed_uids=baseline_allowed_uids,
            holdout_modulo=int(holdout_modulo),
            holdout_bucket=int(holdout_bucket),
            tokenizer=tokenizer,
            max_length=int(max_length),
            token_budget=int(token_budget),
            sampling_hash_seed=int(sampling_hash_seed),
        )
        if baseline_non_holdout_tokens <= 0:
            raise RuntimeError(f"{dataset}: unable to build baseline small-LM probe corpus.")
        _BASELINE_SEQUENCE_CACHE[baseline_cache_key] = (
            baseline_sequences,
            int(baseline_token_count),
            int(baseline_non_holdout_tokens),
            float(sampling_ratio),
        )
    if not baseline_sequences:
        raise RuntimeError(f"{dataset}: sampled baseline small-LM probe corpus is empty.")
    if not eval_sequences:
        raise RuntimeError(f"{dataset}: held-out small-LM probe evaluation set is empty.")

    return SmallLMProbeContext(
        baseline_variant=str(baseline_variant),
        dataset=str(dataset),
        eval_dataset=str(eval_dataset_name),
        model_name=str(model_name),
        train_token_budget=int(token_budget),
        eval_token_budget=int(eval_token_budget),
        holdout_modulo=int(holdout_modulo),
        holdout_bucket=int(holdout_bucket),
        sampling_hash_seed=int(sampling_hash_seed),
        baseline_sampling_ratio=float(sampling_ratio),
        baseline_sequences=baseline_sequences,
        baseline_train_tokens=int(baseline_token_count),
        baseline_non_holdout_tokens=int(baseline_non_holdout_tokens),
        eval_sequences=eval_sequences,
        max_length=int(max_length),
        train_batch_size=int(train_batch_size),
        eval_batch_size=int(eval_batch_size),
        learning_rate=float(learning_rate),
        max_train_steps=int(max_train_steps),
        train_epochs=max(1.0, float(train_epochs)),
        train_audit_token_budget=max(0, int(train_audit_token_budget)),
    )


def score_selected_records(
    context: SmallLMProbeContext,
    selected_records: Iterable[tuple[str, str]],
    *,
    bootstrap_rounds: int = 100,
    seed: int = 42,
    selected_fingerprint: str | None = None,
    selected_sequence_cache: dict[Tuple[Any, ...], tuple[List[List[int]], int, int, float]] | None = None,
) -> Dict[str, Any]:
    tokenizer = _get_tokenizer(context.model_name)
    selected_cache_key = None
    if selected_fingerprint and selected_sequence_cache is not None:
        selected_cache_key = _selected_sequence_cache_key(
            selected_fingerprint=str(selected_fingerprint),
            model_name=str(context.model_name),
            max_length=int(context.max_length),
            holdout_modulo=int(context.holdout_modulo),
            holdout_bucket=int(context.holdout_bucket),
            train_token_budget=int(context.train_token_budget),
            sampling_hash_seed=int(context.sampling_hash_seed),
        )
    cached_selected = selected_sequence_cache.get(selected_cache_key) if selected_cache_key is not None else None
    if cached_selected is not None:
        selected_sequences, selected_token_count, selected_non_eval_tokens, selected_sampling_ratio = cached_selected
    else:
        selected_sequences, selected_token_count, selected_non_eval_tokens, selected_sampling_ratio = _collect_selected_sequences(
            selected_records,
            tokenizer=tokenizer,
            max_length=int(context.max_length),
            holdout_modulo=int(context.holdout_modulo),
            holdout_bucket=int(context.holdout_bucket),
            train_token_budget=int(context.train_token_budget),
            sampling_hash_seed=int(context.sampling_hash_seed),
        )
        if selected_cache_key is not None and selected_sequence_cache is not None:
            selected_sequence_cache[selected_cache_key] = (
                selected_sequences,
                selected_token_count,
                selected_non_eval_tokens,
                selected_sampling_ratio,
            )
    if not selected_sequences:
        raise RuntimeError(f"{context.dataset}: selected subset has no small-LM probe-trainable tokens.")

    baseline_audit_sequences, baseline_audit_tokens = _sample_audit_sequences(
        context.baseline_sequences,
        token_budget=int(context.train_audit_token_budget),
        seed=int(seed) + 104729,
    )
    selected_audit_sequences, selected_audit_tokens = _sample_audit_sequences(
        selected_sequences,
        token_budget=int(context.train_audit_token_budget),
        seed=int(seed) + 130363,
    )

    baseline_model, baseline_device, baseline_tokenizer, baseline_train_stats = _train_model(
        context.baseline_sequences,
        model_name=context.model_name,
        train_batch_size=context.train_batch_size,
        eval_batch_size=context.eval_batch_size,
        learning_rate=context.learning_rate,
        max_train_steps=context.max_train_steps,
        train_epochs=context.train_epochs,
        seed=int(seed),
        audit_sequences=baseline_audit_sequences,
    )
    baseline_nll, baseline_ppl, baseline_doc_loss, doc_token_counts = _eval_model(
        baseline_model,
        baseline_device,
        baseline_tokenizer,
        context.eval_sequences,
        eval_batch_size=context.eval_batch_size,
    )
    selected_eval_cache_key = None
    if selected_cache_key is not None:
        selected_eval_cache_key = (
            *selected_cache_key,
            str(context.eval_dataset),
            int(context.eval_token_budget),
            int(len(context.eval_sequences)),
            int(seed),
            int(context.train_batch_size),
            float(context.learning_rate),
            int(context.max_train_steps),
            round(float(context.train_epochs), 6),
            int(context.train_audit_token_budget),
        )
    cached_selected_eval = _SELECTED_EVAL_CACHE.get(selected_eval_cache_key) if selected_eval_cache_key is not None else None
    if cached_selected_eval is not None:
        selected_nll, selected_ppl, selected_doc_loss, selected_train_stats = cached_selected_eval
    else:
        selected_model, selected_device, selected_tokenizer, selected_train_stats = _train_model(
            selected_sequences,
            model_name=context.model_name,
            train_batch_size=context.train_batch_size,
            eval_batch_size=context.eval_batch_size,
            learning_rate=context.learning_rate,
            max_train_steps=context.max_train_steps,
            train_epochs=context.train_epochs,
            seed=int(seed),
            audit_sequences=selected_audit_sequences,
        )
        selected_nll, selected_ppl, selected_doc_loss, _ = _eval_model(
            selected_model,
            selected_device,
            selected_tokenizer,
            context.eval_sequences,
            eval_batch_size=context.eval_batch_size,
        )
        if selected_eval_cache_key is not None:
            _SELECTED_EVAL_CACHE[selected_eval_cache_key] = (selected_nll, selected_ppl, selected_doc_loss, selected_train_stats)
        del selected_model

    delta_nll = baseline_nll - selected_nll
    relative_nll_gain = delta_nll / baseline_nll if baseline_nll > 0 else 0.0
    score = relative_nll_gain

    rounds = max(50, int(bootstrap_rounds))
    rng = np.random.default_rng(seed)
    n_docs = int(doc_token_counts.shape[0])
    boot_deltas = []
    for _ in range(rounds):
        sample_idx = rng.integers(0, n_docs, size=n_docs)
        sampled_tokens = float(np.sum(doc_token_counts[sample_idx]))
        if sampled_tokens <= 0:
            continue
        sampled_baseline = float(np.sum(baseline_doc_loss[sample_idx])) / sampled_tokens
        sampled_selected = float(np.sum(selected_doc_loss[sample_idx])) / sampled_tokens
        boot_deltas.append(sampled_baseline - sampled_selected)
    if not boot_deltas:
        boot_deltas = [delta_nll]
    boot_delta_array = np.asarray(boot_deltas, dtype=np.float64)
    ci_low, ci_high = np.quantile(boot_delta_array, [0.025, 0.975]).tolist()
    bootstrap_delta_std = float(np.std(boot_delta_array, ddof=1)) if boot_delta_array.shape[0] > 1 else 0.0
    # Minimum detectable effect under the current paired eval protocol. This is
    # not a threshold relaxation; it reports whether observed deltas are large
    # enough to be distinguishable from protocol noise.
    minimum_detectable_delta_nll_95 = float(1.96 * bootstrap_delta_std)
    minimum_detectable_relative_gain_95 = (
        minimum_detectable_delta_nll_95 / float(baseline_nll) if baseline_nll > 0.0 else 0.0
    )
    effect_to_mde_ratio = (
        float(delta_nll) / minimum_detectable_delta_nll_95
        if minimum_detectable_delta_nll_95 > 0.0
        else (math.inf if delta_nll > 0.0 else 0.0)
    )
    selected_train_delta = float(selected_train_stats.get("train_audit_delta_nll") or 0.0)
    baseline_train_delta = float(baseline_train_stats.get("train_audit_delta_nll") or 0.0)
    train_delta_gap = selected_train_delta - baseline_train_delta
    causal_failure_mode = _utility_causal_failure_mode(
        delta_nll=float(delta_nll),
        selected_train_delta=selected_train_delta,
        baseline_train_delta=baseline_train_delta,
        mde=float(minimum_detectable_delta_nll_95),
    )

    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    payload = {
        "baseline_variant": str(context.baseline_variant),
        "small_lm_probe_gain_score": round(float(score), 6),
        "baseline_nll": round(float(baseline_nll), 6),
        "selected_nll": round(float(selected_nll), 6),
        "delta_nll": round(float(delta_nll), 6),
        "delta_nll_ci_low": round(float(ci_low), 6),
        "delta_nll_ci_high": round(float(ci_high), 6),
        "paired_bootstrap": True,
        "paired_bootstrap_rounds": int(rounds),
        "paired_bootstrap_delta_nll_mean": round(float(np.mean(boot_delta_array)), 6),
        "paired_bootstrap_delta_nll_std": round(float(bootstrap_delta_std), 8),
        "minimum_detectable_delta_nll_95": round(float(minimum_detectable_delta_nll_95), 8),
        "minimum_detectable_relative_gain_95": round(float(minimum_detectable_relative_gain_95), 8),
        "effect_to_mde_ratio": round(float(effect_to_mde_ratio), 6) if math.isfinite(effect_to_mde_ratio) else "inf",
        "detectable_effect": bool(float(delta_nll) > float(minimum_detectable_delta_nll_95)),
        "eval_pairing_policy": "paired_same_eval_documents",
        "baseline_perplexity": round(float(baseline_ppl), 6),
        "selected_perplexity": round(float(selected_ppl), 6),
        "delta_perplexity": round(float(baseline_ppl - selected_ppl), 6),
        "relative_nll_gain": round(float(relative_nll_gain), 6),
        "probe_family": "small_causal_lm",
        "probe_model_name": str(context.model_name),
        "probe_device": str(baseline_device),
        "eval_docs": int(n_docs),
        "eval_tokens": int(np.sum(doc_token_counts)),
        "selected_train_tokens": int(selected_token_count),
        "selected_non_eval_tokens": int(selected_non_eval_tokens),
        "baseline_train_tokens": int(context.baseline_train_tokens),
        "baseline_non_holdout_tokens": int(context.baseline_non_holdout_tokens),
        "train_audit_token_budget": int(context.train_audit_token_budget),
        "selected_train_audit_tokens": int(selected_audit_tokens),
        "baseline_train_audit_tokens": int(baseline_audit_tokens),
        "selected_train_audit_pre_nll": selected_train_stats.get("train_audit_pre_nll"),
        "selected_train_audit_post_nll": selected_train_stats.get("train_audit_post_nll"),
        "selected_train_audit_delta_nll": selected_train_stats.get("train_audit_delta_nll"),
        "selected_train_audit_relative_gain": selected_train_stats.get("train_audit_relative_gain"),
        "baseline_train_audit_pre_nll": baseline_train_stats.get("train_audit_pre_nll"),
        "baseline_train_audit_post_nll": baseline_train_stats.get("train_audit_post_nll"),
        "baseline_train_audit_delta_nll": baseline_train_stats.get("train_audit_delta_nll"),
        "baseline_train_audit_relative_gain": baseline_train_stats.get("train_audit_relative_gain"),
        "selected_minus_baseline_train_audit_delta_nll": round(float(train_delta_gap), 6),
        "causal_failure_mode": causal_failure_mode,
        "selected_effective_train_steps": int(selected_train_stats["effective_train_steps"]),
        "baseline_effective_train_steps": int(baseline_train_stats["effective_train_steps"]),
        "selected_target_train_steps": int(selected_train_stats["target_train_steps"]),
        "baseline_target_train_steps": int(baseline_train_stats["target_train_steps"]),
        "selected_one_epoch_train_steps": int(selected_train_stats["one_epoch_train_steps"]),
        "baseline_one_epoch_train_steps": int(baseline_train_stats["one_epoch_train_steps"]),
        "selected_estimated_seen_train_tokens": int(selected_train_stats["estimated_seen_train_tokens"]),
        "baseline_estimated_seen_train_tokens": int(baseline_train_stats["estimated_seen_train_tokens"]),
        "selected_train_token_exposure_ratio": float(selected_train_stats["train_token_exposure_ratio"]),
        "baseline_train_token_exposure_ratio": float(baseline_train_stats["train_token_exposure_ratio"]),
        "selected_target_train_exposure_ratio": float(selected_train_stats["target_train_exposure_ratio"]),
        "baseline_target_train_exposure_ratio": float(baseline_train_stats["target_train_exposure_ratio"]),
        "selected_step_cap_reached": bool(selected_train_stats["step_cap_reached"]),
        "baseline_step_cap_reached": bool(baseline_train_stats["step_cap_reached"]),
        "selected_sampling_ratio": round(float(selected_sampling_ratio), 6),
        "baseline_sampling_ratio": round(float(context.baseline_sampling_ratio), 6),
        "sampling_hash_seed": int(context.sampling_hash_seed),
        "train_token_budget": int(context.train_token_budget),
        "eval_token_budget": int(context.eval_token_budget),
        "holdout_modulo": int(context.holdout_modulo),
        "holdout_bucket": int(context.holdout_bucket),
        "bootstrap_rounds": int(rounds),
        "bootstrap_seed": int(seed),
        "eval_dataset": str(context.eval_dataset),
        "max_length": int(context.max_length),
        "train_batch_size": int(context.train_batch_size),
        "eval_batch_size": int(context.eval_batch_size),
        "learning_rate": float(context.learning_rate),
        "max_train_steps": int(context.max_train_steps),
        "train_epochs": round(float(context.train_epochs), 6),
    }
    # Backward-compatible alias for downstream readers that still expect the old field.
    payload["fixed_token_probe_gain_score"] = payload["small_lm_probe_gain_score"]
    return payload


def aggregate_probe_runs(
    runs: Sequence[Dict[str, Any]],
    *,
    mode: str,
    train_dataset: str,
    eval_dataset: str,
) -> Dict[str, Any]:
    if not runs:
        raise ValueError("aggregate_probe_runs requires at least one run.")

    def _f(run: Dict[str, Any], key: str, default: float = 0.0) -> float:
        return float(run.get(key) if run.get(key) is not None else default)

    scores = [_f(run, "small_lm_probe_gain_score") for run in runs]
    delta_nlls = [_f(run, "delta_nll") for run in runs]
    ci_lows = [_f(run, "delta_nll_ci_low") for run in runs]
    ci_highs = [_f(run, "delta_nll_ci_high") for run in runs]
    rel_gains = [_f(run, "relative_nll_gain") for run in runs]
    baseline_nlls = [_f(run, "baseline_nll") for run in runs]
    selected_nlls = [_f(run, "selected_nll") for run in runs]
    eval_docs = [int(run.get("eval_docs") or 0) for run in runs]
    eval_tokens = [int(run.get("eval_tokens") or 0) for run in runs]
    selected_train_tokens = [int(run.get("selected_train_tokens") or 0) for run in runs]
    baseline_train_tokens = [int(run.get("baseline_train_tokens") or 0) for run in runs]
    selected_train_audit_delta = [_f(run, "selected_train_audit_delta_nll") for run in runs]
    baseline_train_audit_delta = [_f(run, "baseline_train_audit_delta_nll") for run in runs]
    train_audit_gap = [_f(run, "selected_minus_baseline_train_audit_delta_nll") for run in runs]
    selected_train_audit_pre = [_f(run, "selected_train_audit_pre_nll") for run in runs]
    selected_train_audit_post = [_f(run, "selected_train_audit_post_nll") for run in runs]
    baseline_train_audit_pre = [_f(run, "baseline_train_audit_pre_nll") for run in runs]
    baseline_train_audit_post = [_f(run, "baseline_train_audit_post_nll") for run in runs]
    selected_train_audit_tokens = [int(run.get("selected_train_audit_tokens") or 0) for run in runs]
    baseline_train_audit_tokens = [int(run.get("baseline_train_audit_tokens") or 0) for run in runs]
    selected_effective_steps = [int(run.get("selected_effective_train_steps") or 0) for run in runs]
    baseline_effective_steps = [int(run.get("baseline_effective_train_steps") or 0) for run in runs]
    selected_target_steps = [int(run.get("selected_target_train_steps") or 0) for run in runs]
    baseline_target_steps = [int(run.get("baseline_target_train_steps") or 0) for run in runs]
    selected_one_epoch_steps = [int(run.get("selected_one_epoch_train_steps") or 0) for run in runs]
    baseline_one_epoch_steps = [int(run.get("baseline_one_epoch_train_steps") or 0) for run in runs]
    selected_seen_tokens = [int(run.get("selected_estimated_seen_train_tokens") or 0) for run in runs]
    baseline_seen_tokens = [int(run.get("baseline_estimated_seen_train_tokens") or 0) for run in runs]
    selected_exposure = [_f(run, "selected_train_token_exposure_ratio") for run in runs]
    baseline_exposure = [_f(run, "baseline_train_token_exposure_ratio") for run in runs]
    selected_target_exposure = [_f(run, "selected_target_train_exposure_ratio") for run in runs]
    baseline_target_exposure = [_f(run, "baseline_target_train_exposure_ratio") for run in runs]
    paired_delta_stds = [_f(run, "paired_bootstrap_delta_nll_std") for run in runs]
    mde_delta_95 = [_f(run, "minimum_detectable_delta_nll_95") for run in runs]
    mde_relative_95 = [_f(run, "minimum_detectable_relative_gain_95") for run in runs]
    effect_to_mde = [
        float(run.get("effect_to_mde_ratio"))
        for run in runs
        if isinstance(run.get("effect_to_mde_ratio"), (int, float))
    ]
    train_epochs = [_f(run, "train_epochs", 1.0) for run in runs]
    holdout_buckets = [int(run.get("holdout_bucket") or 0) for run in runs]
    bootstrap_seeds = sorted({int(run.get("bootstrap_seed") or 0) for run in runs})
    probe_devices = Counter(str(run.get("probe_device") or "unknown") for run in runs)
    eval_batch_sizes = Counter(str(run.get("eval_batch_size") or "unknown") for run in runs)
    causal_modes = Counter(str(run.get("causal_failure_mode") or "unresolved") for run in runs)
    score_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    delta_std = float(np.std(delta_nlls, ddof=1)) if len(delta_nlls) > 1 else 0.0
    rel_gain_std = float(np.std(rel_gains, ddof=1)) if len(rel_gains) > 1 else 0.0
    mean_delta = float(np.mean(delta_nlls))
    mean_score = float(np.mean(scores))
    noise_floor = max(delta_std, 1e-12)
    positive_run_fraction = sum(1 for value in delta_nlls if value > 0.0) / max(len(delta_nlls), 1)
    ci_positive_fraction = sum(1 for value in ci_lows if value > 0.0) / max(len(ci_lows), 1)
    dominant_causal_mode = causal_modes.most_common(1)[0][0] if causal_modes else "unresolved"
    mean_train_gap = float(np.mean(train_audit_gap)) if train_audit_gap else 0.0
    mean_selected_train_delta = float(np.mean(selected_train_audit_delta)) if selected_train_audit_delta else 0.0
    mean_baseline_train_delta = float(np.mean(baseline_train_audit_delta)) if baseline_train_audit_delta else 0.0
    if dominant_causal_mode == "weaker_selected_training_signal":
        causal_interpretation = "The selected subset lowers train NLL less than the matched baseline, so the Utility loss is consistent with weaker probe-learnable training signal."
    elif dominant_causal_mode == "overfit_or_distribution_shift":
        causal_interpretation = "The selected subset learns its train slice about as well as the baseline but loses on held-out eval, indicating distribution shift or overfitting under the probe."
    elif dominant_causal_mode == "probe_or_training_insensitive":
        causal_interpretation = "Both arms show weak train-loss movement, so the small-LM training protocol may be underpowered for this comparison."
    elif dominant_causal_mode == "inconclusive_near_noise_floor":
        causal_interpretation = "The observed eval delta is close to the paired-bootstrap detectable-effect band."
    elif dominant_causal_mode == "positive_learning_signal":
        causal_interpretation = "The selected subset shows positive held-out learning signal in most probe cells."
    else:
        causal_interpretation = "The current train/eval audit does not isolate a single dominant cause."

    payload = {
        "baseline_variant": str(runs[0].get("baseline_variant") or "baseline_full_random"),
        "mode": str(mode),
        "train_dataset": str(train_dataset),
        "eval_dataset": str(eval_dataset),
        "run_count": int(len(runs)),
        "bucket_count": int(len(set(holdout_buckets))),
        "seed_count": int(len(bootstrap_seeds)),
        "bootstrap_seeds": bootstrap_seeds,
        "holdout_buckets": holdout_buckets,
        "small_lm_probe_gain_score": round(mean_score, 6),
        "small_lm_probe_gain_score_min": round(float(np.min(scores)), 6),
        "delta_nll": round(float(np.mean(delta_nlls)), 6),
        "delta_nll_min": round(float(np.min(delta_nlls)), 6),
        "delta_nll_ci_low": round(float(np.min(ci_lows)), 6),
        "delta_nll_ci_high": round(float(np.max(ci_highs)), 6),
        "paired_bootstrap": bool(all(bool(run.get("paired_bootstrap")) for run in runs)),
        "eval_pairing_policy": "paired_same_eval_documents",
        "paired_bootstrap_delta_nll_std_mean": round(float(np.mean(paired_delta_stds)), 8) if paired_delta_stds else 0.0,
        "paired_bootstrap_delta_nll_std_max": round(float(np.max(paired_delta_stds)), 8) if paired_delta_stds else 0.0,
        "minimum_detectable_delta_nll_95_mean": round(float(np.mean(mde_delta_95)), 8) if mde_delta_95 else 0.0,
        "minimum_detectable_delta_nll_95_max": round(float(np.max(mde_delta_95)), 8) if mde_delta_95 else 0.0,
        "minimum_detectable_relative_gain_95_mean": round(float(np.mean(mde_relative_95)), 8) if mde_relative_95 else 0.0,
        "minimum_detectable_relative_gain_95_max": round(float(np.max(mde_relative_95)), 8) if mde_relative_95 else 0.0,
        "effect_to_mde_ratio_min": round(float(np.min(effect_to_mde)), 6) if effect_to_mde else 0.0,
        "effect_to_mde_ratio_mean": round(float(np.mean(effect_to_mde)), 6) if effect_to_mde else 0.0,
        "detectable_effect_fraction": round(
            float(sum(1 for run in runs if bool(run.get("detectable_effect"))) / max(len(runs), 1)),
            6,
        ),
        "relative_nll_gain": round(float(np.mean(rel_gains)), 6),
        "relative_nll_gain_min": round(float(np.min(rel_gains)), 6),
        "baseline_nll": round(float(np.mean(baseline_nlls)), 6),
        "selected_nll": round(float(np.mean(selected_nlls)), 6),
        "eval_docs": int(np.sum(eval_docs)),
        "eval_tokens": int(np.sum(eval_tokens)),
        "probe_device_counts": dict(sorted(probe_devices.items())),
        "eval_batch_size_counts": dict(sorted(eval_batch_sizes.items())),
        "selected_train_tokens_mean": int(round(float(np.mean(selected_train_tokens)))) if selected_train_tokens else 0,
        "baseline_train_tokens_mean": int(round(float(np.mean(baseline_train_tokens)))) if baseline_train_tokens else 0,
        "selected_train_tokens_min": int(np.min(selected_train_tokens)) if selected_train_tokens else 0,
        "baseline_train_tokens_min": int(np.min(baseline_train_tokens)) if baseline_train_tokens else 0,
        "selected_train_audit_tokens_mean": int(round(float(np.mean(selected_train_audit_tokens)))) if selected_train_audit_tokens else 0,
        "baseline_train_audit_tokens_mean": int(round(float(np.mean(baseline_train_audit_tokens)))) if baseline_train_audit_tokens else 0,
        "selected_train_audit_pre_nll_mean": round(float(np.mean(selected_train_audit_pre)), 6) if selected_train_audit_pre else 0.0,
        "selected_train_audit_post_nll_mean": round(float(np.mean(selected_train_audit_post)), 6) if selected_train_audit_post else 0.0,
        "selected_train_audit_delta_nll_mean": round(mean_selected_train_delta, 6),
        "baseline_train_audit_pre_nll_mean": round(float(np.mean(baseline_train_audit_pre)), 6) if baseline_train_audit_pre else 0.0,
        "baseline_train_audit_post_nll_mean": round(float(np.mean(baseline_train_audit_post)), 6) if baseline_train_audit_post else 0.0,
        "baseline_train_audit_delta_nll_mean": round(mean_baseline_train_delta, 6),
        "selected_minus_baseline_train_audit_delta_nll_mean": round(mean_train_gap, 6),
        "selected_effective_train_steps_mean": int(round(float(np.mean(selected_effective_steps)))) if selected_effective_steps else 0,
        "baseline_effective_train_steps_mean": int(round(float(np.mean(baseline_effective_steps)))) if baseline_effective_steps else 0,
        "selected_effective_train_steps_min": int(np.min(selected_effective_steps)) if selected_effective_steps else 0,
        "baseline_effective_train_steps_min": int(np.min(baseline_effective_steps)) if baseline_effective_steps else 0,
        "selected_target_train_steps_mean": int(round(float(np.mean(selected_target_steps)))) if selected_target_steps else 0,
        "baseline_target_train_steps_mean": int(round(float(np.mean(baseline_target_steps)))) if baseline_target_steps else 0,
        "selected_one_epoch_train_steps_mean": int(round(float(np.mean(selected_one_epoch_steps)))) if selected_one_epoch_steps else 0,
        "baseline_one_epoch_train_steps_mean": int(round(float(np.mean(baseline_one_epoch_steps)))) if baseline_one_epoch_steps else 0,
        "selected_estimated_seen_train_tokens_mean": int(round(float(np.mean(selected_seen_tokens)))) if selected_seen_tokens else 0,
        "baseline_estimated_seen_train_tokens_mean": int(round(float(np.mean(baseline_seen_tokens)))) if baseline_seen_tokens else 0,
        "selected_train_token_exposure_ratio_mean": round(float(np.mean(selected_exposure)), 6) if selected_exposure else 0.0,
        "baseline_train_token_exposure_ratio_mean": round(float(np.mean(baseline_exposure)), 6) if baseline_exposure else 0.0,
        "selected_target_train_exposure_ratio_mean": round(float(np.mean(selected_target_exposure)), 6) if selected_target_exposure else 0.0,
        "baseline_target_train_exposure_ratio_mean": round(float(np.mean(baseline_target_exposure)), 6) if baseline_target_exposure else 0.0,
        "train_epochs_mean": round(float(np.mean(train_epochs)), 6) if train_epochs else 0.0,
        "selected_step_cap_reached_count": int(sum(1 for run in runs if bool(run.get("selected_step_cap_reached")))),
        "baseline_step_cap_reached_count": int(sum(1 for run in runs if bool(run.get("baseline_step_cap_reached")))),
        "stability_diagnostics": {
            "score_std": round(score_std, 8),
            "delta_nll_std": round(delta_std, 8),
            "relative_nll_gain_std": round(rel_gain_std, 8),
            "positive_run_fraction": round(float(positive_run_fraction), 6),
            "ci_positive_fraction": round(float(ci_positive_fraction), 6),
            "mean_delta_nll_to_std_ratio": round(float(mean_delta / noise_floor), 6),
            "mean_score_to_std_ratio": round(float(mean_score / max(score_std, 1e-12)), 6),
            "minimum_detectable_delta_nll_95_max": round(float(np.max(mde_delta_95)), 8) if mde_delta_95 else 0.0,
            "effect_to_mde_ratio_min": round(float(np.min(effect_to_mde)), 6) if effect_to_mde else 0.0,
            "detectable_effect_fraction": round(
                float(sum(1 for run in runs if bool(run.get("detectable_effect"))) / max(len(runs), 1)),
                6,
            ),
            "strict_min_negative": bool(float(np.min(delta_nlls)) <= 0.0),
            "ci_crosses_zero": bool(float(np.min(ci_lows)) <= 0.0 <= float(np.max(ci_highs))),
            "run_count": int(len(runs)),
        },
        "causal_utility_audit": {
            "audit_type": "selected_vs_matched_counterfactual_train_eval_decomposition",
            "dominant_failure_mode": dominant_causal_mode,
            "failure_mode_counts": dict(sorted(causal_modes.items())),
            "interpretation": causal_interpretation,
            "mean_eval_delta_nll": round(float(mean_delta), 6),
            "mean_selected_train_audit_delta_nll": round(mean_selected_train_delta, 6),
            "mean_baseline_train_audit_delta_nll": round(mean_baseline_train_delta, 6),
            "mean_selected_minus_baseline_train_audit_delta_nll": round(mean_train_gap, 6),
            "positive_eval_run_fraction": round(float(positive_run_fraction), 6),
            "selected_train_advantage_fraction": round(
                float(sum(1 for value in train_audit_gap if value > 0.0) / max(len(train_audit_gap), 1)),
                6,
            ),
            "probe_device_counts": dict(sorted(probe_devices.items())),
            "eval_batch_size_counts": dict(sorted(eval_batch_sizes.items())),
            "decision_rule": (
                "If selected learns train slice less than baseline and eval delta is negative, classify as weaker_selected_training_signal; "
                "if selected learns train slice at least comparably but eval delta is negative, classify as overfit_or_distribution_shift; "
                "if both train deltas are small, classify as probe_or_training_insensitive."
            ),
        },
        "per_bucket_runs": list(runs),
    }
    payload["fixed_token_probe_gain_score"] = payload["small_lm_probe_gain_score"]
    payload["fixed_token_probe_gain_score_min"] = payload["small_lm_probe_gain_score_min"]
    return payload
