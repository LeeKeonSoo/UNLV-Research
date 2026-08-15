from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

from semantic_embedding_artifact import (
    EmbeddingArtifact,
    PoolingMode,
    hash_model_snapshot,
    write_embedding_artifact,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SemanticEmbeddingRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EmbeddingProviderSpec:
    provider_id: str
    provider_identity_sha256: str
    model_id: str
    revision: str
    pooling: PoolingMode
    max_length: int
    batch_size: int
    device: str
    cache_dir: Path
    append_eos: bool
    model_path_is_local: bool = False

    def __post_init__(self) -> None:
        if not self.provider_id or not SHA256_RE.fullmatch(self.provider_identity_sha256):
            raise SemanticEmbeddingRuntimeError("Provider identity must be frozen")
        if not self.model_id or not self.revision or self.max_length < 8 or self.batch_size < 1:
            raise SemanticEmbeddingRuntimeError("Embedding provider settings are incomplete")


@dataclass(frozen=True, slots=True)
class EmbeddingDocument:
    uid: str
    text: str

    def __post_init__(self) -> None:
        if not self.uid or not self.text:
            raise SemanticEmbeddingRuntimeError("Embedding documents require ID and text")


def pool_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: PoolingMode,
) -> torch.Tensor:
    match pooling:
        case PoolingMode.CLS:
            return hidden[:, 0]
        case PoolingMode.LAST_TOKEN:
            if bool(torch.all(attention_mask[:, -1] == 1)):
                return hidden[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_indices, sequence_lengths]


def _special_affixes(tokenizer) -> tuple[list[int], list[int]]:
    probe = "semantic coverage probe"
    raw = tokenizer(probe, add_special_tokens=False, truncation=False)["input_ids"]
    prepared = tokenizer(probe, add_special_tokens=True, truncation=False)["input_ids"]
    start = next(
        (index for index in range(len(prepared) - len(raw) + 1) if prepared[index:index + len(raw)] == raw),
        None,
    )
    if start is None:
        raise SemanticEmbeddingRuntimeError("Tokenizer special-token affixes are not recoverable")
    return prepared[:start], prepared[start + len(raw):]


def token_windows(
    tokenizer,
    text: str,
    max_length: int,
    append_eos: bool,
    affixes: tuple[list[int], list[int]] | None = None,
) -> tuple[list[dict], int]:
    token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    eos = tokenizer.eos_token_id if append_eos else None
    prefix, suffix = affixes or _special_affixes(tokenizer)
    reserved = len(prefix) + len(suffix) + int(eos is not None and eos not in suffix)
    width = max_length - reserved
    if width < 1:
        raise SemanticEmbeddingRuntimeError("Embedding window has no payload capacity")
    parts = [token_ids[index:index + width] for index in range(0, max(1, len(token_ids)), width)]
    windows = []
    for part in parts:
        ending = [eos] if eos is not None and eos not in suffix else []
        prepared = [*prefix, *part, *ending, *suffix]
        windows.append({"input_ids": prepared, "attention_mask": [1] * len(prepared)})
    return windows, len(token_ids)


def encode_documents(
    spec: EmbeddingProviderSpec,
    documents: tuple[EmbeddingDocument, ...],
    corpus_sha256: str,
    output_dir: Path,
) -> Path:
    if not documents or len({item.uid for item in documents}) != len(documents):
        raise SemanticEmbeddingRuntimeError("Embedding corpus requires unique documents")
    snapshot = (
        Path(spec.model_id)
        if spec.model_path_is_local
        else Path(
            snapshot_download(
                spec.model_id,
                revision=spec.revision,
                cache_dir=spec.cache_dir,
                local_files_only=True,
            )
        )
    )
    model_source = str(snapshot) if spec.model_path_is_local else spec.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=None if spec.model_path_is_local else spec.revision,
        cache_dir=spec.cache_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left" if spec.pooling is PoolingMode.LAST_TOKEN else "right"
    model = AutoModel.from_pretrained(
        model_source,
        revision=None if spec.model_path_is_local else spec.revision,
        cache_dir=spec.cache_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
    ).to(spec.device)
    model.eval()
    affixes = _special_affixes(tokenizer)
    sums: list[NDArray[np.float32] | None] = [None] * len(documents)
    counts = [0] * len(documents)
    pending: list[dict] = []
    owners: list[int] = []
    maximum_observed_tokens = 0
    windowed_records = 0
    total_windows = 0

    def flush() -> None:
        if not pending:
            return
        encoded = tokenizer.pad(pending, padding=True, return_tensors="pt").to(spec.device)
        hidden = model(**encoded).last_hidden_state
        pooled = pool_hidden_states(hidden, encoded["attention_mask"], spec.pooling)
        vectors = torch.nn.functional.normalize(pooled.float(), p=2, dim=1).cpu().numpy()
        for owner, vector in zip(owners, vectors, strict=True):
            sums[owner] = vector.copy() if sums[owner] is None else sums[owner] + vector
            counts[owner] += 1
        pending.clear()
        owners.clear()

    with torch.inference_mode():
        for index, document in enumerate(documents):
            windows, observed = token_windows(
                tokenizer, document.text, spec.max_length, spec.append_eos, affixes
            )
            maximum_observed_tokens = max(maximum_observed_tokens, observed)
            windowed_records += int(len(windows) > 1)
            total_windows += len(windows)
            for window in windows:
                pending.append(window)
                owners.append(index)
                if len(pending) == spec.batch_size:
                    flush()
            if index == 0 or index + 1 == len(documents) or (index + 1) % 200 == 0:
                print(
                    f"[{spec.provider_id}] prepared {index + 1}/{len(documents)} records",
                    flush=True,
                )
        flush()
    if any(vector is None for vector in sums) or any(count < 1 for count in counts):
        raise SemanticEmbeddingRuntimeError("Every document requires at least one embedding window")
    vectors = np.stack(
        [vector / counts[index] for index, vector in enumerate(sums) if vector is not None]
    ).astype(np.float32, copy=False)
    model_hash = hash_model_snapshot(snapshot)
    del model
    torch.cuda.empty_cache()
    return write_embedding_artifact(
        EmbeddingArtifact(
            provider_id=spec.provider_id,
            provider_identity_sha256=spec.provider_identity_sha256,
            corpus_sha256=corpus_sha256,
            pooling=spec.pooling,
            max_length=spec.max_length,
            uids=tuple(item.uid for item in documents),
            vectors=vectors,
            model_files_sha256=model_hash,
            model_id=spec.model_id,
            revision=spec.revision,
            truncated_records=0,
            maximum_observed_tokens=maximum_observed_tokens,
            windowed_records=windowed_records,
            total_windows=total_windows,
            text_sha256s=tuple(
                hashlib.sha256(document.text.encode("utf-8")).hexdigest()
                for document in documents
            ),
        ),
        output_dir,
    )
