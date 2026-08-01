#!/usr/bin/env python3
"""Tiny LM probe utilities for utility teacher-label generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - import failure is handled at runtime
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


DEFAULT_PROBE_MODEL = "sshleifer/tiny-gpt2"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _flatten_grads(params: List[torch.Tensor]) -> torch.Tensor:
    parts = []
    for p in params:
        if p.grad is None:
            continue
        parts.append(p.grad.detach().reshape(-1).float().cpu())
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


@dataclass
class UtilityProbe:
    model_name: str = DEFAULT_PROBE_MODEL
    max_length: int = 256

    def __post_init__(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError(f"transformers unavailable: {_TRANSFORMERS_IMPORT_ERROR}")
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._tracked_params = self._select_tracked_params()

    def _select_tracked_params(self) -> List[torch.nn.Parameter]:
        names = []
        n_layer = getattr(getattr(self.model, "config", None), "n_layer", None)
        if n_layer is None:
            return [p for _, p in self.model.named_parameters() if p.requires_grad][-8:]
        last_block = f"transformer.h.{n_layer - 1}."
        selected = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad and (name.startswith(last_block) or name.startswith("transformer.ln_f"))
        ]
        return selected

    def _encode(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def gradient_vector(self, text: str) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        batch = self._encode(text)
        outputs = self.model(**batch, labels=batch["input_ids"])
        outputs.loss.backward()
        return _flatten_grads(self._tracked_params)

    def reference_gradient(self, texts: Iterable[str]) -> torch.Tensor:
        grads = []
        for text in texts:
            grads.append(self.gradient_vector(text))
        if not grads:
            return torch.zeros(1)
        size = min(g.numel() for g in grads)
        stacked = torch.stack([g[:size] for g in grads], dim=0)
        return stacked.mean(dim=0)

    def teacher_label(self, text: str, reference_grad: torch.Tensor) -> float:
        grad = self.gradient_vector(text)
        size = min(grad.numel(), reference_grad.numel())
        if size == 0:
            return 0.0
        return _cosine(grad[:size], reference_grad[:size])
