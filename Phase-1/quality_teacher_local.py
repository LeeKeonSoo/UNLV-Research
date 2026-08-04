from __future__ import annotations

from pathlib import Path
from typing import Mapping, TypeVar

from quality_teacher_adapters import CompletionRequest, TeacherAdapterContractError


InputIds = TypeVar("InputIds")


def extract_chat_input_ids(encoded: Mapping[str, InputIds]) -> InputIds:
    return encoded["input_ids"]


class QwenLocalBackend:
    """GPU-backed Qwen backend loaded once and reused across panel requests."""

    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

        if not model_path.is_dir():
            raise TeacherAdapterContractError(
                teacher_id="qwen-local",
                detail=f"local model directory does not exist: {model_path}",
            )
        if not torch.cuda.is_available():
            raise TeacherAdapterContractError(
                teacher_id="qwen-local",
                detail="CUDA is required by the frozen local teacher contract",
            )
        quantization = BitsAndBytesConfig(load_in_8bit=True)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        self._model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            max_memory={0: "15GiB", "cpu": "48GiB"},
            quantization_config=quantization,
            torch_dtype="auto",
        ).eval()

    def complete(self, request: CompletionRequest) -> str:
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]
        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        input_ids = extract_chat_input_ids(encoded).to(self._model.device)
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=request.maximum_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output_ids[0, input_ids.shape[-1] :]
        content = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not content:
            raise TeacherAdapterContractError(
                teacher_id=request.model_id,
                detail="local model returned no textual completion",
            )
        return content
