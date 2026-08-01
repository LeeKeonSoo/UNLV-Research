from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]


def semantic_bucket(embedding: list[float], *, prefix_bits: int = 16) -> str:
    """Return a deterministic LSH bucket used to bound semantic-neighbor comparisons."""
    signs = "".join("1" if value >= 0.0 else "0" for value in embedding[:prefix_bits])
    return f"frozen-lsh-v1:{signs.ljust(prefix_bits, '0')}"


def read_jsonl_records(path: Path) -> tuple[list[JsonMap], int]:
    """Read legacy JSONL while restoring literal newlines found inside a JSON string."""
    rows: list[JsonMap] = []
    repaired_newlines = 0
    buffer: list[str] = []
    in_string = False
    escaped = False
    depth = 0
    for character in path.read_text(encoding="utf-8-sig").replace("\r\n", "\n"):
        if character == "\n" and in_string:
            buffer.append("\\n")
            repaired_newlines += 1
            continue
        buffer.append(character)
        if character == '"' and not escaped:
            in_string = not in_string
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
        if not in_string and character == "{":
            depth += 1
        if not in_string and character == "}":
            depth -= 1
            if depth == 0:
                row = json.loads("".join(buffer).strip())
                if not isinstance(row, dict):
                    raise RuntimeError("Frozen proxy input record must be a JSON object")
                rows.append(row)
                buffer = []
    if buffer and "".join(buffer).strip():
        raise RuntimeError("Frozen proxy input ended with an incomplete JSON record")
    return rows, repaired_newlines


def _read_sample(path: Path, maximum_records: int) -> tuple[list[JsonMap], int]:
    rows, repaired_newlines = read_jsonl_records(path)
    if maximum_records <= 0 or maximum_records >= len(rows):
        return rows, repaired_newlines
    return sorted(rows, key=lambda row: hashlib.sha256(str(row["chunk_uid"]).encode()).hexdigest())[:maximum_records], repaired_newlines


def _snapshot_fingerprint(model_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model_path.iterdir(), key=lambda item: item.name):
        if path.is_file():
            digest.update(f"{path.name}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()


def _unit(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector] if norm else [0.0 for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def score_frozen_proxy_sample(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_path: Path,
    model_id: str,
    maximum_records: int,
    max_length: int,
    device: str,
) -> JsonMap:
    """Generate frozen proxy evidence for one development corpus without runtime-selection authority."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows, repaired_newlines = _read_sample(input_path, maximum_records)
    if not rows:
        raise RuntimeError(f"Frozen proxy scoring input is empty: {input_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model.eval()
    embedding_layer = model.get_input_embeddings()
    scored: list[JsonMap] = []
    gradient_vectors: list[list[float]] = []
    for index, row in enumerate(rows):
        encoded = tokenizer(str(row["text"]), return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.enable_grad():
            inputs_embeds = embedding_layer(input_ids).detach().requires_grad_(True)
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            gradient = torch.autograd.grad(outputs.loss, inputs_embeds)[0]
        mask = attention_mask.unsqueeze(-1)
        token_count = int(attention_mask.sum().item())
        representation = (outputs.hidden_states[-1] * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embedding = representation[0].detach().float().cpu().tolist()
        gradient_vector = gradient[0].mean(dim=0).detach().float().cpu().tolist()
        scored.append(
            {
                "chunk_uid": str(row["chunk_uid"]),
                "semantic_bucket": semantic_bucket(embedding),
                "embedding": embedding,
                "proxy_nll": float(outputs.loss.detach().float().cpu().item()),
                "proxy_tokens": token_count,
            }
        )
        gradient_vectors.append(_unit(gradient_vector))
        del outputs, gradient, inputs_embeds, representation
        if index % 16 == 15:
            torch.cuda.empty_cache()
    centroid = _unit([sum(vector[dimension] for vector in gradient_vectors) for dimension in range(len(gradient_vectors[0]))])
    for row, gradient_vector in zip(scored, gradient_vectors, strict=True):
        row["gradient_alignment"] = _dot(gradient_vector, centroid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "stage-c2-frozen-proxy-raw-score-v1",
        "status": "frozen_proxy_evidence_ready",
        "model_id": model_id,
        "model_sha256": _snapshot_fingerprint(model_path),
        "calibration_snapshot_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "input_path": str(input_path),
        "input_records": len(rows),
        "input_literal_newline_repairs": repaired_newlines,
        "scoring": {"proxy": "frozen_causal_lm", "max_length": max_length, "gradient_alignment": "input_embedding_gradient_to_frozen_sample_centroid", "semantic_index": "last_hidden_state_lsh"},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate candidate-only Stage C-2 frozen proxy evidence.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    manifest = score_frozen_proxy_sample(
        input_path=arguments.input,
        output_path=arguments.output,
        manifest_path=arguments.manifest,
        model_path=arguments.model_path,
        model_id=arguments.model_id,
        maximum_records=arguments.max_records,
        max_length=arguments.max_length,
        device=arguments.device,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
