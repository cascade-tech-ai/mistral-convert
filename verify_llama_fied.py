#!/usr/bin/env python3
"""Verify tokenizer IDs and model logits for a llama-fied Ministral checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_REFERENCE = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
DEFAULT_DATASET = "/home/alvion/valve/services/training/datasets/think2-2025-12-07_gpt-5.4_reasoning.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="Reference HF id or local upstream snapshot.")
    parser.add_argument("--candidate", required=True, help="Converted Llama-compatible checkpoint path.")
    parser.add_argument("--jsonl", default=DEFAULT_DATASET, help="JSONL prompts/messages file.")
    parser.add_argument("--max-rows", type=int, default=None, help="Only verify the first N rows.")
    parser.add_argument("--max-length", type=int, default=2048, help="Tokenizer truncation length for forward passes.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--kl-atol", type=float, default=1e-5, help="Maximum allowed mean token KL.")
    parser.add_argument("--logit-atol", type=float, default=2e-2, help="Maximum allowed absolute logit difference.")
    parser.add_argument(
        "--reference-config-patch",
        action="store_true",
        help="Patch nested text_config.model_type=llama for older Transformers builds that lack ministral3.",
    )
    parser.add_argument("--skip-model", action="store_true", help="Only verify tokenizer parity.")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_reference(
    reference: str, patch_config: bool, tokenizer_only: bool
) -> tuple[Path, contextlib.AbstractContextManager[None]]:
    path = Path(reference)
    if not path.exists():
        allow_patterns = None
        if tokenizer_only:
            allow_patterns = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
            ]
        path = Path(snapshot_download(reference, allow_patterns=allow_patterns))

    if not patch_config:
        return path, contextlib.nullcontext()

    tmp = tempfile.TemporaryDirectory(prefix="ministral3_ref_")
    tmp_path = Path(tmp.name)
    for item in path.iterdir():
        target = tmp_path / item.name
        if item.is_dir():
            shutil.copytree(item, target, symlinks=True)
        else:
            target.symlink_to(item.resolve())

    cfg_path = tmp_path / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("text_config", {}).get("model_type") == "ministral3":
        cfg["text_config"]["model_type"] = "llama"
        cfg_path.unlink()
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    return tmp_path, tmp


def rows(path: Path, max_rows: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            if line.strip():
                yield idx, json.loads(line)


def row_to_text(row: dict[str, Any], tokenizer: Any) -> str:
    if isinstance(row.get("messages"), list):
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    for key in ("prompt", "text", "input"):
        if isinstance(row.get(key), str):
            return row[key]
    raise ValueError("row must contain messages, prompt, text, or input")


def get_logits(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**batch)
    logits = out.logits if hasattr(out, "logits") else out.language_model_logits
    return logits.float()


def main() -> None:
    args = parse_args()
    candidate = Path(args.candidate).resolve()
    jsonl = Path(args.jsonl).resolve()
    dtype = torch_dtype(args.dtype)

    ref_path, ref_context = resolve_reference(args.reference, args.reference_config_patch, args.skip_model)
    with ref_context:
        ref_tok = AutoTokenizer.from_pretrained(ref_path, fix_mistral_regex=True)
        cand_tok = AutoTokenizer.from_pretrained(candidate, fix_mistral_regex=True)

        ref_model = cand_model = None
        if not args.skip_model:
            ref_model = AutoModelForImageTextToText.from_pretrained(ref_path, dtype=dtype, device_map=args.device).eval()
            cand_model = AutoModelForCausalLM.from_pretrained(candidate, dtype=dtype, device_map=args.device).eval()

        checked = 0
        worst_kl = 0.0
        worst_logit = 0.0
        for row_idx, row in rows(jsonl, args.max_rows):
            text = row_to_text(row, ref_tok)
            ref_inputs = ref_tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            cand_inputs = cand_tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)

            if not torch.equal(ref_inputs["input_ids"], cand_inputs["input_ids"]):
                raise AssertionError(f"token IDs differ at row {row_idx}")
            if not torch.equal(ref_inputs["attention_mask"], cand_inputs["attention_mask"]):
                raise AssertionError(f"attention masks differ at row {row_idx}")

            if ref_model is not None and cand_model is not None:
                ref_batch = {k: v.to(args.device) for k, v in ref_inputs.items()}
                cand_batch = {k: v.to(args.device) for k, v in cand_inputs.items()}
                with torch.no_grad():
                    ref_logits = get_logits(ref_model, ref_batch)
                    cand_logits = get_logits(cand_model, cand_batch)

                max_logit = (ref_logits - cand_logits).abs().max().item()
                kl = F.kl_div(
                    F.log_softmax(cand_logits, dim=-1),
                    F.log_softmax(ref_logits, dim=-1),
                    log_target=True,
                    reduction="batchmean",
                ).item()
                if not math.isfinite(kl):
                    raise AssertionError(f"non-finite KL at row {row_idx}: {kl}")
                worst_kl = max(worst_kl, kl)
                worst_logit = max(worst_logit, max_logit)
                if kl > args.kl_atol:
                    raise AssertionError(f"KL too high at row {row_idx}: {kl} > {args.kl_atol}")
                if max_logit > args.logit_atol:
                    raise AssertionError(
                        f"max logit diff too high at row {row_idx}: {max_logit} > {args.logit_atol}"
                    )

            checked += 1

        print(f"verified rows={checked} tokenizer=identical worst_kl={worst_kl:.8g} worst_logit_diff={worst_logit:.8g}")


if __name__ == "__main__":
    main()
