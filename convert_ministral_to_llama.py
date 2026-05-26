#!/usr/bin/env python3
"""Convert Ministral-3 text weights into a plain Llama CausalLM checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
LANGUAGE_PREFIX = "language_model."
FIXED_MISTRAL_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
ALTERNATION_CHECK_BLOCK = """{#- Checks for alternating user/assistant messages. #}
{%- set ns = namespace(index=0) %}
{%- for message in loop_messages %}
    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}
        {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}
            {{- raise_exception('After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results.') }}
        {%- endif %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=DEFAULT_MODEL_ID,
        help="HF model id or local snapshot path. Defaults to the Ministral-3 3B instruct BF16 repo.",
    )
    parser.add_argument("--output", required=True, help="Output directory for the Llama-compatible checkpoint.")
    parser.add_argument(
        "--max-shard-size",
        default=None,
        help=(
            "Accepted for CLI compatibility. Conversion preserves the upstream text shard split "
            "instead of re-sharding, so this value is currently informational."
        ),
    )
    return parser.parse_args()


def resolve_source(source: str) -> Path:
    path = Path(source)
    if path.exists():
        return path.resolve()

    return Path(
        snapshot_download(
            source,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "model.safetensors.index.json",
                "model-*.safetensors",
            ],
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def llama_config_from_mistral3(config: dict[str, Any]) -> dict[str, Any]:
    text = dict(config["text_config"])
    rope = dict(text.get("rope_parameters") or {})
    rope_scaling = {
        "rope_type": rope.get("rope_type", rope.get("type")),
        "factor": rope.get("factor"),
        "original_max_position_embeddings": rope.get("original_max_position_embeddings"),
        "beta_fast": rope.get("beta_fast"),
        "beta_slow": rope.get("beta_slow"),
        "mscale": rope.get("mscale"),
        "mscale_all_dim": rope.get("mscale_all_dim"),
    }
    rope_scaling = {k: v for k, v in rope_scaling.items() if v is not None}

    llama = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": config.get("torch_dtype", config.get("dtype", "bfloat16")),
        "dtype": config.get("dtype", config.get("torch_dtype", "bfloat16")),
        "vocab_size": text["vocab_size"],
        "hidden_size": text["hidden_size"],
        "intermediate_size": text["intermediate_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"],
        "hidden_act": text.get("hidden_act", "silu"),
        "max_position_embeddings": text["max_position_embeddings"],
        "initializer_range": text.get("initializer_range", 0.02),
        "rms_norm_eps": text.get("rms_norm_eps", 1e-5),
        "use_cache": text.get("use_cache", True),
        "tie_word_embeddings": text.get("tie_word_embeddings", True),
        "attention_dropout": text.get("attention_dropout", 0.0),
        "rope_theta": rope.get("rope_theta", text.get("rope_theta", 10000.0)),
        "rope_scaling": rope_scaling,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 11,
        "transformers_version": installed_transformers_version(),
    }
    if "quantization_config" in config:
        quantization_config = dict(config["quantization_config"])
        modules_to_not_convert = quantization_config.get("modules_to_not_convert")
        if modules_to_not_convert:
            quantization_config["modules_to_not_convert"] = [
                name
                for name in modules_to_not_convert
                if "vision_tower" not in name and "multi_modal_projector" not in name
            ]
        llama["quantization_config"] = quantization_config
    return llama


def installed_transformers_version() -> str:
    try:
        return version("transformers")
    except PackageNotFoundError:
        return "5.0.0"


def write_fixed_tokenizer_json(source: Path, output: Path) -> None:
    src = source / "tokenizer.json"
    if not src.exists():
        return

    tokenizer = load_json(src)
    pre_tokenizer = tokenizer.get("pre_tokenizer")
    if isinstance(pre_tokenizer, dict):
        if pre_tokenizer.get("type") == "Sequence":
            pretokenizers = pre_tokenizer.get("pretokenizers") or []
            if pretokenizers and pretokenizers[0].get("type") == "Split":
                pretokenizers[0]["pattern"] = {"Regex": FIXED_MISTRAL_REGEX}
        elif pre_tokenizer.get("type") == "Split":
            pre_tokenizer["pattern"] = {"Regex": FIXED_MISTRAL_REGEX}

    dump_json(output / "tokenizer.json", tokenizer)


def write_tokenizer_files(source: Path, output: Path) -> None:
    write_fixed_tokenizer_json(source, output)
    for name in ["special_tokens_map.json"]:
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)
    chat_template = source / "chat_template.jinja"
    if chat_template.exists():
        template = chat_template.read_text(encoding="utf-8")
        template = template.replace(ALTERNATION_CHECK_BLOCK, "")
        (output / "chat_template.jinja").write_text(template, encoding="utf-8")

    tok_cfg = load_json(source / "tokenizer_config.json")
    tok_cfg.pop("processor_class", None)
    tok_cfg.pop("tokenizer_class", None)
    tok_cfg.pop("fix_mistral_regex", None)
    tok_cfg.pop("chat_template", None)
    dump_json(output / "tokenizer_config.json", tok_cfg)


def transformers_key_from_native(key: str) -> str | None:
    if key == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    if key == "norm.weight":
        return "model.norm.weight"
    if key == "output.weight":
        return "lm_head.weight"
    if key.startswith(("vision_encoder.", "vision_language_adapter.", "patch_merger.", "pre_mm_projector_norm.")):
        return None
    if "_fake_quantizer." in key:
        return None
    if not key.startswith("layers."):
        return None

    parts = key.split(".")
    layer = parts[1]
    rest = ".".join(parts[2:])
    replacements = {
        "attention.wq.weight": f"model.layers.{layer}.self_attn.q_proj.weight",
        "attention.wk.weight": f"model.layers.{layer}.self_attn.k_proj.weight",
        "attention.wv.weight": f"model.layers.{layer}.self_attn.v_proj.weight",
        "attention.wo.weight": f"model.layers.{layer}.self_attn.o_proj.weight",
        "attention.wq.qscale_weight": f"model.layers.{layer}.self_attn.q_proj.weight_scale_inv",
        "attention.wk.qscale_weight": f"model.layers.{layer}.self_attn.k_proj.weight_scale_inv",
        "attention.wv.qscale_weight": f"model.layers.{layer}.self_attn.v_proj.weight_scale_inv",
        "attention.wo.qscale_weight": f"model.layers.{layer}.self_attn.o_proj.weight_scale_inv",
        "attention.wq.qscale_act": f"model.layers.{layer}.self_attn.q_proj.activation_scale",
        "attention.wk.qscale_act": f"model.layers.{layer}.self_attn.k_proj.activation_scale",
        "attention.wv.qscale_act": f"model.layers.{layer}.self_attn.v_proj.activation_scale",
        "attention.wo.qscale_act": f"model.layers.{layer}.self_attn.o_proj.activation_scale",
        "attention_norm.weight": f"model.layers.{layer}.input_layernorm.weight",
        "ffn_norm.weight": f"model.layers.{layer}.post_attention_layernorm.weight",
        "feed_forward.w1.weight": f"model.layers.{layer}.mlp.gate_proj.weight",
        "feed_forward.w2.weight": f"model.layers.{layer}.mlp.down_proj.weight",
        "feed_forward.w3.weight": f"model.layers.{layer}.mlp.up_proj.weight",
        "feed_forward.w1.qscale_weight": f"model.layers.{layer}.mlp.gate_proj.weight_scale_inv",
        "feed_forward.w2.qscale_weight": f"model.layers.{layer}.mlp.down_proj.weight_scale_inv",
        "feed_forward.w3.qscale_weight": f"model.layers.{layer}.mlp.up_proj.weight_scale_inv",
        "feed_forward.w1.qscale_act": f"model.layers.{layer}.mlp.gate_proj.activation_scale",
        "feed_forward.w2.qscale_act": f"model.layers.{layer}.mlp.down_proj.activation_scale",
        "feed_forward.w3.qscale_act": f"model.layers.{layer}.mlp.up_proj.activation_scale",
    }
    return replacements.get(rest)


def convert_sharded_transformers_weights(source: Path, output: Path) -> None:
    index = load_json(source / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]
    input_shards = sorted({filename for filename in weight_map.values()})
    output_name_by_input = {name: name for name in input_shards}
    convert_weight_files(source, output, input_shards, output_name_by_input, lambda k: k.removeprefix(LANGUAGE_PREFIX) if k.startswith(LANGUAGE_PREFIX) else None)


def convert_native_consolidated_weights(source: Path, output: Path) -> None:
    input_name = "consolidated.safetensors"
    output_name = "model.safetensors"
    config = load_json(source / "config.json")
    head_dim = int(config["text_config"]["head_dim"])
    convert_weight_files(
        source,
        output,
        [input_name],
        {input_name: output_name},
        transformers_key_from_native,
        tensor_transform=lambda key, tensor: transform_native_tensor(key, tensor, head_dim),
    )


def convert_single_transformers_weights(source: Path, output: Path) -> None:
    input_name = "model.safetensors"
    convert_weight_files(source, output, [input_name], {input_name: input_name}, lambda k: k.removeprefix(LANGUAGE_PREFIX) if k.startswith(LANGUAGE_PREFIX) else None)


def inverse_rope_permute(weight: Tensor, num_heads: int) -> Tensor:
    return (
        weight.view(num_heads, weight.shape[0] // num_heads // 2, 2, weight.shape[1])
        .transpose(1, 2)
        .reshape(weight.shape)
    )


def transform_native_tensor(key: str, tensor: Tensor, head_dim: int) -> Tensor:
    if key.endswith(("attention.wq.weight", "attention.wk.weight")):
        return inverse_rope_permute(tensor, tensor.shape[0] // head_dim)
    return tensor


def convert_weight_files(
    source: Path,
    output: Path,
    input_files: list[str],
    output_names: dict[str, str],
    key_mapper: Any,
    tensor_transform: Callable[[str, Tensor], Tensor] | None = None,
) -> None:
    new_weight_map: dict[str, str] = {}
    total_size = 0
    for input_name in input_files:
        tensors = {}
        with safe_open(source / input_name, framework="pt", device="cpu") as f:
            for key in f.keys():
                new_key = key_mapper(key)
                if new_key is None:
                    continue
                tensor = f.get_tensor(key)
                if tensor_transform is not None:
                    tensor = tensor_transform(key, tensor)
                tensors[new_key] = tensor
                shard_name = output_names[input_name]
                new_weight_map[new_key] = shard_name
                total_size += tensor.numel() * tensor.element_size()
        if tensors:
            save_file(tensors, output / output_names[input_name], metadata={"format": "pt"})

    if not new_weight_map:
        raise ValueError(f"No text tensors found in {source}")

    dump_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(new_weight_map.items()))},
    )


def convert_weights(source: Path, output: Path) -> None:
    if (source / "model.safetensors.index.json").exists():
        convert_sharded_transformers_weights(source, output)
    elif (source / "consolidated.safetensors").exists():
        convert_native_consolidated_weights(source, output)
    elif (source / "model.safetensors").exists():
        convert_single_transformers_weights(source, output)
    else:
        raise FileNotFoundError(f"No supported safetensors checkpoint found in {source}")


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = load_json(source / "config.json")
    dump_json(output / "config.json", llama_config_from_mistral3(config))

    generation_config = source / "generation_config.json"
    if generation_config.exists():
        shutil.copy2(generation_config, output / "generation_config.json")

    write_tokenizer_files(source, output)
    convert_weights(source, output)
    print(f"Wrote Llama-compatible text checkpoint to {output}")


if __name__ == "__main__":
    main()
