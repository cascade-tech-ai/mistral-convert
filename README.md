# Ministral Llama Convert

Convert text-only Ministral-3 checkpoints into plain `LlamaForCausalLM`
checkpoints that load with standard Transformers `Auto*` APIs.

## Why

Mistral publishes Ministral-3 with Mistral3 multimodal wrappers and tokenizer
metadata. For text-only training and inference, the language model is Llama-like.
This repo strips the wrapper, drops vision tensors, writes Llama config files,
and verifies that tokenization and logits match the reference model.

## Convert

```bash
python convert_ministral_to_llama.py \
  --source /path/to/source_snapshot \
  --output /path/to/llama_text_output
```

Supported source layouts:

- HF-shaped `model.safetensors.index.json` plus shards
- HF-shaped single `model.safetensors`
- native Mistral `consolidated.safetensors`

For native consolidated checkpoints, Q/K attention weights are converted with
the required native-to-HF RoPE permutation.

The exported `tokenizer.json` contains the corrected Mistral pre-tokenizer
regex, and the repo does not force `LlamaTokenizerFast`. Plain
`AutoTokenizer.from_pretrained(...)` loads the generic fast tokenizer backend
and matches the fixed reference tokenizer.

The chat template is kept as `chat_template.jinja`, not embedded in
`tokenizer_config.json`. The converted template removes Mistral's strict
user/assistant alternation assertion; rendering behavior for valid alternating
conversations is otherwise unchanged.

## Verify

```bash
python verify_llama_fied.py \
  --reference /path/to/reference_snapshot \
  --candidate /path/to/llama_text_output \
  --jsonl /path/to/prompts.jsonl \
  --max-rows 3 \
  --max-length 512
```

The verifier:

- renders `messages` rows with the reference chat template
- checks reference and candidate token IDs are identical
- runs both models and reports worst KL and max logit diff

FP8 forward verification requires:

```bash
uv pip install --python .venv/bin/python kernels
```

## Scope

This should work for larger Ministral-3 models with the same text architecture
and checkpoint naming conventions. Treat each model as untrusted until the
verifier reports tokenizer identity and near-zero logits/KL.
