# DecisionFormer

**Fine-tune, load and serve one-pass decision models on any Hugging Face transformers backbone.**

A decision model reads a *state* (text or any JSON value) and answers typed questions about it, with a calibrated
probability for every option, in a single forward pass and with no text generation. That makes it fast, cheap
and easy to calibrate, which suits routing, moderation, classification, judging, entailment and extraction gates.

DecisionFormer is to decision models what `transformers` is to language models:

```python
from decisionformer import AutoDecisionModel

model = AutoDecisionModel.from_pretrained("vllm-sr/d3-edge")   # or your own fine-tune, local or on the Hub
model.decide("The order arrived damaged. The customer has a receipt and wants a replacement today.", {
    "route":   {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"returns": "Refunds and replacements", "billing": "Payments", "technical": "Faults"}},
    "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Routine", "Soon", "Today"]},
})
# {"model": ..., "answers": {"route": {"type": "choice", "choice": "returns", "probabilities": {...}, "confidence": ...},
#                            "receipt": {"type": "noul", "noul": 0.9995}, "urgency": {"type": "score", "score": ...}},
#  "usage": {"input_tokens": ..., "output_tokens": 0}}
```

| | |
|---|---|
| **Train** | Any encoder (BERT, RoBERTa, DeBERTa, ModernBERT, XLM-R, T5) or decoder (Llama, Qwen, Gemma, Mistral, Phi, OLMo, SmolLM, GPT-OSS, DeepSeek-V3, Qwen3.5 and more, including mixture-of-experts and hybrid linear-attention models) |
| **Fine-tuning** | `full`, `lora` (every `peft.LoraConfig` option), `qlora` (4- or 8-bit), `head_only` |
| **Multi-GPU** | DDP through `torchrun`, with gradients identical to single-GPU training, optional ZeRO-1, and exact resume |
| **Calibration** | Temperature per question type, fitted on a separate fold; ECE, Brier and NLL; constrained checkpoint selection |
| **Checkpoints** | The shared standard of AutoJev and Decision 3.0: a plain transformers backbone, `decision_config.json` and `readout.safetensors`. Upload with `decisionformer push`, load with `from_pretrained` |
| **Serve** | `decisionformer serve` on `/v1/systemone`, with transformers or vLLM computing the model |
| **Benchmark** | A Decision Index engine, and a server that speaks the Index's `http` engine protocol |

## Install

```bash
pip install -e .              # from a clone: Python >= 3.10, torch >= 2.6, transformers >= 5.20
pip install -e ".[serve]"     # the /v1/systemone server (FastAPI, uvicorn)
pip install -e ".[qlora]"     # bitsandbytes for QLoRA
pip install -e ".[vllm]"      # the vLLM serving backend (a vLLM build from 2026-10-15 or later)
pip install -e ".[fla]"       # fast Gated DeltaNet kernels for Qwen3.5 backbones
pip install -e ".[data]"      # datasets, for the example converters and hf: inputs
```

## The System One format

A request has a `state` and named `questions`. Each question has a `type`:

| type | options | answer |
|---|---|---|
| `choice` | `criteria`: an object of option → description (or `null`), or a list of option names | `choice`, `probabilities`, `confidence` |
| `noul` | none (yes/no) | `noul`: the probability of yes |
| `score` | `criteria`: 2 to 10 ordered levels | `score` (expected level), `probabilities`, `confidence`, `legend` |

Training data is the same request with the gold answers: one JSON object per line.

```json
{"id": "t-17", "family": "ticket-17",
 "state": {"ticket": 17, "text": "I was charged twice this month."},
 "questions": {"route": {"type": "choice", "criteria": {"billing": "Payments and invoices", "returns": null}},
               "angry": {"type": "noul", "instructions": "Is the customer angry?"}},
 "gold": {"route": "billing", "angry": false},
 "targets": {"route": {"billing": 0.9, "returns": 0.1}}}
```

`gold` may also be called `expected` or `answers`. `targets` (soft labels, such as vote shares) and `teachers`
(probabilities to distill from) are optional, per question. A record with a single `question` and a `label` also
works. `family` keeps related records together in splits, data checks and bootstrap intervals. Files can be
`.jsonl`, `.jsonl.gz`, or a Hub dataset written as `hf:<name>[:<config>]:<split>`.
`decisionformer data check test=test.jsonl dev=dev.jsonl train=train.jsonl` finds leakage between splits.

## Fine-tune

```bash
python examples/prepare_datasets.py mnli --output data/mnli --train-size 20000
decisionformer train configs/recipes/decoder_lora_code_readout.yaml
torchrun --nproc-per-node 8 -m decisionformer train configs/recipes/decoder_lora_code_readout.yaml train.lr=1e-4
```

Every setting lives in a YAML config, and any value can be overridden with dotted keys. `decisionformer config
FILE` prints the resolved configuration. A minimal config:

```yaml
model:
  backbone: Qwen/Qwen3.5-0.8B        # any Hub id or local path
  attention: causal                  # causal | bidirectional | prefix_bidirectional
  head: {type: code_readout}
finetune:
  mode: lora                         # full | lora | qlora | head_only
  lora: {r: 16, lora_alpha: 32, target_modules: all-linear}
data:
  train: data/train.jsonl
  calibration: data/calibration.jsonl
  eval: {dev: data/dev.jsonl}
train:
  output_dir: runs/my-decider
  lr: 2.0e-4
  push_to_hub: my-org/my-decider     # optional: uploads best/ when training ends (private by default)
```

A run writes `runs/<name>/best` and `last`: finished, merged checkpoints in the standard layout, plus
`metrics.jsonl`, per-example predictions at every evaluation, resumable state and `summary.json`.

### Fine-tuning modes

```yaml
finetune:
  mode: qlora
  gradient_checkpointing: true
  quantization: {bits: 4, quant_type: nf4, double_quant: true, compute_dtype: bfloat16}
  lora:                          # passed to peft.LoraConfig with peft's own field names
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: all-linear   # list | regex | all-linear | attention | mlp | attention_mlp
    use_dora: false
    use_rslora: false
    init_lora_weights: true      # true | gaussian | pissa | olora | loftq | ...
    rank_pattern: {q_proj: 32}
  export: merged                 # merged (default: plain weights, no peft needed) | adapter
```

The LoRA block is checked against peft when the config loads. `all-linear` leaves mixture-of-experts routers
out. The decision head always trains fully, with its own `train.head_lr`. Full fine-tuning also supports
`freeze_embeddings`, `freeze_layers: "0-7"` and `layerwise_lr_decay`.

### Heads

| head | what it does | good for |
|---|---|---|
| `code_readout` | Scores the answer codes (`A`, `B`, ... `ZZ`) at the last prompt token, initialized from the LM head | Decoders; up to 255 options; strong from step 0; the standard, servable format |
| `lm_verbalizer` | The frozen LM head restricted to the codes, with no new parameters | Zero-shot baselines |
| `option_scoring` | Pools each option's own tokens and scores it with a shared MLP | Encoders; any number of options; order-invariant |
| `fixed_softmax` | A classifier over a fixed `labels` list | A fixed label set |
| `binary` | One logit for yes | `noul` questions |
| `ordinal` | A cumulative-link model with increasing thresholds | `score` questions |

Use `model.head` for one task or `model.heads: {task_name: {...}, default: {...}}` for several. A head can set
its own pooler (`decision_token`, `last_token`, `cls`, `mean`, `attention`, `option_spans`).

### Prompts

`data.prompt.style` picks the prompt: `d3` (the default: Decision 3.0's byte-exact prompt, servable by vLLM) or
`autojev` (AutoJev's). Decoders use the tokenizer's own chat template, and `chat_template_kwargs` defaults to
`{enable_thinking: false}`. Prompts are never truncated: a request that does not fit is refused, as the Decision
Index requires. Register your own format with `decisionformer.register_format`.

### Decoder attention

| `model.attention` | behaviour | saved as |
|---|---|---|
| `causal` (default) | Standard left-to-right attention | `causal` |
| `bidirectional` | Full-attention layers see the whole prompt; recurrent and linear-attention layers (Qwen3.5's Gated DeltaNet) stay causal | `noncausal_full_attention` |
| `prefix_bidirectional` | The prompt is bidirectional and the answer suffix causal | `prefix_bidirectional` |

Bidirectional mode uses transformers' own switches (`is_causal=False` on the text decoder only), so SDPA,
flash and sliding-window kernels all respect it, and a vision tower keeps its own attention. Conformance tests
check every family in `tests/test_attention.py`. A causally pretrained decoder needs real training to benefit:
in our MNLI tests, switching SmolLM2-360M to bidirectional without adaptation stayed near chance where causal
reached 0.80 (`docs/benchmarks.md`). Keep `causal` unless you have a reason and a baseline to compare against.

## Load and use

```python
from decisionformer import AutoDecisionModel

model = AutoDecisionModel.from_pretrained("runs/my-decider/best", device="cuda")
response = model.decide(state, questions)              # one request
responses = model.decide_many([{"state": ..., "questions": ...}, ...])
```

`from_pretrained` loads decisionformer checkpoints and the other checkpoints of the standard, such as Decision
3.0 (`vllm-sr/d3*`) and AutoJev exports, with each one's own prompt, attention mode and temperatures. On
`vllm-sr/d3-edge` it reproduces the checkpoint's reference runtime exactly (identical probabilities, BF16, one
question per pass). Batching several questions per pass changes BF16 rounding by up to 0.008, as in the runtime itself.

```bash
decisionformer predict runs/my-decider/best requests.jsonl --output responses.jsonl
decisionformer eval runs/my-decider/best data/dev.jsonl --predictions dev.predictions.jsonl
decisionformer compare a.predictions.jsonl b.predictions.jsonl     # paired bootstrap over families
```

## Upload

Checkpoints are plain Hugging Face model repos: the backbone's `config.json` and safetensors, the tokenizer,
`decision_config.json`, `readout.safetensors` (or `heads.safetensors` for the other heads) and a model card.

```bash
decisionformer push runs/my-decider/best my-org/my-decider          # private unless --public
```

```python
model.push_to_hub("my-org/my-decider")
```

## Serve

```bash
pip install -e ".[serve]"
decisionformer serve my-org/my-decider --port 8000                  # transformers backend
curl localhost:8000/v1/systemone -H 'content-type: application/json' \
  -d '{"state": "I was charged twice.", "questions": {"angry": {"type": "noul"}}}'
```

The server takes the System One request and returns the answers in the same shape as the other System One
servers. It refuses over-long requests with HTTP 422 and the phrases the Decision Index counts as unsupported,
and it can require an API key: `--api-key-env MY_KEY_VAR` reads the key from that variable.

**vLLM backend.** For code-readout checkpoints on a Qwen3.5 backbone (the Decision 3.0 contract), vLLM can
compute the model while DecisionFormer keeps the prompt, temperatures and answers, so both backends answer the
same way:

```bash
pip install -e ".[vllm]"
decisionformer serve my-org/my-qwen35-decider --backend vllm
```

Other checkpoints (encoders, other decoder families, non-readout heads, prefix attention) fail with a clear
`UnsupportedBackend` error that points to the default backend. On `vllm-sr/d3-edge` in causal mode (RTX 2060
SUPER, FP16), the vLLM backend matched the FP32 transformers backend to within 0.001 per probability, with
identical choices, at a median of 30 ms per 3-question request. vLLM's non-causal attention kernels need more
shared memory than Turing GPUs have, so non-causal checkpoints need an Ampere or newer GPU on this backend.

## Decision Index

Run any checkpoint through the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
kit in-process:

```bash
pip install "decision-index @ git+https://github.com/apolinario/decision-index"
python -m decision_index pipeline --engine decisionformer.integrations.decision_index:DecisionFormerEngine \
    --option model=my-org/my-decider --out runs/index-my-decider
```

Or serve it with `decisionformer serve` and use the kit's `http` engine. Requests that do not fit are reported as
unsupported, never truncated.

## Tested models

`tests/test_zoo.py` fine-tunes a tiny random checkpoint of each family below, exports it, reloads it with
`AutoDecisionModel` and checks the response against the Decision Index's rules:

* **Encoders:** BERT, RoBERTa, XLM-RoBERTa, ELECTRA, ALBERT, DeBERTa, DeBERTa-v2/v3, DistilBERT, ModernBERT,
  EuroBERT, T5 and mT5 (encoder).
* **Decoders:** Llama, Mistral, Ministral, Qwen2, Qwen3, Gemma 1–4, Phi, Phi-3, Granite, OLMo 2, SmolLM3, GLM-4,
  StableLM, GPT-2, GPT-NeoX, Falcon, Cohere 2, EXAONE 4, Seed-OSS, Hunyuan.
* **Mixture of experts:** Mixtral, Qwen3-MoE, OLMoE, Granite-MoE, Phi-MoE, Llama 4, GPT-OSS, DeepSeek-V3.
* **Hybrid linear attention:** Qwen3-Next, Qwen3.5 (saved whole, with its vision tower).

Encoder-decoder models without a standalone text encoder in transformers (BART) are refused with a clear error.

## Measured results

On one 8 GB RTX 2060 SUPER in FP16, after one epoch of 8k MNLI examples: Qwen3-0.6B with LoRA and `code_readout`
went from 0.34 zero-shot to **0.80** dev accuracy (calibrated ECE 0.033). QLoRA reached 0.795, SmolLM2-360M with
`option_scoring` 0.796, and ModernBERT-base 0.675. Configs, full tables and the bidirectional results are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Multi-GPU (DDP)

Launch with `torchrun`. NCCL is used on GPUs and gloo on CPU. Every rank computes the same step plan with
balanced token counts. Ranks with fewer micro-batches run zero-weight padding batches, so synchronization never
deadlocks, and the loss is scaled so that DDP averaging yields the global-batch mean (tested against one
process for 1, 2 and 3 ranks). `train.shard_optimizer_states: true` adds ZeRO-1, and `train.save_every` with
`train.resume: auto` resumes exactly, data order and RNG included.

| mode (per 80 GB GPU, approximate) | fits about |
|---|---|
| full, mixed precision | ≤ 3B |
| full + `shard_optimizer_states` on 8 GPUs | 5–7B |
| LoRA (bf16 base) | 14B comfortably, 30B tight |
| QLoRA (4-bit base) | 30B comfortably, 70B tight |

## Development

```bash
uv venv && uv pip install -e ".[dev,data]"
pytest -q                 # offline, tiny random models: about 2 minutes on a laptop CPU
pytest -q -m "not slow"   # unit and attention-conformance tests only
ruff check src tests
```

Experimental: `decisionformer adapt` runs masked next-token prediction (as in LLM2Vec) before bidirectional
training (`configs/recipes/adapt_mntp.yaml`). A short stage did not help in our tests.

## Roadmap

FSDP2 for full fine-tuning beyond the DDP limits, image and video inputs, several questions per sequence, and
more heads (`multi_label`, `regression`).

## License and acknowledgements

Apache-2.0. The option-code readout, prompt layouts, calibration metrics and near-duplicate checks are adapted
from [AutoJev](https://github.com/denis-pplx/autojev) (MIT; see `NOTICE`). The `d3` prompt and checkpoint
contract follow Decision 3.0 ([`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3)).
