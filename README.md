# DecisionFormer

![DecisionFormer](https://raw.githubusercontent.com/FrontiersMindAI/decisionformer/main/assets/decisionformer.svg)

The simplest way to fine-tune and serve decision models on any Hugging Face model. Give it a *state* and typed
*questions*; it returns a calibrated probability for every option in a single forward pass, with no text
generation. Three lines to decide, one command to train, one command to serve.

## Installation

Python 3.10 or newer, with PyTorch 2.6+ and transformers 5.20+.

```bash
pip install decisionformer                 # the library and CLI
pip install "decisionformer[serve]"        # with the API server
```

The latest code, or a development install:

```bash
pip install "decisionformer[serve] @ git+https://github.com/FrontiersMindAI/decisionformer.git"

git clone https://github.com/FrontiersMindAI/decisionformer.git
cd decisionformer
pip install -e ".[dev]"
```

Optional extras, combined as `pip install "decisionformer[serve,qlora]"`:

| extra | adds |
|---|---|
| `serve` | the `/v1/systemone` API server (FastAPI, uvicorn) |
| `qlora` | bitsandbytes, for QLoRA |
| `vllm` | vLLM, for the vLLM serving backend; Decision 3.0 models need its nightly build (see [Serve as an API](#serve-as-an-api)) |
| `fla` | fast Gated DeltaNet kernels for Qwen3.5 backbones |
| `data` | datasets, for Hub datasets (`hf:` inputs) and the example converters |

## Quick start

### Inference in Python

```python
from decisionformer import AutoDecisionModel

model = AutoDecisionModel.from_pretrained("vllm-sr/d3-edge")   # or your own fine-tune, local or on the Hub

response = model.decide(
    state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults",
            },
        },
        "receipt": {
            "type": "noul",
            "instructions": "Does the customer have a receipt?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["Routine", "Soon", "Today"],
        },
    },
)
# {
#   "model": "vllm-sr/d3-edge",
#   "answers": {
#     "route": {
#       "type": "choice",
#       "choice": "returns",
#       "probabilities": {"returns": 0.9927, "billing": 0.0001, "technical": 0.0073},
#       "confidence": 0.9602
#     },
#     "receipt": {
#       "type": "noul",
#       "noul": 0.9995
#     },
#     "urgency": {
#       "type": "score",
#       "score": 0.5777,
#       "probabilities": {"0": 0.5393, "1": 0.3438, "2": 0.1169},
#       "confidence": 0.1343,
#       "legend": {"0": "Routine", "1": "Soon", "2": "Today"}
#     }
#   },
#   "usage": {"input_tokens": 345, "output_tokens": 0}
# }
```

`choice` picks one of the `criteria`, `noul` answers yes/no with the probability of yes, and `score` picks one of
2 to 10 ordered levels; every question needs a `type`. `confidence` is not the top probability: it is 1 minus the
normalized entropy of the probabilities, from 0 (all options equally likely) to 1 (certain), so a 0.51/0.49 choice
has a confidence near 0. Checkpoints in the AutoJev format report the top probability's margin over chance instead.
`model.decide_many([...])` answers several requests in shared batches.

A model refuses requests longer than the length it was trained on (`data.max_length`, 1,024 tokens by default) with
a `ContextLengthError` (HTTP 422 from the server) instead of truncating them; pass `max_length=` to
`from_pretrained`, or `--max-length` to `serve`, to allow longer ones. On a CPU, `vllm-sr/d3-edge` (0.6B parameters)
answers a request in several seconds; the Decision Index's latency rule (a median of at most 1,000 ms per request)
needs a GPU.

### Serve as an API

```bash
decisionformer serve vllm-sr/d3-edge --port 8000
```

```bash
curl localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for my subscription this month.",
  "questions": {
    "angry": {"type": "noul", "instructions": "Is the customer angry?"}
  }
}'
```

The server takes the same `state` (required, may be empty) and `questions` and returns the same JSON as `decide`, with `id`, `object` and
`created` added. It is the System One API used by Jev, vLLM and the Decision Index; the request and response
schemas are at `/docs`. Over-long requests are refused with HTTP 422, never truncated, and
`--api-key-env MY_KEY_VAR` requires `Authorization: Bearer <key>`. A request that arrives while the model is idle
runs at once; requests that arrive while it is busy are answered together, up to `--max-batch` (16) at a time.

For code-readout checkpoints on Qwen3.5 (the Decision 3.0 layout), vLLM can run the model behind the same API.
This needs a vLLM nightly build from 2026-10-15 or later (vLLM 0.32.0 on PyPI predates its decision-model support)
and a CUDA GPU, Ampere or newer for non-causal checkpoints such as `vllm-sr/d3-edge`. Other checkpoints fail with a
clear error that points to the default backend:

```bash
pip install "decisionformer[serve]"
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
decisionformer serve vllm-sr/d3-edge --backend vllm
```

## Supported backbones

Every family below is fine-tuned, exported, reloaded and checked against the Decision Index's response rules in
`tests/test_zoo.py`, using tiny randomly initialized models of each architecture, newest first within each group.
Real checkpoints fine-tuned, exported and reloaded so far: Qwen3-0.6B, Qwen3.5-0.8B, SmolLM2-135M and 360M,
ModernBERT-base and T5-small; `vllm-sr/d3-edge` is loaded and served (`docs/benchmarks.md`).

| | families |
|---|---|
| **Decoders, dense** | Gemma 4, Gemma 3, Qwen3, SmolLM3, Seed-OSS, EXAONE 4, GLM-4, Ministral, Cohere 2, Hunyuan, OLMo 2, Granite, Phi-3, Qwen2, Gemma 2, Gemma, Mistral, Llama, Phi, StableLM, Falcon, GPT-NeoX, GPT-2 |
| **Decoders, mixture of experts** | GPT-OSS, Llama 4, DeepSeek-V3, Qwen3-MoE, Granite-MoE, OLMoE, Phi-MoE, Mixtral |
| **Decoders, hybrid linear attention** | Qwen3.5 (saved whole, with its vision tower), Qwen3-Next |
| **Encoders** | ModernBERT, EuroBERT, DeBERTa-v3 (and v1/v2), XLM-RoBERTa, RoBERTa, ELECTRA, ALBERT, DistilBERT, BERT, T5 and mT5 (the encoder) |

Backbones are the architectures built into Hugging Face transformers, loaded with its `Auto` classes and saved with
`save_pretrained`; decisionformer has no model code of its own. Hub models that need their own modeling code
(`trust_remote_code`) are not supported. Decoders read the answer as a code token (`A`, `B`, ...) at the end of
the prompt, which works from step 0; encoders score each option. Encoder-decoder models without a standalone text
encoder in transformers (BART) are refused with a clear error. Fine-tuning Qwen3.5 with the answer-code readout
has not learned reliably in our runs yet (`docs/benchmarks.md`); loading and serving Qwen3.5 checkpoints such as
`vllm-sr/d3-edge` works.

## Fine-tune

```bash
decisionformer data split hf:FrontiersMind/deciformer-15k:train --out data/deciformer-15k
decisionformer train deciformer15k_qwen3                  # Qwen3-0.6B + LoRA, one epoch
torchrun --nproc-per-node 8 -m decisionformer train deciformer15k_qwen3                     # multi-GPU
```

`deciformer15k_qwen3` is a bundled recipe: `decisionformer recipes` lists them all, and
`decisionformer init deciformer15k_qwen3` copies one to a YAML file you can edit. Every setting lives in that YAML,
and any value can be overridden on the command line with dotted keys (`train.lr=1e-4`).
`decisionformer config FILE` prints the resolved configuration. A minimal config:

```yaml
model:
  backbone: Qwen/Qwen3-0.6B          # any Hub id or local path
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
  push_to_hub: my-org/my-decider     # optional: uploads best/ when training ends (private by default)
```

Everything else has a default. The learning rate follows the fine-tuning mode (2e-5 full, 2e-4 LoRA and QLoRA,
1e-3 head only). Training evaluates and saves resumable state every 500 steps, and rerunning the same command
resumes where it stopped; the resumable state is removed once the run finishes. A run writes `runs/<name>/best` and `last`: finished checkpoints in the standard Hugging
Face layout, ready for `from_pretrained`, `serve` and `push`. It also writes `metrics.jsonl`, per-example
predictions at every evaluation and `summary.json`. Training warns when the training data shares ids, families or
identical states with the evaluation or calibration data.

The same from Python, with a recipe name, a YAML file or a dict:

```python
import decisionformer

summary = decisionformer.train("deciformer15k_qwen3", {"train.max_steps": 200})
model = decisionformer.AutoDecisionModel.from_pretrained("runs/deciformer15k-qwen3-0.6b/best")
```

### On a laptop (Mac or Linux)

A small model trains on a laptop. This recipe fully fine-tunes SmolLM2-135M-Instruct on deciformer-15k for 300
steps:

```bash
pip install "decisionformer[data]"
decisionformer data split hf:FrontiersMind/deciformer-15k:train --out data/deciformer-15k
decisionformer train local_smollm2
decisionformer serve runs/local-smollm2-135m/best        # try it on /v1/systemone
```

The device is picked automatically: a CUDA GPU, then Apple silicon (MPS), then the CPU. Set
`DECISIONFORMER_DEVICE=cpu` to force the CPU. On MPS and CPU, training runs in float32. The recipe keeps memory
low with micro-batches of 4, at most 4,096 tokens per micro-batch and gradient checkpointing: on a 6-core CPU it
peaked at 3.7 GB of memory and took about 15 s per step. If memory is still tight, lower them:
`train.micro_batch_size=2 train.token_budget=2048`. Resumable state is saved every 100 steps, and rerunning the
same command resumes from it.

### Training data

Training data is the request with the gold answers, one JSON object per line. One record, spread out here for
reading:

```json
{
  "id": "t-17",
  "family": "ticket-17",
  "state": {"ticket": 17, "text": "I was charged twice this month."},
  "questions": {
    "route": {
      "type": "choice",
      "criteria": {"billing": "Payments and invoices", "returns": null}
    },
    "angry": {
      "type": "noul",
      "instructions": "Is the customer angry?"
    }
  },
  "gold": {"route": "billing", "angry": false},
  "targets": {"route": {"billing": 0.9, "returns": 0.1}}
}
```

`gold` may also be called `expected` or `answers`. `targets` (soft labels, such as vote shares) and `teachers`
(probabilities to distill from) are optional, per question. A record with a single `question` and a `label` also
works, and so do flat rows with one question each, as the deciformer pipeline writes them: `state`, `question`
(the instructions), `question_type`, `options`, `option_descriptions`, `label` and `soft_target`. `family` keeps
related records together in splits, data checks and bootstrap intervals. Files can be `.jsonl`, `.jsonl.gz`, or a
Hub dataset written as `hf:<name>[:<config>]:<split>`.

`decisionformer data split` holds out 10% for calibration and 10% for dev by default (`--holdout NAME=FRACTION`),
keeping whole families together and each task's share equal in every split; the source is not modified.
`decisionformer data check dev=dev.jsonl calibration=calibration.jsonl train=train.jsonl` finds leakage between
splits.

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

### Precision

Training runs in bf16 by default. Every part is configurable:

```yaml
train:
  mixed_precision: bf16      # bf16 (default) | fp16 (with loss scaling) | no (float32)
finetune:
  model_dtype: auto          # weights while training: float32 | bfloat16 | float16
  export_dtype: auto         # saved checkpoints: auto = the base checkpoint's dtype | float32 | bfloat16 | float16
  quantization:
    compute_dtype: auto      # QLoRA compute: auto = follows mixed_precision
```

With `model_dtype: auto`, LoRA keeps the frozen base in bf16 and QLoRA in 4-bit, and full fine-tuning keeps
float32 master weights while the compute runs in bf16, as PyTorch AMP does: pure bf16 weights lose small optimizer
updates. Set `model_dtype: bfloat16` for pure bf16 training. GPUs without bf16 (Turing, V100) fall back to fp16
with loss scaling and log a warning; CPU runs use float32. Exports are saved in the base checkpoint's dtype (bf16
for most current models), so a fine-tuned model is the same size as the one it started from.

### Heads

| head | what it does | good for |
|---|---|---|
| `code_readout` | Scores the answer codes (`A`, `B`, ... `ZZ`) at the last prompt token, initialized from the LM head | Decoders; up to 255 options; strong from step 0; the standard, servable layout |
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
`{enable_thinking: false}`. Register your own format with `decisionformer.register_format`. Training examples
longer than `data.max_length` have the end of their state cut (`data.overflow`); loaded and served models never
truncate and refuse what does not fit instead.

### Decoder attention

| `model.attention` | behaviour | saved as |
|---|---|---|
| `causal` (default) | Standard left-to-right attention | `causal` |
| `bidirectional` | Full-attention layers see the whole prompt; recurrent and linear-attention layers (Qwen3.5's Gated DeltaNet) stay causal | `noncausal_full_attention` |
| `prefix_bidirectional` | The prompt is bidirectional and the answer suffix causal | `prefix_bidirectional` |

Bidirectional mode uses transformers' own switches (`is_causal=False` on the text decoder only), so SDPA,
flash and sliding-window kernels all respect it, and a vision tower keeps its own attention. Conformance tests
check every family in `tests/test_attention.py`. Compare against a causal baseline before adopting it: a causally
pretrained decoder made bidirectional loses most of what it knows. On deciformer-15k, Qwen3-0.6B reached 0.45 after
50 steps with bidirectional attention against 0.67 with causal attention, and on MNLI SmolLM2-360M stayed near
chance where causal reached 0.80 (`docs/benchmarks.md`).

### Multi-GPU (DDP)

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

## Evaluate, upload and benchmark

```bash
decisionformer eval runs/my-decider/best data/dev.jsonl --predictions dev.predictions.jsonl
decisionformer predict runs/my-decider/best requests.jsonl --output responses.jsonl
decisionformer compare a.predictions.jsonl b.predictions.jsonl       # paired bootstrap over families
decisionformer push runs/my-decider/best my-org/my-decider            # private unless --public
```

Checkpoints are plain Hugging Face model repos: the backbone's `config.json` and safetensors, the tokenizer,
`decision_config.json`, `readout.safetensors` (or `heads.safetensors` for the other heads) and a model card. In
Python, `model.push_to_hub("my-org/my-decider")` does the same.

`from_pretrained` also loads the other checkpoints of this standard, such as Decision 3.0 (`vllm-sr/d3*`) and
AutoJev exports, with each one's own prompt, attention mode and temperatures. On `vllm-sr/d3-edge` it reproduces
the checkpoint's reference runtime exactly (identical probabilities in BF16, one question per pass).

Run any checkpoint through the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
kit in-process, or serve it and use the kit's `http` engine:

```bash
pip install "decision-index @ git+https://github.com/apolinario/decision-index"
python -m decision_index pipeline --engine decisionformer.integrations.decision_index:DecisionFormerEngine \
    --option model=my-org/my-decider --out runs/index-my-decider
```

## Measured results

On one 8 GB RTX 2060 SUPER in FP16, the starting recipe `deciformer15k_qwen3` (Qwen3-0.6B with LoRA, one epoch of
deciformer-15k, 30 minutes) reaches **0.824** dev accuracy across its 9 tasks, from 0.75 (VitaminC) to 0.90 (Civil
Comments), with calibrated ECE 0.023. On MNLI: Qwen3-0.6B with LoRA and `code_readout` went from 0.36 zero-shot to
**0.83** dev accuracy in 250 steps (calibrated ECE 0.042), and Qwen3.5-0.8B with QLoRA and bidirectional attention
reached **0.84** in 150 steps. The vLLM backend matched the transformers backend to within 0.001 per probability, with identical choices,
at a median of 30 ms per 3-question request. Configs and full tables are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Development

```bash
pip install -e ".[dev,data]"
pytest -q                 # offline, tiny random models: about 2 minutes on a laptop CPU
pytest -q -m "not slow"   # unit and attention-conformance tests only
ruff check src tests
```

Experimental: `decisionformer adapt` runs masked next-token prediction (as in LLM2Vec) before bidirectional
training (the `adapt_mntp` recipe). A short stage did not help in our tests.

## Roadmap

FSDP2 for full fine-tuning beyond the DDP limits, encoding a request's state once for all of its questions,
image and video inputs, and more heads (`multi_label`, `regression`).

## License and acknowledgements

Apache-2.0. The option-code readout, prompt layouts, calibration metrics and near-duplicate checks are adapted
from [AutoJev](https://github.com/denis-pplx/autojev) (MIT; see `NOTICE`). The `d3` prompt and checkpoint
contract follow Decision 3.0 ([`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3)).

## Citation

If you use DecisionFormer in your work, please cite it:

```bibtex
@software{decisionformer2026,
  title  = {{DecisionFormer}: Fine-tune, load and serve one-pass decision models on Hugging Face transformers},
  author = {{FrontiersMind AI}},
  year   = {2026},
  url    = {https://github.com/FrontiersMindAI/decisionformer},
  note   = {Version 0.2}
}
```
