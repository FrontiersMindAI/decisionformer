# deciformer

**Fine-tune any Hugging Face encoder or decoder into a calibrated, one-pass decision model.**

A decision model reads a *state* (some text) and a *question* with a set of *options*, and returns a probability
for every option in a single forward pass, with no text generation. That makes it fast, cheap and easy to
calibrate, which suits routing, moderation, classification, judging, entailment and extraction gates.

deciformer turns any `transformers` backbone into one, and everything is chosen in a YAML config:

| | |
|---|---|
| **Backbones** | Encoders (BERT, RoBERTa, DeBERTa, ModernBERT, XLM-R, the T5 encoder) and decoders (Llama, Qwen, Mistral, Gemma, Phi, Granite, OLMo, SmolLM, ...) |
| **Decoder attention** | `causal` (default), `bidirectional`, or `prefix_bidirectional`, verified per model family by conformance tests |
| **Heads** | `code_readout`, `lm_verbalizer`, `option_scoring`, `fixed_softmax`, `binary`, `ordinal`, with one head per task for multitask training |
| **Poolers** | `decision_token`, `last_token`, `cls`, `mean`, `attention`, `option_spans` |
| **Fine-tuning** | `full`, `lora`, `qlora`, `head_only`; every `peft.LoraConfig` option; layer freezing and layer-wise LR decay |
| **Multi-GPU** | DDP through `torchrun`, with gradients identical to single-GPU training, optional ZeRO optimizer-state sharding, and exact resume |
| **Calibration** | Temperature fitted per task on a separate fold; ECE, Brier and NLL; checkpoint selection under constraints such as `ece <= 0.05` |

## Install

```bash
pip install -e .            # from a clone; Python >= 3.10, transformers >= 5.20
pip install -e ".[qlora]"   # adds bitsandbytes for 4-bit QLoRA
pip install -e ".[data]"    # adds datasets for the example converters and hf: inputs
```

## Quickstart

```bash
python examples/prepare_datasets.py mnli --output data/mnli --train-size 20000
deciformer train configs/recipes/decoder_lora_code_readout.yaml
deciformer eval runs/qwen3-0.6b-lora-code/best data/mnli/dev.jsonl --predictions dev.predictions.jsonl
```

Multi-GPU uses the same config with `torchrun`:

```bash
torchrun --nproc-per-node 8 -m deciformer train configs/recipes/decoder_lora_code_readout.yaml train.lr=1e-4
```

Any config value can be overridden on the command line with dotted keys, as above. `deciformer config FILE`
prints the fully resolved configuration.

## Data format

One JSON object per line. `family` groups related records, such as several questions about one passage, so
that splits, data checks and bootstrap intervals keep them together.

```json
{"id": "ex-1", "task": "nli", "family": "premise-17",
 "state": "Premise: A man plays a guitar.\nHypothesis: A person makes music.",
 "question": {"type": "choice", "instructions": "How does the premise relate to the hypothesis?",
              "options": {"entailment": "It must be true.", "neutral": null, "contradiction": "It must be false."}},
 "label": "entailment",
 "target": {"entailment": 0.9, "neutral": 0.1, "contradiction": 0.0},
 "teacher": {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05}}
```

* `question.type`: `choice` (options are an object of key → description, or a list of keys), `binary` (the label
  is a boolean; `options` may describe `true` and `false`), or `score` (options are a list of ordered level
  descriptions, and the label is the level index).
* `target` is optional. It holds soft labels (for example human vote shares) and replaces the one-hot target in
  the loss. `teacher` is optional and holds probabilities to distill from (`train.loss.distill_alpha`).
* `state` can be a string or any JSON value. AutoJev-style records (`criteria`, `noul`) are accepted as-is.

`deciformer data check test=data/test.jsonl dev=data/dev.jsonl train=data/train.jsonl` reports duplicate ids,
families or identical states that cross splits, near-duplicate states, and label balance. It exits non-zero if
it finds leakage.

## How the model works

```
prompt (state + question + coded options) → backbone → pooler → head → one logit per option → softmax / T
```

The prompt lists options with codes (`A: entailment: ...`). Decoders use the tokenizer's chat template, or plain
text for base models. Encoders use a sentence pair (state, then question and options). Character offsets locate
every option's tokens, so option-level poolers and heads know exactly where each option is.

### Heads

| head | pooler level | what it does | good for |
|---|---|---|---|
| `code_readout` | example | Scores the answer-code tokens at the decision position, initialized from the LM-head rows | Decoders; any option count up to `data.max_options`; strong from step 0 |
| `lm_verbalizer` | example | The frozen LM head restricted to the codes, with zero new parameters | Zero-shot baselines |
| `option_scoring` | option | Pools each option's own tokens and scores it with a shared MLP | Any number of options; order-invariant; encoders and bidirectional decoders |
| `fixed_softmax` | example | A classifier over `labels`; options are looked up by key | A fixed label set |
| `binary` | example | One logit for `true`, returned as `[0, z]` so the softmax equals the sigmoid | Yes/no questions |
| `ordinal` | example | Cumulative-link model with increasing thresholds | `score` questions with `num_levels` ordered levels |

Use `model.head` for one task or `model.heads: {task_name: {...}, default: {...}}` for several. Each record's
`task` field picks its head, and any task without its own head falls back to `default`. A head can override the
pooler with `pooler: {type: ...}`. When the pooler is `auto` (the default), encoders use `cls`, causal decoders
use `decision_token` (the last prompt token), bidirectional decoders use `mean`, and option heads use
`option_spans`.

### Decoder attention modes

| `model.attention` | behaviour |
|---|---|
| `causal` (default) | Standard left-to-right attention |
| `bidirectional` | Every token attends to every non-padding token, as in an encoder |
| `prefix_bidirectional` | The prompt (state, question, options) is bidirectional, and the answer suffix stays causal |

Bidirectional mode relies on transformers' own support: `config.is_causal = False` makes mask creation
bidirectional (sliding-window layers get a bidirectional window), and each attention module's `is_causal` flag
is cleared so SDPA and flash kernels don't re-apply causality. Prefix mode builds a 4D mask that transformers
passes through unchanged. Conformance tests check every supported family (`tests/test_attention.py`):
causal never leaks future tokens, bidirectional does see them, prefix behaves as described, and padding never
changes outputs. Recurrent and linear-attention layers (Mamba, DeltaNet, for example in Qwen3-Next and Qwen3.5)
are causal by construction, so non-causal modes refuse those models instead of silently doing the wrong thing.

A causally pretrained decoder needs a little adaptation before bidirectional attention pays off. `deciformer
adapt` runs masked next-token prediction (as in LLM2Vec) and saves a checkpoint you can use as `model.backbone`
(see `configs/recipes/adapt_mntp.yaml`).

## Fine-tuning modes and LoRA

```yaml
finetune:
  mode: lora               # full | lora | qlora | head_only
  gradient_checkpointing: true
  lora:                    # passed to peft.LoraConfig: every field works, using peft's own names
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: all-linear   # list | regex | all-linear | preset: attention | mlp | attention_mlp
    exclude_modules: null
    bias: none
    use_rslora: false
    use_dora: false
    init_lora_weights: true      # true | gaussian | pissa | olora | loftq | orthogonal | ...
    layers_to_transform: null
    rank_pattern: {q_proj: 32}
    alpha_pattern: {}
    trainable_token_indices: auto  # auto = rows of any tokens deciformer adds (e.g. prompt.marker_token)
    modules_to_save: []
  quantization: {bits: 4, quant_type: nf4, double_quant: true, compute_dtype: bfloat16}  # qlora only
  export: adapter          # adapter (small) | merged (full weights, no peft needed to serve)
```

The LoRA block is checked against peft when the config loads, so a typo fails immediately. The decision head is
never wrapped by peft: it always trains fully, with its own `train.head_lr`. Full fine-tuning also supports
`freeze_embeddings`, `freeze_layers: "0-7"` and `layerwise_lr_decay`.

## Multi-GPU (DDP)

Launch with `torchrun`. NCCL is used on GPUs and gloo on CPU (`distributed.backend` overrides this). The design
makes DDP training match single-GPU training:

* Every rank computes the same step plan. Each global batch is shuffled, split across ranks with balanced token
  counts, and cut into micro-batches by `train.micro_batch_size` and `train.token_budget`.
* Ranks that need fewer micro-batches run zero-weight padding micro-batches, so gradient synchronization never
  deadlocks.
* The loss is the per-example sum × `world_size` / global example count, so DDP's gradient averaging yields the
  global-batch mean. A test checks gradients against a single process for 1, 2 and 3 ranks.
* Evaluation is sharded across ranks and gathered on rank 0, which fits temperatures, selects and exports.
* `train.shard_optimizer_states: true` shards AdamW state across ranks (ZeRO-1) while staying on DDP.
* `train.save_every` with `train.resume: auto` gives exact resume, including the data order and RNG state.

Rough sizing per 80 GB GPU, to be confirmed for your sequence lengths:

| mode | fits about |
|---|---|
| full, mixed precision | ≤ 3B |
| full + `shard_optimizer_states` on 8 GPUs | 5–7B |
| LoRA (bf16 base) | 14B comfortably, 30B tight |
| QLoRA (4-bit base) | 30B comfortably, 70B tight |

## Calibration, selection and outputs

At each evaluation, a temperature is fitted per task on `data.calibration` (never on the evaluation data) and
metrics are reported raw and calibrated: accuracy, ECE, Brier (summed over classes) and NLL, plus macro averages
when there are several tasks. The best checkpoint is chosen by `eval.selection_metric` among those that satisfy
`eval.constraints`.

```
runs/<name>/
  config.yaml        resolved config
  metrics.jsonl      train and eval events
  predictions/       per-example predictions at every evaluation
  checkpoints/       resumable training state
  best/  last/       self-contained exported models (deciformer eval / predict / load_decision_model)
  summary.json
```

```python
from deciformer.model import load_decision_model
model = load_decision_model("runs/x/best", device="cuda")
```

`deciformer compare a.jsonl b.jsonl` gives a paired bootstrap over families for accuracy, ECE, Brier and NLL.

## Development

```bash
uv venv && uv pip install -e ".[dev,data]"
pytest -q                 # all tests are offline and use tiny random models; about 2 minutes on a laptop CPU
pytest -q -m "not slow"   # unit and attention-conformance tests only
ruff check src tests
```

## Roadmap

FSDP2 and ZeRO-3 for full fine-tuning beyond the DDP limits, vision-language backbones, packing several
questions about one state into a single sequence, `bi_encoder`, `multi_label` and `regression` heads, and an
inference server.

## License and acknowledgements

Apache-2.0. The option-code readout, prompt layout, calibration metrics and near-duplicate checks are adapted
from [AutoJev](https://github.com/denis-pplx/autojev) (MIT); see `NOTICE`.
