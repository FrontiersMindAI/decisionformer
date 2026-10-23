# Benchmarks: MNLI on an 8 GB GPU

## v0.2 checks (2026-10-23)

Same GPU and software as below, decisionformer 0.2 (`d3` prompt by default). MNLI dev and calibration sets of 500
records each (`dev_small.jsonl`, `calibration_small.jsonl`, separated by whole families). Single seed.

| run | backbone | attention | head | fine-tuning | dev accuracy (calibrated) | calibrated ECE |
|---|---|---|---|---|---|---|
| `e2_qwen3_lora_code` (unchanged v0.1 config, 250 steps) | Qwen3-0.6B | causal | `code_readout` | LoRA r16 | 0.364 → 0.802 (step 125) → **0.828** (step 250) | 0.042 |
| QLoRA, 150 steps | Qwen3.5-0.8B | bidirectional | `code_readout` (mean pooler)¹ | QLoRA 4-bit nf4 | 0.362 → 0.804 (step 75) → **0.840** (step 150) | 0.038 |

¹ Before v0.2's final pooler default: code-readout heads now read the last token in every attention mode (the
Decision 3.0 layout).

**deciformer-15k, the starting recipe.** `decisionformer train deciformer15k_qwen3` (Qwen3-0.6B, LoRA r16, causal,
one epoch of 11,982 records = 375 updates, fp16 on the 2060 SUPER, 30 minutes with evaluations) on the family-level
split from `decisionformer data split` (1,509 dev records):

| step | dev accuracy (calibrated) | ECE | Brier | NLL |
|---|---|---|---|---|
| 125 | 0.792 | 0.039 | 0.314 | 0.585 |
| 250 | 0.805 | 0.030 | 0.281 | 0.533 |
| 375 | **0.824** | 0.023 | 0.252 | 0.467 |

Per task at step 375: Civil Comments 0.904, PAWS 0.875, MASSIVE en-US 0.864, SQuAD 2.0 0.860, MASSIVE de-DE 0.823,
Aegis 2.0 0.790, MultiNLI 0.774, BoolQ 0.772, VitaminC 0.750.

**Known issue: Qwen3.5 with causal attention.** Qwen3.5-0.8B fine-tuned with causal attention stayed at chance on
MNLI (0.34-0.37) in four runs: LoRA with an fp16 base (150 steps, twice), QLoRA (60 steps), and LoRA on an fp32 base
(60 steps). Each run's loss rose near step 25 and then settled at the class prior. Before training, the code
readout's logits equal the language model's own next-token logits for the answer codes exactly, so the prompt,
answer position and readout rows are correct. Qwen3-0.6B learns with causal attention, and Qwen3.5 learned with
bidirectional attention and a mean pooler (0.84 above). With bidirectional attention and the last-token readout
(the Decision 3.0 layout, the default since v0.2), a 40-step QLoRA check on deciformer-15k showed the same loss spike
near step 20 (1.55 → 2.04 → 1.23) and 0.41 dev accuracy on 128 records, so no Qwen3.5 recipe ships yet.

**Bidirectional attention on a smaller decoder.** Three 50-step LoRA runs of the `deciformer15k_qwen3` recipe
(Qwen3-0.6B, deciformer-15k, 256 dev records), identical except for attention and pooling:

| attention | pooler | train loss, step 10 → 50 | dev accuracy |
|---|---|---|---|
| causal | last token | 1.55 → 0.73 | **0.672** |
| bidirectional | last token (the Decision 3.0 layout) | 2.89 → 1.27 | 0.449 |
| bidirectional | mean | 2.63 → 1.25 | 0.441 |

Switching a causally pretrained decoder to bidirectional attention starts it from a much worse point (gradient
norms 3-5x larger with the last-token readout) and it learns far more slowly, on Qwen3 as on Qwen3.5 and SmolLM2
(v0.1, below). Bidirectional decoders need a long adaptation stage before they compete; keep `causal` otherwise.

**Parity with the Decision 3.0 reference runtime.** `AutoDecisionModel.from_pretrained("vllm-sr/d3-edge")` against
the checkpoint's own `trust_remote_code` runtime, BF16, four requests (choice, noul and score questions, an empty
state, a JSON state, a 32-option choice): one question per forward pass, the probabilities are identical (maximum
difference 0.0). With several questions per pass, padding changes BF16 rounding by up to 0.008 in both
implementations; in FP32, decisionformer's batched and unbatched results agree within 2e-6.

**vLLM backend.** `vllm-sr/d3-edge` with its attention mode set to causal, served by `decisionformer serve
--backend vllm` (vLLM 0.32.1 nightly, FP16 on Turing) against decisionformer's transformers backend in FP32: maximum
probability difference 7.2e-4, identical choices, median 29.6 ms and p80 36.6 ms per 3-question request. vLLM's
non-causal attention kernels (Triton, FlexAttention) need more shared memory than Turing's 64 KB, so non-causal
checkpoints need an Ampere or newer GPU on the vLLM backend.

## v0.1 results (2026-10-22)

Measured on 2026-10-22 on one **NVIDIA RTX 2060 SUPER (8 GB, Turing)** with torch 2.14.1+cu130, transformers 5.20.2
and peft 0.21.2. Turing has no native bf16 and no flash-attention, so every run uses fp16 autocast with SDPA.
These runs are short (one epoch of 8,000 examples, 250 updates) and use a single seed. Read them as a check that
each code path learns, not as tuned results.

**Data.** Built with `python examples/prepare_datasets.py mnli --output data/mnli --train-size 8000`. The
converter removed 356 training records whose premise also appears in `validation_matched`, and `decisionformer data
check` then reported no cross-split overlap. Evaluation used the first 1,000 records of `dev.jsonl` and
temperatures were fitted on the first 1,000 of `calibration.jsonl`. The two are separated by whole families.
Chance accuracy is 0.33.

**Configs.** `configs/benchmarks/mnli_8gb/`. Run from the repository root, for example
`decisionformer train configs/benchmarks/mnli_8gb/e2_qwen3_lora_code.yaml`.

## Results

| config | backbone | attention | head | fine-tuning | dev accuracy (step 0 → 250) | calibrated ECE | time |
|---|---|---|---|---|---|---|---|
| `e1_qwen3_zeroshot` | Qwen3-0.6B | causal | `lm_verbalizer` | none (zero-shot) | 0.337 | 0.084 | 31 s |
| `e2_qwen3_lora_code` | Qwen3-0.6B | causal | `code_readout` | LoRA r16 | 0.337 → **0.803** | 0.033 | 11 min |
| `q1_qwen3_qlora_full` | Qwen3-0.6B | causal | `code_readout` | QLoRA 4-bit nf4 | 0.342 → **0.795** | 0.031 | 10 min |
| `e4_smollm2_causal` | SmolLM2-360M-Instruct | causal | `option_scoring` | LoRA r32 | 0.306 → **0.796** | 0.022 | 8 min |
| `e3_modernbert_full` | ModernBERT-base | encoder | `fixed_softmax` | full | 0.309 → 0.675 | 0.043 | 2 min |
| `e8_ddp_gloo_gpu` | SmolLM2-135M-Instruct | causal | `code_readout` | LoRA r8, 2-rank DDP + ZeRO, 40 steps | 0.309 → 0.356 | 0.012 | 81 s |

* **Calibration.** The fitted temperature brings ECE to 0.02–0.04 for every model that learned. The zero-shot
  verbalizer's raw ECE is 0.61 and falls to 0.08 once a temperature is fitted.
* **QLoRA** matches LoRA within a point on the same schedule. An earlier 80-step QLoRA run compressed the cosine
  schedule and stayed at chance, a reminder that `max_steps` also shortens the schedule.
* **DDP.** Two ranks shared the GPU over gloo with ZeRO optimizer sharding. A run paused at step 20 and resumed
  produced weights bit-identical to an uninterrupted run.
* **fp16.** The first update overflows while the GradScaler calibrates its scale, and that step is skipped. This
  is expected.

## Bidirectional attention on decoders: correct, but it needs real adaptation

Every bidirectional or prefix-bidirectional run on SmolLM2-360M stayed at chance:

| config | change from `e4_smollm2_causal` | dev accuracy at step 250 |
|---|---|---|
| `e4_smollm2_bidirectional` | `attention: bidirectional` | diverged in fp16 near step 100 (NaN) |
| `b2_bidir_option_lowlr` | bidirectional, lr 1e-4 | 0.309 |
| `b1_bidir_mean_fixed` | bidirectional, mean pooler + `fixed_softmax` | 0.344 |
| `b4_bidir_code_readout` | bidirectional, `code_readout` | 0.347 |
| `b3_prefix_option` | `attention: prefix_bidirectional` | stopped: non-finite loss at step 26 in fp16 |
| `e7_smollm2_bidirectional_after_mntp` | bidirectional after 150 MNTP steps (`e6`) | 0.309 |

To tell an implementation bug from a modeling effect, `probe.py` compares the fp16 SDPA bidirectional forward
on CUDA with an fp32 eager reference, and fits a linear probe on frozen features (2,000 train / 1,000 dev):

| backbone | attention | fp16 SDPA vs fp32 eager, max relative error | probe accuracy, last token | probe accuracy, mean |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | causal | 1.2% | 0.506 | 0.435 |
| SmolLM2-360M-Instruct | bidirectional | 0.5% | 0.401 | 0.388 |
| Qwen3-0.6B | causal | 0.9% | 0.676 | 0.545 |
| Qwen3-0.6B | bidirectional | 0.3% | 0.437 | 0.422 |
| SmolLM2 after 150 MNTP steps | bidirectional | 0.4% | 0.396 | 0.376 |

* **The implementation is correct.** The fast fp16 kernel path agrees with the fp32 reference. The CPU
  conformance tests independently check causality, future visibility and padding invariance for 11 model
  families.
* **The failure is a modeling effect.** Making a causally pretrained decoder bidirectional removes most of the
  task information from its frozen features, with Qwen3's probe accuracy dropping from 0.68 to 0.44. A short MNTP
  stage (150 LoRA steps) restored the masked-token loss from 10.9, close to uniform, to 3.5, but did not restore
  task information. Bidirectional decoders need a substantial adaptation budget before they compete with causal
  ones.
* **fp16 is fragile in these modes.** Both fp16 divergences happened in bidirectional or prefix mode, never in
  causal mode. Prefer bf16 or fp32 for them.

Recommendation: keep `attention: causal` (the default) unless you can afford a long adaptation stage, and
compare against the causal baseline with `decisionformer compare` before adopting a bidirectional model.
