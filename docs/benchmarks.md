# Benchmarks: MNLI on an 8 GB GPU

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
