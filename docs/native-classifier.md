# Export a trained decision model as a native classifier

The sequence-classification export packages a trained Qwen3 decision model as a standard Hugging Face
`Qwen3ForSequenceClassification` checkpoint. It copies the trained backbone into `model.*` and the independent
answer-code readout into `score.weight`.

The original checkpoint stays in place. The export writes to a new directory and refuses an existing destination.
The first supported layout is a merged, unquantized, causal Qwen3 model with one `code_readout` head and last-token
pooling. Adapter-only exports, other backbones, multiple heads and other attention or pooling modes are outside
this conversion's scope.

## What must remain the same

The readout starts as a copy of selected language-model output rows, but training updates it independently.
Reloading the original language-model head would discard those updates. The classifier export copies the learned
matrix into the classifier's bias-free `score` layer.

DecisionFormer still renders the saved prompt and formats the response. For each question, it takes the raw
scores for that question's active options, divides by the saved question-type temperature, and applies softmax
over those options. This preserves `choice`, `noul` and `score` response semantics, including confidence and legends.
The tokenizer, chat template, answer-code order, input limit and calibration metadata travel with the export.

The export uses CPU FP32 weights for the first implementation. Stored FP16/BF16 values are promoted exactly,
which approximately doubles their weight storage. A later inference engine may cast the backbone to another
dtype. Compare that engine's outputs at the selected precision before relying on numerical parity.

## Export and load

Start with a trained checkpoint, such as the merged `best` directory from the Qwen3 recipe:

```bash
decisionformer export runs/deciformer15k-qwen3-0.6b/best runs/my-classifier --format sequence-classification
```

The Python equivalent is:

```python
from decisionformer.classifier import export_sequence_classifier

export_sequence_classifier("runs/deciformer15k-qwen3-0.6b/best", "runs/my-classifier")
```

The new directory has the usual Hugging Face config, tokenizer and model weights. Its weights contain
`model.*` and `score.weight`. It also retains `decision_config.json`, marked with
`decisionformer.export_format: "sequence_classification"`. It has no duplicate `readout.safetensors` file.

Load it through either interface:

```python
from transformers import AutoModelForSequenceClassification
from decisionformer import AutoDecisionModel

classifier = AutoModelForSequenceClassification.from_pretrained("runs/my-classifier")
decider = AutoDecisionModel.from_pretrained("runs/my-classifier", device="cpu")
```

The standard Transformers model returns raw classifier logits. The DecisionFormer interface additionally
renders questions, applies the stored temperatures and returns the System One answer objects.

## Compare the outputs

Put representative requests in a JSONL file. Use requests the trained model was designed to answer. For a
ticket model, a small `requests.jsonl` might contain:

```jsonl
{"state":"The customer was charged twice.","questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and invoices","technical":"Product faults"}},"urgent":{"type":"noul","instructions":"Does this need an immediate response?"}}}
{"state":"The customer cannot sign in after resetting a password.","questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and invoices","technical":"Product faults"}},"priority":{"type":"score","instructions":"How urgent is this?","criteria":["Routine","Soon","Immediate"]}}}
```

Run the comparison from a repository checkout:

```bash
python examples/compare_classifier.py \
  --source runs/deciformer15k-qwen3-0.6b/best \
  --exported runs/my-classifier \
  --requests requests.jsonl \
  --backend transformers --dtype float32 --atol 1e-5
```

The reference always runs on CPU in FP32. The command compares exact prompt tokens and code mapping, the saved
input limit and temperatures, active raw logits, probabilities, categorical choices and numeric answer fields.
It prints both sets of answers and a JSON summary, including the request file's SHA256 digest. Exit code `0` means all checks passed at the supplied tolerance,
`1` means a comparison failed, and `2` reports an input or loading error.

Choose a tolerance before testing another precision. Inspect raw-score errors and close decisions instead of
loosening the tolerance until a result passes. Agreement says the conversion preserved the tested behavior.
It does not measure the model's accuracy.

## Use the actual vLLM adapter

In an environment with a compatible vLLM build installed, use the same request file:

```bash
python examples/compare_classifier.py \
  --source runs/deciformer15k-qwen3-0.6b/best \
  --exported runs/my-classifier \
  --requests requests.jsonl \
  --backend vllm --dtype float32 --atol 1e-5 \
  --engine-options '{"enforce_eager":true}'
```

This calls the actual vLLM engine. The adapter requests raw classifier logits with activation disabled, then
DecisionFormer applies option slicing and temperature scaling. Conversion from original LM rows and extra
logit-normalization settings must stay disabled because they would change the trained decision function.

Serve the exported model through DecisionFormer's existing API:

```bash
decisionformer serve runs/my-classifier --backend vllm
```

Using vLLM's stock `/v1/systemone` endpoint with another read strategy is a different integration. Its prompt,
labels, probability handling and confidence definitions must be checked independently.

### Isolated Linux CPU execution

An official x86 CPU image can be used on a Docker host without installing vLLM into the host's Python environment:

```text
vllm/vllm-openai-cpu@sha256:827ef38187f9b52f2bb838695ef815ec5b65f4bf3e4c061b002b2b0a19cb0701
```

The image is the `v0.32.0-x86_64` release for `linux/amd64`. Its bundled Transformers version predates the
DecisionFormer requirement. The following derived image supplied the dependencies for the CPU run recorded below:

```dockerfile
FROM vllm/vllm-openai-cpu@sha256:827ef38187f9b52f2bb838695ef815ec5b65f4bf3e4c061b002b2b0a19cb0701
RUN python -m pip install --no-cache-dir transformers==5.20.2 peft==0.21.2 omegaconf==2.3.0
```

Build it with a local tag such as `decisionformer-vllm-cpu`. From the repository root, this POSIX-shell example
uses existing local checkpoints and requests, exposes no server port, and disables network access during inference:

The resource limits below are the tiny-fixture validation settings. Larger checkpoints may need more memory,
including space for the CPU FP32 reference and the engine model.

```bash
docker run --rm --platform linux/amd64 --cpus 4 --memory 8g --shm-size 1g --network none \
  --entrypoint python \
  -e PYTHONPATH=/workspace/src -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e OMP_NUM_THREADS=4 -e VLLM_CPU_KVCACHE_SPACE=1 -e VLLM_CPU_OMP_THREADS_BIND=nobind \
  -v "$PWD:/workspace:ro" -w /workspace decisionformer-vllm-cpu \
  examples/compare_classifier.py \
  --source runs/deciformer15k-qwen3-0.6b/best --exported runs/my-classifier \
  --requests requests.jsonl --backend vllm --dtype float32 --atol 1e-4 \
  --engine-options '{"enforce_eager":true,"max_num_seqs":16,"max_num_batched_tokens":1024}'
```

Keep the source checkout, model artifacts and requests mounted read-only. Confirm the installed versions inside
the container rather than assuming they match a host environment.

For a small CPU check, the official setup documents `VLLM_CPU_KVCACHE_SPACE` and `VLLM_CPU_OMP_THREADS_BIND`.
Set explicit container CPU, memory and shared-memory limits. Verify that the Linux environment exposes supported
CPU instructions. Follow the [official CPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
for the runtime requirements.

## Recorded CPU engine check

A bounded run used the actual `VLLMDecider` integration in the derived image above, with no network access during
inference. The container had a four-CPU limit, an 8 GiB memory limit and AVX512 available. Its packages were vLLM
`0.32.0+cpu`, PyTorch `2.13.0+cpu`, Transformers `5.20.2`, PEFT `0.21.2` and OmegaConf `2.3.0`.

The source was a randomly initialized, two-layer Qwen3 fixture with hidden size 64 and head dimension 32, trained
for 80 LoRA updates on synthetic data. Its checkpoint had eight answer codes and a 256-token input limit. Four
requests covered plain text, a longer state, a JSON state and an empty state. Each request contained four-choice,
yes/no, three-level score and eight-choice questions, for 16 decisions in total.

The FP32 comparison used an absolute tolerance of `1e-4`, selected before the run:

| Check | Observed result |
| --- | --- |
| Maximum active raw-logit error | `6.705522537231445e-8` |
| Maximum probability error | `2.150599187622504e-8` |
| Maximum numeric answer-field error | `3.2019130458138534e-8` |
| Prompt mismatches | `0` |
| Categorical choice mismatches | `0` |
| Process result | Exit `0`, parity checks passed |

This establishes interoperability for that CPU fixture and package configuration. The run does not establish
real-task accuracy, reduced-precision parity, GPU support or a speedup.

## Opt-in actual-engine test

The default test suite skips the actual-engine check. In a compatible vLLM environment with pytest installed,
enable it explicitly:

```bash
DECISIONFORMER_RUN_VLLM=1 pytest -q -s tests/test_classifier_runtime.py
```

This test creates its tokenizer and tiny Qwen3 weights locally. It deliberately changes the readout after random
initialization, exports the checkpoint and compares all three question types through the actual vLLM adapter.
It uses a preselected FP32 tolerance of `1e-4` and prints the comparison JSON. It downloads no model weights.
This test fixture is a conversion check and has no model-quality claim.

When running the test in Docker, use the same resource and network limits as the comparison command. Add pytest
to the derived test image, set `DECISIONFORMER_RUN_VLLM=1`, and replace the final command with
`-m pytest -q -s -p no:cacheprovider tests/test_classifier_runtime.py`.

## Validation boundaries

The repository's offline tests use small locally constructed models and a fake engine adapter. They check
conversion behavior and the adapter contract without downloading model weights or requiring vLLM.

A separate actual-engine comparison is required for each claimed runtime configuration. Record the model and
engine versions, image digest where applicable, hardware, dtype, requests, tolerances and full output.
CPU engine parity does not establish GPU compatibility or acceleration. The
[vLLM pooling-model guide](https://docs.vllm.ai/en/latest/models/pooling_models/) does not guarantee a speedup over Transformers.
Measure matched workloads, precision, batching, latency,
throughput and memory before making a performance claim.
