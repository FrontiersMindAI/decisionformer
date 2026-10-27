"""Explicit, CPU-FP32 export of a merged Qwen3 decision model as an HF classifier.

The trained answer matrix becomes ``score.weight``. Promoting stored FP16/BF16
values to FP32 loses no information, but approximately doubles their storage.
This conversion does not claim reduced-precision or accelerated-engine parity.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

CLASSIFIER_FORMAT = "sequence_classification"
_DECISION_CONFIG = "decision_config.json"
_LOGIT_TRANSFORMS = ("classifier_from_token", "method", "logit_mean", "logit_sigma")


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing checkpoint file: {path.name}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return result


def _check_contract(config: dict[str, Any], hf: dict[str, Any]) -> tuple[int, int]:
    if config.get("format_version") != 1:
        raise ValueError("Classifier export requires decision format_version 1")
    if config.get("attention_mode", "causal") != "causal":
        raise ValueError("Classifier export supports causal attention only")
    if config.get("pooling", "last") != "last":
        raise ValueError("Classifier export requires last-token pooling")
    ext = config.get("decisionformer") or {}
    if not isinstance(ext, dict):
        raise ValueError("decisionformer metadata must be an object")
    if ext.get("backbone") == "adapter":
        raise ValueError("Classifier export requires a merged backbone; merge the adapter checkpoint first")
    if hf.get("model_type") != "qwen3" or ext.get("kind", "decoder") != "decoder":
        raise ValueError("Classifier export supports dense Qwen3 decoder checkpoints only")
    if hf.get("quantization_config") is not None:
        raise ValueError("Classifier export does not support quantized backbones")
    for field in _LOGIT_TRANSFORMS:
        if hf.get(field) is not None:
            raise ValueError(f"Classifier checkpoints must leave {field} unset")
    heads = ext.get("heads")
    if heads and (not isinstance(heads, dict) or set(heads) != {"default"} or
                  not isinstance(heads["default"], dict) or heads["default"].get("type") not in
                  ("code_readout", "lm_verbalizer")):
        raise ValueError("Classifier export requires a single answer-code readout")
    codes, token_ids = config.get("codes"), config.get("token_ids")
    if (not isinstance(codes, list) or len(codes) < 2 or
            any(not isinstance(code, str) or not code for code in codes) or len(set(codes)) != len(codes)):
        raise ValueError("Classifier checkpoints need at least two unique answer codes")
    if (not isinstance(token_ids, list) or len(token_ids) != len(codes) or
            any(type(token) is not int or token < 0 for token in token_ids) or len(set(token_ids)) != len(token_ids)):
        raise ValueError("Classifier checkpoints need one unique token id per answer code")
    temperatures = config.get("temperature_by_type") or {}
    if not isinstance(temperatures, dict):
        raise ValueError("temperature_by_type must be an object")
    for value in [config.get("temperature", 1.0), *temperatures.values()]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Classifier temperatures must be finite and positive")
    hidden_size = hf.get("hidden_size")
    if type(hidden_size) is not int or hidden_size < 1:
        raise ValueError("Qwen3 config needs a positive hidden_size")
    return len(codes), hidden_size


def _check_readout(weight: torch.Tensor, shape: tuple[int, int]) -> None:
    if tuple(weight.shape) != shape:
        raise ValueError(f"Answer readout shape {tuple(weight.shape)} does not match {shape}")
    if not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("Answer readout must contain finite floating-point values")


def classifier_readout(root: Path) -> torch.Tensor:
    """Read only ``score.weight`` from a single or indexed sharded safetensors export."""
    root = Path(root)
    path = root / "model.safetensors"
    if not path.is_file():
        index = _json(root / "model.safetensors.index.json")
        filename = (index.get("weight_map") or {}).get("score.weight")
        if not isinstance(filename, str):
            raise ValueError("Classifier safetensors index has no score.weight")
        path = (root / filename).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Classifier weight shard must stay inside its checkpoint directory")
    if not path.is_file():
        raise ValueError("Classifier checkpoint has no safetensors file for score.weight")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if "score.weight" not in handle.keys():  # noqa: SIM118
            raise ValueError("Classifier safetensors file has no score.weight")
        return handle.get_tensor("score.weight")


def validate_classifier_checkpoint(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the native layout without deriving any head weights from an LM."""
    root = Path(root)
    ext = config.get("decisionformer")
    if not isinstance(ext, dict) or ext.get("export_format") != CLASSIFIER_FORMAT:
        raise ValueError("Not a DecisionFormer sequence_classification export")
    hf = _json(root / "config.json")
    shape = _check_contract(config, hf)
    if hf.get("architectures") != ["Qwen3ForSequenceClassification"]:
        raise ValueError("Native classifier architecture must be Qwen3ForSequenceClassification")
    labels = hf.get("num_labels", len(hf.get("id2label") or {}))
    if labels != shape[0]:
        raise ValueError("Classifier num_labels must equal the number of saved answer codes")
    if type(hf.get("pad_token_id")) is not int or hf["pad_token_id"] < 0:
        raise ValueError("Native classifier needs a configured pad_token_id for last-token pooling")
    if (root / "readout.safetensors").exists() or (root / "heads.safetensors").exists():
        raise ValueError("Native classifier must use score.weight without duplicate readout/head sidecars")
    weight = classifier_readout(root)
    _check_readout(weight, shape)
    if weight.dtype != torch.float32:
        raise ValueError("Native classifier score.weight must be float32")
    return hf


def export_sequence_classifier(source: str | Path, output_dir: str | Path, *, revision: str | None = None) -> Path:
    """Copy a merged causal Qwen3 code-readout checkpoint into a new CPU-FP32 HF classifier.

    All backbone values and the independent readout are promoted to float32.
    The original checkpoint is never modified and an existing destination is
    refused. Native HF loading returns raw code scores; DecisionFormer keeps
    prompt rendering, active-option normalization and calibrated responses.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    from decisionformer.api import _local_dir, _prompt_name
    from decisionformer.formats import get_format
    from decisionformer.prompting import prepare_tokenizer

    out = Path(output_dir).expanduser()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Export destination already exists: {out}")
    root = _local_dir(source, revision, None)
    config = _json(root / _DECISION_CONFIG)
    hf = _json(root / "config.json")
    shape = _check_contract(config, hf)
    if hf.get("architectures") != ["Qwen3Model"]:
        raise ValueError("Classifier export requires a merged bare Qwen3Model backbone")
    if not (root / "readout.safetensors").is_file() or (root / "heads.safetensors").exists():
        raise ValueError("Classifier export requires a single answer-code readout.safetensors")
    tensors = load_file(str(root / "readout.safetensors"))
    if "weight" not in tensors:
        raise ValueError("Answer readout.safetensors has no weight tensor")
    _check_readout(tensors["weight"], shape)
    try:
        backbone, loading = AutoModel.from_pretrained(root, dtype=torch.float32, attn_implementation="sdpa",
                                                       trust_remote_code=False, output_loading_info=True)
    except RuntimeError as error:
        raise ValueError(f"Backbone checkpoint weights do not match the configuration: {error}") from error
    problems = {key: loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading.get(key)}
    if problems:
        raise ValueError(f"Backbone checkpoint has missing, unexpected or mismatched weights: {problems}")
    tokenizer_dir = root if (root / "tokenizer_config.json").is_file() else root / "processor"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)
    prepare_tokenizer(tokenizer)
    get_format(_prompt_name(config))
    hf_config = AutoConfig.from_pretrained(root, trust_remote_code=False)
    hf_config.num_labels = shape[0]
    hf_config.id2label = dict(enumerate(config["codes"]))
    hf_config.label2id = {code: i for i, code in enumerate(config["codes"])}
    hf_config.pad_token_id = tokenizer.pad_token_id
    hf_config.problem_type = "single_label_classification"
    hf_config.use_cache = False
    with torch.random.fork_rng(devices=[]):
        native = AutoModelForSequenceClassification.from_config(hf_config).float()
    native.model.load_state_dict(backbone.state_dict(), strict=True)
    with torch.no_grad():
        native.score.weight.copy_(tensors["weight"].float())
    native.eval()
    payload = copy.deepcopy(config)
    payload["decisionformer"] = {**(payload.get("decisionformer") or {}), "export_format": CLASSIFIER_FORMAT,
                                "export_dtype": "float32", "backbone": "full"}
    payload["readout_dtype"] = "float32"
    out.parent.mkdir(parents=True, exist_ok=True)
    parent = out.parent.resolve()
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=parent))
    try:
        native.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        (staging / _DECISION_CONFIG).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
        (staging / "README.md").write_text(
            "# DecisionFormer native sequence classifier\n\n"
            "A merged causal Qwen3 decision checkpoint, exported in float32. The learned answer-code readout is "
            "stored as `score.weight`, independently from input embeddings.\n\n"
            "Use the saved tokenizer and DecisionFormer prompt and calibration metadata. HF logits are raw code "
            "scores: slice to the question's active options, apply its saved temperature, then normalize. "
            "Do not substitute vocabulary-token scores or normalize unused code rows.\n\n"
            "This file conversion establishes no accelerator compatibility or performance claim.\n",
            encoding="utf-8")
        validate_classifier_checkpoint(staging, payload)
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"Export destination already exists: {out}")
        staging.rename(out)
    finally:
        if staging.exists():
            if staging.is_symlink() or staging.resolve().parent != parent:
                raise RuntimeError("Refusing to remove an export staging directory outside its parent")
            shutil.rmtree(staging)
    return out
