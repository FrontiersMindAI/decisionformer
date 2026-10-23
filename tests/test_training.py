"""End-to-end training on tiny models: every head type, attention mode and fine-tuning mode, plus export/reload."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from conftest import build_decoder, build_encoder, synthetic_records, write_jsonl
from safetensors.torch import load_file

from decisionformer.config import load_config
from decisionformer.trainer import train

pytestmark = pytest.mark.slow


def make_config(tmp_path: Path, backbone: Path, data_dir: Path, *, head: dict, out: str = "run", steps: int = 12,
                extra: dict | None = None, overrides: list[str] | None = None):
    config = {
        "model": {"backbone": str(backbone), "head": head},
        "data": {"train": str(data_dir / "train.jsonl"), "eval": {"dev": str(data_dir / "dev.jsonl")},
                 "calibration": str(data_dir / "calibration.jsonl"), "max_length": 256},
        "train": {"output_dir": str(tmp_path / out), "max_steps": steps, "global_batch_size": 16,
                  "micro_batch_size": 8, "lr": 3e-3, "log_every": 4, "seed": 3},
        "eval": {"micro_batch_size": 16},
    }
    for section, values in (extra or {}).items():
        config.setdefault(section, {}).update(values)
    path = tmp_path / f"{out}.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_config(path, overrides or [])


def _losses(output: Path) -> list[float]:
    lines = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    return [line["loss"] for line in lines if line["event"] == "train"]


def _moved(output: Path, base: Path) -> bool:
    """True if training changed the backbone: LoRA B matrices left zero, or full weights left the base checkpoint."""
    last = output / "last"
    if (last / "adapter").exists():
        adapter = load_file(str(last / "adapter" / "adapter_model.safetensors"))
        return any(float(t.abs().sum()) > 0 for k, t in adapter.items() if "lora_B" in k)
    trained = load_file(str(last / "model.safetensors"))
    original = _unprefixed(load_file(str(base / "model.safetensors")))
    shared = [k for k in trained if k in original and trained[k].shape == original[k].shape]
    return bool(shared) and any(not torch.equal(trained[k], original[k]) for k in shared)


def _unprefixed(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Base checkpoints nest the backbone (`model.layers...`, `bert.encoder...`); exports store it bare."""
    result = dict(state)
    for key, value in state.items():
        result.setdefault(key.split(".", 1)[1] if "." in key else key, value)
    return result


def _accuracy(output: Path) -> float:
    return _evals(output)[-1]["sets"]["dev"]["raw"]["overall"]["accuracy"]


def _evals(output: Path) -> list[dict]:
    lines = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    return [line for line in lines if line["event"] == "eval"]


def test_encoder_full_fixed_softmax_learns_and_reloads(model_root, data_dir, tmp_path):
    config = make_config(tmp_path, build_encoder(model_root, "bert"), data_dir, steps=60,
                         head={"type": "fixed_softmax", "labels": ["red", "blue", "green", "yellow"]},
                         extra={"train": {"eval_every": 30}})
    summary = train(config)
    output = Path(config.train.output_dir)
    final = _evals(output)[-1]["sets"]["dev"]["calibrated"]["overall"]
    assert final["accuracy"] >= 0.6, final
    assert summary["best_step"] in (30, 60)
    assert (output / "best" / "model.safetensors").exists() and (output / "best" / "heads.safetensors").exists()
    assert (output / "predictions").exists()

    from decisionformer.cli import main

    predictions = tmp_path / "reloaded.jsonl"
    main(["eval", str(output / "last"), str(data_dir / "dev.jsonl"), "--predictions", str(predictions),
          "--device", "cpu"])
    reloaded = [json.loads(line) for line in predictions.read_text().splitlines()]
    trained = [json.loads(line) for line in (output / "predictions" / "dev-step000060.jsonl").read_text().splitlines()]
    for a, b in zip(reloaded, trained):
        assert a["id"] == b["id"]
        assert a["probabilities"] == pytest.approx(b["probabilities"], abs=1e-5)


@pytest.mark.parametrize("encoder", ["modernbert", "t5"])
def test_other_encoders_train(model_root, data_dir, tmp_path, encoder):
    base = build_encoder(model_root, encoder)
    config = make_config(tmp_path, base, data_dir, head={"type": "option_scoring"})
    train(config)
    assert _moved(Path(config.train.output_dir), base)


@pytest.mark.parametrize("attention, head", [
    ("causal", {"type": "code_readout"}),
    ("bidirectional", {"type": "option_scoring", "mlp_hidden": 16}),
    ("prefix_bidirectional", {"type": "code_readout"}),
    ("bidirectional", {"type": "fixed_softmax", "labels": ["red", "blue", "green", "yellow"],
                       "pooler": {"type": "attention"}}),
])
def test_decoder_attention_modes_with_lora(model_root, data_dir, tmp_path, attention, head):
    config = make_config(tmp_path, build_decoder(model_root, "qwen3"), data_dir, head=head,
                         extra={"model": {"attention": attention},
                                "finetune": {"mode": "lora", "lora": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}}})
    train(config)
    output = Path(config.train.output_dir)
    assert _moved(output, build_decoder(model_root, "qwen3"))
    from decisionformer import AutoDecisionModel

    assert not (output / "last" / "adapter").exists(), "LoRA exports are merged by default"
    loaded = AutoDecisionModel.from_pretrained(output / "last", device="cpu")
    assert loaded.model.backbone.attention == attention
    expected = {"causal": "causal", "bidirectional": "noncausal_full_attention"}.get(attention, attention)
    assert loaded.config["attention_mode"] == expected


def test_plain_decoder_full_finetune_with_layer_freezing(model_root, data_dir, tmp_path):
    base = build_decoder(model_root, "llama", plain=True)
    config = make_config(tmp_path, base, data_dir, head={"type": "code_readout"},
                         extra={"finetune": {"mode": "full", "freeze_layers": "0", "freeze_embeddings": True,
                                             "layerwise_lr_decay": 0.8}})
    train(config)
    output = Path(config.train.output_dir)
    assert _moved(output, base)
    trained = load_file(str(output / "last" / "model.safetensors"))
    original = _unprefixed(load_file(str(base / "model.safetensors")))
    for key in trained:
        if ".layers.0." in key or "embed_tokens" in key:
            assert torch.equal(trained[key], original[key]), f"{key} should be frozen"


def test_multitask_heads_and_binary(model_root, data_dir, tmp_path):
    mixed = synthetic_records(40, 7) + synthetic_records(40, 8, kind="binary")
    write_jsonl(tmp_path / "mixed.jsonl", mixed)
    config = make_config(tmp_path, build_decoder(model_root, "llama"), data_dir, head={"type": "binary"},
                         overrides=["data.train=" + str(tmp_path / "mixed.jsonl")])
    config.model.heads = {"color": config.model.heads["default"].model_copy(update={"type": "option_scoring"}),
                          "is_red": config.model.heads["default"]}
    train(config)
    assert _moved(Path(config.train.output_dir), build_decoder(model_root, "llama"))


def test_ordinal_head_on_score_questions(model_root, tmp_path):
    levels = ["tiny", "small", "medium", "large", "huge"]
    records = [{"id": str(i), "state": f"the object is {['small', 'large'][i % 2]} .", "label": (i % 2) * 3,
                "question": {"type": "score", "instructions": "How big ?", "options": levels}} for i in range(48)]
    write_jsonl(tmp_path / "score.jsonl", records)
    config = make_config(tmp_path, build_encoder(model_root, "bert"), tmp_path, head={"type": "ordinal", "num_levels": 5},
                         overrides=[f"data.train={tmp_path / 'score.jsonl'}", f"data.eval.dev={tmp_path / 'score.jsonl'}",
                                    "data.calibration=null"])
    train(config)
    assert _moved(Path(config.train.output_dir), build_encoder(model_root, "bert"))


def test_lm_verbalizer_zero_shot_eval_needs_no_training(model_root, data_dir, tmp_path):
    config = make_config(tmp_path, build_decoder(model_root, "llama"), data_dir, head={"type": "lm_verbalizer"},
                         extra={"finetune": {"mode": "head_only"}})
    summary = train(config)
    output = Path(config.train.output_dir)
    assert summary["steps"] == 0 and summary["best_step"] == 0
    assert _evals(output)[0]["sets"]["dev"]["raw"]["overall"]["count"] == 32
    assert not _losses(output) and (output / "best" / "readout.safetensors").exists()
    config_file = json.loads((output / "best" / "decision_config.json").read_text())
    assert config_file["format_id"] == "d3-code-readout-v1" and len(config_file["codes"]) == 255


def test_eval_at_start_records_the_baseline(model_root, data_dir, tmp_path):
    config = make_config(tmp_path, build_encoder(model_root, "bert"), data_dir, steps=4,
                         head={"type": "fixed_softmax", "labels": ["red", "blue", "green", "yellow"]},
                         extra={"eval": {"at_start": True}})
    train(config)
    assert [e["step"] for e in _evals(Path(config.train.output_dir))] == [0, 4]


def test_merged_export_matches_adapter(model_root, data_dir, tmp_path):
    import shutil

    from decisionformer import AutoDecisionModel
    from decisionformer.model import merge_checkpoint

    config = make_config(tmp_path, build_decoder(model_root, "mistral"), data_dir, head={"type": "code_readout"},
                         extra={"finetune": {"mode": "lora", "lora": {"r": 4, "lora_dropout": 0.0}, "export": "adapter"}})
    train(config)
    adapter_dir = Path(config.train.output_dir) / "last"
    assert (adapter_dir / "adapter").exists()
    merged_dir = tmp_path / "merged"
    shutil.copytree(adapter_dir, merged_dir)
    merge_checkpoint(merged_dir)
    assert (merged_dir / "model.safetensors").exists() and not (merged_dir / "adapter").exists()
    questions = {"color": {"type": "choice", "instructions": "What is the color ?", "criteria": dict.fromkeys(
        ["red", "blue", "green", "yellow"])}, "red": {"type": "noul", "instructions": "Is it red ?"}}
    answers = [AutoDecisionModel.from_pretrained(d, device="cpu", dtype=torch.float32).decide(
        "the small circle is blue .", questions)["answers"] for d in (adapter_dir, merged_dir)]
    for key in questions:
        first, second = answers[0][key], answers[1][key]
        if key == "red":
            assert first["noul"] == pytest.approx(second["noul"], abs=1e-4)
        else:
            assert first["choice"] == second["choice"]
            for option, value in first["probabilities"].items():
                assert value == pytest.approx(second["probabilities"][option], abs=1e-4)


def test_pause_and_resume_is_exact(model_root, data_dir, tmp_path):
    def run(name: str, stops: list[int | None]):
        config = None
        for stop in stops:
            config = make_config(tmp_path, build_decoder(model_root, "llama"), data_dir, out=name, steps=8,
                                 head={"type": "code_readout"},
                                 extra={"train": {"save_every": 4, "stop_after": stop, "resume": "auto"},
                                        "data": {"eval": {}}},
                                 overrides=["data.calibration=null"])
            train(config)
        return load_file(str(Path(config.train.output_dir) / "last" / "readout.safetensors")), \
            load_file(str(Path(config.train.output_dir) / "last" / "model.safetensors"))

    straight_heads, straight_backbone = run("straight", [None])
    resumed_heads, resumed_backbone = run("resumed", [4, None])
    for key in straight_heads:
        assert torch.equal(straight_heads[key], resumed_heads[key]), key
    for key in straight_backbone:
        assert torch.equal(straight_backbone[key], resumed_backbone[key]), key


def test_mntp_adaptation_then_bidirectional_training(model_root, data_dir, tmp_path):
    from decisionformer.adapt import adapt

    config = make_config(tmp_path, build_decoder(model_root, "llama"), data_dir, head={"type": "option_scoring"},
                         out="adapt", steps=4,
                         extra={"model": {"attention": "bidirectional"},
                                "finetune": {"mode": "lora", "lora": {"r": 4, "lora_dropout": 0.0}}})
    adapted = adapt(config)
    assert (adapted / "config.json").exists()
    follow = make_config(tmp_path, adapted, data_dir, head={"type": "code_readout"}, out="after",
                         extra={"model": {"attention": "bidirectional"}})
    train(follow)
    assert _moved(Path(follow.train.output_dir), adapted)


@pytest.mark.parametrize("name, backbone, head, extra, minimum", [
    ("bidirectional-option-scoring", "qwen3", {"type": "option_scoring", "mlp_hidden": 16},
     {"model": {"attention": "bidirectional"}, "train": {"seed": 1},  # tiny random models are seed-sensitive
      "finetune": {"mode": "lora", "lora": {"r": 4, "lora_dropout": 0.0}}},
     0.5),
    ("causal-code-readout", "llama", {"type": "code_readout"},
     {"data": {"shuffle_options": False}, "train": {"lr": 1e-2},
      "finetune": {"mode": "lora", "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0}}}, 0.6),
])
def test_decoders_learn_the_task(model_root, data_dir, tmp_path, name, backbone, head, extra, minimum):
    config = make_config(tmp_path, build_decoder(model_root, backbone), data_dir, head=head, out=name, steps=48,
                         extra=extra)
    train(config)
    assert _accuracy(Path(config.train.output_dir)) >= minimum  # chance is 0.25


def _bf16_base(model_root: Path, tmp_path: Path) -> Path:
    from transformers import AutoModelForCausalLM

    base = tmp_path / "llama-bf16"
    if not base.exists():
        source = build_decoder(model_root, "llama")
        AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.bfloat16).save_pretrained(str(base))
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy(source / name, base)
    return base


def _saved_dtypes(checkpoint: Path) -> tuple[set[torch.dtype], str]:
    from safetensors import safe_open

    with safe_open(str(checkpoint / "model.safetensors"), "pt") as handle:
        dtypes = {handle.get_tensor(k).dtype for k in handle.keys()}  # noqa: SIM118
    return dtypes, json.loads((checkpoint / "config.json").read_text())["dtype"]


@pytest.mark.parametrize("mode, export_dtype, expected", [
    ("lora", "auto", torch.bfloat16),   # merged in fp32, saved like the base checkpoint
    ("full", "auto", torch.bfloat16),   # fp32 master weights, saved like the base checkpoint
    ("full", "float32", torch.float32),
])
def test_exports_are_saved_in_the_export_dtype(model_root, data_dir, tmp_path, mode, export_dtype, expected):
    lora = {"lora": {"r": 4, "lora_dropout": 0.0}} if mode == "lora" else {}
    config = make_config(tmp_path, _bf16_base(model_root, tmp_path), data_dir, head={"type": "code_readout"}, steps=2,
                         extra={"finetune": {"mode": mode, "export_dtype": export_dtype, **lora}})
    train(config)
    best = Path(config.train.output_dir) / "best"
    assert not (best / "adapter").exists()
    assert _saved_dtypes(best) == ({expected}, str(expected).removeprefix("torch."))
    from decisionformer import AutoDecisionModel

    AutoDecisionModel.from_pretrained(best, device="cpu", dtype=torch.float32).decide(
        "the small circle is blue .", {"red": {"type": "noul", "instructions": "Is it red ?"}})
