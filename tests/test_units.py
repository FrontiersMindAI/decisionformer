"""Unit tests for schema normalization, config loading, prompts, heads, losses, metrics, and batch planning."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from conftest import COLORS, make_tokenizer

from deciformer.config import HeadConfig, LossConfig, PromptConfig, load_config
from deciformer.data import plan_epoch
from deciformer.heads import BinaryHead, FixedSoftmaxHead, OrdinalHead
from deciformer.losses import decision_loss
from deciformer.metrics import compute_metrics, fit_temperature, paired_bootstrap, satisfies, softmax
from deciformer.prompting import PromptBuilder, prepare_tokenizer
from deciformer.schema import normalize

# ---------------------------------------------------------------------------------------------------- schema


def test_normalize_choice_binary_score_and_autojev_aliases():
    choice = normalize({"id": "a", "state": {"x": 1}, "question": {"type": "choice", "options": {"p": "Pos", "n": None}},
                        "label": "n", "target": {"p": 0.2, "n": 0.8}})
    assert choice.option_keys == ["p", "n"] and choice.option_texts == ["p: Pos", "n"]
    assert choice.label_index == 1 and choice.target == pytest.approx([0.2, 0.8]) and choice.state == '{"x": 1}'
    binary = normalize({"state": "s", "question": {"type": "noul", "criteria": {"true": "It is"}}, "label": True,
                        "teacher": 0.9})
    assert binary.option_keys == ["false", "true"] and binary.option_texts == ["No", "It is"]
    assert binary.label_index == 1 and binary.teacher == pytest.approx([0.1, 0.9])
    score = normalize({"state": "s", "question": {"type": "score", "options": ["bad", "ok", "good"]}, "label": 2})
    assert score.option_keys == ["0", "1", "2"] and score.label_index == 2
    autojev = normalize({"id": "j", "state": "s", "suite": "bbh", "question": {"type": "choice", "criteria": {"A": None}},
                         "label": "A", "target": "A", "source": {"dataset": "bbh"}})
    assert autojev.task == "bbh" and autojev.target is None and autojev.meta["source"] == {"dataset": "bbh"}


@pytest.mark.parametrize("record, message", [
    ({"state": "s", "question": {"type": "choice", "options": {"a": None}}, "label": "z"}, "not one of the options"),
    ({"state": "s", "question": {"type": "score", "options": ["a", "b"]}, "label": 5}, "score label"),
    ({"state": "s", "question": {"type": "choice", "options": {"a": None, "b": None}}, "target": {"a": 0.7, "b": 0.7}},
     "sum to 1"),
    ({"state": "s", "question": {"type": "weird"}}, "unknown question type"),
])
def test_normalize_errors(record, message):
    with pytest.raises(ValueError, match=message):
        normalize(record)

# ---------------------------------------------------------------------------------------------------- config


def _write(path, text):
    path.write_text(text)
    return path


def test_config_inheritance_overrides_and_validation(tmp_path):
    _write(tmp_path / "base.yaml", "model: {backbone: x, head: {type: option_scoring}}\ntrain: {output_dir: out, lr: 1.0e-3}\n")
    child = _write(tmp_path / "child.yaml", "base: base.yaml\nfinetune: {mode: lora, lora: {r: 8, use_dora: true}}\n")
    config = load_config(child, ["train.lr=5e-5", "model.attention=bidirectional"])
    assert config.train.lr == 5e-5 and config.model.attention == "bidirectional"
    assert list(config.model.heads) == ["default"] and config.finetune.lora["use_dora"] is True
    with pytest.raises(ValueError, match="extra"):
        load_config(child, ["train.learning_rate=1"])
    with pytest.raises(ValueError, match="unknown arguments"):
        load_config(child, ["model.head.hidden=3"])
    with pytest.raises(ValueError, match="Invalid finetune.lora"):
        load_config(child, ["finetune.lora.not_a_field=1"])
    with pytest.raises(ValueError, match="missing required"):
        load_config(child, ["model.head.type=fixed_softmax"])


def test_lora_presets_and_auto_trainable_tokens():
    from deciformer.finetune import lora_config_from

    config = lora_config_from({"target_modules": "attention", "rank_pattern": {"q_proj": 4}}, added_token_ids=[7])
    assert "q_proj" in config.target_modules and config.trainable_token_indices == [7]
    assert config.rank_pattern == {"q_proj": 4}
    assert lora_config_from({}, added_token_ids=[]).trainable_token_indices is None

# ---------------------------------------------------------------------------------------------------- prompts


def _decision(n=4):
    return normalize({"id": "d", "state": "the circle is red .", "question": {
        "type": "choice", "instructions": "What is the color ?", "options": dict.fromkeys(COLORS[:n])}, "label": "red"})


@pytest.mark.parametrize("kind, fmt", [("decoder", "chat"), ("decoder", "plain"), ("encoder", "pair")])
def test_prompt_spans_cover_each_option(kind, fmt):
    tokenizer = make_tokenizer(kind)
    if fmt == "plain":
        tokenizer.chat_template = None
    prepare_tokenizer(tokenizer, PromptConfig(), kind)
    builder = PromptBuilder(tokenizer, PromptConfig(), kind, max_length=256, overflow="error", max_options=26,
                            need_code_tokens=kind == "decoder")
    assert builder.format == fmt
    decision = _decision()
    order = [2, 0, 3, 1]
    encoded = builder.encode(decision, order)
    for slot, (start, end) in enumerate(encoded.option_spans):
        text = tokenizer.decode(encoded.input_ids[start:end])
        assert builder.codes[slot] in text and decision.option_keys[order[slot]] in text
    if kind == "decoder":
        assert encoded.decision_index == len(encoded.input_ids) - 1
        assert 0 < encoded.prefix_end < len(encoded.input_ids)
        assert len(set(builder.code_token_ids)) == 26
    else:
        assert encoded.decision_index == 0 and encoded.token_type_ids is not None


def test_prompt_overflow_truncates_state_and_marker_is_pooled():
    tokenizer = make_tokenizer("decoder")
    prompt = PromptConfig(marker_token="<|decision|>")
    prepare_tokenizer(tokenizer, prompt, "decoder")
    long = normalize({"state": " ".join(["red"] * 300), "question": {"type": "binary"}, "label": True})
    strict = PromptBuilder(tokenizer, prompt, "decoder", max_length=120, overflow="error", max_options=4,
                           need_code_tokens=False)
    with pytest.raises(ValueError, match="exceed"):
        strict.encode(long)
    loose = PromptBuilder(tokenizer, prompt, "decoder", max_length=120, overflow="truncate_state_end", max_options=4,
                          need_code_tokens=False)
    encoded = loose.encode(long)
    assert len(encoded.input_ids) <= 120
    assert encoded.input_ids[encoded.decision_index] == tokenizer.convert_tokens_to_ids("<|decision|>")

# ---------------------------------------------------------------------------------------------------- heads + losses


def _batch(keys, types=None):
    width = max(len(k) for k in keys)
    mask = torch.tensor([[i < len(k) for i in range(width)] for k in keys])
    return SimpleNamespace(option_keys=keys, option_mask=mask, ids=[str(i) for i in range(len(keys))],
                           question_types=types or ["choice"] * len(keys),
                           num_options=torch.tensor([len(k) for k in keys]))


def test_fixed_softmax_maps_options_by_key():
    head = FixedSoftmaxHead(4, labels=["a", "b", "c"])
    pooled = torch.randn(2, 4)
    batch = _batch([["c", "a"], ["b", "a", "c"]])
    full = head.proj(pooled)
    out = head(pooled, batch, torch.tensor([0, 1]))
    assert torch.allclose(out[0, :2], full[0, [2, 0]]) and torch.allclose(out[1], full[1, [1, 0, 2]])


def test_binary_and_ordinal_heads_are_proper_distributions():
    binary = BinaryHead(4)
    pooled = torch.randn(3, 4)
    logits = binary(pooled, _batch([["false", "true"]] * 3, ["binary"] * 3), torch.arange(3))
    assert torch.allclose(logits.softmax(-1)[:, 1], torch.sigmoid(binary.proj(pooled).squeeze(-1)))
    ordinal = OrdinalHead(4, num_levels=5)
    probs = ordinal(pooled, _batch([list("01234")] * 3, ["score"] * 3), torch.arange(3)).exp()
    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-5)
    assert (ordinal.thresholds().diff() > 0).all()


def test_loss_masks_padding_and_supports_smoothing_focal_distillation():
    logits = torch.tensor([[2.0, 0.0, 99.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    plain = decision_loss(logits, targets, mask, LossConfig())
    assert plain[0] == pytest.approx(math.log(1 + math.exp(-2)), rel=1e-5)
    smooth = decision_loss(logits, targets, mask, LossConfig(label_smoothing=0.2))
    assert smooth[0] > plain[0]
    focal = decision_loss(logits, targets, mask, LossConfig(type="focal"))
    assert (focal < plain).all()
    teacher = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
    distilled = decision_loss(logits, targets, mask, LossConfig(distill_alpha=1.0), teacher,
                              torch.tensor([True, False]))
    assert distilled[1] == pytest.approx(float(plain[1])) and distilled[0] != pytest.approx(float(plain[0]))

# ---------------------------------------------------------------------------------------------------- metrics


def test_temperature_recovers_known_scale():
    generator = torch.Generator().manual_seed(0)
    true_logits = torch.randn(4000, 3, generator=generator) * 2
    labels = torch.multinomial(true_logits.softmax(-1), 1, generator=generator).squeeze(-1)
    fitted = fit_temperature((true_logits * 3).tolist(), labels.tolist())
    assert fitted == pytest.approx(3.0, rel=0.1)


def test_metrics_constraints_and_bootstrap():
    probabilities = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
    metrics = compute_metrics(probabilities, [0, 0, 1, 1], bins=10)
    assert metrics["accuracy"] == 0.75 and metrics["count"] == 4
    assert metrics["brier"] == pytest.approx((0.02 + 0.08 + 0.18 + 0.72) / 4)
    assert satisfies(metrics, ["accuracy >= 0.7", "ece <= 1"]) and not satisfies(metrics, ["accuracy > 0.8"])
    rows = [{"id": str(i), "probabilities": p, "label_index": label}
            for i, (p, label) in enumerate(zip(probabilities, [0, 0, 1, 1]))]
    result = paired_bootstrap(rows, rows, replicates=50)
    assert result["metrics"]["accuracy"]["difference"] == 0 and result["metrics"]["accuracy"]["ci95"] == [0, 0]
    assert softmax([0, 0]).tolist() == [0.5, 0.5]

# ---------------------------------------------------------------------------------------------------- planning


@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_plan_covers_each_example_once_with_equal_micro_batches(world_size):
    lengths = [((i * 37) % 50) + 5 for i in range(103)]
    plans = plan_epoch(lengths, global_batch_size=16, world_size=world_size, micro_batch_size=4, token_budget=120,
                       seed=1, epoch=0, drop_last=False)
    seen = []
    for plan in plans:
        counts = {len(rank) for rank in plan.per_rank}
        assert len(counts) == 1, "every rank must run the same number of micro-batches"
        real = [i for rank in plan.per_rank for mb in rank if not mb.dummy for i in mb.indices]
        assert len(real) == plan.global_count
        for rank in plan.per_rank:
            for mb in rank:
                assert len(mb.indices) <= 4 and (len(mb.indices) == 1 or max(lengths[i] for i in mb.indices)
                                                 * len(mb.indices) <= 120)
        seen += real
    assert sorted(seen) == list(range(103))
    again = plan_epoch(lengths, global_batch_size=16, world_size=world_size, micro_batch_size=4, token_budget=120,
                       seed=1, epoch=0, drop_last=False)
    assert [p.global_count for p in again] == [p.global_count for p in plans]
    assert again[0].per_rank[0][0].indices == plans[0].per_rank[0][0].indices


def test_head_config_extra_arguments_roundtrip():
    config = HeadConfig(type="fixed_softmax", labels=["a"], mlp_hidden=8)
    assert config.kwargs() == {"labels": ["a"], "mlp_hidden": 8}


@pytest.mark.parametrize("recipe", sorted(str(p) for p in
                                          (__import__("pathlib").Path(__file__).parents[1] / "configs" / "recipes")
                                          .glob("*.yaml")))
def test_shipped_recipes_validate(recipe):
    config = load_config(recipe)
    assert config.train.output_dir and config.model.heads
