"""Attention-mode conformance: proves causal, bidirectional and prefix-bidirectional behave as claimed per family."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from conftest import DECODERS, build_decoder

from deciformer.attention import apply_attention_mode, check_supported, prefix_attention_mask
from deciformer.backbone import Backbone

IMPLEMENTATIONS = ["sdpa", "eager"]
LENGTH = 12


def _load(model_root, model_type: str, implementation: str, mode: str) -> Backbone:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(build_decoder(model_root, model_type)), attn_implementation=implementation)
    model.eval()
    try:
        apply_attention_mode(model, mode)
    except NotImplementedError as error:  # e.g. prefix mode on models mixing sliding and full layers
        pytest.skip(str(error))
    return Backbone(model, "decoder", mode)


def _hidden(backbone: Backbone, ids: torch.Tensor, mask: torch.Tensor | None = None,
            prefix_end: int | None = None) -> torch.Tensor:
    mask = torch.ones_like(ids) if mask is None else mask
    prefix = torch.full((ids.shape[0],), prefix_end if prefix_end is not None else ids.shape[1])
    batch = SimpleNamespace(input_ids=ids, attention_mask=mask, token_type_ids=None, prefix_end=prefix)
    with torch.no_grad():
        return backbone(batch)


def _ids(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(20, 100, (1, LENGTH), generator=generator)


def _perturb(ids: torch.Tensor, position: int) -> torch.Tensor:
    changed = ids.clone()
    changed[0, position] = 20 + (int(ids[0, position]) - 20 + 7) % 80
    return changed


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", DECODERS)
def test_causal_and_bidirectional(model_root, model_type, implementation):
    ids = _ids()
    changed = _perturb(ids, 7)
    causal = _load(model_root, model_type, implementation, "causal")
    a, b = _hidden(causal, ids), _hidden(causal, changed)
    assert _same(a[:, :7], b[:, :7]), "causal: a later token changed an earlier hidden state"
    assert not _same(a[:, 7:], b[:, 7:])

    bidirectional = _load(model_root, model_type, implementation, "bidirectional")
    a, b = _hidden(bidirectional, ids), _hidden(bidirectional, changed)  # no padding: SDPA may skip the mask
    assert not _same(a[:, :7], b[:, :7]), "bidirectional: earlier tokens did not see a later token"


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("mode", ["causal", "bidirectional", "prefix_bidirectional"])
@pytest.mark.parametrize("model_type", DECODERS)
def test_right_padding_is_invisible(model_root, model_type, implementation, mode):
    backbone = _load(model_root, model_type, implementation, mode)
    short, long = _ids(1)[:, :8], _ids(2)
    padded = torch.zeros((2, LENGTH), dtype=torch.long)
    padded[0, :8], padded[1] = short[0], long[0]
    mask = torch.zeros_like(padded)
    mask[0, :8], mask[1] = 1, 1
    prefix = 6 if mode == "prefix_bidirectional" else None
    alone = _hidden(backbone, short, prefix_end=prefix)
    batched = _hidden(backbone, padded, mask, prefix_end=prefix)
    assert _same(alone[0], batched[0, :8])
    assert _same(_hidden(backbone, long, prefix_end=prefix)[0], batched[1])


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", DECODERS)
def test_prefix_bidirectional(model_root, model_type, implementation):
    backbone = _load(model_root, model_type, implementation, "prefix_bidirectional")
    ids, prefix = _ids(), 6
    base = _hidden(backbone, ids, prefix_end=prefix)
    inside = _hidden(backbone, _perturb(ids, 4), prefix_end=prefix)
    assert not _same(base[:, :4], inside[:, :4]), "prefix tokens must see later prefix tokens"
    suffix = _hidden(backbone, _perturb(ids, 9), prefix_end=prefix)
    assert _same(base[:, :9], suffix[:, :9]), "suffix tokens must stay causal"
    assert not _same(base[:, 9:], suffix[:, 9:])


@pytest.mark.parametrize("model_type", ["gemma2", "gemma3_text"])
def test_prefix_refuses_mixed_sliding_models(model_root, model_type):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(build_decoder(model_root, model_type)))
    model.config.layer_types = ["sliding_attention", "full_attention"]
    with pytest.raises(NotImplementedError):
        apply_attention_mode(model, "prefix_bidirectional")


def test_bidirectional_checkpoint_reloads_as_causal(model_root, tmp_path):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(build_decoder(model_root, "llama")))
    apply_attention_mode(model, "bidirectional")
    model.save_pretrained(str(tmp_path))
    reloaded = AutoModel.from_pretrained(str(tmp_path))
    assert reloaded.config.is_causal is False
    apply_attention_mode(reloaded, "causal")
    backbone = Backbone(reloaded.eval(), "decoder", "causal")
    ids = _ids()
    a, b = _hidden(backbone, ids), _hidden(backbone, _perturb(ids, 7))
    assert _same(a[:, :7], b[:, :7])


def test_hybrid_and_recurrent_models():
    hybrid = torch.nn.Module()
    hybrid.config = SimpleNamespace(layer_types=["full_attention", "linear_attention"], _attn_implementation="sdpa")
    check_supported(hybrid, "bidirectional")  # noncausal_full_attention: full layers open, linear layers causal
    with pytest.raises(NotImplementedError):
        check_supported(hybrid, "prefix_bidirectional")
    recurrent = torch.nn.Module()
    recurrent.config = SimpleNamespace(layer_types=["linear_attention", "mamba"], _attn_implementation="sdpa")
    with pytest.raises(ValueError, match="needs attention layers"):
        check_supported(recurrent, "bidirectional")
    check_supported(recurrent, "causal")


def test_vision_tower_keeps_its_attention(model_root):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(build_decoder(model_root, "qwen3_5")))
    apply_attention_mode(model, "bidirectional")
    apply_attention_mode(model, "causal")
    flags = {name: m.is_causal for name, m in model.named_modules() if isinstance(getattr(m, "is_causal", None), bool)}
    assert all(v is False for k, v in flags.items() if "visual" in k)
    assert all(v is True for k, v in flags.items() if "language_model" in k)


def test_prefix_mask_shapes_and_values():
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    allowed = prefix_attention_mask(mask, torch.tensor([2]), implementation="sdpa", dtype=torch.float32)[0, 0]
    assert allowed.shape == (5, 5)
    assert allowed[0, 1] and not allowed[0, 2] and allowed[3, 2] and not allowed[2, 3] and not allowed[3, 4]
    additive = prefix_attention_mask(mask, torch.tensor([2]), implementation="eager", dtype=torch.float32)[0, 0]
    assert additive[0, 1] == 0 and additive[0, 2] < -1e30
