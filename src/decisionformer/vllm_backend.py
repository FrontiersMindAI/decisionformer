"""vLLM as an API backend: `decisionformer serve <model> --backend vllm`.

vLLM computes the answer-code logits; decisionformer renders the prompt (byte-identical to the transformers
backend) and turns logits into answers with the checkpoint's per-type temperatures, so both backends answer the
same way. vLLM's decision-model support covers code-readout checkpoints whose backbone is a bare Qwen3.5 model
(`Qwen3_5Model` plus `decision_config.json` and `readout.safetensors`): decisionformer exports on those bases,
Decision 3.0 (`vllm-sr/d3*`) and AutoJev-format exports such as FrontiersMind/Lumma-fev-27B. Anything else raises
`UnsupportedBackend`; serve it with the default transformers backend. Needs a vLLM nightly from 2026-10-15
on (vLLM 0.32.0 on PyPI predates its decision-model support) and a CUDA or ROCm GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from decisionformer.api import Decider, _local_dir, _prompt_name, display_name
from decisionformer.formats import get_format
from decisionformer.model import DECISION_CONFIG, HEADS, READOUT
from decisionformer.prompting import Encoded, PromptBuilder, prepare_tokenizer
from decisionformer.schema import Decision

_ATTENTION_MODES = ("causal", "noncausal_full_attention")


class UnsupportedBackend(ValueError):
    """The checkpoint (or this machine's vLLM) cannot run on the vLLM backend."""


def check_vllm_support(root: Path, config: dict[str, Any]) -> None:
    """Raise UnsupportedBackend unless vLLM's decision-model path can load this checkpoint as it is."""
    def refuse(reason: str) -> None:
        raise UnsupportedBackend(f"vLLM cannot serve {root}: {reason}. vLLM serves code-readout decision checkpoints "
                                 "with a bare Qwen3.5 backbone (Qwen3_5Model). Use the transformers backend "
                                 "(the default) for this model.")

    if not (root / READOUT).is_file() or (root / HEADS).is_file():
        refuse("it is not a single answer-code readout (option_scoring, fixed_softmax, binary and ordinal heads "
               "run on the transformers backend only)")
    if config.get("format_version") != 1:
        refuse(f"decision_config.json format_version {config.get('format_version')!r} is not 1")
    if config.get("pooling", "last") != "last":
        refuse(f"pooling {config.get('pooling')!r} is not 'last'")
    if config.get("attention_mode", "causal") not in _ATTENTION_MODES:
        refuse(f"attention_mode {config.get('attention_mode')!r} is not one of {_ATTENTION_MODES}")
    if not config.get("codes") or len(config.get("codes", [])) != len(config.get("token_ids") or []):
        refuse("decision_config.json does not list one token id per answer code")
    backbone = json.loads((root / "config.json").read_text()) if (root / "config.json").is_file() else {}
    if backbone.get("model_type") != "qwen3_5" or backbone.get("architectures") != ["Qwen3_5Model"]:
        refuse(f"its backbone is {backbone.get('architectures') or backbone.get('model_type')}")
    try:
        import vllm.model_executor.models.d3  # noqa: F401
    except ImportError as error:
        raise UnsupportedBackend("this vLLM has no decision-model support; install a vLLM build from 2026-10-15 or "
                                 "later (`pip install -U vllm --pre --extra-index-url "
                                 "https://wheels.vllm.ai/nightly`)") from error


class VLLMDecider(Decider):
    """A `Decider` whose answer-code logits come from vLLM; prompts, temperatures and answers are decisionformer's."""

    def __init__(self, llm: Any, builder: PromptBuilder, tokenizer: Any, config: dict[str, Any], *, name: str,
                 max_input_tokens: int | None, path: Path | None = None) -> None:
        self.llm = llm
        self.model = None
        self.builder = builder
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cpu")  # answers are formatted on the host; vLLM owns the GPU
        self.name = name
        self.max_input_tokens = max_input_tokens
        self.path = path
        self.temperature = float(config.get("temperature", 1.0))
        self.temperature_by_type = {k: float(v) for k, v in (config.get("temperature_by_type") or {}).items()}

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, revision: str | None = None, token: str | None = None,
                        prompt: str | None = None, max_length: int | None = None,
                        gpu_memory_utilization: float = 0.85, **engine_options: Any) -> VLLMDecider:
        from transformers import AutoTokenizer

        root = _local_dir(name_or_path, revision, token)
        if not (root / DECISION_CONFIG).is_file():
            raise UnsupportedBackend(f"{name_or_path} is not a decision checkpoint (no {DECISION_CONFIG})")
        config = json.loads((root / DECISION_CONFIG).read_text())
        check_vllm_support(root, config)
        from vllm import LLM

        # Text-only: no image or video budget, so the vision tower takes no profiling memory.
        options = {"model": str(root), "gpu_memory_utilization": gpu_memory_utilization,
                   "limit_mm_per_prompt": {"image": 0, "video": 0}, **engine_options}
        if max_length:
            options["max_model_len"] = max_length
        llm = LLM(**options)
        tokenizer = AutoTokenizer.from_pretrained(str(root))
        prepare_tokenizer(tokenizer)
        limit = max_length or llm.llm_engine.model_config.max_model_len
        ext = config.get("decisionformer") or {}
        builder = PromptBuilder(tokenizer, get_format(prompt or _prompt_name(config)), "decoder", max_length=limit,
                                overflow="error", codes=config["codes"], code_token_ids=config["token_ids"],
                                chat_template_kwargs=ext.get("chat_template_kwargs"))
        name = ext.get("name") or display_name(name_or_path, root)
        return cls(llm, builder, tokenizer, config, name=name, max_input_tokens=limit, path=root)

    def code_logits(self, decisions: list[Decision], encoded: list[Encoded]) -> list[torch.Tensor]:
        from vllm import PoolingParams
        from vllm.inputs import TokensPrompt

        # Raw code logits (vLLM's D3 default, requested explicitly): probabilities exist only over each question's
        # own options, at the checkpoint's per-type temperature.
        outputs = self.llm.classify([TokensPrompt(prompt_token_ids=e.input_ids) for e in encoded],
                                    pooling_params=PoolingParams(use_activation=False), use_tqdm=False)
        return [torch.tensor(output.outputs.probs, dtype=torch.float32) for output in outputs]
