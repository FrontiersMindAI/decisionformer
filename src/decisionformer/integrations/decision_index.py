"""Decision Index engine: run any decision checkpoint through the Index kit in-process.

    pip install "decision-index @ git+https://github.com/apolinario/decision-index"
    python -m decision_index pipeline --engine decisionformer.integrations.decision_index:DecisionFormerEngine \\
        --option model=org/my-decider --out runs/my-decider

In-process is the fastest path the maintainers can measure for the latency rule (median, mean and p80 at most
1,000 ms per request). Requests that do not fit are refused as `Unsupported`, never truncated.
"""

from __future__ import annotations

from typing import Any

import torch

from decisionformer.api import Decider, UnsupportedInput
from decisionformer.prompting import ContextLengthError

try:  # the kit's own base classes when it is installed
    from decision_index.engines.base import Engine, Unsupported
except ImportError:  # pragma: no cover - same interface, so the engine also works without the kit

    class Unsupported(ValueError):  # type: ignore[no-redef]
        pass

    class Engine:  # type: ignore[no-redef]
        name = "engine"
        provenance: dict[str, Any] = {}
        latency = ""

        def __init__(self, **options: Any) -> None:
            self.options = options

        def runtime(self) -> dict[str, Any]:
            return {}

        def synchronize(self) -> None:
            pass

        def close(self) -> None:
            pass


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class DecisionFormerEngine(Engine):
    name = "decisionformer"
    latency = ("Device-synchronized in-process request wall time including prompt construction; excludes model "
               "loading.")

    def __init__(self, model: str, revision: str | None = None, device: str | None = None, dtype: str | None = None,
                 batch_size: int | str = 16, max_length: int | str | None = None, **options: Any) -> None:
        super().__init__(**options)
        self.decider = Decider.from_pretrained(model, revision=revision, device=device,
                                               dtype=_DTYPES[dtype] if dtype else None, batch_size=int(batch_size),
                                               max_length=int(max_length) if max_length else None)
        self.provenance = {"kind": "decisionformer", "repo": model, "revision": revision,
                           "prompt_format": self.decider.builder.format.name,
                           "attention_mode": self.decider.config.get("attention_mode", "causal"),
                           "max_input_tokens": self.decider.max_input_tokens,
                           "max_options": len(self.decider.builder.codes),
                           "policy": "One forward pass per question with the checkpoint's own prompt and answer-code "
                                     "readout; questions of a request share batches. Over-long requests and option "
                                     "sets beyond the answer codes are refused as unsupported, never truncated."}

    def __call__(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], None]:
        try:
            return self.decider.decide(state, questions), None
        except (ContextLengthError, UnsupportedInput) as error:
            raise Unsupported(str(error)) from error

    def runtime(self) -> dict[str, Any]:
        info: dict[str, Any] = {"torch": torch.__version__, "device": str(self.decider.device)}
        if self.decider.device.type == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(self.decider.device))
        return info

    def synchronize(self) -> None:
        if self.decider.device.type == "cuda":
            torch.cuda.synchronize(self.decider.device)
