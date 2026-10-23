"""decisionformer: train, load, serve and benchmark one-pass decision models on any Hugging Face backbone.

    from decisionformer import AutoDecisionModel
    model = AutoDecisionModel.from_pretrained("org/my-decider")
    model.decide("My card was charged twice.", {"team": {"type": "choice", "criteria": {"billing": None,
                                                                                         "shipping": None}}})
"""

__version__ = "0.2.0.dev0"

from decisionformer.api import AutoDecisionModel, Decider, UnsupportedInput  # noqa: E402
from decisionformer.config import RunConfig, load_config  # noqa: E402
from decisionformer.formats import PromptFormat, get_format, register_format  # noqa: E402
from decisionformer.prompting import ContextLengthError  # noqa: E402
from decisionformer.schema import DecisionInputError  # noqa: E402

__all__ = [
    "AutoDecisionModel", "ContextLengthError", "Decider", "DecisionInputError", "PromptFormat", "RunConfig",
    "UnsupportedInput", "__version__", "get_format", "load_config", "register_format",
]
