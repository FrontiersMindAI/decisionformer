# Contributing

Thanks for helping. A few conventions keep the platform dependable:

* **Everything is configurable, nothing is magic.** New behaviour gets a config field with a safe default, and
  `decisionformer config` must show it.
* **Prove claims with tests.** Attention behaviour is checked per model family in `tests/test_attention.py`.
  Adding a family means adding it to `DECODERS` in `tests/conftest.py`. Distributed changes must keep
  `tests/test_distributed.py` passing (single-process and multi-process gradients must match).
* **Offline tests.** Tests build tiny random models and a word-level tokenizer locally, so CI never downloads
  weights.

## Adding a head or pooler

Register it and it becomes available from YAML:

```python
from decisionformer.heads import Head
from decisionformer.registry import HEADS

@HEADS.register("my_head")
class MyHead(Head):
    pooler_level = "example"          # or "option"
    question_types = ("choice",)

    def __init__(self, hidden_size, my_arg: int = 4, max_options=26, code_rows=None):
        super().__init__()
        ...

    def forward(self, pooled, batch, rows):   # -> logits (len(rows), options) in display order
        ...
```

Unknown or missing constructor arguments are reported when the config loads.

## Development loop

```bash
uv venv && uv pip install -e ".[dev,data]"
ruff check src tests examples
pytest -q
```
