"""The decision record format: a state, a question, its options, and a label.

Every input file is normalized into `Decision` objects. The format also accepts
AutoJev-style records (`criteria` for options, `noul` for binary questions).

    {"id": "1", "state": "...", "task": "sentiment", "family": "doc-17",
     "question": {"type": "choice", "instructions": "...",
                  "options": {"positive": "Favourable", "negative": null}},
     "label": "positive",
     "target": {"positive": 0.8, "negative": 0.2},     # optional soft target
     "teacher": {"positive": 0.7, "negative": 0.3}}    # optional distillation target
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUESTION_TYPES = ("choice", "binary", "score")
_TYPE_ALIASES = {"choice": "choice", "binary": "binary", "noul": "binary", "bool": "binary", "score": "score"}
_BINARY_KEYS = ["false", "true"]
_BINARY_DEFAULT_TEXT = ["No", "Yes"]


@dataclass
class Decision:
    id: str
    state: str
    question_type: str
    instructions: str | None
    option_keys: list[str]
    option_texts: list[str]
    label_index: int | None
    target: list[float] | None = None
    teacher: list[float] | None = None
    task: str = "default"
    family: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_options(self) -> int:
        return len(self.option_keys)


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _distribution(value: Any, keys: list[str], question_type: str, where: str) -> list[float] | None:
    if value is None:
        return None
    if question_type == "binary" and isinstance(value, (int, float)) and not isinstance(value, bool):
        values = [1.0 - float(value), float(value)]
    elif isinstance(value, dict):
        lookup = {str(k).lower() if question_type == "binary" else str(k): v for k, v in value.items()}
        if set(lookup) != set(keys):
            raise ValueError(f"{where}: distribution keys {sorted(lookup)} must match options {keys}")
        values = [float(lookup[k]) for k in keys]
    elif isinstance(value, list):
        values = [float(v) for v in value]
    else:
        return None  # hard labels repeated in `target` (AutoJev style) are ignored
    if len(values) != len(keys) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"{where}: distribution must have one finite, nonnegative value per option")
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=1e-4):
        raise ValueError(f"{where}: distribution must sum to 1 (got {total:.6f})")
    return [v / total for v in values]


def _binary_label(value: Any, where: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.lower() in ("true", "false", "yes", "no", "1", "0"):
        return int(value.lower() in ("true", "yes", "1"))
    raise ValueError(f"{where}: binary label must be a boolean, got {value!r}")


def normalize(record: dict[str, Any], index: int = 0) -> Decision:
    identifier = str(record.get("id", index))
    where = f"record {identifier}"
    if "state" not in record or "question" not in record:
        raise ValueError(f"{where}: needs 'state' and 'question'")
    question = record["question"]
    if not isinstance(question, dict):
        raise ValueError(f"{where}: 'question' must be an object")
    question_type = _TYPE_ALIASES.get(str(question.get("type", "choice")))
    if question_type is None:
        raise ValueError(f"{where}: unknown question type {question.get('type')!r}; use one of {QUESTION_TYPES}")
    options = question.get("options", question.get("criteria"))
    instructions = question.get("instructions")
    instructions = None if instructions is None else _as_text(instructions)

    if question_type == "choice":
        if isinstance(options, list):
            options = dict.fromkeys(map(str, options))
        if not isinstance(options, dict) or not options:
            raise ValueError(f"{where}: choice questions need a nonempty 'options' object")
        keys = [str(k) for k in options]
        texts = [k if v is None else f"{k}: {_as_text(v)}" for k, v in zip(keys, options.values())]
    elif question_type == "binary":
        given = {str(k).lower(): v for k, v in (options or {}).items()}
        if set(given) - set(_BINARY_KEYS):
            raise ValueError(f"{where}: binary options may only define 'true' and 'false'")
        keys = list(_BINARY_KEYS)
        texts = [_as_text(given[k]) if given.get(k) is not None else d for k, d in zip(keys, _BINARY_DEFAULT_TEXT)]
    else:
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"{where}: score questions need a list of at least two level descriptions")
        keys = [str(i) for i in range(len(options))]
        texts = [_as_text(v) for v in options]

    label = record.get("label")
    label_index: int | None = None
    if label is not None:
        if question_type == "binary":
            label_index = _binary_label(label, where)
        elif question_type == "score":
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(keys):
                raise ValueError(f"{where}: score label must be an integer level in [0, {len(keys) - 1}]")
            label_index = label
        else:
            if str(label) not in keys:
                raise ValueError(f"{where}: label {label!r} is not one of the options {keys}")
            label_index = keys.index(str(label))

    target = _distribution(record.get("target"), keys, question_type, f"{where} target")
    teacher = _distribution(record.get("teacher"), keys, question_type, f"{where} teacher")
    if label_index is None and target is not None:
        label_index = max(range(len(target)), key=target.__getitem__)
    known = {"id", "state", "question", "label", "target", "teacher", "task", "family", "source", "suite"}
    meta = {k: v for k, v in record.items() if k not in known}
    if "source" in record:
        meta["source"] = record["source"]
    family = record.get("family")
    return Decision(
        id=identifier, state=_as_text(record["state"]), question_type=question_type, instructions=instructions,
        option_keys=keys, option_texts=texts, label_index=label_index, target=target, teacher=teacher,
        task=str(record.get("task", record.get("suite", "default")) or "default"),
        family=None if family is None else str(family), meta=meta,
    )


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, or a Hugging Face dataset written as `hf:<name>[:<config>]:<split>`."""
    spec = str(path)
    if spec.startswith("hf:"):
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError("Reading 'hf:' datasets needs `pip install deciformer[data]`") from error
        parts = spec[3:].split(":")
        if len(parts) not in (2, 3):
            raise ValueError("Use hf:<name>:<split> or hf:<name>:<config>:<split>")
        name, config, split = (parts[0], None, parts[1]) if len(parts) == 2 else parts
        return [dict(row) for row in load_dataset(name, config, split=split)]
    records = []
    with Path(spec).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{spec}:{number}: invalid JSON ({error.msg})") from error
    return records


def load_decisions(path: str | Path) -> list[Decision]:
    decisions = [normalize(record, index) for index, record in enumerate(read_records(path))]
    seen: set[str] = set()
    for decision in decisions:
        if decision.id in seen:
            raise ValueError(f"{path}: duplicate id {decision.id!r}")
        seen.add(decision.id)
    return decisions
