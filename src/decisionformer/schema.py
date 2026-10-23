"""Requests and questions in the System One format used by Jev, vLLM `/v1/systemone` and the Decision Index.

A request is one state plus named, typed questions. Training files add the gold answer per question:

    {"id": "t1", "state": "My card was charged twice.", "family": "ticket-17",
     "questions": {
       "team":   {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": "payments and refunds", "shipping": null}},
       "urgent": {"type": "noul", "instructions": "Reply within the hour?"},
       "tone":   {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]}},
     "gold":     {"team": "billing", "urgent": false, "tone": 1},
     "targets":  {"team": {"billing": 0.9, "shipping": 0.1}},     # optional soft labels
     "teachers": {"urgent": 0.2}}                                  # optional distillation targets

`expected` (Decision Index rows) and `answers` are accepted for `gold`. A single-question shorthand is also
accepted: `{"state", "question": {...}, "label", "target", "teacher"}`, and `options`/`binary` as aliases of
`criteria`/`noul`. Flat rows with one question each, as the deciformer pipeline writes them (for example
FrontiersMind/deciformer-15k), are read as they are: `{"state", "question": "<instructions>", "question_type",
"options", "option_descriptions", "label", "soft_target"}`. Each question becomes one `Decision`; questions of one
request share its family.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUESTION_TYPES = ("choice", "noul", "score")
_TYPE_ALIASES = {"choice": "choice", "noul": "noul", "binary": "noul", "bool": "noul", "score": "score"}
NOUL_KEYS = ("false", "true")
MAX_SCORE_LEVELS = 10


class DecisionInputError(ValueError):
    """A request or question that does not follow the format (HTTP 400)."""


@dataclass
class Question:
    key: str
    type: str
    instructions: Any
    option_keys: list[str]
    descriptions: list[Any]  # per option: a description (string or JSON) or None; score: the level values
    criteria: Any  # as given, for formats that need it
    task: str | None = None  # optional head to answer with, in multi-head models

    @property
    def num_options(self) -> int:
        return len(self.option_keys)


@dataclass
class Decision:
    id: str
    request_id: str
    state: Any
    question: Question
    label_index: int | None = None
    target: list[float] | None = None
    teacher: list[float] | None = None
    task: str = "default"
    family: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def question_type(self) -> str:
        return self.question.type

    @property
    def option_keys(self) -> list[str]:
        return self.question.option_keys

    @property
    def num_options(self) -> int:
        return self.question.num_options


def parse_question(key: str, raw: Any) -> Question:
    where = f"question {key!r}"
    if not isinstance(raw, dict):
        raise DecisionInputError(f"{where}: must be an object")
    kind = _TYPE_ALIASES.get(str(raw.get("type", "choice")))
    if kind is None:
        raise DecisionInputError(f"{where}: unknown type {raw.get('type')!r}; use one of {QUESTION_TYPES}")
    criteria = raw.get("criteria", raw.get("options"))
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(map(str, criteria))
        if not isinstance(criteria, dict) or not criteria:
            raise DecisionInputError(f"{where}: a choice needs a nonempty `criteria` object (option -> description)")
        if any(not str(k) for k in criteria):
            raise DecisionInputError(f"{where}: option names must be nonempty")
        keys, descriptions = [str(k) for k in criteria], list(criteria.values())
    elif kind == "noul":
        given = criteria or {}
        if not isinstance(given, dict) or set(map(str, given)) - set(NOUL_KEYS):
            raise DecisionInputError(f"{where}: noul criteria may only describe 'false' and 'true'")
        keys, descriptions = list(NOUL_KEYS), [given.get(k) for k in NOUL_KEYS]
    else:
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
            raise DecisionInputError(f"{where}: a score takes 2 to {MAX_SCORE_LEVELS} levels as an ordered list")
        keys, descriptions = [str(i) for i in range(len(criteria))], list(criteria)
    task = raw.get("task")
    return Question(key, kind, raw.get("instructions"), keys, descriptions, criteria,
                    None if task is None else str(task))


def parse_questions(questions: Any) -> list[Question]:
    if not isinstance(questions, dict) or not questions:
        raise DecisionInputError("`questions` must be a nonempty object of question id -> question")
    return [parse_question(str(key), raw) for key, raw in questions.items()]


def _label_index(question: Question, value: Any, where: str) -> int | None:
    if value is None:
        return None
    if question.type == "noul":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        if isinstance(value, str) and value.lower() in ("true", "false", "yes", "no", "1", "0"):
            return int(value.lower() in ("true", "yes", "1"))
        raise DecisionInputError(f"{where}: a noul answer must be a boolean, got {value!r}")
    if question.type == "score":
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < question.num_options:
            raise DecisionInputError(f"{where}: a score answer must be a level index in [0, {question.num_options - 1}]")
        return value
    if str(value) not in question.option_keys:
        raise DecisionInputError(f"{where}: answer {value!r} is not one of the options")
    return question.option_keys.index(str(value))


def _distribution(question: Question, value: Any, where: str) -> list[float] | None:
    if value is None:
        return None
    keys = question.option_keys
    if question.type == "noul" and isinstance(value, (int, float)) and not isinstance(value, bool):
        values = [1.0 - float(value), float(value)]
    elif isinstance(value, dict):
        lookup = {str(k).lower() if question.type == "noul" else str(k): v for k, v in value.items()}
        if set(lookup) != set(keys):
            raise DecisionInputError(f"{where}: distribution keys {sorted(lookup)} must match the options {keys}")
        values = [float(lookup[k]) for k in keys]
    elif isinstance(value, list):
        values = [float(v) for v in value]
    else:
        return None  # a hard answer repeated as `target` (AutoJev style) adds nothing
    if len(values) != len(keys) or any(not math.isfinite(v) or v < 0 for v in values):
        raise DecisionInputError(f"{where}: a distribution needs one finite, nonnegative value per option")
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=1e-4):
        raise DecisionInputError(f"{where}: a distribution must sum to 1 (got {total:.6f})")
    return [v / total for v in values]


FLAT_FIELDS = ("question_type", "options", "option_descriptions", "label_index", "label_code", "soft_target")


def _flat_question(key: str, record: dict[str, Any], request_id: str) -> tuple[Question, Any, Any]:
    """A flat row (one question per row, as the deciformer pipeline writes them): the instruction text in `question`,
    with `question_type`, `options`, `option_descriptions`, `label` (or `label_index`) and `soft_target`, both in
    the order of `options`."""
    where = f"record {request_id}"
    options = record["options"]
    descriptions = record.get("option_descriptions") or [None] * len(options or [])
    if not isinstance(options, list) or not isinstance(descriptions, list) or len(descriptions) != len(options):
        raise DecisionInputError(f"{where}: `options` and `option_descriptions` must be lists of the same length")
    kind = record.get("question_type", "choice")
    criteria = list(options) if _TYPE_ALIASES.get(str(kind)) == "score" else dict(zip(map(str, options), descriptions))
    question = parse_question(key, {"type": kind, "instructions": record["question"], "criteria": criteria})
    label = record.get("label")
    if question.type == "score" or label is None:
        label = record.get("label_index")
    soft = record.get("soft_target")
    if isinstance(soft, list) and question.type != "score":
        if len(soft) != len(options):
            raise DecisionInputError(f"{where}: `soft_target` needs one value per option")
        soft = dict(zip(map(str, options), soft))  # by name, so the order of `options` does not matter
    return question, label, soft


def _first(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def parse_request(record: dict[str, Any], index: int = 0) -> list[Decision]:
    """One record (request) -> one Decision per question."""
    request_id = str(record.get("id", index))
    if "state" not in record:
        raise DecisionInputError(f"record {request_id}: needs `state`")
    if "questions" in record:
        questions = parse_questions(record["questions"])
        gold = _first(record, "gold", "expected", "answers") or {}
        targets = record.get("targets") or {}
        teachers = record.get("teachers") or {}
        if not isinstance(record.get("teacher"), (dict, type(None))):
            raise DecisionInputError(f"record {request_id}: use `teachers` (per question) in multi-question records")
        per_question = [(q, gold.get(q.key), targets.get(q.key), teachers.get(q.key)) for q in questions]
        task_of = {q.key: q.task or str(record.get("task") or "default") for q in questions}
        ids = {q.key: f"{request_id}:{q.key}" for q in questions}
    elif isinstance(record.get("question"), str) and "options" in record:
        key = str(record.get("question_key", "decision"))
        question, label, target = _flat_question(key, record, request_id)
        per_question = [(question, label, target, record.get("teacher"))]
        task_of = {key: str(record.get("task", record.get("suite", "default")) or "default")}
        ids = {key: request_id}
    elif "question" in record:
        key = str(record.get("question_key", "decision"))
        question = parse_question(key, record["question"])
        per_question = [(question, record.get("label"), record.get("target"), record.get("teacher"))]
        task_of = {key: str(record.get("task", record.get("suite", "default")) or "default")}
        ids = {key: request_id}
    else:
        raise DecisionInputError(f"record {request_id}: needs `questions` (or a single `question`)")
    family = record.get("family")
    family = request_id if family is None else str(family)
    known = {"id", "state", "questions", "question", "question_key", "gold", "expected", "answers", "targets",
             "teachers", "label", "target", "teacher", "task", "family", "suite", *FLAT_FIELDS}
    meta = {k: v for k, v in record.items() if k not in known}
    decisions = []
    for question, label, target, teacher in per_question:
        where = f"record {request_id} question {question.key!r}"
        label_index = _label_index(question, label, where)
        target_values = _distribution(question, target, f"{where} target")
        if label_index is None and target_values is not None:
            label_index = max(range(len(target_values)), key=target_values.__getitem__)
        decisions.append(Decision(
            id=ids[question.key], request_id=request_id, state=record["state"], question=question,
            label_index=label_index, target=target_values,
            teacher=_distribution(question, teacher, f"{where} teacher"),
            task=task_of[question.key], family=family, meta=meta))
    return decisions


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL (optionally .gz), or a Hugging Face dataset written as `hf:<name>[:<config>]:<split>`."""
    spec = str(path)
    if spec.startswith("hf:"):
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError("Reading 'hf:' datasets needs `pip install decisionformer[data]`") from error
        parts = spec[3:].split(":")
        if len(parts) not in (2, 3):
            raise ValueError("Use hf:<name>:<split> or hf:<name>:<config>:<split>")
        name, config, split = (parts[0], None, parts[1]) if len(parts) == 2 else parts
        return [dict(row) for row in load_dataset(name, config, split=split)]
    opener = gzip.open if spec.endswith(".gz") else open
    records = []
    with opener(spec, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{spec}:{number}: invalid JSON ({error.msg})") from error
    return records


def load_decisions(path: str | Path) -> list[Decision]:
    decisions = [d for index, record in enumerate(read_records(path)) for d in parse_request(record, index)]
    seen: set[str] = set()
    for decision in decisions:
        if decision.id in seen:
            raise ValueError(f"{path}: duplicate id {decision.id!r}")
        seen.add(decision.id)
    return decisions
