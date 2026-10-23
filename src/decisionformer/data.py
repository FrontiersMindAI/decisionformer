"""Datasets, collation, and deterministic token-budget batch planning for single- and multi-GPU training."""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, fields

import torch

from decisionformer.prompting import Encoded, PromptBuilder
from decisionformer.schema import Decision, load_decisions


def _seed(*parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    decision_index: torch.Tensor
    prefix_end: torch.Tensor
    option_spans: torch.Tensor  # (batch, options, 2)
    option_mask: torch.Tensor  # (batch, options) bool
    num_options: torch.Tensor
    targets: torch.Tensor  # (batch, options) probabilities in display order
    label_index: torch.Tensor  # display-order index of the hard label, -1 if unlabeled
    teacher: torch.Tensor
    has_teacher: torch.Tensor
    weights: torch.Tensor  # 0 for padding micro-batches in distributed training
    indices: list[int] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    families: list[str | None] = field(default_factory=list)
    question_types: list[str] = field(default_factory=list)
    option_keys: list[list[str]] = field(default_factory=list)

    def to(self, device: torch.device | str) -> Batch:
        moved = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        return Batch(**moved)

    def __len__(self) -> int:
        return self.input_ids.shape[0]


class DecisionDataset:
    def __init__(self, decisions: list[Decision], builder: PromptBuilder) -> None:
        self.decisions = decisions
        self.lengths = [len(builder.encode(d).input_ids) for d in decisions]

    @classmethod
    def from_paths(cls, paths: str | Sequence[str], builder: PromptBuilder, *,
                   labeled_only: bool = False) -> DecisionDataset:
        """Load requests and expand them into one decision per question. `labeled_only` drops questions without
        a gold answer or target (Decision Index rows mark unscored questions with a null answer)."""
        paths = [paths] if isinstance(paths, str) else list(paths)
        decisions: list[Decision] = []
        for path in paths:
            decisions.extend(load_decisions(path))
        ids = [d.id for d in decisions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate ids across {paths}")
        if labeled_only:
            kept = [d for d in decisions if d.label_index is not None or d.target is not None]
            if not kept:
                raise ValueError(f"No labeled questions in {paths}")
            dropped = len(decisions) - len(kept)
            if dropped:
                logging.getLogger("decisionformer").info("Skipped %d unlabeled questions in %s", dropped, paths)
            decisions = kept
        return cls(decisions, builder)

    def __len__(self) -> int:
        return len(self.decisions)


class Collator:
    def __init__(self, builder: PromptBuilder, pad_token_id: int, *, shuffle_options: bool = False, seed: int = 0) -> None:
        self.builder = builder
        self.pad_token_id = pad_token_id
        self.shuffle_options = shuffle_options
        self.seed = seed

    def order_for(self, decision: Decision, index: int, epoch: int) -> list[int]:
        order = list(range(decision.num_options))
        if self.shuffle_options and decision.question_type == "choice":
            random.Random(_seed(self.seed, epoch, index)).shuffle(order)
        return order

    def __call__(self, dataset: DecisionDataset, indices: Sequence[int], *, epoch: int = 0,
                 dummy: bool = False) -> Batch:
        decisions = [dataset.decisions[i] for i in indices]
        orders = [self.order_for(d, i, epoch) for d, i in zip(decisions, indices)]
        encoded = [self.builder.encode(d, o) for d, o in zip(decisions, orders)]
        return make_batch(decisions, orders, encoded, self.pad_token_id, indices=list(indices), dummy=dummy)


def make_batch(decisions: Sequence[Decision], orders: Sequence[Sequence[int]], encoded: Sequence[Encoded],
               pad_token_id: int, *, indices: Sequence[int] | None = None, dummy: bool = False) -> Batch:
    """Pad encoded prompts into a batch; targets and labels follow each prompt's option order."""
    size, length = len(decisions), max(len(e.input_ids) for e in encoded)
    width = max(d.num_options for d in decisions)
    input_ids = torch.full((size, length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((size, length), dtype=torch.long)
    has_types = any(e.token_type_ids is not None for e in encoded)
    token_types = torch.zeros((size, length), dtype=torch.long) if has_types else None
    spans = torch.zeros((size, width, 2), dtype=torch.long)
    option_mask = torch.zeros((size, width), dtype=torch.bool)
    targets = torch.zeros((size, width))
    teacher = torch.zeros((size, width))
    has_teacher = torch.zeros(size, dtype=torch.bool)
    label_index = torch.full((size,), -1, dtype=torch.long)
    for row, (decision, order, item) in enumerate(zip(decisions, orders, encoded)):
        n = len(item.input_ids)
        input_ids[row, :n] = torch.tensor(item.input_ids)
        attention[row, :n] = 1
        if token_types is not None and item.token_type_ids is not None:
            token_types[row, :n] = torch.tensor(item.token_type_ids)
        k = decision.num_options
        spans[row, :k] = torch.tensor(item.option_spans)
        option_mask[row, :k] = True
        if decision.target is not None:
            targets[row, :k] = torch.tensor([decision.target[i] for i in order])
        elif decision.label_index is not None:
            targets[row, order.index(decision.label_index)] = 1.0
        if decision.label_index is not None:
            label_index[row] = order.index(decision.label_index)
        if decision.teacher is not None:
            teacher[row, :k] = torch.tensor([decision.teacher[i] for i in order])
            has_teacher[row] = True
    return Batch(
        input_ids=input_ids, attention_mask=attention, token_type_ids=token_types,
        decision_index=torch.tensor([e.decision_index for e in encoded]),
        prefix_end=torch.tensor([e.prefix_end for e in encoded]),
        option_spans=spans, option_mask=option_mask,
        num_options=torch.tensor([d.num_options for d in decisions]), targets=targets, label_index=label_index,
        teacher=teacher, has_teacher=has_teacher,
        weights=torch.zeros(size) if dummy else torch.ones(size),
        indices=list(indices) if indices is not None else list(range(size)), ids=[d.id for d in decisions],
        tasks=[d.task for d in decisions], families=[d.family for d in decisions],
        question_types=[d.question_type for d in decisions],
        option_keys=[[d.option_keys[i] for i in o] for d, o in zip(decisions, orders)],
    )


def split_micro_batches(indices: Sequence[int], lengths: Sequence[int], micro_batch_size: int,
                        token_budget: int | None) -> list[list[int]]:
    """Greedy split of length-sorted indices so each micro-batch fits `micro_batch_size` and the padded token budget."""
    ordered = sorted(indices, key=lambda i: (-lengths[i], i))
    result: list[list[int]] = []
    current: list[int] = []
    for index in ordered:
        longest = lengths[current[0]] if current else lengths[index]
        too_many = len(current) >= micro_batch_size
        too_long = token_budget is not None and current and longest * (len(current) + 1) > token_budget
        if current and (too_many or too_long):
            result.append(current)
            current = []
        current.append(index)
    if current:
        result.append(current)
    return result


@dataclass
class MicroBatch:
    indices: list[int]
    dummy: bool = False


@dataclass
class StepPlan:
    per_rank: list[list[MicroBatch]]
    global_count: int


def plan_epoch(lengths: Sequence[int], *, global_batch_size: int, world_size: int, micro_batch_size: int,
               token_budget: int | None, seed: int, epoch: int, drop_last: bool) -> list[StepPlan]:
    """Identical on every rank: shuffle, cut global batches, deal them to ranks by length, pad micro-batch counts."""
    order = list(range(len(lengths)))
    random.Random(_seed(seed, "epoch", epoch)).shuffle(order)
    plans: list[StepPlan] = []
    for start in range(0, len(order), global_batch_size):
        chunk = order[start:start + global_batch_size]
        if drop_last and len(chunk) < global_batch_size:
            break
        by_length = sorted(chunk, key=lambda i: (-lengths[i], i))
        shares = [by_length[rank::world_size] for rank in range(world_size)]
        micro = [split_micro_batches(share, lengths, micro_batch_size, token_budget) for share in shares]
        steps = max(len(m) for m in micro)
        per_rank: list[list[MicroBatch]] = []
        for rank_micro in micro:
            batches = [MicroBatch(m) for m in rank_micro]
            filler = rank_micro[-1] if rank_micro else [by_length[-1]]
            batches += [MicroBatch(list(filler), dummy=True) for _ in range(steps - len(batches))]
            per_rank.append(batches)
        plans.append(StepPlan(per_rank, len(chunk)))
    return plans


def eval_batches(indices: Sequence[int], lengths: Sequence[int], micro_batch_size: int,
                 token_budget: int | None) -> list[list[int]]:
    return split_micro_batches(indices, lengths, micro_batch_size, token_budget)
