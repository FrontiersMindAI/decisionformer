"""torchrun worker: accumulate one step of gradients and save the synchronized result from rank 0.

The step is the first one where a 3-rank plan needs zero-weight padding micro-batches, so the padding path is
exercised; every world size accumulates that same step (the global batch is identical for any world size).
"""

import sys

import torch

from decisionformer import distributed
from decisionformer.config import load_config
from decisionformer.data import plan_epoch
from decisionformer.trainer import Trainer


def padding_step(trainer: Trainer, world_size: int = 3) -> tuple[int, int]:
    train = trainer.config.train
    plans = plan_epoch(trainer.train_set.lengths, global_batch_size=train.global_batch_size, world_size=world_size,
                       micro_batch_size=train.micro_batch_size, token_budget=train.token_budget, seed=train.seed,
                       epoch=0, drop_last=train.drop_last)
    for step, plan in enumerate(plans):
        padding = sum(micro.dummy for rank in plan.per_rank for micro in rank)
        if padding:
            return step, padding
    return 0, 0


if __name__ == "__main__":
    config_path, output_path, *overrides = sys.argv[1:]
    trainer = Trainer(load_config(config_path, overrides))
    step, padding_at_three = padding_step(trainer)
    _, plan = trainer._plan_for(step)
    padding = sum(micro.dummy for rank in plan.per_rank for micro in rank)
    loss_sum, count = trainer.accumulate_gradients(step)
    totals = distributed.all_reduce_sum([loss_sum, count], trainer.state.device)
    gradients = {name: p.grad.detach().cpu().clone() for name, p in trainer.model.named_parameters()
                 if p.requires_grad and p.grad is not None}
    if trainer.state.is_main:
        torch.save({"gradients": gradients, "loss": totals[0] / totals[1], "count": totals[1], "padding": padding,
                    "step": step, "padding_at_three": padding_at_three}, output_path)
    distributed.teardown()
