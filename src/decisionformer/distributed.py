"""Thin helpers over torch.distributed for DDP training launched with torchrun."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def pick_device(local_rank: int = 0, world_size: int = 1) -> torch.device:
    """`DECISIONFORMER_DEVICE` (cuda, mps or cpu) when set; otherwise CUDA, then Apple's MPS for single-process
    runs, then CPU."""
    wanted = os.environ.get("DECISIONFORMER_DEVICE", "")
    if wanted in ("", "cuda") and torch.cuda.is_available():
        index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    if wanted in ("", "mps") and world_size == 1 and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup(backend: str = "auto", timeout_minutes: int = 30) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = pick_device(local_rank, world_size)
    use_cuda = device.type == "cuda"
    if world_size > 1 and not dist.is_initialized():
        chosen = backend if backend != "auto" else ("nccl" if use_cuda else "gloo")
        kwargs: dict[str, Any] = {"backend": chosen, "timeout": datetime.timedelta(minutes=timeout_minutes)}
        if chosen == "nccl":
            kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    return DistributedState(rank, world_size, local_rank, device)


def teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(value: Any, source: int = 0) -> Any:
    if not dist.is_initialized():
        return value
    holder = [value]
    dist.broadcast_object_list(holder, src=source)
    return holder[0]


def gather_objects(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def no_forward_buffer_sync() -> dict[str, bool]:
    """DDP keyword for 'sync buffers at init only' across torch versions (renamed in torch 2.14)."""
    import inspect

    from torch.nn.parallel import DistributedDataParallel

    if "forward_sync_buffers" in inspect.signature(DistributedDataParallel.__init__).parameters:
        return {"forward_sync_buffers": False}
    return {"broadcast_buffers": False}


def step_scheduler(scheduler: Any) -> None:
    """Advance the LR schedule every update, including fp16 steps the GradScaler skipped (expected early on)."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before")
        scheduler.step()


def all_reduce_sum(values: list[float], device: torch.device) -> list[float]:
    if not dist.is_initialized():
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.tolist()
