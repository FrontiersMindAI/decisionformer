"""Single- and multi-GPU (DDP) training with periodic evaluation, calibration, selection, export and exact resume."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from decisionformer import distributed as dist_utils
from decisionformer.config import RunConfig, dump_config, load_config
from decisionformer.data import Collator, DecisionDataset, plan_epoch
from decisionformer.finetune import count_parameters, effective_precision
from decisionformer.inference import (
    autocast,
    collect_logits,
    fit_type_temperatures,
    grouped_metrics,
    iterate_jsonl,
    prediction_records,
    shard,
    temperature_for,
)
from decisionformer.losses import decision_loss
from decisionformer.metrics import is_improvement, satisfies
from decisionformer.model import BuildResult, build_model, export_checkpoint, export_dtype, merge_checkpoint
from decisionformer.optim import build_optimizer, lr_lambda

logger = logging.getLogger("decisionformer")


def _setup_logging(is_main: bool) -> None:
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "huggingface_hub", "urllib3", "filelock"):  # Hub requests
        logging.getLogger(name).setLevel(logging.WARNING)


PRECISION_NAMES = {"no": "float32", "bf16": "bf16 autocast", "fp16": "fp16 autocast with loss scaling"}


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _gather_to_main(value: Any) -> list[Any]:
    import torch.distributed as dist

    if not dist.is_initialized():
        return [value]
    gathered: list[Any] | None = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(value, gathered, dst=0)
    return gathered or []


def _warn_about_leakage(config: RunConfig) -> None:
    """Warn when training data shares ids, families or identical states with the evaluation or calibration data
    (`decisionformer data check` also looks for near-duplicates)."""
    from decisionformer.datacheck import check_splits

    splits = {name: path for name, path in config.data.eval.items() if isinstance(path, str)}
    if isinstance(config.data.calibration, str):
        splits["calibration"] = config.data.calibration
    if not splits or not isinstance(config.data.train, str):
        return
    try:
        report = check_splits({**splits, "train": config.data.train}, near_duplicates=False)
    except Exception as error:  # noqa: BLE001 - a check must never stop training
        logger.debug("Leakage check skipped: %s", error)
        return
    for problem in report["problems"]:
        logger.warning("%s shares %d %s value(s) with an earlier split (e.g. %s); evaluation on it will be "
                       "optimistic. Run `decisionformer data check` for details.", problem["split"],
                       problem["count"], problem["overlap"], ", ".join(map(str, problem["examples"][:3])))


class Trainer:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.state = dist_utils.setup(config.distributed.backend, config.distributed.timeout_minutes)
        _setup_logging(self.state.is_main)
        self.output = Path(config.train.output_dir)
        if self.state.is_main:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "config.yaml").write_text(dump_config(config))
        if config.data.train is None:
            raise ValueError("data.train is required for training")
        _seed_everything(config.train.seed)
        self.result: BuildResult = build_model(config, self.state.device, training=True)
        self.model = self.result.model
        builder = self.result.builder
        self.train_set = DecisionDataset.from_paths(config.data.train, builder, labeled_only=True)
        self.eval_sets = {name: DecisionDataset.from_paths(paths, builder) for name, paths in config.data.eval.items()}
        self.calibration = DecisionDataset.from_paths(config.data.calibration, builder) if config.data.calibration else None
        if self.state.is_main:
            _warn_about_leakage(config)
        pad = self.result.tokenizer.pad_token_id
        self.train_collator = Collator(builder, pad, shuffle_options=config.data.shuffle_options, seed=config.train.seed)
        self.eval_collator = Collator(builder, pad)
        trainable, total = count_parameters(self.model)
        logger.info("Backbone %s (%s, attention=%s, finetune=%s): %s trainable / %s parameters", config.model.backbone,
                    self.result.kind, self.result.attention, config.finetune.mode, f"{trainable:,}", f"{total:,}")
        _seed_everything(config.train.seed + self.state.rank)  # distinct dropout noise per rank
        # Nothing to train (e.g. head_only + lm_verbalizer): evaluate and export, a zero-shot baseline.
        self.evaluation_only = trainable == 0
        self.ddp: torch.nn.Module = self.model
        if self.state.enabled and not self.evaluation_only:
            find_unused = config.train.find_unused_parameters
            if find_unused == "auto":
                find_unused = len(self.model.heads) > 1
            self.ddp = DistributedDataParallel(self.model, find_unused_parameters=bool(find_unused),
                                               **dist_utils.no_forward_buffer_sync())
        self.steps_per_epoch = len(self._plans(0))
        if self.steps_per_epoch == 0:
            raise ValueError("The training set is smaller than one global batch with drop_last=True")
        self.total_steps = 0 if self.evaluation_only else (config.train.max_steps
                                                           or config.train.epochs * self.steps_per_epoch)
        self.optimizer: Any = None
        self.scheduler: Any = None
        if not self.evaluation_only:
            self.optimizer = build_optimizer(self.model, config.train, config.finetune, distributed=self.state.enabled)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                               lr_lambda(config.train.scheduler, self.total_steps))
        self.precision = effective_precision(config.train.mixed_precision, self.state.device)
        if self.precision != config.train.mixed_precision and self.state.device.type == "cuda":
            logger.warning("This GPU has no bf16 support: training in fp16 with loss scaling "
                           "(set train.mixed_precision to choose)")
        self.scaler = torch.amp.GradScaler(self.state.device.type, enabled=self.precision == "fp16")
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.step = 0
        self.best: dict[str, Any] | None = None
        self.best_step: int | None = None
        self.temperatures: dict[str, float] = {}  # fitted at the best checkpoint
        self.last_temperatures: dict[str, float] = {}  # fitted at the latest evaluation
        self._plan_cache: dict[int, list[Any]] = {}
        self._resume()

    # ----------------------------------------------------------------------------------------------- planning
    def _plans(self, epoch: int) -> list[Any]:
        train = self.config.train
        return plan_epoch(self.train_set.lengths, global_batch_size=train.global_batch_size,
                          world_size=self.state.world_size, micro_batch_size=train.micro_batch_size,
                          token_budget=train.token_budget, seed=train.seed, epoch=epoch, drop_last=train.drop_last)

    def _plan_for(self, step: int) -> tuple[int, Any]:
        epoch, position = divmod(step, self.steps_per_epoch)
        if epoch not in self._plan_cache:
            self._plan_cache = {epoch: self._plans(epoch)}
        return epoch, self._plan_cache[epoch][position]

    # ----------------------------------------------------------------------------------------------- training
    def train(self) -> dict[str, Any]:
        config, state = self.config, self.state
        if self.evaluation_only:
            logger.info("No trainable parameters: evaluation-only run (zero-shot baseline)")
            self.evaluate()
            return self._finish()
        logger.info("Training %d steps (%d per epoch) on %d process(es), device %s, %s", self.total_steps,
                    self.steps_per_epoch, state.world_size, state.device.type, PRECISION_NAMES[self.precision])
        if config.eval.at_start and self.step == 0:
            self.evaluate()
        window_loss, window_count, window_start = 0.0, 0.0, time.monotonic()
        started, first_step = time.monotonic(), self.step
        while self.step < self.total_steps:
            epoch, _ = self._plan_for(self.step)
            loss_sum, count = self.accumulate_gradients(self.step)
            window_loss += loss_sum
            window_count += count
            grad_norm = self.apply_update()
            self.step += 1
            if self.step % config.train.log_every == 0 or self.step in (1, self.total_steps):
                loss_sum, count = dist_utils.all_reduce_sum([window_loss, window_count], state.device)
                per_step = (time.monotonic() - started) / (self.step - first_step)
                self._log({"event": "train", "step": self.step, "total_steps": self.total_steps, "epoch": epoch,
                           "loss": loss_sum / max(count, 1), "lr": self.scheduler.get_last_lr()[0],
                           "grad_norm": grad_norm, "seconds": time.monotonic() - window_start,
                           "eta_seconds": per_step * (self.total_steps - self.step)})
                window_loss, window_count, window_start = 0.0, 0.0, time.monotonic()
            evaluate_now = config.train.eval_every and self.step % config.train.eval_every == 0
            if evaluate_now or self.step == self.total_steps:
                self.evaluate()
            pausing = config.train.stop_after is not None and self.step >= config.train.stop_after
            if self.step < self.total_steps and (pausing or (config.train.save_every
                                                             and self.step % config.train.save_every == 0)):
                self.save_checkpoint()  # resumable state; a finished run needs none (best/ and last/ are exported)
            if pausing:
                logger.info("Pausing at step %d (train.stop_after); resume with train.resume=auto", self.step)
                break
        return self._finish()

    def accumulate_gradients(self, step: int) -> tuple[float, float]:
        """Forward/backward every micro-batch of `step` on this rank; gradients end up as the global-batch mean."""
        config, state = self.config, self.state
        epoch, plan = self._plan_for(step)
        micro_batches = plan.per_rank[state.rank]
        self.ddp.train()
        loss_sum, count = 0.0, 0.0
        for i, micro in enumerate(micro_batches):
            batch = self.train_collator(self.train_set, micro.indices, epoch=epoch, dummy=micro.dummy).to(state.device)
            last = i == len(micro_batches) - 1
            sync = contextlib.nullcontext() if (last or not state.enabled) else self.ddp.no_sync()
            with sync:
                with autocast(state.device, self.precision):
                    logits = self.ddp(batch)
                per_example = decision_loss(logits, batch.targets, batch.option_mask, config.train.loss,
                                            batch.teacher, batch.has_teacher)
                weighted = (per_example * batch.weights).sum()
                if not torch.isfinite(weighted):
                    raise FloatingPointError(
                        f"Non-finite loss at step {step + 1} (rank {state.rank}); the forward pass overflowed or "
                        "diverged. Try a lower train.lr, mixed_precision bf16 or no (fp16 has a narrow range), or "
                        "resume from an earlier checkpoint.")
                # DDP averages gradients over ranks, so scale by world_size / global examples to get the mean.
                self.scaler.scale(weighted * state.world_size / plan.global_count).backward()
            loss_sum += float(weighted.detach())
            count += float(batch.weights.sum())
        return loss_sum, count

    def apply_update(self) -> float:
        self.scaler.unscale_(self.optimizer)
        max_norm = self.config.train.max_grad_norm if self.config.train.max_grad_norm is not None else math.inf
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.trainable, max_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        dist_utils.step_scheduler(self.scheduler)
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    # ----------------------------------------------------------------------------------------------- evaluation
    def _rows(self, dataset: DecisionDataset) -> list[dict[str, Any]]:
        indices = shard(range(len(dataset)), self.state.rank, self.state.world_size)
        rows = collect_logits(self.model, dataset, self.eval_collator, indices, device=self.state.device,
                              micro_batch_size=self.config.eval.micro_batch_size,
                              token_budget=self.config.eval.token_budget,
                              mixed_precision=self.precision)
        return [row for part in _gather_to_main(rows) for row in part]

    def evaluate(self) -> dict[str, Any] | None:
        config = self.config
        calibration_rows = self._rows(self.calibration) if self.calibration else None
        set_rows = {name: self._rows(dataset) for name, dataset in self.eval_sets.items()}
        report: dict[str, Any] | None = None
        if self.state.is_main:
            temperatures = fit_type_temperatures(self.calibration, calibration_rows) if self.calibration else {}
            self.last_temperatures = temperatures
            report = {"event": "eval", "step": self.step, "temperatures": temperatures, "sets": {}}
            for name, rows in set_rows.items():
                dataset = self.eval_sets[name]
                raw = prediction_records(dataset, rows, lambda decision: 1.0)
                entry = {"raw": grouped_metrics(raw, bins=config.eval.ece_bins)}
                records = raw
                if temperatures:
                    records = prediction_records(
                        dataset, rows, lambda decision: temperature_for(temperatures, decision.question_type))
                    entry["calibrated"] = grouped_metrics(records, bins=config.eval.ece_bins)
                report["sets"][name] = entry
                if config.eval.save_predictions:
                    folder = self.output / "predictions"
                    folder.mkdir(exist_ok=True)
                    (folder / f"{name}-step{self.step:06d}.jsonl").write_text("".join(iterate_jsonl(records)))
            self._log(report)
            self._select(report, temperatures)
        dist_utils.barrier()
        return report

    def _select(self, report: dict[str, Any], temperatures: dict[str, float]) -> None:
        config = self.config
        if not report["sets"]:
            return
        set_name = config.eval.selection_set or next(iter(report["sets"]))
        entry = report["sets"][set_name]
        flavor = "calibrated" if config.eval.use_calibrated and "calibrated" in entry else "raw"
        metrics = entry[flavor].get("overall")
        if not metrics:
            return
        if not satisfies(metrics, config.eval.constraints):
            logger.info("Step %d does not satisfy selection constraints %s", self.step, config.eval.constraints)
            return
        if is_improvement(metrics, self.best, config.eval.selection_metric, config.eval.selection_mode):
            self.best, self.best_step, self.temperatures = metrics, self.step, temperatures
            export_checkpoint(self.result, config, self.output / "best", temperatures=temperatures,
                              metrics={"step": self.step, "set": set_name, "flavor": flavor, **metrics})
            logger.info("New best at step %d: %s=%.4f", self.step, config.eval.selection_metric,
                        metrics[config.eval.selection_metric])

    # ----------------------------------------------------------------------------------------------- checkpoints
    def save_checkpoint(self) -> None:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        state, folder = self.state, self.output / "checkpoints"
        final, tmp = folder / f"step-{self.step:06d}", folder / f"step-{self.step:06d}.tmp"
        if isinstance(self.optimizer, ZeroRedundancyOptimizer):
            self.optimizer.consolidate_state_dict(to=0)
        if state.is_main:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            payload = {"model": {k: v.cpu() for k, v in self.model.trainable_state().items()},
                       "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                       "scaler": self.scaler.state_dict(), "step": self.step, "best": self.best,
                       "best_step": self.best_step, "temperatures": self.temperatures,
                       "world_size": state.world_size}
            torch.save(payload, tmp / "state.pt")
        dist_utils.barrier()
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(rng, tmp / f"rng-rank{state.rank}.pt")
        dist_utils.barrier()
        if state.is_main:
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
            kept = sorted(p for p in folder.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
            for old in kept[: max(0, len(kept) - self.config.train.keep_last)]:
                shutil.rmtree(old)
            logger.info("Saved checkpoint %s", final)
        dist_utils.barrier()

    def _resume(self) -> None:
        target = self.config.train.resume
        if not target or self.evaluation_only:
            return
        if target == "auto":
            found = sorted(p for p in (self.output / "checkpoints").glob("step-*")
                           if p.is_dir() and not p.name.endswith(".tmp"))
            if not found:
                finished = self.output / "summary.json"
                if finished.exists() and json.loads(finished.read_text()).get("complete"):
                    logger.warning("%s holds a finished run; training it again from step 0 overwrites best/ and "
                                   "last/ (set train.output_dir to keep it)", self.output)
                else:
                    logger.info("resume=auto: no checkpoint found, starting fresh")
                return
            path = found[-1]
        else:
            path = Path(target)
        payload = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        expected = set(self.model.trainable_state())
        if set(payload["model"]) != expected:
            raise ValueError("Checkpoint parameters do not match this model's trainable parameters")
        self.model.load_state_dict(payload["model"], strict=False)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.step, self.best, self.best_step = payload["step"], payload["best"], payload["best_step"]
        self.temperatures = payload.get("temperatures", {})
        rng_file = path / f"rng-rank{self.state.rank}.pt"
        if payload.get("world_size") == self.state.world_size and rng_file.exists():
            rng = torch.load(rng_file, map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            logger.warning("World size changed since the checkpoint; data order is exact but dropout RNG is not")
        logger.info("Resumed from %s at step %d", path, self.step)

    # ----------------------------------------------------------------------------------------------- wrap-up
    def _log(self, record: dict[str, Any]) -> None:
        if not self.state.is_main:
            return
        line = json.dumps(record)
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")
        if record.get("event") == "train":
            logger.info("step %d/%d loss %.4f lr %.3g grad_norm %.3f, %s left (training only)", record["step"],
                        record["total_steps"], record["loss"], record["lr"], record["grad_norm"],
                        _duration(record["eta_seconds"]))
        else:
            for name, entry in record.get("sets", {}).items():
                for flavor, metrics in entry.items():
                    overall = metrics.get("overall", {})
                    if overall:
                        logger.info("eval %s [%s] step %d: acc %.4f ece %.4f brier %.4f nll %.4f", name, flavor,
                                    record["step"], overall["accuracy"], overall["ece"], overall["brier"],
                                    overall["nll"])

    def _finish(self) -> dict[str, Any]:
        config = self.config
        summary: dict[str, Any] = {}
        if self.state.is_main:
            export_checkpoint(self.result, config, self.output / "last", temperatures=self.last_temperatures,
                              metrics={"step": self.step})
            if self.best is None and not self.eval_sets:
                shutil.copytree(self.output / "last", self.output / "best", dirs_exist_ok=True)
            if config.finetune.export == "merged" and config.finetune.mode in ("lora", "qlora"):
                self._merge_exports()
            if self.step == self.total_steps and (self.output / "checkpoints").exists():
                shutil.rmtree(self.output / "checkpoints")  # only needed to resume an unfinished run
            summary = {"steps": self.step, "complete": self.step == self.total_steps, "best_step": self.best_step,
                       "best": self.best, "temperatures": self.temperatures, "output_dir": str(self.output)}
            if config.train.push_to_hub and (self.output / "best").exists():
                from decisionformer.hub import push_folder

                summary["hub_url"] = push_folder(self.output / "best", config.train.push_to_hub,
                                                 private=config.train.hub_private)
                logger.info("Pushed the best checkpoint to %s", summary["hub_url"])
            (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        dist_utils.barrier()
        return summary

    def _merge_exports(self) -> None:
        """Turn LoRA exports into full weights (the standard layout), saved in the base checkpoint's dtype. Exact
        merges in float32 on CPU for models up to ~3B parameters; larger ones merge in bfloat16 to bound memory."""
        size = sum(p.numel() for p in self.model.parameters())
        dtype = torch.float32 if size < 3e9 else torch.bfloat16
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for name in ("best", "last"):
            if (self.output / name).exists():
                merge_checkpoint(self.output / name, device="cpu", dtype=dtype, save_dtype=export_dtype(self.config))


def train(config: RunConfig | str | Path | Mapping[str, Any],
          overrides: Sequence[str] | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fine-tune and export a decision model; returns the run summary.

        decisionformer.train("deciformer15k_qwen3")                                  # a bundled recipe
        decisionformer.train("my.yaml", {"train.lr": 1e-4, "train.max_steps": 200})  # a file, with overrides
        decisionformer.train({"model": {...}, "data": {...}, "train": {"output_dir": "runs/x"}})
    """
    if not isinstance(config, RunConfig):
        config = load_config(config, overrides)
    elif overrides:
        config = load_config(config.model_dump(mode="json"), overrides)
    trainer = Trainer(config)
    try:
        return trainer.train()
    finally:
        dist_utils.teardown()
