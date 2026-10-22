"""Masked next-token prediction (MNTP, as in LLM2Vec) to adapt a causal decoder to bidirectional attention.

A causally pretrained decoder loses quality when its attention is suddenly made bidirectional. A short MNTP stage
fixes most of that: mask a fraction of tokens and predict each masked token from the hidden state of the token
before it. The adapted model is saved as a regular checkpoint; point `model.backbone` at it for decision training.
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from deciformer import distributed as dist_utils
from deciformer.attention import apply_attention_mode
from deciformer.backbone import load_tokenizer
from deciformer.config import RunConfig
from deciformer.data import _seed, plan_epoch
from deciformer.finetune import count_parameters, is_peft, prepare_for_finetuning, resolve_dtype
from deciformer.inference import autocast
from deciformer.optim import lr_lambda
from deciformer.schema import read_records

logger = logging.getLogger("deciformer.adapt")


def _texts(config: RunConfig) -> list[str]:
    sources = config.adapt.texts or config.data.train
    if sources is None:
        raise ValueError("Set adapt.texts (or data.train) to the text used for adaptation")
    paths = [sources] if isinstance(sources, str) else list(sources)
    texts = []
    for path in paths:
        for record in read_records(path):
            value = record.get(config.adapt.text_field)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    if not texts:
        raise ValueError(f"No '{config.adapt.text_field}' text found in {paths}")
    return texts


def _mask_token_id(tokenizer: Any) -> int:
    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)
    ids = tokenizer("_", add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return int(ids[0])
    return int(tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.pad_token_id)


def adapt(config: RunConfig) -> Path:
    from transformers import AutoModelForCausalLM

    if config.finetune.mode not in ("full", "lora"):
        raise ValueError("adapt supports finetune.mode 'full' or 'lora'")
    state = dist_utils.setup(config.distributed.backend, config.distributed.timeout_minutes)
    logging.basicConfig(level=logging.INFO if state.is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    torch.manual_seed(config.train.seed)
    model_cfg = config.model
    tokenizer = load_tokenizer(model_cfg.backbone, model_cfg.revision, model_cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.backbone, revision=model_cfg.revision, trust_remote_code=model_cfg.trust_remote_code,
        dtype=resolve_dtype(config.finetune, config.train.mixed_precision),
        attn_implementation=model_cfg.attn_implementation)
    apply_attention_mode(model.base_model, "bidirectional")
    model = prepare_for_finetuning(model, config.finetune, added_token_ids=[]).to(state.device)
    trainable, total = count_parameters(model)
    logger.info("MNTP adaptation of %s: %s trainable / %s parameters", model_cfg.backbone, f"{trainable:,}",
                f"{total:,}")
    encoded = [tokenizer(text, truncation=True, max_length=config.adapt.max_length)["input_ids"]
               for text in _texts(config)]
    encoded = [ids for ids in encoded if len(ids) >= 4]
    lengths = [len(ids) for ids in encoded]
    mask_id, pad_id = _mask_token_id(tokenizer), int(tokenizer.pad_token_id)
    specials = set(tokenizer.all_special_ids)
    ddp: torch.nn.Module = (DistributedDataParallel(model, **dist_utils.no_forward_buffer_sync())
                            if state.enabled else model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    train = config.train
    optimizer = torch.optim.AdamW(parameters, lr=train.lr, weight_decay=train.weight_decay,
                                  betas=tuple(train.adam_betas), eps=train.adam_eps)

    def plans(epoch: int) -> list[Any]:
        return plan_epoch(lengths, global_batch_size=train.global_batch_size, world_size=state.world_size,
                          micro_batch_size=train.micro_batch_size, token_budget=train.token_budget, seed=train.seed,
                          epoch=epoch, drop_last=train.drop_last)

    steps_per_epoch = len(plans(0))
    total_steps = train.max_steps or train.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(train.scheduler, total_steps))
    scaler = torch.amp.GradScaler(state.device.type, enabled=train.mixed_precision == "fp16")

    def collate(indices: list[int], epoch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = max(lengths[i] for i in indices)
        input_ids = torch.full((len(indices), length), pad_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for row, index in enumerate(indices):
            ids = encoded[index]
            rng = random.Random(_seed(train.seed, "mntp", epoch, index))
            input_ids[row, :len(ids)] = torch.tensor(ids)
            attention[row, :len(ids)] = 1
            candidates = [p for p in range(1, len(ids)) if ids[p] not in specials]
            chosen = [p for p in candidates if rng.random() < config.adapt.mask_probability] or candidates[:1]
            for position in chosen:
                labels[row, position] = ids[position]
                input_ids[row, position] = mask_id
        return input_ids, attention, labels

    step = 0
    while step < total_steps:
        epoch, position = divmod(step, steps_per_epoch)
        plan = plans(epoch)[position]
        local = [(collate(m.indices, epoch), m.dummy) for m in plan.per_rank[state.rank]]
        local_tokens = sum(float((labels != -100).sum()) for (_, _, labels), dummy in local if not dummy)
        global_tokens = dist_utils.all_reduce_sum([local_tokens], state.device)[0]
        ddp.train()
        loss_total = 0.0
        for i, ((input_ids, attention, labels), dummy) in enumerate(local):
            input_ids, attention, labels = (t.to(state.device) for t in (input_ids, attention, labels))
            sync = contextlib.nullcontext() if (i == len(local) - 1 or not state.enabled) else ddp.no_sync()
            with sync:
                with autocast(state.device, train.mixed_precision):
                    logits = ddp(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
                # MNTP: the hidden state at position i-1 predicts the masked token at position i.
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
                loss = loss * (0.0 if dummy else 1.0)
                scaler.scale(loss * state.world_size / max(global_tokens, 1.0)).backward()
            loss_total += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, train.max_grad_norm or math.inf)
        scaler.step(optimizer)
        scaler.update()
        dist_utils.step_scheduler(scheduler)
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % train.log_every == 0 or step in (1, total_steps):
            summed = dist_utils.all_reduce_sum([loss_total], state.device)[0]
            logger.info("adapt step %d/%d mntp_loss %.4f lr %.3g", step, total_steps,
                        summed / max(global_tokens, 1.0), scheduler.get_last_lr()[0])
    output = Path(train.output_dir) / "adapted"
    if state.is_main:
        to_save = model.merge_and_unload() if is_peft(model) else model
        to_save.save_pretrained(str(output), safe_serialization=True)
        tokenizer.save_pretrained(str(output))
        logger.info("Saved adapted model to %s; set model.backbone to it and model.attention to 'bidirectional'",
                    output)
    dist_utils.teardown()
    return output
