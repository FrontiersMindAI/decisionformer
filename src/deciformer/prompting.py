"""Render decisions into token sequences with a prompt format, and record where options and the answer sit."""

from __future__ import annotations

import itertools
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from deciformer.formats import PromptFormat, describe, get_format
from deciformer.schema import Decision

MAX_CODES = 255
_CODE_PROBE = "Choose an option."
_PLAIN_SUFFIX = "\nAnswer:"


class ContextLengthError(ValueError):
    """A prompt longer than the model accepts. Never truncated at inference (HTTP 422)."""


@dataclass
class Encoded:
    input_ids: list[int]
    token_type_ids: list[int] | None
    decision_index: int
    prefix_end: int
    option_spans: list[tuple[int, int]]


def candidate_codes() -> list[str]:
    letters = string.ascii_uppercase
    return [*letters, *("".join(pair) for pair in itertools.product(letters, repeat=2))]


def prepare_tokenizer(tokenizer: Any) -> None:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("The tokenizer has neither a pad nor an eos token")
        tokenizer.pad_token = tokenizer.eos_token


def render_mode(tokenizer: Any, kind: str) -> str:
    if kind == "encoder":
        return "pair"
    return "chat" if getattr(tokenizer, "chat_template", None) else "plain"


def _chat(tokenizer: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def answer_codes(tokenizer: Any, mode: str, chat_template_kwargs: dict[str, Any],
                 limit: int = MAX_CODES) -> tuple[list[str], list[int]]:
    """Up to `limit` codes (A..Z, AA..) that are one distinct token alone and right after the answer prompt."""
    if mode == "chat":
        prefix, spacer = _chat(tokenizer, [{"role": "user", "content": _CODE_PROBE}], chat_template_kwargs), ""
    else:
        prefix, spacer = "x" + _PLAIN_SUFFIX, " "
    base = tokenizer.encode(prefix, add_special_tokens=False)
    codes, ids, taken = [], [], {tokenizer.unk_token_id}  # a code must never be the unknown token
    for code in candidate_codes():
        alone = tokenizer.encode(spacer + code, add_special_tokens=False)
        if len(alone) != 1 or alone[0] in taken:
            continue
        if tokenizer.encode(prefix + spacer + code, add_special_tokens=False) != [*base, alone[0]]:
            continue
        codes.append(code)
        ids.append(alone[0])
        taken.add(alone[0])
        if len(codes) == limit:
            break
    return codes, ids


class PromptBuilder:
    def __init__(self, tokenizer: Any, fmt: PromptFormat | str, kind: str, *, max_length: int | None,
                 overflow: str = "error", max_options: int = MAX_CODES, need_code_tokens: bool = True,
                 chat_template_kwargs: dict[str, Any] | None = None, codes: Sequence[str] | None = None,
                 code_token_ids: Sequence[int] | None = None) -> None:
        self.tokenizer = tokenizer
        self.format = get_format(fmt) if isinstance(fmt, str) else fmt
        self.kind = kind
        self.mode = render_mode(tokenizer, kind)
        self.max_length = max_length
        self.overflow = overflow
        self.chat_template_kwargs = dict(chat_template_kwargs if chat_template_kwargs is not None
                                         else {"enable_thinking": False})
        self.system_in_user = False
        if self.mode == "chat":
            try:
                _chat(tokenizer, [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                      self.chat_template_kwargs)
            except Exception:  # templates such as Gemma's reject a system role
                self.system_in_user = True
        if codes is not None:
            self.codes = list(codes)
            self.code_token_ids = list(code_token_ids) if code_token_ids is not None else None
        elif need_code_tokens:
            if self.mode == "pair":
                raise ValueError("Code-token heads need a decoder backbone")
            self.codes, self.code_token_ids = answer_codes(tokenizer, self.mode, self.chat_template_kwargs,
                                                           max_options)
            if len(self.codes) < max_options:
                raise ValueError(f"The tokenizer has only {len(self.codes)} single-token answer codes; "
                                 "lower data.max_options")
        else:
            self.codes, self.code_token_ids = candidate_codes()[:max_options], None

    def encode(self, decision: Decision, order: Sequence[int] | None = None,
               max_length: int | None = None) -> Encoded:
        order = list(range(decision.num_options)) if order is None else list(order)
        if len(order) > len(self.codes):
            raise ContextLengthError(f"question {decision.question.key!r} has {len(order)} options; this model takes "
                                     f"at most {len(self.codes)} options per choice")
        limit = max_length or self.max_length
        state = decision.state
        for _ in range(6):
            encoded = self._encode_once(decision, state, order)
            excess = len(encoded.input_ids) - limit if limit else 0
            if excess <= 0:
                return encoded
            if self.overflow == "error":
                raise ContextLengthError(f"question {decision.question.key!r} needs {len(encoded.input_ids)} tokens, "
                                         f"over the maximum context length of {limit} tokens")
            state_ids = self.tokenizer(describe(state), add_special_tokens=False)["input_ids"]
            keep = len(state_ids) - excess - 2
            if keep <= 0:
                raise ContextLengthError("the prompt without its state exceeds the maximum context length")
            kept = state_ids[:keep] if self.overflow == "truncate_state_end" else state_ids[-keep:]
            state = self.tokenizer.decode(kept, skip_special_tokens=True)
        raise ContextLengthError("could not truncate the state to fit the maximum context length")

    def _encode_once(self, decision: Decision, state: Any, order: list[int]) -> Encoded:
        question, fmt = decision.question, self.format
        if self.mode == "pair":
            block, spans = fmt.question_block(question, self.codes, order)
            encoding = self.tokenizer(fmt.state_text(state), block, add_special_tokens=True,
                                      return_offsets_mapping=True, return_token_type_ids=True)
            ids = list(encoding["input_ids"])
            offsets = [tuple(o) for o in encoding["offset_mapping"]]
            sequence_ids = encoding.sequence_ids()
            option_spans = [_token_span(offsets, s, e, sequence_ids, 1) for s, e in spans]
            types = encoding.get("token_type_ids")
            return Encoded(ids, list(types) if types is not None else None, 0, len(ids), option_spans)
        user, spans = fmt.user_text(state, question, self.codes, order)
        system = fmt.system
        if self.mode == "plain" or self.system_in_user:
            lead = system + "\n\n"
            user, spans, system = lead + user, [(s + len(lead), e + len(lead)) for s, e in spans], None
        if self.mode == "chat":
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
            text = _chat(self.tokenizer, messages, self.chat_template_kwargs)
            start = text.find(user)
            if start < 0:
                raise ValueError("The chat template changed the user message; deciformer cannot locate option spans")
            encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        else:
            text, start = user + _PLAIN_SUFFIX, 0
            encoding = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        ids = list(encoding["input_ids"])
        offsets = [tuple(o) for o in encoding["offset_mapping"]]
        option_spans = [_token_span(offsets, start + s, start + e, None, None) for s, e in spans]
        user_end = start + len(user)
        prefix_end = 1 + max(i for i, (s, e) in enumerate(offsets) if e > s and e <= user_end)
        return Encoded(ids, None, len(ids) - 1, prefix_end, option_spans)


def _token_span(offsets: list[tuple[int, int]], start: int, end: int, sequence_ids: list[int | None] | None,
                sequence: int | None) -> tuple[int, int]:
    hits = [i for i, (s, e) in enumerate(offsets)
            if e > s and s < end and e > start and (sequence_ids is None or sequence_ids[i] == sequence)]
    if not hits:
        raise ValueError("An option was truncated away or rendered to no tokens")
    return hits[0], hits[-1] + 1
