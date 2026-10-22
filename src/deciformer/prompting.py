"""Render decisions into token sequences and record where each option and the decision point sit."""

from __future__ import annotations

import itertools
import string
from dataclasses import dataclass
from typing import Any

from deciformer.config import PromptConfig
from deciformer.schema import Decision

_SENTINEL = "OPTIONS"


@dataclass
class Encoded:
    input_ids: list[int]
    token_type_ids: list[int] | None
    decision_index: int
    prefix_end: int
    option_spans: list[tuple[int, int]]


def candidate_codes(kind: str) -> list[str]:
    if kind == "numbers":
        return [str(i) for i in range(1, 703)]
    letters = list(string.ascii_uppercase)
    return letters + ["".join(pair) for pair in itertools.product(letters, repeat=2)]


def prepare_tokenizer(tokenizer: Any, prompt: PromptConfig, kind: str) -> list[int]:
    """Set padding, add the optional marker token, and return the ids of any tokens that were added."""
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("The tokenizer has neither a pad nor an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    added: list[int] = []
    if prompt.marker_token and prompt.marker_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [prompt.marker_token]},
                                     replace_additional_special_tokens=False)
        added.append(tokenizer.convert_tokens_to_ids(prompt.marker_token))
    return added


class PromptBuilder:
    def __init__(self, tokenizer: Any, prompt: PromptConfig, kind: str, *, max_length: int, overflow: str,
                 max_options: int, need_code_tokens: bool) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.kind = kind
        self.max_length = max_length
        self.overflow = overflow
        self.format = self._resolve_format()
        self.system_in_user = False
        if self.format == "chat" and prompt.system:
            try:
                self._chat_text(prompt.system, "probe")
            except Exception:  # templates such as Gemma's reject a system role
                self.system_in_user = True
        self.marker_id = tokenizer.convert_tokens_to_ids(prompt.marker_token) if prompt.marker_token else None
        self.codes, self.code_token_ids = self._select_codes(max_options, need_code_tokens)

    def _resolve_format(self) -> str:
        requested = self.prompt.format
        if requested == "auto":
            if self.kind == "encoder":
                return "pair"
            return "chat" if getattr(self.tokenizer, "chat_template", None) else "plain"
        if requested == "pair" and self.kind != "encoder":
            raise ValueError("prompt.format 'pair' is for encoders")
        if requested in ("chat", "plain") and self.kind == "encoder":
            raise ValueError("Encoders use prompt.format 'pair' (or 'auto')")
        if requested == "chat" and not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("prompt.format is 'chat' but the tokenizer has no chat template; use 'plain'")
        return requested

    def _chat_text(self, system: str | None, user: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **self.prompt.chat_template_kwargs)

    def _answer_prefix(self) -> tuple[str, str]:
        if self.format == "chat":
            return self._chat_text(None, "x"), ""
        return "x" + self.prompt.plain_suffix, " "

    def _select_codes(self, max_options: int, need_tokens: bool) -> tuple[list[str], list[int] | None]:
        candidates = candidate_codes(self.prompt.codes)
        if not need_tokens:
            return candidates[:max_options], None
        if self.format == "pair":
            raise ValueError("Code-token heads need a decoder backbone")
        prefix, spacer = self._answer_prefix()
        base = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        codes, ids = [], []
        for code in candidates:
            tokens = self.tokenizer(prefix + spacer + code, add_special_tokens=False)["input_ids"]
            if len(tokens) == len(base) + 1 and tokens[:-1] == base and tokens[-1] not in ids:
                codes.append(code)
                ids.append(tokens[-1])
            if len(codes) == max_options:
                break
        if len(codes) < max_options:
            raise ValueError(f"The tokenizer has only {len(codes)} single-token answer codes; lower data.max_options")
        return codes, ids

    def _option_block(self, decision: Decision, order: list[int]) -> tuple[str, list[tuple[int, int]]]:
        if len(order) > len(self.codes):
            raise ValueError(f"record {decision.id}: {len(order)} options exceed data.max_options={len(self.codes)}")
        lines, spans, position = [], [], 0
        separator = self.prompt.option_separator
        for slot, index in enumerate(order):
            line = self.prompt.option_template.format(code=self.codes[slot], text=decision.option_texts[index],
                                                      key=decision.option_keys[index])
            spans.append((position, position + len(line)))
            lines.append(line)
            position += len(line) + len(separator)
        return separator.join(lines), spans

    def _fill(self, template: str, decision: Decision, state: str, order: list[int]) -> tuple[str, list[tuple[int, int]]]:
        block, spans = self._option_block(decision, order)
        values = {"state": state, "options": _SENTINEL,
                  "instructions": decision.instructions or self.prompt.default_instructions,
                  "answer_instruction": self.prompt.answer_instruction}
        rendered = template.format_map(values)
        at = rendered.index(_SENTINEL)
        text = rendered.replace(_SENTINEL, block, 1)
        if self.prompt.marker_token:
            text += self.prompt.marker_token
        return text, [(at + start, at + end) for start, end in spans]

    def encode(self, decision: Decision, order: list[int] | None = None) -> Encoded:
        order = list(range(decision.num_options)) if order is None else order
        state = decision.state
        for _ in range(6):
            encoded = self._encode_once(decision, state, order)
            excess = len(encoded.input_ids) - self.max_length
            if excess <= 0:
                return encoded
            if self.overflow == "error":
                raise ValueError(f"record {decision.id}: {len(encoded.input_ids)} tokens exceed data.max_length="
                                 f"{self.max_length}; raise max_length or set data.overflow to truncate the state")
            state_ids = self.tokenizer(state, add_special_tokens=False)["input_ids"]
            keep = len(state_ids) - excess - 2
            if keep <= 0:
                raise ValueError(f"record {decision.id}: the prompt without its state exceeds data.max_length")
            kept = state_ids[:keep] if self.overflow == "truncate_state_end" else state_ids[-keep:]
            state = self.tokenizer.decode(kept, skip_special_tokens=True)
        raise ValueError(f"record {decision.id}: could not truncate the state to fit data.max_length")

    def _encode_once(self, decision: Decision, state: str, order: list[int]) -> Encoded:
        if self.format == "pair":
            return self._encode_pair(decision, state, order)
        system = self.prompt.system
        user, spans = self._fill(self.prompt.user_template, decision, state, order)
        if system and (self.system_in_user or self.format == "plain"):
            lead = system + "\n\n"
            user, spans, system = lead + user, [(s + len(lead), e + len(lead)) for s, e in spans], None
        if self.format == "chat":
            text = self._chat_text(system, user)
            start = text.find(user)
            if start < 0:
                raise ValueError("The chat template changed the user message; deciformer cannot locate option spans")
            encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        else:
            text, start = user + self.prompt.plain_suffix, 0
            encoding = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        ids = list(encoding["input_ids"])
        offsets = [tuple(o) for o in encoding["offset_mapping"]]
        user_end = start + len(user)
        option_spans = [_token_span(offsets, start + s, start + e, None, None) for s, e in spans]
        prefix_end = 1 + max(i for i, (s, e) in enumerate(offsets) if e > s and e <= user_end)
        decision_index = self._marker_index(ids) if self.marker_id is not None else len(ids) - 1
        return Encoded(ids, None, decision_index, prefix_end, option_spans)

    def _encode_pair(self, decision: Decision, state: str, order: list[int]) -> Encoded:
        text_b, spans = self._fill(self.prompt.pair_template, decision, state, order)
        encoding = self.tokenizer(state, text_b, add_special_tokens=True, return_offsets_mapping=True,
                                  return_token_type_ids=True)
        ids = list(encoding["input_ids"])
        offsets = [tuple(o) for o in encoding["offset_mapping"]]
        sequence_ids = encoding.sequence_ids()
        option_spans = [_token_span(offsets, s, e, sequence_ids, 1) for s, e in spans]
        token_types = encoding.get("token_type_ids")
        decision_index = self._marker_index(ids) if self.marker_id is not None else 0
        return Encoded(ids, list(token_types) if token_types is not None else None, decision_index, len(ids),
                       option_spans)

    def _marker_index(self, ids: list[int]) -> int:
        positions = [i for i, token in enumerate(ids) if token == self.marker_id]
        if not positions:
            raise ValueError("The marker token was truncated away")
        return positions[-1]


def _token_span(offsets: list[tuple[int, int]], start: int, end: int, sequence_ids: list[int | None] | None,
                sequence: int | None) -> tuple[int, int]:
    hits = [i for i, (s, e) in enumerate(offsets)
            if e > s and s < end and e > start and (sequence_ids is None or sequence_ids[i] == sequence)]
    if not hits:
        raise ValueError("An option was truncated away or rendered to no tokens")
    return hits[0], hits[-1] + 1
