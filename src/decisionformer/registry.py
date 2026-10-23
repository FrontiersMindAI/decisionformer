"""Name-to-class registries that let configs pick poolers and heads, and let plugins add new ones."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, *names: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            for name in names:
                if name in self._items:
                    raise ValueError(f"{self.kind} '{name}' is already registered")
                self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise ValueError(f"Unknown {self.kind} '{name}'. Available: {', '.join(sorted(self._items))}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


POOLERS: Registry = Registry("pooler")
HEADS: Registry = Registry("head")
