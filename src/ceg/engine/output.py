"""Output: an Object whose value is written by another object's handler.

An Output declares no Dependencies — it is never triggered to compute itself.
Instead, some upstream handler computes Output's value as a side effect of
its own work and writes it directly via the value KV (and sets the Output's
snap).

Output's id is derived from its parent plus a key/index slot, so each
output of a fan-out handler has a stable, distinguishable address.

Concrete storage flavours (chunked Daily series etc.) are provided by
subclasses that mix in the relevant storage trait.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from .objects import Object
from .depends import Dependency


@dataclass(frozen=True)
class Output(Object):

    parent: Object | str
    key: str | None = None
    index: int | None = None

    @classmethod
    def _suffix(cls, key: str | None, index: int | None) -> str:
        if key is not None and index is not None:
            return f"{key}-{index}"
        if key is not None:
            return key
        return str(index)

    @classmethod
    def new(
        cls,
        parent: Object,
        key: str | None = None,
        index: int | None = None,
        **kwargs,
    ):
        assert key is not None or index is not None, (cls, key, index)
        return cls(
            id=parent.extend_id(cls._suffix(key, index)),
            parent=parent,
            key=key,
            index=index,
            **kwargs,
        )

    def depends(self) -> list[Dependency]:
        return []


__all__ = ["Output"]
