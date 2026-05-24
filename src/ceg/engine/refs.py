"""Typed references into a Graph.

A Ref addresses an object in the graph without being one. Three forms:

  IdRef(id)                              — absolute id in Graph.objects
  AliasRef(name)                         — session-local alias
  MemberRef(parent, field, slot)         — a slot inside another object's
                                           ref-bearing field (parent may itself
                                           be any Ref — recursion allowed)

Refs round-trip as tagged JSON ({"type": "...", ...}), so any Ref stored
inside a persisted Object field decodes back to its concrete subclass.

The id/alias maps are independent — IdRef("x") and AliasRef("x") are
distinct addresses even when their string contents coincide.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Annotated, Union
import functools

from pydantic import TypeAdapter, Field

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Ref:

    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        """Use as a default for a union-typed field: `r: A | B | C = Ref.union_field()`."""
        return Field(discriminator="type")

    @classmethod
    def union_field_optional(cls):
        """Use as a default for an optional union-typed field."""
        return Field(default=None, discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    def dump_bytes(self) -> bytes:
        return _ref_adapter().dump_json(self)

    @classmethod
    def parse_json(cls, data: str | bytes | bytearray) -> Ref:
        return _ref_adapter().validate_json(data)


@dataclass(frozen=True)
class IdRef(Ref):
    id: str
    type: Literal["IdRef"] = Ref.type_field("IdRef")


@dataclass(frozen=True)
class AliasRef(Ref):
    name: str
    type: Literal["AliasRef"] = Ref.type_field("AliasRef")


@dataclass(frozen=True)
class MemberRef(Ref):
    parent: Annotated[
        Union[IdRef, AliasRef, "MemberRef"],
        Field(discriminator="type"),
    ]
    field: str
    slot: str | int
    type: Literal["MemberRef"] = Ref.type_field("MemberRef")


@functools.lru_cache(maxsize=1)
def _ref_adapter() -> TypeAdapter:
    return TypeAdapter(
        Annotated[
            Union[IdRef, AliasRef, MemberRef],
            Field(discriminator="type"),
        ]
    )


__all__ = ["Ref", "IdRef", "AliasRef", "MemberRef"]
