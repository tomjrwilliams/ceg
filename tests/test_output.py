"""Output: an Object whose value is written by its parent's handler.

Declares no Dependencies — never triggers work directly. Id derives from
parent + key|index. Used today by aggregation/combination derivs; designed to
absorb both into a single pattern.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ceg.engine import Output, Object


@dataclass(frozen=True)
class _Parent(Object):
    parent: str | None = None
    type: Literal["_Parent"] = "_Parent"


@dataclass(frozen=True)
class _Out(Output):
    parent: _Parent | str
    type: Literal["_Out"] = "_Out"


# --- depends() always empty ---

def test_output_has_no_dependencies():
    p = _Parent(id="p")
    o = _Out.new(parent=p, key="x")
    assert o.depends() == []


# --- id construction ---

def test_id_from_key_only():
    p = _Parent(id="p")
    o = _Out.new(parent=p, key="rx")
    assert o.id == "p-rx"

def test_id_from_index_only():
    p = _Parent(id="p")
    o = _Out.new(parent=p, index=3)
    assert o.id == "p-3"

def test_id_from_key_and_index():
    p = _Parent(id="p")
    o = _Out.new(parent=p, key="w", index=0)
    assert o.id == "p-w-0"


# --- requires at least one of key/index ---

def test_requires_key_or_index():
    p = _Parent(id="p")
    with pytest.raises(AssertionError):
        _Out.new(parent=p)


# --- distinctness ---

def test_distinct_keys_yield_distinct_ids():
    p = _Parent(id="p")
    assert _Out.new(parent=p, key="a").id != _Out.new(parent=p, key="b").id

def test_distinct_indices_yield_distinct_ids():
    p = _Parent(id="p")
    assert _Out.new(parent=p, index=0).id != _Out.new(parent=p, index=1).id

def test_key_and_index_distinct_from_key_only():
    # Output(key="x", index=0) and Output(key="x") must not collide.
    p = _Parent(id="p")
    a = _Out.new(parent=p, key="x", index=0)
    b = _Out.new(parent=p, key="x")
    assert a.id != b.id


# --- parent preserved at construction ---

def test_parent_stored():
    p = _Parent(id="p")
    o = _Out.new(parent=p, key="x")
    assert o.parent == p
