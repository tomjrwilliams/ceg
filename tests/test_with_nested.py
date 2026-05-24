"""Graph.with_ref(parent, field, child, ...) — the primitive that:

  1. introspects parent's `field` annotation to decide tuple vs dict slot shape
  2. derives child.id as `parent.id + "-" + field + "-" + (index|key)`
  3. stamps that id (and the parent ref) onto the child
  4. mutates parent's slot field to hold the child's IdRef
  5. writes both back to the graph (re-running parent.depends())
  6. returns (new_graph, IdRef-to-child)

Two slot shapes accepted, detected via the field type annotation:
  - `tuple[<RefTypes>, ...]`           — positional
  - `t.frozen.dict[str, <RefTypes>]`   — named (key required)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import pytest
from frozendict import frozendict

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ceg.engine import (
    Graph,
    Object,
    IdRef,
    AliasRef,
    MemberRef,
)
from ceg.apis import t


@dataclass(frozen=True)
class _TupleNS(Object):
    parent: str | None = None
    members: tuple[IdRef | AliasRef | MemberRef, ...] = ()
    type: Literal["_TupleNS"] = "_TupleNS"


@dataclass(frozen=True)
class _DictNS(Object):
    parent: str | None = None
    outputs: t.frozen.dict[str, IdRef | AliasRef | MemberRef] = t.frozen.dict()
    type: Literal["_DictNS"] = "_DictNS"


@dataclass(frozen=True)
class _MultiNS(Object):
    """A namespace with both a tuple slot and a dict slot."""
    parent: str | None = None
    members: tuple[IdRef | AliasRef | MemberRef, ...] = ()
    outputs: t.frozen.dict[str, IdRef | AliasRef | MemberRef] = t.frozen.dict()
    type: Literal["_MultiNS"] = "_MultiNS"


@dataclass(frozen=True)
class _LeafNS(Object):
    parent: str | None = None
    type: Literal["_LeafNS"] = "_LeafNS"


@dataclass(frozen=True)
class _SingleNS(Object):
    """A namespace whose slot holds at most one ref (not a list, not a dict)."""
    parent: str | None = None
    rx: IdRef | AliasRef | MemberRef | None = None
    px: IdRef | None = None
    type: Literal["_SingleNS"] = "_SingleNS"


# --- tuple slots: positional append ---

def test_tuple_slot_appends_at_end():
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    g, ref = g.with_ref(IdRef("p"), "members", _LeafNS(id="ignored"))
    assert ref == IdRef("p-members-0")

def test_tuple_slot_two_appends_get_distinct_ids():
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    g, r1 = g.with_ref(IdRef("p"), "members", _LeafNS(id="ignored"))
    g, r2 = g.with_ref(IdRef("p"), "members", _LeafNS(id="ignored"))
    assert r1 == IdRef("p-members-0")
    assert r2 == IdRef("p-members-1")

def test_tuple_slot_parent_field_holds_id_refs():
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    g, ref = g.with_ref(IdRef("p"), "members", _LeafNS(id="ignored"))
    parent_after = g.get(IdRef("p"), _TupleNS)
    assert parent_after.members == (IdRef("p-members-0"),)

def test_tuple_slot_child_registered_in_graph():
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    g, _ = g.with_ref(IdRef("p"), "members", _LeafNS(id="ignored"))
    child = g.get(IdRef("p-members-0"), _LeafNS)
    assert child is not None


# --- dict slots: named ---

def test_dict_slot_requires_key():
    g = Graph(name="t").with_obj(_DictNS(id="p"))
    with pytest.raises((AssertionError, ValueError)):
        g.with_ref(IdRef("p"), "outputs", _LeafNS(id="ignored"))

def test_dict_slot_uses_key_in_id():
    g = Graph(name="t").with_obj(_DictNS(id="p"))
    g, ref = g.with_ref(IdRef("p"), "outputs", _LeafNS(id="ignored"), key="rx")
    assert ref == IdRef("p-outputs-rx")

def test_dict_slot_parent_field_holds_keyed_refs():
    g = Graph(name="t").with_obj(_DictNS(id="p"))
    g, _ = g.with_ref(IdRef("p"), "outputs", _LeafNS(id="ignored"), key="rx")
    parent_after = g.get(IdRef("p"), _DictNS)
    assert parent_after.outputs == frozendict({"rx": IdRef("p-outputs-rx")})


# --- multiple slots on one parent ---

def test_multi_slot_tuple_and_dict_independent():
    g = Graph(name="t").with_obj(_MultiNS(id="p"))
    g, _ = g.with_ref(IdRef("p"), "members", _LeafNS(id="i"))
    g, _ = g.with_ref(IdRef("p"), "outputs", _LeafNS(id="i"), key="rx")
    parent_after = g.get(IdRef("p"), _MultiNS)
    assert parent_after.members == (IdRef("p-members-0"),)
    assert parent_after.outputs == frozendict({"rx": IdRef("p-outputs-rx")})


# --- ids are field-distinguishing ---

def test_same_position_different_field_yields_different_id():
    # Critical: with field name in the id, slot 0 of `members` doesn't collide
    # with slot 0 of `outputs` (if both were positional).
    g = Graph(name="t").with_obj(_MultiNS(id="p"))
    g, r1 = g.with_ref(IdRef("p"), "members", _LeafNS(id="i"))
    g, r2 = g.with_ref(IdRef("p"), "outputs", _LeafNS(id="i"), key="0")
    assert r1.id != r2.id


# --- ref returned is resolvable via Graph.get ---

def test_returned_ref_is_resolvable():
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    g, ref = g.with_ref(IdRef("p"), "members", _LeafNS(id="ignored"))
    assert g.get(ref, _LeafNS) is not None


def test_memberref_to_added_slot_resolves():
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    g, _ = g.with_ref(IdRef("p"), "members", _LeafNS(id="ignored"))
    leaf = g.get(MemberRef(parent=IdRef("p"), field="members", slot=0), _LeafNS)
    assert leaf is not None
    assert leaf.id == "p-members-0"


# --- errors ---

def test_unknown_field_raises():
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    with pytest.raises((ValueError, KeyError, AttributeError)):
        g.with_ref(IdRef("p"), "nonexistent", _LeafNS(id="i"))

def test_non_slot_field_raises():
    # `parent` field is str|None, not a ref-bearing slot — must reject.
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    with pytest.raises((ValueError, TypeError)):
        g.with_ref(IdRef("p"), "parent", _LeafNS(id="i"))


# --- child receives parent ref ---

def test_child_parent_set_to_parent_id():
    g = Graph(name="t").with_obj(_TupleNS(id="p"))
    g, ref = g.with_ref(IdRef("p"), "members", _LeafNS(id="ignored"))
    child = g.get(ref, _LeafNS)
    assert child.parent == "p"


# --- single-ref slots: no index/key needed ---

def test_single_slot_id_is_parent_dash_field():
    g = Graph(name="t").with_obj(_SingleNS(id="p"))
    g, ref = g.with_ref(IdRef("p"), "rx", _LeafNS(id="ignored"))
    assert ref == IdRef("p-rx")

def test_single_slot_parent_field_holds_ref():
    g = Graph(name="t").with_obj(_SingleNS(id="p"))
    g, _ = g.with_ref(IdRef("p"), "rx", _LeafNS(id="ignored"))
    parent_after = g.get(IdRef("p"), _SingleNS)
    assert parent_after.rx == IdRef("p-rx")

def test_single_slot_two_different_fields_yield_two_children():
    g = Graph(name="t").with_obj(_SingleNS(id="p"))
    g, r_rx = g.with_ref(IdRef("p"), "rx", _LeafNS(id="ignored"))
    g, r_px = g.with_ref(IdRef("p"), "px", _LeafNS(id="ignored"))
    assert r_rx == IdRef("p-rx")
    assert r_px == IdRef("p-px")
    parent_after = g.get(IdRef("p"), _SingleNS)
    assert parent_after.rx == IdRef("p-rx")
    assert parent_after.px == IdRef("p-px")

def test_single_slot_does_not_accept_index_or_key():
    # A single-ref slot can't be indexed/keyed; passing should raise or be ignored.
    g = Graph(name="t").with_obj(_SingleNS(id="p"))
    # Index/key are nonsense here; expect either ignore or error.
    # Implementation choice: error to keep contract clean.
    with pytest.raises((AssertionError, ValueError, TypeError)):
        g.with_ref(IdRef("p"), "rx", _LeafNS(id="ignored"), index=0)

def test_single_slot_overwrites_on_re_call():
    # Single-ref slots replace, not accumulate.
    g = Graph(name="t").with_obj(_SingleNS(id="p"))
    g, _ = g.with_ref(IdRef("p"), "rx", _LeafNS(id="ignored"))
    g, ref2 = g.with_ref(IdRef("p"), "rx", _LeafNS(id="ignored"))
    # Same id (since the slot string is just "rx"); the second call replaces.
    assert ref2 == IdRef("p-rx")
    parent_after = g.get(IdRef("p"), _SingleNS)
    assert parent_after.rx == IdRef("p-rx")
