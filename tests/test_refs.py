"""Tests for the typed Ref family.

Each test pins one invariant. Failure modes do not share a single root cause —
e.g. a regression that collapsed every Ref to a string would fail multiple
tests below (cross-type equality, encoding distinctness, type preservation),
not just one.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import pytest
from frozendict import frozendict

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ceg.engine.refs import Ref, IdRef, AliasRef, MemberRef
from ceg.engine import Object, ObjectParser


# --- equality ---

def test_idref_equal_when_id_matches():
    assert IdRef("foo") == IdRef("foo")

def test_idref_unequal_when_id_differs():
    assert IdRef("foo") != IdRef("bar")

def test_aliasref_equal_when_name_matches():
    assert AliasRef("foo") == AliasRef("foo")

def test_idref_and_aliasref_with_same_string_are_unequal():
    # Load-bearing: id and alias maps must be addressable independently
    # even when their string contents happen to coincide.
    assert IdRef("foo") != AliasRef("foo")

def test_memberref_unequal_when_field_differs():
    a = MemberRef(parent=IdRef("c"), field="members", slot="0")
    b = MemberRef(parent=IdRef("c"), field="outputs", slot="0")
    assert a != b

def test_memberref_unequal_when_slot_differs():
    a = MemberRef(parent=IdRef("c"), field="members", slot="0")
    b = MemberRef(parent=IdRef("c"), field="members", slot="1")
    assert a != b

def test_memberref_unequal_when_parent_differs():
    a = MemberRef(parent=IdRef("c1"), field="members", slot="0")
    b = MemberRef(parent=IdRef("c2"), field="members", slot="0")
    assert a != b


# --- recursive parents ---

def test_memberref_parent_can_be_alias():
    r = MemberRef(parent=AliasRef("strats"), field="members", slot="0")
    assert isinstance(r.parent, AliasRef)
    assert r.parent.name == "strats"

def test_memberref_parent_can_be_memberref():
    inner = MemberRef(parent=IdRef("p"), field="children", slot="0")
    outer = MemberRef(parent=inner, field="members", slot="rx")
    assert outer.parent == inner
    assert isinstance(outer.parent, MemberRef)


# --- hashability / use as dict keys ---

def test_refs_hashable_in_frozendict():
    d = frozendict({IdRef("a"): 1, IdRef("b"): 2})
    assert d[IdRef("a")] == 1
    assert d[IdRef("b")] == 2

def test_id_and_alias_with_same_string_are_distinct_dict_keys():
    # Critical for the "alias and id can share a string" design:
    # a dict keyed on Ref must not collapse them.
    d = frozendict({IdRef("x"): "id", AliasRef("x"): "alias"})
    assert len(d) == 2
    assert d[IdRef("x")] == "id"
    assert d[AliasRef("x")] == "alias"


# --- JSON round-trip ---

def test_idref_json_roundtrip():
    r = IdRef("test-foo")
    decoded = Ref.parse_json(r.dump_bytes())
    assert decoded == r
    assert isinstance(decoded, IdRef)

def test_aliasref_json_roundtrip():
    r = AliasRef("my-alias")
    decoded = Ref.parse_json(r.dump_bytes())
    assert decoded == r
    assert isinstance(decoded, AliasRef)

def test_memberref_json_roundtrip():
    r = MemberRef(parent=IdRef("c"), field="members", slot="0")
    decoded = Ref.parse_json(r.dump_bytes())
    assert decoded == r
    assert isinstance(decoded, MemberRef)
    assert isinstance(decoded.parent, IdRef)

def test_memberref_int_slot_roundtrip():
    r = MemberRef(parent=IdRef("c"), field="members", slot=3)
    decoded = Ref.parse_json(r.dump_bytes())
    assert decoded == r
    assert decoded.slot == 3
    assert isinstance(decoded.slot, int)

def test_memberref_nested_parent_roundtrip():
    inner = MemberRef(parent=IdRef("p"), field="children", slot="0")
    outer = MemberRef(parent=inner, field="members", slot="rx")
    decoded = Ref.parse_json(outer.dump_bytes())
    assert decoded == outer
    assert isinstance(decoded, MemberRef)
    assert isinstance(decoded.parent, MemberRef)
    assert isinstance(decoded.parent.parent, IdRef)


# --- encoding distinctness (guards against "everything is a string") ---

def test_idref_and_aliasref_encode_differently():
    # If a regression made both encode as bare strings, this would fail —
    # IdRef("foo") and AliasRef("foo") must round-trip to different concrete types,
    # which requires them to encode to different JSON.
    assert IdRef("foo").dump_bytes() != AliasRef("foo").dump_bytes()

def test_idref_decodes_only_as_idref():
    decoded = Ref.parse_json(IdRef("foo").dump_bytes())
    assert isinstance(decoded, IdRef)
    assert not isinstance(decoded, (AliasRef, MemberRef))


# --- Ref inside an Object field round-trips ---

@dataclass(frozen=True)
class _Holder(Object):
    parent: str | None = None
    target: IdRef | AliasRef | MemberRef | None = Ref.union_field_optional()
    type: Literal["_Holder"] = "_Holder"


@dataclass(frozen=True)
class _HolderParser(ObjectParser):
    obj: _Holder


# initialise parser type
_ = _HolderParser(_Holder(id="x", target=None))


def test_idref_inside_object_field_roundtrips():
    o = _Holder(id="x", target=IdRef("test-target"))
    decoded = Object.parse_json(o.dump_bytes(), _HolderParser)
    assert isinstance(decoded.target, IdRef)
    assert decoded.target.id == "test-target"

def test_aliasref_inside_object_field_preserves_type():
    # An AliasRef inside an Object's field must NOT decay to IdRef or str.
    o = _Holder(id="x", target=AliasRef("my-alias"))
    decoded = Object.parse_json(o.dump_bytes(), _HolderParser)
    assert isinstance(decoded.target, AliasRef)
    assert not isinstance(decoded.target, IdRef)
    assert decoded.target.name == "my-alias"

def test_memberref_inside_object_field_roundtrips():
    o = _Holder(
        id="x",
        target=MemberRef(parent=IdRef("c"), field="members", slot="0"),
    )
    decoded = Object.parse_json(o.dump_bytes(), _HolderParser)
    assert isinstance(decoded.target, MemberRef)
    assert decoded.target.parent == IdRef("c")
    assert decoded.target.field == "members"
    assert decoded.target.slot == "0"
