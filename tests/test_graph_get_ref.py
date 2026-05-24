"""Graph.get must accept Refs alongside the existing str | Object surface.

Tests in this file build a minimal Graph by hand (no scheduling) — focus is
purely on the resolution semantics of the four supported input forms:

  - bare str        (back-compat: equivalent to IdRef)
  - Object          (back-compat: pass-through after type check)
  - IdRef           (look up Graph.objects)
  - AliasRef        (look up Graph.aliases, then recurse)
  - MemberRef       (resolve parent, walk field, index by slot, recurse)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import pytest
from frozendict import frozendict

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass
from dataclasses import replace

from ceg.engine.refs import Ref, IdRef, AliasRef, MemberRef
from ceg.engine import Graph, Object
from ceg.apis import t


@dataclass(frozen=True)
class _Leaf(Object):
    parent: str | None = None
    type: Literal["_Leaf"] = "_Leaf"


@dataclass(frozen=True)
class _Bag(Object):
    parent: str | None = None
    members: tuple[IdRef | AliasRef | MemberRef, ...] = ()
    outputs: t.frozen.dict[str, IdRef | AliasRef | MemberRef] = t.frozen.dict()
    type: Literal["_Bag"] = "_Bag"


def _graph_with(*objs: Object, aliases=None) -> Graph:
    g = Graph(name="t")
    for o in objs:
        g = replace(g, objects=g.objects.set(o.id, o))
    if aliases:
        g = replace(g, aliases=frozendict(aliases))
    return g


# --- bare str / Object back-compat ---

def test_get_str_returns_object():
    leaf = _Leaf(id="a")
    g = _graph_with(leaf)
    assert g.get("a", _Leaf) is leaf

def test_get_object_passes_through():
    leaf = _Leaf(id="a")
    g = _graph_with(leaf)
    assert g.get(leaf, _Leaf) is leaf


# --- IdRef ---

def test_get_idref_returns_object():
    leaf = _Leaf(id="a")
    g = _graph_with(leaf)
    assert g.get(IdRef("a"), _Leaf) is leaf

def test_get_idref_missing_id_raises():
    g = _graph_with()
    with pytest.raises((AssertionError, KeyError, ValueError)):
        g.get(IdRef("missing"), _Leaf)


# --- AliasRef ---

def test_get_aliasref_resolves_via_aliases_map():
    leaf = _Leaf(id="a")
    g = _graph_with(leaf, aliases={"short": IdRef("a")})
    assert g.get(AliasRef("short"), _Leaf) is leaf

def test_get_aliasref_unknown_alias_raises():
    leaf = _Leaf(id="a")
    g = _graph_with(leaf)
    with pytest.raises((AssertionError, KeyError, ValueError)):
        g.get(AliasRef("missing"), _Leaf)

def test_get_aliasref_does_not_fall_back_to_id_lookup():
    # Even if an Object with id == alias-name exists, AliasRef must consult
    # the aliases map only, not the objects map.
    leaf = _Leaf(id="x")
    g = _graph_with(leaf)  # aliases empty
    with pytest.raises((AssertionError, KeyError, ValueError)):
        g.get(AliasRef("x"), _Leaf)


# --- MemberRef ---

def test_get_memberref_indexes_tuple_slot():
    a = _Leaf(id="a")
    b = _Leaf(id="b")
    bag = _Bag(id="bag", members=(IdRef("a"), IdRef("b")))
    g = _graph_with(a, b, bag)
    assert g.get(MemberRef(parent=IdRef("bag"), field="members", slot=0), _Leaf) is a
    assert g.get(MemberRef(parent=IdRef("bag"), field="members", slot=1), _Leaf) is b

def test_get_memberref_indexes_dict_slot():
    a = _Leaf(id="a")
    bag = _Bag(id="bag", outputs=frozendict({"rx": IdRef("a")}))
    g = _graph_with(a, bag)
    assert g.get(MemberRef(parent=IdRef("bag"), field="outputs", slot="rx"), _Leaf) is a

def test_get_memberref_through_alias_parent():
    a = _Leaf(id="a")
    bag = _Bag(id="bag", members=(IdRef("a"),))
    g = _graph_with(a, bag, aliases={"the-bag": IdRef("bag")})
    assert g.get(
        MemberRef(parent=AliasRef("the-bag"), field="members", slot=0),
        _Leaf,
    ) is a

def test_get_memberref_through_memberref_parent():
    # Nested: bag2.members[0] == bag1, bag1.members[0] == leaf
    leaf = _Leaf(id="leaf")
    bag1 = _Bag(id="b1", members=(IdRef("leaf"),))
    bag2 = _Bag(id="b2", members=(IdRef("b1"),))
    g = _graph_with(leaf, bag1, bag2)
    inner = MemberRef(parent=IdRef("b2"), field="members", slot=0)
    outer = MemberRef(parent=inner, field="members", slot=0)
    assert g.get(outer, _Leaf) is leaf

def test_get_memberref_bad_field_raises():
    bag = _Bag(id="bag")
    g = _graph_with(bag)
    with pytest.raises((AttributeError, KeyError, ValueError, AssertionError)):
        g.get(MemberRef(parent=IdRef("bag"), field="nonexistent", slot=0), _Leaf)

def test_get_memberref_slot_out_of_range_raises():
    bag = _Bag(id="bag", members=())
    g = _graph_with(bag)
    with pytest.raises((IndexError, KeyError, ValueError, AssertionError)):
        g.get(MemberRef(parent=IdRef("bag"), field="members", slot=0), _Leaf)


# --- type check still applies ---

def test_get_idref_wrong_type_raises():
    bag = _Bag(id="bag")
    g = _graph_with(bag)
    with pytest.raises(AssertionError):
        g.get(IdRef("bag"), _Leaf)  # bag is _Bag, not _Leaf


# --- None pass-through ---

def test_get_none_returns_none():
    g = _graph_with()
    assert g.get(None, None) is None
