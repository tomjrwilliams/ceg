"""DailyComposition: the unified replacement for DailyAggregate +
DailyCombination.

Holds inputs as typed Refs directly (was integer indices into Container.agg
or Container.comb). Method-driven handler dispatch via the HANDLERS registry.

DailyOutput: the unified deriv type that fans out from a composition's
handler. Already a thin alias over engine.Output for the Daily/chunked
storage flavour.
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

from ceg.engine import IdRef, AliasRef, MemberRef, Object, Graph, Dependency
from ceg.objs.dev.agg.composition import (
    DailyComposition,
    DailyOutput,
    get_composition_dfs,
)


# A minimal namespace-shaped object usable as a parent for compositions.
# Lives only in the test; the real flow uses Container or similar.
@dataclass(frozen=True)
class _Bag(Object):
    parent: str | None = None
    members: tuple[IdRef, ...] = ()
    type: Literal["_Bag"] = "_Bag"


# --- id construction is deterministic & captures inputs ---

def test_id_includes_method():
    bag = _Bag(id="bag")
    a = DailyComposition.new(parent=bag, method="hsum",  refs=(IdRef("x"),))
    b = DailyComposition.new(parent=bag, method="hprod", refs=(IdRef("x"),))
    assert a.id != b.id

def test_id_includes_refs():
    bag = _Bag(id="bag")
    a = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),))
    b = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("y"),))
    assert a.id != b.id

def test_id_is_deterministic():
    bag = _Bag(id="bag")
    a = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"), IdRef("y")))
    b = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"), IdRef("y")))
    assert a.id == b.id

def test_id_preserves_ref_order():
    # different order → different id; positional semantics
    bag = _Bag(id="bag")
    a = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"), IdRef("y")))
    b = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("y"), IdRef("x")))
    assert a.id != b.id

def test_id_includes_params():
    bag = _Bag(id="bag")
    a = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),))
    b = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),), params={"span": 4})
    assert a.id != b.id

def test_id_param_values_distinguish():
    bag = _Bag(id="bag")
    a = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),), params={"span": 4})
    b = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),), params={"span": 8})
    assert a.id != b.id


# --- refs storage ---

def test_refs_preserved_as_refs():
    bag = _Bag(id="bag")
    c = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"), IdRef("y")))
    assert c.refs == (IdRef("x"), IdRef("y"))

def test_string_refs_accepted_as_legacy_form():
    # Strings are accepted alongside Refs for migration ergonomics.
    bag = _Bag(id="bag")
    c = DailyComposition.new(parent=bag, method="hsum", refs=("legacy-id",))
    # Stored as-is; id encoding handles both
    assert "legacy-id" in c.id


# --- depends() ---

def test_depends_includes_parent():
    # DailyComposition extends DependsParentValue → at least one dep on parent.
    bag = _Bag(id="bag")
    c = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),))
    parent_deps = [d for d in c.depends() if d.depends == "bag"]
    assert len(parent_deps) == 1
    assert "ValueUpdate" in parent_deps[0].events

def test_depends_includes_peers():
    bag = _Bag(id="bag")
    c = DailyComposition.new(
        parent=bag, method="hsum",
        refs=(IdRef("x"),),
        peers=(IdRef("peer-1"), IdRef("peer-2")),
    )
    peer_targets = {d.depends for d in c.depends()}
    assert "peer-1" in peer_targets
    assert "peer-2" in peer_targets

def test_depends_peer_strings_accepted():
    bag = _Bag(id="bag")
    c = DailyComposition.new(
        parent=bag, method="hsum", refs=(IdRef("x"),),
        peers=("peer-1",),
    )
    peer_targets = {d.depends for d in c.depends()}
    assert "peer-1" in peer_targets


# --- handler dispatch by method name ---

def test_handler_dispatches_known_method():
    # `hsum` is a registered method in aggs.py HANDLERS
    from ceg.engine import ValueUpdate, context
    from ceg.objs.dev.agg.aggs import HandleNewDailyHSum

    bag = _Bag(id="bag")
    with context.graph("t_hsum"):
        context.get().with_obj(bag)
        c = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),))
        context.get().with_obj(c)
        h = c.handler(ValueUpdate("bag"), c.Snap.new(c.id))
    assert isinstance(h, HandleNewDailyHSum)

def test_handler_rejects_unknown_method():
    from ceg.engine import ValueUpdate, context
    bag = _Bag(id="bag")
    with context.graph("t_unknown"):
        context.get().with_obj(bag)
        c = DailyComposition.new(parent=bag, method="no-such-method", refs=(IdRef("x"),))
        context.get().with_obj(c)
        with pytest.raises((AssertionError, ValueError, KeyError, TypeError)):
            c.handler(ValueUpdate("bag"), c.Snap.new(c.id))


# --- DailyOutput is the unified deriv ---

def test_daily_output_parents_any_object():
    # Unlike DailyAggregateDeriv / DailyCombinationDeriv which narrowed parent,
    # DailyOutput accepts any Object.
    bag = _Bag(id="bag")
    o = DailyOutput.new(parent=bag, key="rx")
    assert o.id == "bag-rx"

def test_daily_output_parents_composition():
    bag = _Bag(id="bag")
    c = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),))
    o = DailyOutput.new(parent=c, key="weight", index=0)
    assert o.id == c.extend_id("weight-0")
    assert o.depends() == []


# --- deriv() shortcut on composition ---

def test_composition_deriv_returns_daily_output():
    bag = _Bag(id="bag")
    c = DailyComposition.new(parent=bag, method="hsum", refs=(IdRef("x"),))
    o = c.deriv(key="w", index=0)
    assert isinstance(o, DailyOutput)
    assert o.parent == c


# --- get_composition_dfs error path ---

def test_get_composition_dfs_empty_refs_raises():
    g = Graph(name="t")
    with pytest.raises(ValueError):
        get_composition_dfs((), g)
