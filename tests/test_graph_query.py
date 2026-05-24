"""Graph absorbs Collection — session-local aliases / tags / info plus a
where() query DSL.

The Graph now records (per session, not persisted):
  - aliases: alias name → Ref
  - tags:    tag name   → set of object ids
  - info:    object id  → frozendict of annotations

with_obj accepts alias / tag / info kwargs; with_alias / with_tag / with_info
add to an existing entry. where() filters by type / tag / alias / lambda /
dot-path string lookup.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import pytest
from frozendict import frozendict

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ceg.engine import Graph, Object, IdRef, AliasRef


@dataclass(frozen=True)
class _Tags:
    symbol: str = ""
    adjust: str = ""

@dataclass(frozen=True)
class _A(Object):
    parent: str | None = None
    tags: _Tags = _Tags()
    type: Literal["_A"] = "_A"

@dataclass(frozen=True)
class _B(Object):
    parent: str | None = None
    symbol: str = ""
    type: Literal["_B"] = "_B"


# --- alias via with_obj ---

def test_with_obj_alias_kwarg_records_alias():
    g = Graph(name="t").with_obj(_A(id="long-id-1"), alias="short")
    assert g.get_alias("short") is not None
    assert g.get_alias("short").id == "long-id-1"

def test_alias_unknown_raises():
    g = Graph(name="t").with_obj(_A(id="x"))
    with pytest.raises((KeyError, AssertionError, ValueError)):
        g.get_alias("missing")

def test_alias_and_id_share_string_distinct_resolution():
    # IdRef("x") and AliasRef("x") must resolve independently.
    g = Graph(name="t").with_obj(_A(id="x"))
    g = g.with_obj(_A(id="other"), alias="x")
    assert g.get(IdRef("x"), _A).id == "x"
    assert g.get(AliasRef("x"), _A).id == "other"


# --- tag via with_obj ---

def test_with_obj_tag_kwarg_records_tag():
    g = (
        Graph(name="t")
        .with_obj(_A(id="a"), tag="x")
        .with_obj(_A(id="b"), tag="x")
        .with_obj(_A(id="c"), tag="y")
    )
    by_x = {o.id for o in g.where(tag="x")}
    by_y = {o.id for o in g.where(tag="y")}
    assert by_x == {"a", "b"}
    assert by_y == {"c"}

def test_with_obj_multi_tag():
    g = Graph(name="t").with_obj(_A(id="a"), tag=("x", "y"))
    assert {o.id for o in g.where(tag="x")} == {"a"}
    assert {o.id for o in g.where(tag="y")} == {"a"}


# --- info via with_obj ---

def test_with_obj_info_kwarg_recorded():
    g = Graph(name="t").with_obj(_A(id="a"), info={"k": "v"})
    assert g.info["a"]["k"] == "v"

def test_info_not_in_objects():
    # info is session-local; the object's persisted state is unaffected
    g = Graph(name="t").with_obj(_A(id="a"), info={"k": "v"})
    obj = g.get(IdRef("a"), _A)
    assert not hasattr(obj, "k")


# --- with_alias / with_tag / with_info on existing objects ---

def test_with_alias_after_creation():
    g = Graph(name="t").with_obj(_A(id="a"))
    g = g.with_alias(IdRef("a"), "alpha")
    assert g.get_alias("alpha").id == "a"

def test_with_tag_after_creation():
    g = Graph(name="t").with_obj(_A(id="a"))
    g = g.with_tag(IdRef("a"), "x")
    assert {o.id for o in g.where(tag="x")} == {"a"}

def test_with_info_after_creation():
    g = Graph(name="t").with_obj(_A(id="a"))
    g = g.with_info(IdRef("a"), {"k": "v"})
    assert g.info["a"]["k"] == "v"

def test_with_info_merges():
    g = (
        Graph(name="t")
        .with_obj(_A(id="a"), info={"k1": "v1"})
        .with_info(IdRef("a"), {"k2": "v2"})
    )
    assert g.info["a"]["k1"] == "v1"
    assert g.info["a"]["k2"] == "v2"


# --- where() filter forms ---

def test_where_lambda():
    g = (
        Graph(name="t")
        .with_obj(_A(id="a"))
        .with_obj(_A(id="b"))
        .with_obj(_A(id="c"))
    )
    res = {o.id for o in g.where(lambda o: o.id in ("a", "c"))}
    assert res == {"a", "c"}

def test_where_type():
    g = (
        Graph(name="t")
        .with_obj(_A(id="a"))
        .with_obj(_B(id="b"))
    )
    res = {o.id for o in g.where(type=_A)}
    assert res == {"a"}

def test_where_type_union():
    g = (
        Graph(name="t")
        .with_obj(_A(id="a"))
        .with_obj(_B(id="b"))
    )
    res = {o.id for o in g.where(type=(_A, _B))}
    assert res == {"a", "b"}

def test_where_alias_filter():
    g = (
        Graph(name="t")
        .with_obj(_A(id="a"), alias="alpha")
        .with_obj(_A(id="b"), alias="beta")
        .with_obj(_A(id="c"))
    )
    res = {o.id for o in g.where(alias="alpha")}
    assert res == {"a"}
    res = {o.id for o in g.where(alias=("alpha", "beta"))}
    assert res == {"a", "b"}

def test_where_dot_path_shallow_field():
    g = (
        Graph(name="t")
        .with_obj(_B(id="a", symbol="SPY"))
        .with_obj(_B(id="b", symbol="VYM"))
    )
    res = {o.id for o in g.where(paths={"symbol": "SPY"})}
    assert res == {"a"}

def test_where_dot_path_nested_field():
    g = (
        Graph(name="t")
        .with_obj(_A(id="a", tags=_Tags(symbol="SPY", adjust="pct")))
        .with_obj(_A(id="b", tags=_Tags(symbol="VYM", adjust="raw")))
        .with_obj(_A(id="c", tags=_Tags(symbol="SPY", adjust="raw")))
    )
    res = {o.id for o in g.where(paths={"tags.symbol": "SPY"})}
    assert res == {"a", "c"}

def test_where_combined_filters():
    g = (
        Graph(name="t")
        .with_obj(_A(id="a", tags=_Tags(symbol="SPY")), tag="x")
        .with_obj(_A(id="b", tags=_Tags(symbol="VYM")), tag="x")
        .with_obj(_A(id="c", tags=_Tags(symbol="SPY")))
    )
    res = {o.id for o in g.where(tag="x", paths={"tags.symbol": "SPY"})}
    assert res == {"a"}


# --- alias preserves session-locality (not persisted info) ---

def test_aliases_field_is_a_separate_map():
    g = Graph(name="t").with_obj(_A(id="a"), alias="alpha")
    # aliases lives on Graph; objects dict still keyed by id only
    assert "alpha" not in g.objects
    assert "a" in g.objects
