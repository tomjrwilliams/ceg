"""BasketTyped + DailyBtUvarTyped + DailyBtMvarTyped — the new bt shape that
takes typed Strategy/Basket parents directly (no DailyCombination/Container.comb
indirection).

These tests pin the construction + dependency-wiring contract. Handler
execution (full df pipeline) is covered separately once data fixtures land.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ceg.engine import (
    Graph,
    IdRef,
    ValueUpdate,
    ObjectChange,
    HandleUpdateLatestUpstream,
)
from ceg.objs.dev.strat import StrategyTyped, BasketTyped, StrategyParams
from ceg.objs.dev.strat.assets import AssetConcrete
from ceg.objs.dev.pfo.d.bt import (
    DailyBtUvarTyped,
    DailyBtMvarTyped,
    HandleNewDailyBtUvarTyped,
    HandleNewDailyBtMvarTyped,
)


_ASSET = AssetConcrete(product="FUT", symbol="ES")


# --- BasketTyped ---

def test_basket_holds_member_refs():
    b = BasketTyped(
        id="basket-0",
        parent="p",
        members=(IdRef("s-0"), IdRef("s-1")),
    )
    assert b.members == (IdRef("s-0"), IdRef("s-1"))

def test_basket_default_empty_members():
    b = BasketTyped(id="basket-0", parent="p")
    assert b.members == ()

def test_basket_upstream_paths_lists_member_ids():
    b = BasketTyped(
        id="basket-0",
        parent="p",
        members=(IdRef("s-0"), IdRef("s-1")),
    )
    assert set(b.upstream_paths()) == {"s-0", "s-1"}

def test_basket_handler_returns_triage_on_value_update():
    b = BasketTyped(id="basket-0", parent="p", members=(IdRef("s-0"),))
    h = b.handler(ValueUpdate("s-0"), b.Snap.new(b.id))
    assert isinstance(h, HandleUpdateLatestUpstream)


# --- DailyBtUvarTyped ---

def test_btuvar_constructed_with_strategy_parent_string():
    bt = DailyBtUvarTyped(id="s-bt-uv", parent="s")
    assert bt.parent == "s"
    assert bt.id == "s-bt-uv"

def test_btuvar_new_constructs_id_from_parent():
    strat = StrategyTyped(id="s", parent="p", asset=_ASSET)
    bt = DailyBtUvarTyped.new(parent=strat)
    assert bt.id == "s-bt-uv"

def test_btuvar_depends_on_parent_value_update():
    bt = DailyBtUvarTyped(id="s-bt-uv", parent="s")
    deps = bt.depends()
    assert len(deps) == 1
    assert deps[0].obj == "s-bt-uv"
    assert deps[0].depends == "s"
    assert "ValueUpdate" in deps[0].events

def test_btuvar_handler_returns_handler_on_value_update():
    strat = StrategyTyped(id="s", parent="p", asset=_ASSET, rx=IdRef("rx"))
    bt = DailyBtUvarTyped.new(parent=strat)
    h = bt.handler(ValueUpdate("s"), bt.Snap.new(bt.id))
    assert isinstance(h, HandleNewDailyBtUvarTyped)

def test_btuvar_handler_rejects_unrelated_event():
    bt = DailyBtUvarTyped(id="s-bt-uv", parent="s")
    with pytest.raises(ValueError):
        bt.handler(ObjectChange("s", diff=None), bt.Snap.new(bt.id))


# --- DailyBtMvarTyped ---

def test_btmvar_constructed_with_basket_parent_string():
    bt = DailyBtMvarTyped(id="b-bt-mv", parent="b")
    assert bt.parent == "b"

def test_btmvar_new_constructs_id_from_parent():
    basket = BasketTyped(id="b", parent="p", members=(IdRef("s-0"),))
    bt = DailyBtMvarTyped.new(parent=basket)
    assert bt.id == "b-bt-mv"

def test_btmvar_depends_on_basket_value_update():
    bt = DailyBtMvarTyped(id="b-bt-mv", parent="b")
    deps = bt.depends()
    assert len(deps) == 1
    assert deps[0].depends == "b"
    assert "ValueUpdate" in deps[0].events

def test_btmvar_handler_returns_handler_on_value_update():
    basket = BasketTyped(id="b", parent="p", members=(IdRef("s-0"),))
    bt = DailyBtMvarTyped.new(parent=basket)
    h = bt.handler(ValueUpdate("b"), bt.Snap.new(bt.id))
    assert isinstance(h, HandleNewDailyBtMvarTyped)


# --- graph round-trip ---

def test_strategy_basket_bt_round_trip_in_graph():
    strat = StrategyTyped(id="s", parent="p", asset=_ASSET, rx=IdRef("rx"))
    basket = BasketTyped(id="b", parent="p", members=(IdRef("s"),))
    btu = DailyBtUvarTyped.new(parent=strat)
    btm = DailyBtMvarTyped.new(parent=basket)
    g = (
        Graph(name="t")
        .with_obj(strat)
        .with_obj(basket)
        .with_obj(btu)
        .with_obj(btm)
    )
    # alias_parents flattens Object parent refs to id strings on insert;
    # check ids/relevant fields rather than full equality.
    assert g.get(IdRef("s"), StrategyTyped).id == "s"
    assert g.get(IdRef("b"), BasketTyped).id == "b"
    assert g.get(IdRef("s-bt-uv"), DailyBtUvarTyped).parent == "s"
    assert g.get(IdRef("b-bt-mv"), DailyBtMvarTyped).parent == "b"
