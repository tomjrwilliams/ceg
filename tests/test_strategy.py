"""StrategyTyped: the new Strategy as a typed-Ref-slot graph namespace.

A StrategyTyped is a TriageUpstream Object whose data inputs live as
direct IdRef fields (rx/px/gross/signal/scale/cap). Downstream bt types
will consume strategy fields directly, with no Container.comb indirection.

These tests pin down the *shape* of the new Strategy. Bind orchestration
(param→object factories, registering children into the graph via with_ref)
is deferred to a follow-up — once that lands, this file gains end-to-end
construction tests.
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
from ceg.objs.dev.strat import StrategyTyped, StrategyParams
from ceg.objs.dev.strat.assets import AssetConcrete


_ASSET = AssetConcrete(product="FUT", symbol="ES")


# --- field shape ---

def test_typed_slots_default_none():
    s = StrategyTyped(id="s", parent="p", asset=_ASSET)
    for field in ("rx", "px", "gross", "signal", "scale", "cap", "bt", "perf"):
        assert getattr(s, field) is None, field

def test_typed_slots_hold_id_refs():
    s = StrategyTyped(
        id="s", parent="p", asset=_ASSET,
        rx=IdRef("rx-id"),
        px=IdRef("px-id"),
        gross=IdRef("gross-id"),
    )
    assert s.rx == IdRef("rx-id")
    assert s.px == IdRef("px-id")
    assert s.gross == IdRef("gross-id")

def test_signal_scale_cap_optional():
    s = StrategyTyped(
        id="s", parent="p", asset=_ASSET,
        signal=IdRef("sig"),
        scale=IdRef("sc"),
        cap=IdRef("cap"),
    )
    assert s.signal == IdRef("sig")
    assert s.scale == IdRef("sc")
    assert s.cap == IdRef("cap")


# --- TriageUpstream contract ---

def test_upstream_paths_only_filled_slots():
    s = StrategyTyped(
        id="s", parent="p", asset=_ASSET,
        rx=IdRef("rx"), px=IdRef("px"),
        # gross/signal/scale/cap unset
    )
    assert set(s.upstream_paths()) == {"rx", "px"}

def test_upstream_paths_empty_when_no_slots_filled():
    s = StrategyTyped(id="s", parent="p", asset=_ASSET)
    assert s.upstream_paths() == []

def test_upstream_paths_includes_all_when_filled():
    s = StrategyTyped(
        id="s", parent="p", asset=_ASSET,
        rx=IdRef("rx"), px=IdRef("px"), gross=IdRef("gross"),
        signal=IdRef("sig"), scale=IdRef("sc"), cap=IdRef("cap"),
    )
    assert set(s.upstream_paths()) == {"rx", "px", "gross", "sig", "sc", "cap"}


# --- handler ---

def test_handler_returns_triage_handler_on_value_update():
    s = StrategyTyped(id="s", parent="p", asset=_ASSET, rx=IdRef("rx"))
    h = s.handler(ValueUpdate("rx"), s.Snap.new(s.id))
    assert isinstance(h, HandleUpdateLatestUpstream)

def test_handler_rejects_unrelated_event():
    s = StrategyTyped(id="s", parent="p", asset=_ASSET, rx=IdRef("rx"))
    with pytest.raises(ValueError):
        s.handler(ObjectChange("s", diff=None), s.Snap.new(s.id))


# --- graph round-trip ---

def test_round_trips_in_graph():
    s = StrategyTyped(id="s", parent="p", asset=_ASSET, rx=IdRef("rx"))
    g = Graph(name="t").with_obj(s)
    fetched = g.get(IdRef("s"), StrategyTyped)
    assert fetched.rx == IdRef("rx")
    assert fetched.asset == _ASSET

def test_params_field_default():
    s = StrategyTyped(id="s", parent="p", asset=_ASSET)
    # params defaults to StrategyParams() — all-None scaling/signal/cap
    assert isinstance(s.params, StrategyParams)


# --- bt/perf ref slots ---

def test_bt_and_perf_default_none():
    s = StrategyTyped(id="s", parent="p", asset=_ASSET)
    assert s.bt is None
    assert s.perf is None

def test_bt_and_perf_set_to_id_refs():
    s = StrategyTyped(
        id="s", parent="p", asset=_ASSET,
        bt=IdRef("s-bt-uv"),
        perf=IdRef("s-bt-uv-rx-perf"),
    )
    assert s.bt == IdRef("s-bt-uv")
    assert s.perf == IdRef("s-bt-uv-rx-perf")
