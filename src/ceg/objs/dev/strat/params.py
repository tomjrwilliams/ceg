from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    overload,
    cast,
    final,
)
import datetime
import datetime as dt

import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

import numpy as np
import polars as pl

from .... import apis
from ....apis import t, kv, assert_is
from ....engine import *

from ..bars.d import *
from ..calc.d import *
from ..eco.d import *

from ..agg import *
from ..pfo import *

from .assets import *
from .calcs import *

from functools import partial

#  ------------------

TSignal = (
    SignalOsc | SignalEco | SignalFactor | SignalWmThold
)

TScale = ScaleNotional | ScaleVol | ScaleFactor
TCap = CapNotional | CapVol


def unpack_paths(obj: Daily | tuple[Daily, ...] | None):
    return (
        obj.id
        if isinstance(obj, Daily)
        else tuple((o.id for o in obj))
        if isinstance(obj, tuple)
        else None
    )


@dataclass(frozen=True)
class StrategyParams:
    scale: ScaleNotional | ScaleVol | ScaleFactor | None = (
        None
    )
    signal: TSignal | None = None
    weight: WeightFactor | None = None
    cap: CapNotional | CapVol | None = None

    rebal: str | None = None

    scale_id: str | tuple[str, ...] | None = None
    signal_id: str | tuple[str, ...] | None = None
    weight_id: str | tuple[str, ...] | None = None
    cap_id: str | tuple[str, ...] | None = None
    rebal_id: str | tuple[str, ...] | None = None

    # unpack() removed — Bound/Collection-based ref resolution machinery deleted
    # during the Bound/Collection migration. If declarative ref-resolution is
    # needed again, build against the new Graph.with_obj alias/tag/info API.

    def with_weight(self, weight: WeightFactor | None):
        return replace(self, weight=weight)

    def label(self):
        return (
            f"{assert_is(self.scale).label()}"
            + (
                ""
                if self.signal is None
                else f"-{self.signal.label()}"
            )
            + (
                ""
                if self.weight is None
                else f"-{self.weight.label()}"
            )
            + (
                ""
                if self.rebal is None
                else f"-{self.rebal}"
            )
            + (
                ""
                if self.cap is None
                else f"-{self.cap.label()}"
            )
        )

    def bind(
        self,
        spec: Combination,
        params: frozendict[str, TParamValue],
        obj: Strategy | Basket,
        univ: Container,
    ):

        # if assert_scale:
        #     assert params.scale is not None, self

        if self.scale is not None:
            scale, kvs = self.scale.obj(obj)
            for k, v in kvs.items():
                params = params.set(
                    f"scale_{k}", assert_is(v, TParamValue)
                )
            spec, univ = spec.with_objects(
                univ,
                scale=assert_is(scale),
            )
            self = replace(
                self, scale_id=unpack_paths(scale)
            )

        if self.signal is not None:
            signal, kvs = self.signal.obj(obj)
            for k, v in kvs.items():
                params = params.set(
                    f"signal_{k}", assert_is(v, TParamValue)
                )
            spec, univ = spec.with_objects(
                univ,
                signal=assert_is(signal),
            )
            self = replace(
                self, signal_id=unpack_paths(signal)
            )

        if self.weight is not None:
            weight, kvs = self.weight.obj(obj)
            for k, v in kvs.items():
                params = params.set(
                    f"weight_{k}", assert_is(v, TParamValue)
                )
            spec, univ = spec.with_objects(
                univ,
                weight=assert_is(weight),
            )
            self = replace(
                self, weight_id=unpack_paths(weight)
            )

        if self.cap is not None:
            cap, kvs = self.cap.obj(obj)
            for k, v in kvs.items():
                params = params.set(
                    f"cap_{k}", assert_is(v, TParamValue)
                )
            spec, univ = spec.with_objects(
                univ,
                cap=assert_is(cap),
            )
            self = replace(self, cap_id=unpack_paths(cap))

        # TODO: if rebalance is not None:
        # as above

        univ, spec = spec.with_params(
            univ, t=Container, **params
        )

        return spec, self, univ


@dataclass(frozen=True)
class HasBackTest(HasAsset):
    params: StrategyParams
    name: str | None = None

    bt: DailyBtUvar | DailyBtMvar | None = None
    perf: DailyBtPerf | None = None

    perf_variants: tuple[DailyBtPerf] | None = None

    pfo: DailyPfolio | None = None
    ntl: DailyPfolioNotional | None = None
    ntl_cap: DailyPfolioNotionalCap | None = None

    def universe(self):
        return assert_is(self.bt).get_parent()

    def with_perf(self):
        pos = assert_is(self.bt).pos()
        context.get().with_obj(pos)

        rx = assert_is(self.bt).rx()
        context.get().with_obj(rx)

        gross = assert_is(self.bt).gross()
        context.get().with_obj(gross)

        perf = DailyBtPerf.new(rx)
        context.get().with_obj(perf)  # TODO: take gross?

        return replace(self, perf=perf)

    def with_perf_variants(
        self,
        *variants: DailyBtPerf,
    ):
        curr = self.perf_variants or ()
        return replace(self, perf_variants=curr + variants)

    def with_pfo(self):
        pfo = DailyPfolio.new(assert_is(self.bt))
        context.get().with_obj(pfo, force=True)

        pfo_rx = DailyPfolioRx.new(pfo)
        context.get().with_obj(pfo_rx, force=True)
        # NOTE: because ntl depends on pfo rx

        # by key?

        return replace(self, pfo=pfo)

    def with_pfo_ntl(
        self,
        # univ: Container,
        prices: dict[str, Daily],
        cap: float = 1.0,
        t: Type[TContainer] = Container,
    ):
        pfo = assert_is(self.pfo)
        univ = self.universe()
        # assert univ.id == self.universe().id, dict(
        #     given=univ.id,
        #     curr=self.universe().id,
        # )
        univ_px = {}

        for k, px in prices.items():
            univ, i = univ.with_object(px)
            univ_px[k] = assert_is(i, int)

        context.get().with_obj(univ, force=True)

        ntl = DailyPfolioNotional.new(
            pfo,
            **univ_px,
            cap=cap,
        )
        context.get().with_obj(ntl, force=True)

        return assert_is(univ, t), replace(self, ntl=ntl)


@dataclass(frozen=True)
class Strategy(HasBackTest, HasAsset, AssetLike):
    asset: TAsset

    params: StrategyParams
    name: str | None = None

    # TODO: separate the params from the objects?

    bt: DailyBtUvar | None = None
    perf: DailyBtPerf | None = None

    pfo: DailyPfolio | None = None
    ntl: DailyPfolioNotional | None = None
    ntl_cap: DailyPfolioNotionalCap | None = None

    def get_key(self):
        return None  # pfolio compression key

    def rx(self):
        return assert_is(self.bt).rx()

    def px(self):
        return assert_is(self.bt).px()

    def gross(self):
        return assert_is(self.bt).gross()

    def pos(self):
        return assert_is(self.bt).pos()

    def price_symbols(self):
        return self.asset.price_symbols()

    # TODO: with pfo, with ntl, with ntl_cap etc.

    def label(self):
        return f"{self.asset.label()}-{self.params.label()}"

    # unpack() removed — see StrategyParams.unpack comment.

    def bind_spec(
        self,
        univ: Container,
    ):

        assert isinstance(
            self.asset,
            (
                AssetConcrete
                | AssetSynthetic
                | Strategy
                | Basket
            ),
        ), self

        spec = Combination(
            key=self.asset.get_key(),
        )
        params: frozendict[str, TParamValue] = frozendict()
        params = params.set("unit", 1.0)

        rx = assert_is(self.asset.rx(), Daily)
        spec, univ = spec.with_objects(
            univ,
            rx=rx,
        )
        px = assert_is(self.asset.px(), Daily)
        spec, univ = spec.with_objects(
            univ,
            px=px,
        )
        gross = assert_is(self.asset.gross(), Daily)
        spec, univ = spec.with_objects(
            univ,
            gross=gross,
        )

        spec, self_params, univ = self.params.bind(
            spec, params, self, univ
        )
        self = replace(self, params=self_params)

        univ, spec = univ.with_comb(spec)

        return univ, self, spec

    @overload
    def bind(
        self,
        univ: Container,
        t: Type[TContainer],
        into: None = None,
    ) -> tuple[TContainer, Strategy]: ...

    @overload
    def bind(
        self,
        univ: Container,
        t: Type[TContainer],
        into: list[Strategy | Basket],
    ) -> tuple[TContainer, list[Strategy | Basket]]: ...

    def bind(
        self,
        univ: Container,
        t: Type[TContainer],
        into: list[Strategy | Basket] | None = None,
    ):
        univ, self, spec = self.bind_spec(univ)

        bt = DailyBtUvar.new(univ, spec)
        context.get().with_obj(bt)

        univ = assert_is(univ, t)
        self = assert_is(
            replace(self, bt=bt).with_perf(), Strategy
        )

        context.get().with_obj(univ, force=True)

        if into is None:
            return univ, self

        return univ, into + [self]

    def vis_spec_perf(self):
        return (
            assert_is(self.perf).id,
            self.label(),
            dict(group="perf"),
        )

    def vis_spec_rx(self):
        return (
            assert_is(self.rx()).id,
            self.label(),
            dict(group="rx"),
        )

    def vis_spec_signal(self):
        # TODO: get the object? not the spec
        assert isinstance(self.params.signal, str), self
        return (
            self.params.signal,
            self.label(),
            dict(group="signal"),
        )


@dataclass(frozen=True)
class Basket(HasBackTest, HasAsset, AssetLike):
    # always strat, even if basket of basket (wrap the basket in a strat, so we have scale etc.)
    asset: tuple[Strategy, ...]

    params: StrategyParams

    name: str = ""

    bt: DailyBtMvar | None = None

    def price_symbols(self):
        res = set()
        for a in self.asset:
            res = res.union(a.price_symbols())
        return res

    def label(self):
        assert self.name != "", self
        return f"{self.name}-{self.params.label()}"

    def get_key(self):
        return None

    def rx(self, i: int | None = None):
        return assert_is(self.bt).rx(i)

    def px(self, i: int | None = None):
        return assert_is(self.bt).px(i)

    def gross(self, i: int | None = None):
        return assert_is(self.bt).gross(i)

    def universe(self):
        return assert_is(self.bt).get_parent()

    # unpack() removed — see StrategyParams.unpack comment.

    def bind_specs(
        self,
        univ: Container,
        t: Type[TContainer],
        **mv_params,
    ):

        specs = []
        spec = Combination(
            key=None,
        )
        params: frozendict[str, TParamValue] = frozendict(
            **mv_params
        )
        params = params.set("unit", 1.0)

        spec, self_params, univ = self.params.bind(
            spec, params, self, univ
        )
        self = replace(self, params=self_params)
        univ, spec = univ.with_comb(spec)
        specs.append(spec)
        strats = []

        for strat in self.asset:
            univ, strat, strat_spec = strat.bind_spec(univ)
            specs.append(strat_spec)
            strats.append(strat)

        self = replace(self, asset=tuple(strats))

        return assert_is(univ, t), self, specs

    @overload
    def bind(
        self,
        univ: Container,
        t: Type[TContainer],
        into: None = None,
    ) -> tuple[TContainer, Basket]: ...

    @overload
    def bind(
        self,
        univ: Container,
        t: Type[TContainer],
        into: list[Strategy | Basket],
    ) -> tuple[TContainer, list[Strategy | Basket]]: ...

    def bind(
        self,
        univ: Container,
        t: Type[TContainer],
        into: list[Strategy | Basket] | None = None,
    ):
        univ, self, specs = self.bind_specs(univ, t)

        assert self.name != "", self

        bt = DailyBtMvar.new(univ, tuple(specs))
        context.get().with_obj(bt)

        univ = assert_is(univ, t)
        self = assert_is(
            replace(self, bt=bt).with_perf(), Basket
        )
        context.get().with_obj(univ, force=True)

        if into is None:
            return univ, self
        return univ, into + [self]

    def vis_spec_perf(self):
        return (
            assert_is(self.perf).id,
            self.label(),
            dict(group="perf"),
        )

    def vis_spec_rx(self, i: int | None = None):
        return (
            assert_is(self.rx(i)).id,
            self.label(),
            dict(group="rx"),
        )

    def vis_spec_signal(self):
        assert isinstance(self.params.signal, str), self
        return (
            self.params.signal,
            self.label(),
            dict(group="signal"),
        )


TAsset = AssetConcrete | AssetSynthetic | Strategy | Basket


@dataclass(frozen=True)
class Strategies(Container):
    rxs: t.frozen.dict[TAsset, int] = t.frozen.dict()

    type: Literal["Strategies"] = "Strategies"

    @classmethod
    def under_project(
        cls,
        project: Project,
        name: str,  # of container
        depth: int | None = None,
    ):
        if depth is not None:
            name = f"{name}-{depth}"
        return cls.new(project, name=name)

    def with_rx(
        self,
        asset: TAsset,
    ):
        rx = asset.rx()
        self, i = self.with_object(rx, Strategies)
        return replace(self, rxs=self.rxs.set(asset, i))


def new_container(
    project: Project,
    name: str,  # of container
    depth: int | None = None,
):
    return Strategies.under_project(project, name, depth=depth)


def cv(
    asset: AssetConcrete | tuple[str, str] | str,
    span: int,
    signal: TSignal | None = None,
    cap: CapNotional | CapVol | None = None,
    rebal: str | None = None,
    scale: float = 1.0,
    notional: float | None = None
):
    if isinstance(asset, str):
        ac, sym = asset.split("-")
        asset = (ac, sym)
    if isinstance(asset, tuple):
        asset = AssetConcrete(*asset)
    if notional is not None:
        cap = CapNotional(notional)
    return Strategy(
        asset,
        StrategyParams(
            scale=ScaleVol(span=span, val=scale),
            signal=signal,
            cap=cap,
            rebal=rebal,
        ),
    )


def cv_osc(
    asset: AssetConcrete | tuple[str, str],
    span: int,
    fast: int,
    ratio: float,
    vol: int | None = None,
    cap: CapNotional | CapVol | None = None,
    rebal: str | None = None,
    activation: str = "tanh",
    qile: SignalQile | SignalZile | None = None,
    scale: float = 1.0,
    variant: str | None = None,
    notional: float | None = None
):
    vol = vol or span
    return cv(
        asset,
        span=span,
        signal=SignalOsc(
            fast,
            int(fast * ratio),
            vol,
            activation=activation,
            qile=qile
            if isinstance(qile, SignalQile)
            else None,
            zile=qile
            if isinstance(qile, SignalZile)
            else None,
            constraint=variant,
        ),
        cap=cap,
        rebal=rebal,
        scale=scale,
        notional=notional,
    )


# --------------------------------------------------------------------------
# New "typed namespace" form of Strategy — runs alongside the legacy Strategy
# above during migration. Strategy is a graph Object (TriageUpstream namespace)
# with typed IdRef slots for its data inputs (rx/px/gross/signal/scale/cap).
# bt and perf are stored as IdRefs to the downstream backtest objects.
#
# Bind orchestration / param→object factories are deferred to a follow-up;
# the new type can already be constructed and registered in a graph.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyTyped(TriageUpstream, UseRootStorage):

    parent: Project | Container | str

    asset: AssetConcrete | AssetSynthetic

    # typed input refs
    rx:     IdRef | None = None
    px:     IdRef | None = None
    gross:  IdRef | None = None
    signal: IdRef | None = None
    scale:  IdRef | None = None
    cap:    IdRef | None = None

    params: StrategyParams = StrategyParams()
    name: str | None = None

    # downstream backtest / perf — also IdRefs once attached
    bt:   IdRef | None = None
    perf: IdRef | None = None

    version: int = 0
    type: Literal["StrategyTyped"] = "StrategyTyped"

    @dataclass(frozen=True)
    class Snap(TriageUpstream.Snap):
        type: Literal["StrategyTypedSnap"] = "StrategyTypedSnap"

    def upstream_paths(self) -> list[str]:
        return [
            r.id for r in (
                self.rx, self.px, self.gross,
                self.signal, self.scale, self.cap,
            ) if r is not None
        ]

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            return HandleUpdateLatestUpstream()
        raise ValueError((self, event))


@dataclass(frozen=True)
class BasketTyped(TriageUpstream, UseRootStorage):
    """A typed-namespace Basket: holds Refs to constituent Strategies (or
    nested BasketTyped). TriageUpstream fires one ValueUpdate when every
    member has first-published."""

    parent: Project | Container | str

    members: tuple[IdRef, ...] = ()

    params: StrategyParams = StrategyParams()
    name: str | None = None

    bt:   IdRef | None = None
    perf: IdRef | None = None

    version: int = 0
    type: Literal["BasketTyped"] = "BasketTyped"

    @dataclass(frozen=True)
    class Snap(TriageUpstream.Snap):
        type: Literal["BasketTypedSnap"] = "BasketTypedSnap"

    def upstream_paths(self) -> list[str]:
        return [r.id for r in self.members]

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            return HandleUpdateLatestUpstream()
        raise ValueError((self, event))


__all__ = [
    "TSignal",
    "TAsset",
    "TScale",
    "TCap",
    "Strategy",
    "StrategyTyped",
    "Basket",
    "BasketTyped",
    "StrategyParams",
    "Strategies",
    "new_container",
    "cv",
    "cv_osc",
]
