from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
    Iterable,
    Union,
    overload,
    cast,
    final,
)
from types import UnionType
import datetime
import datetime as dt

import itertools
import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

import numpy as np
import polars as pl

from ..... import apis
from .....apis import t, kv, assert_is
from .....engine import *

from ...bars.d import *
from ...calc.d import *
from ...eco.d import *

from ...agg import *
from ...pfo import *

from ..assets import *
from ..calcs import *
from ..params import *
from ..domain import *

from .agg import (
    unpack_signal_param,
    unpack_scale_param,
    unpack_cap_param,
)

from .....vis import explorer


@dataclass(frozen=True)
class PfoWeightParams:
    metrics: str | tuple[str, ...]
    samples: str | int | tuple[str | int, ...]
    windows: float | tuple[float, ...]
    smoothings: float | tuple[float, ...]
    norms: str | tuple[str, ...]
    tops: int | tuple[int, ...]

    def encode(
        self,
    ):

        if isinstance(self.metrics, list):
            self = replace(
                self, metrics=tuple(self.metrics)
            )
        if isinstance(self.samples, list):
            self = replace(
                self, samples=tuple(self.samples)
            )
        if isinstance(self.windows, list):
            self = replace(
                self, windows=tuple(self.windows)
            )
        if isinstance(self.smoothings, list):
            self = replace(
                self, smoothings=tuple(self.smoothings)
            )
        if isinstance(self.norms, list):
            self = replace(self, norms=tuple(self.norms))
        if isinstance(self.tops, list):
            self = replace(self, tops=tuple(self.tops))
        return self

    def weight_keys(self, symbols: tuple) -> list[str]:
        metrics: str | tuple[str, ...] = self.metrics
        if not isinstance(metrics, tuple):
            metrics = (metrics,)
        samples: str | int | tuple[str | int, ...] = (
            self.samples
        )
        if not isinstance(samples, tuple):
            samples = (samples,)
        windows: float | tuple[float, ...] = self.windows
        if not isinstance(windows, tuple):
            windows = (windows,)
        smoothings: float | tuple[float, ...] = (
            self.smoothings
        )
        if not isinstance(smoothings, tuple):
            smoothings = (smoothings,)
        norms: str | tuple[str, ...] = self.norms
        if not isinstance(norms, tuple):
            norms = (norms,)
        tops: int | tuple[int, ...] = self.tops
        if not isinstance(tops, tuple):
            tops = (tops,)
        return [
            "-".join(
                map(
                    str,
                    [
                        metric,
                        sample,
                        window,
                        smoothing,
                        norm,
                        top,
                    ],
                )
            )
            for metric in metrics
            for sample in samples
            for window in windows
            for smoothing in smoothings
            for norm in norms
            for top in tops
            if top <= len(symbols)
        ]


# TODO: dont pass the signals etc. into the first basket generation
# then its just the rx itself

# so we can then apply signals to just the rx, with the weights given (and nothing else from the weight calc)
# as a standalone step

# eg. calculate the hrp weights over asset universe, then apply oscillator etc. over those weights

# in that case, we want one to one weights to symbol universe
# and then many to many (weighted) symbol universe to signal


# ie. if you want one to one weight to signal, pass directly as universe
# if you want to calc weights, then for each asset, a range of signals
# you pass in the universe as just rx, and then the signals separately


@dataclass(frozen=True)
class BasketSpec:
    symbols: tuple[str, ...]

    params: PfoWeightParams

    vol_norm: int | None = None
    not_cap: float | None = None

    params_strats: tuple[t.frozen.dict, ...] | None = None

    name: str | None = None

    w: DailyBtWeights | None = None

    def drop_particulars(self):
        return replace(self, symbols=(), name=None)

    def with_name(self, name: str):
        return replace(self, name=name)

    def with_symbols(self, symbols: Iterable[str]):
        # eg. anonymous for rank_baskets_dot
        return replace(self, symbols=tuple(symbols))

    def arity(self):
        # per weight_key
        return len(self.symbols) * (
            1
            if self.params_strats is None
            else len(self.params_strats)
        )

    @classmethod
    def new(
        cls,
        s: str | Iterable[str],
        *symbols: str,
        params: PfoWeightParams | None = None,
        vol_norm: int | None = None,
        not_cap: float | None = None,
        params_strats: tuple[dict, ...] | None = None,
        name: str | None = None,
    ):
        symbols = (
            (s,) if isinstance(s, str) else tuple(s)
        ) + symbols
        return cls(
            symbols,
            params=assert_is(params).encode(),
            vol_norm=vol_norm,
            not_cap=not_cap,
            params_strats=(
                params_strats
                if params_strats is None
                else tuple(map(frozendict, params_strats))
            ),
            name=name,
        )

    def update(self, **kwargs):
        return replace(self, **kwargs)

    def weight_objs(self, weight_key: str):
        w = assert_is(self.w)
        return [
            w.weight(weight_key, i)
            for i in range(self.arity())
        ]

    def with_weights(self):
        for w_key in self.weight_keys():
            ws = self.weight_objs(w_key)
            context.get().with_objs(*ws)
        return self

    def unpack_strat_params(self):
        if self.params_strats is None:
            return [StrategyParams()]
        p = self.params_strats
        res: list[StrategyParams] = []
        for p in self.params_strats:
            scale = p.get("scale", ())
            signal = p.get("signal", ())
            cap = p.get("cap", ())
            if not isinstance(scale, (list, tuple)):
                scale = (scale,)
            if not isinstance(signal, (list, tuple)):
                signal = (signal,)
            if not isinstance(cap, (list, tuple)):
                cap = (cap,)
            res.extend(
                *[  # type: ignore
                    StrategyParams(
                        scale=sc,
                        signal=sig,
                        cap=cp,
                    )
                    for sc in scale
                    for sig in signal
                    for cp in cap
                ]
            )
        return res

    def strat_params(self, weight_key: str | None):
        if weight_key is None:
            if self.params_strats is None:
                return [StrategyParams()]
            return self.unpack_strat_params()
        # NOTE: params are many to many with symbols
        # hence, we cycle through the params, binding to each weight
        # as weights are one to one with symbol-param product
        ps = itertools.cycle(self.strat_params(None))
        weight_objs = self.weight_objs(weight_key)
        return [
            p.with_weight(w)
            for w, p in zip(weight_objs, ps)
        ]

    def weight_keys(self) -> list[str]:
        return self.params.weight_keys(self.symbols)

    def to_basket(
        self,
        univ: Strategies,
        strats: dict[str, Strategy | Basket],
    ):
        rxs: list[str] = [
            strats[s].rx().id for s in self.symbols
        ]
        inds: list[int] = []  # of rx in univ

        for rx in rxs:
            univ, i = univ.with_object(rx, Strategies)
            inds.append(i)

        if self.params_strats is not None:
            inds = sum(
                [inds for _ in self.strat_params(None)], []
            )

        return (
            univ,
            Basket(
                asset=tuple(
                    [
                        Strategy(
                            AssetSynthetic(rx),
                            params=StrategyParams(),
                        )
                        for rx in rxs
                    ]
                ),
                params=StrategyParams(),
            ),
            inds,
        )

    def bind(
        self,
        univ: Strategies,
        strats: dict[str, Strategy | Basket],
    ):
        comb = Combination()

        univ, basket, inds = self.to_basket(univ, strats)

        univ, basket, specs = basket.bind_specs(
            univ,
            Strategies,
            metrics=self.params.metrics,
            samples=self.params.samples,
            windows=self.params.windows,
            smoothings=self.params.smoothings,
            norms=self.params.norms,
            tops=self.params.tops,
        )

        w = DailyBtWeights.new(univ, tuple(specs))
        context.get().with_obj(w)

        univ = assert_is(univ, Strategies)
        self = assert_is(
            replace(self, w=w).with_weights(),
            # with child keys?
            BasketSpec,
        )
        context.get().with_obj(univ, force=True)

        return comb, inds, self, univ


@dataclass(frozen=True)
class RankBasket(BasketSpec):
    pass


@dataclass(frozen=True)
class WeightBasket(BasketSpec):
    pass


comb_rank_basket = RankBasket.new
comb_weight_basket = WeightBasket.new


def param_str(o):
    try:
        return o.label()
    except:
        return str(o)


def build_baskets(
    baskets: dict[str, RankBasket | WeightBasket],
    project: Project,
    univ_prefix: str,
    cv: dict[str, Strategy | Basket],
    prices: dict[str, Daily],
):

    required = set().union(
        *[b.symbols for b in baskets.values()]
    )
    strats: dict[str, Strategy | Basket] = {
        **cv,
    }
    done = {}
    depth = 0

    while len(done) < len(baskets):
        n_curr = len(done)

        univ = Strategies.under_project(
            project,
            f"{univ_prefix}-rx",
            depth=depth,
        )
        univ_rank = Strategies.under_project(
            project,
            f"{univ_prefix}-rw",
            depth=depth,
        )
        univ_weight = Strategies.under_project(
            project,
            f"{univ_prefix}-w",
            depth=depth,
        )
        univ_r_norm: Strategies | None = None
        univ_r_cap: Strategies | None = None
        univ_w_norm: Strategies | None = None
        univ_w_cap: Strategies | None = None

        baskets_by_shared: dict[
            RankBasket | WeightBasket,
            list[RankBasket] | list[WeightBasket],
        ] = {}

        for b_key, b in baskets.items():
            if b_key in done:
                continue
            if any(
                [
                    symbol not in strats
                    for symbol in b.symbols
                ]
            ):
                continue

            b_params = b.drop_particulars()

            if b not in baskets_by_shared:
                baskets_by_shared[b_params] = []
            baskets_by_shared[b_params].append(
                b.with_name(b_key)  # type: ignore
            )

        for b_params, bs in baskets_by_shared.items():
            vol_norm = b_params.vol_norm
            not_cap = b_params.not_cap

            # TODO: b.use_gross

            if isinstance(b_params, RankBasket):
                bs = cast(list[RankBasket], bs)

                if vol_norm is not None:
                    if univ_r_norm is None:
                        univ_r_norm = (
                            Strategies.under_project(
                                project,
                                f"{univ_prefix}-rw-norm",
                                depth=depth,
                            )
                        )
                if not_cap is not None:
                    if univ_r_cap is None:
                        univ_r_cap = (
                            Strategies.under_project(
                                project,
                                f"{univ_prefix}-rw-cap",
                                depth=depth,
                            )
                        )

                if vol_norm is None and not_cap is None:
                    (univ, univ_rank, b_strats) = (
                        rank_baskets(
                            univ,
                            univ_rank,
                            b_params,
                            bs,
                            strats,
                            vol_norm=vol_norm,
                            not_cap=not_cap,
                        )
                    )
                    strats = {
                        **strats,
                        **b_strats,
                    }
                elif (
                    vol_norm is not None and not_cap is None
                ):
                    (
                        univ,
                        univ_rank,
                        univ_r_norm,
                        b_strats,
                        b_strats_norm,
                    ) = rank_baskets(
                        univ,
                        univ_rank,
                        b_params,
                        bs,
                        strats,
                        vol_norm=vol_norm,
                        not_cap=not_cap,
                        univ_norm=assert_is(univ_r_norm),
                    )
                    strats = {
                        **strats,
                        **b_strats,
                        **b_strats_norm,
                    }
                elif (
                    vol_norm is None and not_cap is not None
                ):
                    (
                        univ,
                        univ_rank,
                        univ_r_cap,
                        b_strats,
                        b_strats_cap,
                    ) = rank_baskets(
                        univ,
                        univ_rank,
                        b_params,
                        bs,
                        strats,
                        vol_norm=vol_norm,
                        not_cap=not_cap,
                        univ_cap=assert_is(univ_r_cap),
                        prices=prices,
                    )
                    strats = {
                        **strats,
                        **b_strats,
                        **b_strats_cap,
                    }

                elif (
                    vol_norm is not None
                    and not_cap is not None
                ):
                    (
                        univ,
                        univ_rank,
                        univ_r_norm,
                        univ_r_cap,
                        b_strats,
                        b_strats_norm,
                        b_strats_cap,
                    ) = rank_baskets(
                        univ,
                        univ_rank,
                        b_params,
                        bs,
                        strats,
                        vol_norm=vol_norm,
                        not_cap=not_cap,
                        univ_norm=assert_is(univ_r_norm),
                        univ_cap=assert_is(univ_r_cap),
                        prices=prices,
                    )
                    strats = {
                        **strats,
                        **b_strats,
                        **b_strats_norm,
                        **b_strats_cap,
                    }

                else:
                    raise ValueError(b_params)

            elif isinstance(b_params, WeightBasket):
                bs = cast(list[WeightBasket], bs)
                #     (
                #     basket_name,
                #     basket_var,
                # )
                if vol_norm is not None:
                    if univ_w_norm is None:
                        univ_w_norm = (
                            Strategies.under_project(
                                project,
                                f"{univ_prefix}-w-norm",
                                depth=depth,
                            )
                        )
                if not_cap is not None:
                    if univ_w_cap is None:
                        univ_w_cap = (
                            Strategies.under_project(
                                project,
                                f"{univ_prefix}-w-cap",
                                depth=depth,
                            )
                        )

                if vol_norm is None and not_cap is None:
                    (univ, univ_weight, b_strats) = (
                        weight_baskets(
                            univ,
                            univ_weight,
                            b_params,
                            bs,
                            strats,
                            vol_norm=vol_norm,
                            not_cap=not_cap,
                        )
                    )
                    strats = {
                        **strats,
                        **b_strats,
                    }
                elif (
                    vol_norm is not None and not_cap is None
                ):
                    (
                        univ,
                        univ_weight,
                        univ_w_norm,
                        b_strats,
                        b_strats_norm,
                    ) = weight_baskets(
                        univ,
                        univ_weight,
                        b_params,
                        bs,
                        strats,
                        vol_norm=vol_norm,
                        not_cap=not_cap,
                        univ_norm=assert_is(univ_w_norm),
                    )
                    strats = {
                        **strats,
                        **b_strats,
                        **b_strats_norm,
                    }
                elif (
                    vol_norm is None and not_cap is not None
                ):
                    (
                        univ,
                        univ_weight,
                        univ_w_cap,
                        b_strats,
                        b_strats_cap,
                    ) = weight_baskets(
                        univ,
                        univ_weight,
                        b_params,
                        bs,
                        strats,
                        vol_norm=vol_norm,
                        not_cap=not_cap,
                        univ_cap=assert_is(univ_w_cap),
                        prices=prices,
                    )
                    strats = {
                        **strats,
                        **b_strats,
                        **b_strats_cap,
                    }

                elif (
                    vol_norm is not None
                    and not_cap is not None
                ):
                    (
                        univ,
                        univ_weight,
                        univ_w_norm,
                        univ_w_cap,
                        b_strats,
                        b_strats_norm,
                        b_strats_cap,
                    ) = weight_baskets(
                        univ,
                        univ_weight,
                        b_params,
                        bs,
                        strats,
                        vol_norm=vol_norm,
                        not_cap=not_cap,
                        univ_norm=assert_is(univ_w_norm),
                        univ_cap=assert_is(univ_w_cap),
                        prices=prices,
                    )
                    strats = {
                        **strats,
                        **b_strats,
                        **b_strats_norm,
                        **b_strats_cap,
                    }

                else:
                    raise ValueError(b_params)
            else:
                raise ValueError(b_params)

            done = {**done, **{b.name: True for b in bs}}

        if n_curr == len(done):
            raise ValueError(
                dict(
                    universe=list(strats.keys()),
                    baskets_done=list(done.keys()),
                    baskets_todo=[
                        k for k in baskets if k not in done
                    ],
                    strats_found=[
                        k for k in required if k in strats
                    ],
                    strats_missing=[
                        k
                        for k in required
                        if not k in strats
                    ],
                )
            )

        context.get().with_obj(univ, force=True)
        context.get().with_obj(univ_rank, force=True)
        context.get().with_obj(univ_weight, force=True)

        if univ_w_norm is not None:
            context.get().with_obj(univ_w_norm, force=True)
        if univ_w_cap is not None:
            context.get().with_obj(univ_w_cap, force=True)
        if univ_r_cap is not None:
            context.get().with_obj(univ_r_cap, force=True)
        if univ_r_norm is not None:
            context.get().with_obj(univ_r_norm, force=True)

        depth += 1

    return strats


@overload
def rank_baskets_dot(
    project: Project,
    baskets: dict[str, list[Strategy]],
    rank_params: RankBasket,
    scale: TScale | None = None,
    cap: TCap | None = None,
    vol_norm: None = None,
) -> tuple[
    dict[str, Strategies],
    dict[str, Strategies],
    dict[str, Strategy | Basket],
]: ...


@overload
def rank_baskets_dot(
    project: Project,
    baskets: dict[str, list[Strategy]],
    rank_params: RankBasket,
    scale: TScale | None = None,
    cap: TCap | None = None,
    vol_norm: int = 256,
) -> tuple[
    dict[str, Strategies],
    dict[str, Strategies],
    dict[str, Strategies],
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
]: ...


def rank_baskets_dot(
    project: Project,
    baskets: dict[str, list[Strategy]],
    rank_params: RankBasket,
    scale: TScale | None = None,
    cap: TCap | None = None,
    vol_norm: int | None = None,
):
    weight_objs = {}

    res = {}
    res_norm = {}
    res_cap = {}

    univs = {}
    univs_rank = {}
    univs_norm = {}

    univ_prefix = project.name

    for name, strats in baskets.items():
        assert len(strats), name

        basket = rank_params.with_name(name)

        univ_prefix = project.name

        univ = Strategies.under_project(
            project,
            f"{name}-c-rw-panel",
        )
        univ_rank = Strategies.under_project(
            project,
            f"{name}-c-rw",
        )

        univ_norm = (
            None
            if vol_norm is None
            else Strategies.under_project(
                project,
                f"{name}-c-rw-norm",
            )
        )

        name = assert_is(basket.name)

        spec, inds, basket, univ = basket.with_symbols(
            [f"strat_{i}" for i in range(len(strats))]
        ).bind(
            univ,
            {
                f"strat_{i}": cast(Strategy | Basket, s)
                for i, s in enumerate(strats)
            },
        )

        assert len(inds), name

        for weight_key in basket.weight_keys():
            mvar_key = f"{univ_prefix}-{name}-{weight_key}"

            strat_inds = inds
            v_strats = strats

            assert len(v_strats), (
                weight_key,
                strats,
            )

            dot_sig = SignalFactor(
                factors=tuple(
                    [
                        assert_is(s.params.signal_path, str)
                        for s in v_strats
                    ]
                ),
                key=f"{mvar_key}-dot",
                weights=(
                    tuple(
                        [
                            vw.id
                            for vw in basket.weight_objs(
                                weight_key
                            )
                        ]
                    ),
                ),
            )

            uvar = Strategy(
                asset=v_strats[0].asset,
                params=StrategyParams(
                    signal=dot_sig,
                    cap=cap,
                    scale=scale,
                    # rebal=... TODO
                ),
            )

            univ_rank, uvar = uvar.bind(
                univ_rank, t=Strategies
            )
            context.get().with_obj(univ_rank, force=True)

            res[mvar_key] = uvar

            if vol_norm is not None:
                assert univ_norm is not None, (
                    univ_norm,
                    vol_norm,
                )
                mvar_norm = Strategy(
                    asset=AssetSynthetic(
                        uvar.rx().id,
                    ),
                    params=StrategyParams(
                        scale=ScaleVol(
                            span=vol_norm,
                            rx=uvar.rx().id,
                        ),
                        cap=cap,
                    ),
                    name=f"{mvar_key}_norm",
                )
                univ_norm, bt_norm = mvar_norm.bind(
                    univ_norm,
                    Strategies,
                )
                res_norm[f"{mvar_key}_norm"] = bt_norm

                context.get().with_obj(
                    univ_norm, force=True
                )

        univs[name] = univ
        univs_rank[name] = univ_rank
        if univ_norm is not None:
            univs_norm[name] = univ_norm

    if vol_norm is not None:
        return (
            univs,
            univs_rank,
            univs_norm,
            res,
            res_norm,
            # weight_objs
        )
    return univs, univs_rank, res
    # , weight_objs


@overload
def rank_baskets(
    univ: Strategies,
    univ_rank: Strategies,
    # indices in univ.objects
    basket_params: RankBasket,
    rank_params: list[RankBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: int,
    univ_norm: Strategies,
    not_cap: float,
    univ_cap: Strategies,
    prices: dict[str, Daily],
) -> tuple[
    Strategies,
    Strategies,
    Strategies,
    Strategies,
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
]: ...
@overload
def rank_baskets(
    univ: Strategies,
    univ_rank: Strategies,
    # indices in univ.objects
    basket_params: RankBasket,
    rank_params: list[RankBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: int,
    univ_norm: Strategies,
    not_cap: float | None = None,
    univ_cap: Strategies | None = None,
    prices: dict[str, Daily] | None = None,
) -> tuple[
    Strategies,
    Strategies,
    Strategies,
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
]: ...
@overload
def rank_baskets(
    univ: Strategies,
    univ_rank: Strategies,
    # indices in univ.objects
    basket_params: RankBasket,
    rank_params: list[RankBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: None = None,
    univ_norm: None = None,
    not_cap: float = 10.0,
    univ_cap: Strategies | None = None,
    prices: dict[str, Daily] = {},
) -> tuple[
    Strategies,
    Strategies,
    Strategies,
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
]: ...


@overload
def rank_baskets(
    univ: Strategies,
    univ_rank: Strategies,
    # indices in univ.objects
    basket_params: RankBasket,
    rank_params: list[RankBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: None = None,
    univ_norm: None = None,
    not_cap: None = None,
    univ_cap: None = None,
    prices: None = None,
) -> tuple[
    Strategies, Strategies, dict[str, Strategy | Basket]
]: ...


def rank_baskets(
    univ: Strategies,
    univ_rank: Strategies,
    # indices in univ.objects
    basket_params: RankBasket,
    rank_params: list[RankBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: int | None = None,
    univ_norm: Strategies | None = None,
    not_cap: float | None = None,
    univ_cap: Strategies | None = None,
    prices: dict[str, Daily] | None = None,
):
    weight_objs = {}

    res = {}
    res_norm = {}
    res_cap = {}

    for basket in rank_params:
        name = assert_is(basket.name)

        spec, inds, basket, univ = basket.bind(univ, strats)

        assert len(inds), name

        for weight_key in basket.weight_keys():
            mvar_key = f"{name}-{weight_key}"
            uvars = [
                # replace(
                #     i_to_params[i],
                #     weight=WeightFactor(
                #         vw.id,
                #         f"{weight_key}_{ii}"
                #     )
                # )
                Strategy(
                    asset=AssetSynthetic(
                        obj_path(univ.objects[i])
                    ),
                    params=StrategyParams(
                        signal=SignalFactor(
                            vw.id, f"{weight_key}_{ii}"
                        )
                    ),
                )
                for i, (
                    ii,
                    vw,
                ) in zip(
                    inds,
                    enumerate(
                        basket.weight_objs(weight_key)
                    ),
                )
            ]
            mvar = Basket(
                tuple(uvars),
                params=StrategyParams(),
                name=mvar_key,
            )
            univ_rank, bt = mvar.bind(
                univ_rank,
                Strategies,
            )

            res[mvar_key] = bt

            if vol_norm is not None:
                assert univ_norm is not None, (
                    univ_norm,
                    vol_norm,
                )
                mvar_norm = Strategy(
                    asset=AssetSynthetic(
                        bt.rx().id,
                    ),
                    params=StrategyParams(
                        scale=ScaleVol(
                            span=vol_norm,
                            rx=bt.rx().id,
                        ),
                    ),
                    name=f"{mvar_key}_norm",
                )
                univ_norm, bt_norm = mvar_norm.bind(
                    univ_norm,
                    Strategies,
                )
                res_norm[f"{mvar_key}_norm"] = bt_norm

                bt = bt_norm

            if not_cap is not None:
                univ_cap = assert_is(univ_cap)
                prices = assert_is(prices)

                bt = bt.with_pfo()

                if vol_norm is not None:
                    univ_norm, bt = bt.with_pfo_ntl(
                        cap=not_cap,
                        prices=prices,
                        t=Strategies,
                    )
                else:
                    univ_rank, bt = bt.with_pfo_ntl(
                        cap=not_cap,
                        prices=prices,
                        t=Strategies,
                    )

                cap_ratio = DailyPfolioNotionalCap.new(
                    assert_is(bt.ntl), key="gross"
                )
                context.get().with_obj(
                    cap_ratio, force=True
                )

                mvar_cap = Strategy(
                    asset=AssetSynthetic(
                        bt.rx().id,
                    ),
                    params=StrategyParams(
                        scale=ScaleFactor(
                            cap_ratio.id,
                            key=f"cap-n{not_cap}",
                        )
                    ),
                    name=f"{mvar_key}_cap",
                )
                univ_cap, bt_cap = mvar_cap.bind(
                    univ_cap, Strategies
                )
                res_cap[f"{mvar_key}_cap"] = bt_cap

    if not_cap is not None and vol_norm is not None:
        return (
            univ,
            univ_rank,
            univ_norm,
            univ_cap,
            res,
            res_norm,
            res_cap,
            # weight_objs
        )
    elif not_cap is not None:
        return (
            univ,
            univ_rank,
            univ_cap,
            res,
            res_cap,
            # weight_objs
        )
    elif vol_norm is not None:
        return (
            univ,
            univ_rank,
            univ_norm,
            res,
            res_norm,
            # weight_objs
        )
    return univ, univ_rank, res
    # , weight_objs


@overload
def weight_baskets(
    univ: Strategies,
    univ_weight: Strategies,
    # indices in univ.objects
    basket_params: WeightBasket,
    weight_params: list[WeightBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: int,
    univ_norm: Strategies,
    not_cap: float,
    univ_cap: Strategies,
    prices: dict[str, Daily],
) -> tuple[
    Strategies,
    Strategies,
    Strategies,
    Strategies,
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
]: ...


@overload
def weight_baskets(
    univ: Strategies,
    univ_weight: Strategies,
    # indices in univ.objects
    basket_params: WeightBasket,
    weight_params: list[WeightBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: int,
    univ_norm: Strategies,
    not_cap: float | None = None,
    univ_cap: Strategies | None = None,
    prices: dict[str, Daily] | None = None,
) -> tuple[
    Strategies,
    Strategies,
    Strategies,
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
]: ...


@overload
def weight_baskets(
    univ: Strategies,
    univ_weight: Strategies,
    # indices in univ.objects
    basket_params: WeightBasket,
    weight_params: list[WeightBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: None = None,
    univ_norm: None = None,
    not_cap: float = 10.0,
    univ_cap: Strategies | None = None,
    prices: dict[str, Daily] = {},
) -> tuple[
    Strategies,
    Strategies,
    Strategies,
    dict[str, Strategy | Basket],
    dict[str, Strategy | Basket],
]: ...


@overload
def weight_baskets(
    univ: Strategies,
    univ_weight: Strategies,
    # indices in univ.objects
    basket_params: WeightBasket,
    weight_params: list[WeightBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: None = None,
    univ_norm: None = None,
    not_cap: None = None,
    univ_cap: None = None,
    prices: None = None,
) -> tuple[
    Strategies, Strategies, dict[str, Strategy | Basket]
]: ...


def weight_baskets(
    univ: Strategies,
    univ_weight: Strategies,
    # indices in univ.objects
    basket_params: WeightBasket,
    weight_params: list[WeightBasket],
    strats: dict[str, Strategy | Basket],
    vol_norm: int | None = None,
    univ_norm: Strategies | None = None,
    not_cap: float | None = None,
    univ_cap: Strategies | None = None,
    prices: dict[str, Daily] | None = None,
):
    weight_objs = {}

    res = {}
    res_norm = {}
    res_cap = {}

    for basket in weight_params:
        name = assert_is(basket.name)

        spec, inds, basket, univ = basket.bind(univ, strats)

        assert len(inds), name

        for weight_key in basket.weight_keys():
            mvar_key = f"{name}-{weight_key}"
            uvars = [
                Strategy(
                    asset=AssetSynthetic(
                        obj_path(univ.objects[i])
                    ),
                    params=ps,
                )
                for i, ps in zip(
                    inds, basket.strat_params(weight_key)
                )
            ]
            if len(uvars) == 1:
                mvar = uvars[0]
            else:
                mvar = Basket(
                    tuple(uvars),
                    params=StrategyParams(),
                    name=mvar_key,
                )
            univ_weight, bt = mvar.bind(
                univ_weight,
                Strategies,
            )
            res[mvar_key] = bt

            # TODO: notional using gross not pfo

            # TODO: find a way to only apply the variant (long only etc.) after the norm, cap, etc.?

            if vol_norm is not None:
                assert univ_norm is not None, (
                    univ_norm,
                    vol_norm,
                )
                mvar_norm = Strategy(
                    asset=AssetSynthetic(
                        bt.rx().id,
                    ),
                    params=StrategyParams(
                        scale=ScaleVol(
                            span=vol_norm,
                            rx=bt.rx().id,
                        ),
                    ),
                    name=f"{mvar_key}_norm",
                )
                univ_norm, bt_norm = mvar_norm.bind(
                    univ_norm,
                    Strategies,
                )
                res_norm[f"{mvar_key}_norm"] = bt_norm

                bt = bt_norm

            if not_cap is not None:
                univ_cap = assert_is(univ_cap)
                prices = assert_is(prices)

                bt = bt.with_pfo()

                if vol_norm is not None:
                    univ_norm, bt = bt.with_pfo_ntl(
                        cap=not_cap,
                        prices=prices,
                        t=Strategies,
                    )
                else:
                    univ_weight, bt = bt.with_pfo_ntl(
                        cap=not_cap,
                        prices=prices,
                        t=Strategies,
                    )

                cap_ratio = DailyPfolioNotionalCap.new(
                    assert_is(bt.ntl), key="gross"
                )
                context.get().with_obj(
                    cap_ratio, force=True
                )

                mvar_cap = Strategy(
                    asset=AssetSynthetic(
                        bt.rx().id,
                    ),
                    params=StrategyParams(
                        scale=ScaleFactor(
                            cap_ratio.id,
                            key=f"cap-n{not_cap}",
                        )
                    ),
                    name=f"{mvar_key}_cap",
                )
                univ_cap, bt_cap = mvar_cap.bind(
                    univ_cap, Strategies
                )
                res_cap[f"{mvar_key}_cap"] = bt_cap

    if not_cap is not None and vol_norm is not None:
        return (
            univ,
            univ_weight,
            univ_norm,
            univ_cap,
            res,
            res_norm,
            res_cap,
            # weight_objs
        )
    elif not_cap is not None:
        return (
            univ,
            univ_weight,
            univ_cap,
            res,
            res_cap,
            # weight_objs
        )
    elif vol_norm is not None:
        return (
            univ,
            univ_weight,
            univ_norm,
            res,
            res_norm,
            # weight_objs
        )
    return univ, univ_weight, res
    # , weight_objs


__all__ = [
    # "rank_baskets",
    "RankBasket",
    "WeightBasket",
    "build_baskets",
    "rank_baskets_dot",
    "comb_rank_basket",
    "comb_weight_basket",
]
