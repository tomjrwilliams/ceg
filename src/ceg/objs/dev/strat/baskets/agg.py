
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

from .....vis import explorer

@overload
def rank_baskets_dot(
    project: Project,
    baskets: dict[str, list[Strategy]] | dict[
        str, dict[str, list[Strategy]]
        # NOTE if dict, assume key is weight key
    ],
    rank_params: list[dict],
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
    baskets: dict[str, list[Strategy]] | dict[
        str, dict[str, list[Strategy]]
        # NOTE if dict, assume key is weight key
    ],
    rank_params: list[dict],
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
    baskets: dict[str, list[Strategy]] | dict[
        str, dict[str, list[Strategy]]
        # NOTE if dict, assume key is weight key
    ],
    rank_params: list[dict],
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

        univ_prefix = project.name

        univ = Strategies.under_project( 
            project,
            f"{name}-rw-panel",
        )
        univ_rank = Strategies.under_project( 
            project,
            f"{name}-rw",
        )

        univ_norm = (
            None
            if vol_norm is None
            else Strategies.under_project( 
                project,
                f"{name}-rw-norm",
            )
        )

        # TODO: assert shared symbol?
        if isinstance(strats, list):
            inds = []
            for strat in strats:
                univ, i = univ.with_object(
                    strat.rx(),
                    t=Strategies
                )
                inds.append(i)

            univ, i_agg = univ.with_agg(
                tuple(inds),
                t=Strategies
            )
            context.get().with_obj(univ, force=True)
        else:
            inds_d = {}
            for weight_key, k_strats in strats.items():
                inds_d[weight_key] = []
                for strat in k_strats:
                    univ, i = univ.with_object(
                        strat.rx(),
                        t=Strategies
                    )
                    inds_d[weight_key].append(i)

        for w_params in rank_params:
            top = w_params["top"]

            weight_key = "_".join(map(
                param_str, w_params.values()
            ))
            mvar_key = (
                f"{univ_prefix}-{name}-{weight_key}"
            )

            if isinstance(strats, list):
                strat_inds = inds
                v_strats = strats
            else:
                strat_inds = inds_d[weight_key]
                univ, i_agg = univ.with_agg(
                    tuple(strat_inds),
                    t=Strategies
                )
                context.get().with_obj(univ, force=True)
                v_strats = strats[weight_key]

            assert len(v_strats), (
                weight_key,
                strats,
            )

            if top > len(v_strats):
                continue

            weights = DailyAggregate.new(
                univ,
                agg="weights_rank",
                aggs=i_agg,
                **w_params,
            )
            context.get().with_obj(weights, force=True)
            # res[(
            #     sym, t, w_params
            # )] = weights

            v_weights = {}
            for k in [
                # "metric",
                # "metric_smooth",
                "weight",
            ]:
                if k not in weight_objs:
                    weight_objs[k] = []
                vws = [
                    DailyAggregateDeriv.new(
                        weights,
                        index=i,
                        key=k
                    )
                    for i in range(len(strat_inds))
                ]
                v_weights[k] = vws
                for ii, v_w in enumerate(vws):
                    weight_objs[k].append((
                        v_w.path,
                        f"{mvar_key}_{ii}"
                    ))

                    # TODO: label with params

                context.get().with_objs(
                    *vws, force=True
                )

            dot_sig = SignalFactor(
                factors = tuple([
                    assert_is(s.params.signal_path, str)
                    for s in v_strats
                ]),
                key=f"{mvar_key}-dot",
                weights = (
                    tuple([
                        vw.path for vw
                        in v_weights["weight"]
                    ]),
                )
            )

            uvar = Strategy(
                asset=v_strats[0].asset,
                params=StrategyParams(
                    signal=dot_sig,
                    cap=cap,
                    scale=scale,
                    # rebal=... TODO
                )
            )

            univ_rank, uvar = uvar.bind(
                univ_rank, t = Strategies
            )
            context.get().with_obj(univ_rank, force=True)

            res[mvar_key] = uvar

            if vol_norm is not None:
                assert univ_norm is not None, (
                    univ_norm, vol_norm
                )
                mvar_norm = Strategy(
                    asset=AssetSynthetic(
                        uvar.rx().path,
                    ),
                    params=StrategyParams(
                        scale=ScaleVol(
                            span=vol_norm,
                            rx=uvar.rx().path,
                        ),
                        cap=cap,
                    ),
                    name=f"{mvar_key}_norm"
                )
                univ_norm, bt_norm = mvar_norm.bind(
                    univ_norm, Strategies, 
                )
                res_norm[f"{mvar_key}_norm"] = bt_norm

                context.get().with_obj(univ_norm, force=True)

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
    basket_indices: dict[str, list[int]],
    rank_params: list[dict],
    vol_norm: int,
    univ_norm: Strategies,
    ntl_cap: float,
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
    basket_indices: dict[str, list[int]],
    rank_params: list[dict],
    vol_norm: int,
    univ_norm: Strategies,
    ntl_cap: float | None = None,
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
    basket_indices: dict[str, list[int]],
    rank_params: list[dict],
    vol_norm: None = None,
    univ_norm: None = None,
    ntl_cap: float = 10.,
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
    basket_indices: dict[str, list[int]],
    rank_params: list[dict],
    vol_norm: None = None,
    univ_norm: None = None,
    ntl_cap: None = None,
    univ_cap: None = None,
    prices: None = None,
) -> tuple[
    Strategies,
    Strategies,
    dict[str, Strategy | Basket]
]: ...

def rank_baskets(
    univ: Strategies,
    univ_rank: Strategies,
    # indices in univ.objects
    basket_indices: dict[str, list[int]],
    rank_params: list[dict],
    vol_norm: int | None = None,
    univ_norm: Strategies | None = None,
    ntl_cap: float | None = None,
    univ_cap: Strategies | None = None,
    prices: dict[str, Daily] | None = None,
):
    weight_objs = {}

    res = {}
    res_norm = {}
    res_cap = {}

    for name, inds in basket_indices.items():
        assert len(inds), name

        univ, i_agg = univ.with_agg(
            tuple(inds),
            t=Strategies
        )

        for w_params in rank_params:
            top = w_params["top"]
            if top > len(inds):
                continue

            weight_key = "_".join(map(
                param_str,
                w_params.values()
            ))
            mvar_key = (
                f"{name}-{weight_key}"
            )

            weights = DailyAggregate.new(
                univ,
                agg="weights_rank",
                aggs=i_agg,
                **w_params,
            )
            context.get().with_obj(weights, force=True)
            # res[(
            #     sym, t, w_params
            # )] = weights

            v_weights = {}
            for k in [
                # "metric",
                # "metric_smooth",
                "weight",
            ]:
                if k not in weight_objs:
                    weight_objs[k] = []
                vws = [
                    DailyAggregateDeriv.new(
                        weights,
                        index=i,
                        key=k
                    )
                    for i in range(len(inds))
                ]
                v_weights[k] = vws
                for ii, v_w in enumerate(vws):
                    weight_objs[k].append((
                        v_w.path,
                        f"{mvar_key}_{ii}"
                    ))

                    # TODO: label with params

                context.get().with_objs(
                    *vws, force=True
                )

            uvars = [
                # replace(
                #     i_to_params[i],
                #     weight=WeightFactor(
                #         vw.path,
                #         f"{weight_key}_{ii}"
                #     )
                # )
                Strategy(
                    asset=AssetSynthetic(
                        obj_path(univ.objects[i])
                    ),
                    params=StrategyParams(
                        signal=SignalFactor(
                            vw.path,
                            f"{weight_key}_{ii}"
                        )
                    )
                )
                for i, (
                    ii, vw, 
                ) in zip(
                    inds,
                    enumerate(
                        v_weights["weight"]
                    )
                )
            ]
            mvar = Basket(
                tuple(uvars),
                params=StrategyParams(),
                name = mvar_key
            )
            univ_rank, bt = mvar.bind(
                univ_rank,
                Strategies,
            )

            res[mvar_key] = bt

            if vol_norm is not None:
                assert univ_norm is not None, (
                    univ_norm, vol_norm
                )
                mvar_norm = Strategy(
                    asset=AssetSynthetic(
                        bt.rx().path,
                    ),
                    params=StrategyParams(
                        scale=ScaleVol(
                            span=vol_norm,
                            rx=bt.rx().path,
                        ),
                    ),
                    name=f"{mvar_key}_norm"
                )
                univ_norm, bt_norm = mvar_norm.bind(
                    univ_norm, Strategies, 
                )
                res_norm[f"{mvar_key}_norm"] = bt_norm

                bt = bt_norm

            if ntl_cap is not None:
                univ_cap = assert_is(univ_cap)
                prices = assert_is(prices)

                bt = bt.with_pfo()

                if vol_norm is not None:
                    univ_norm, bt = bt.with_pfo_ntl(
                        cap=ntl_cap,
                        prices=prices,
                        t=Strategies,
                    )
                else:
                    univ_rank, bt = bt.with_pfo_ntl(
                        cap=ntl_cap,
                        prices=prices,
                        t=Strategies
                    )
                
                cap_ratio = DailyPfolioNotionalCap.new(
                    assert_is(bt.ntl), key="gross"
                )
                context.get().with_obj(
                    cap_ratio, force=True
                )

                mvar_cap = Strategy(
                    asset=AssetSynthetic(
                        bt.rx().path,
                    ),
                    params = StrategyParams(
                        scale=ScaleFactor(
                            cap_ratio.path,
                            key=f"cap-n{ntl_cap}",
                        )
                    ),
                    name=f"{mvar_key}_cap"
                )
                univ_cap, bt_cap = mvar_cap.bind(
                    univ_cap, Strategies
                )
                res_cap[f"{mvar_key}_cap"] = bt_cap

    if ntl_cap is not None and vol_norm is not None:
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
    elif ntl_cap is not None:
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
    basket_indices: dict[str, list[int]],
    weight_params: list[dict],
    vol_norm: int,
    univ_norm: Strategies,
    ntl_cap: float,
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
    basket_indices: dict[str, list[int]],
    weight_params: list[dict],
    vol_norm: int,
    univ_norm: Strategies,
    ntl_cap: float | None = None,
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
    basket_indices: dict[str, list[int]],
    weight_params: list[dict],
    vol_norm: None = None,
    univ_norm: None = None,
    ntl_cap: float = 10.,
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
    basket_indices: dict[str, list[int]],
    weight_params: list[dict],
    vol_norm: None = None,
    univ_norm: None = None,
    ntl_cap: None = None,
    univ_cap: None = None,
    prices: None = None,
) -> tuple[
    Strategies,
    Strategies,
    dict[str, Strategy | Basket]
]: ...

def weight_baskets(
    univ: Strategies,
    univ_weight: Strategies,
    # indices in univ.objects
    basket_indices: dict[str, list[int]],
    weight_params: list[dict],
    vol_norm: int | None = None,
    univ_norm: Strategies | None = None,
    ntl_cap: float | None = None,
    univ_cap: Strategies | None = None,
    prices: dict[str, Daily] | None = None,
):
    weight_objs = {}

    res = {}
    res_norm = {}
    res_cap = {}

    for name, inds in basket_indices.items():
        assert len(inds), name
            
        for w_params in weight_params:
            # top = w_params["top"]
            # if top > len(inds):
            #     continue

            weight_key = "_".join(map(
                param_str, w_params.values()
            ))
            mvar_key = (
                f"{name}-{weight_key}"
            )
            uvars = [
                Strategy(
                    asset=AssetSynthetic(
                        obj_path(univ.objects[i])
                    ),
                    params=ps
                )
                for i, ps in zip(
                    inds,
                    WeightBasket.build_params(
                        [
                            context.get().get(
                                univ.objects[i], Daily
                            )
                            for i in inds
                        ],
                        w_params,
                    )
                )
            ]
            if len(uvars) == 1:
                mvar = uvars[0]
            else:
                mvar = Basket(
                    tuple(uvars),
                    params=StrategyParams(),
                    name = mvar_key
                )
            univ_weight, bt = mvar.bind(
                univ_weight,
                Strategies,
            )
            res[mvar_key] = bt

            # TODO: find a way to only apply the variant (long only etc.) after the norm, cap, etc.?

            if vol_norm is not None:
                assert univ_norm is not None, (
                    univ_norm, vol_norm
                )
                mvar_norm = Strategy(
                    asset=AssetSynthetic(
                        bt.rx().path,
                    ),
                    params=StrategyParams(
                        scale=ScaleVol(
                            span=vol_norm,
                            rx=bt.rx().path,
                        ),
                    ),
                    name=f"{mvar_key}_norm"
                )
                univ_norm, bt_norm = mvar_norm.bind(
                    univ_norm, Strategies, 
                )
                res_norm[f"{mvar_key}_norm"] = bt_norm

                bt = bt_norm

            if ntl_cap is not None:
                univ_cap = assert_is(univ_cap)
                prices = assert_is(prices)

                bt = bt.with_pfo()
                
                if vol_norm is not None:
                    univ_norm, bt = bt.with_pfo_ntl(
                        cap=ntl_cap,
                        prices=prices,
                        t=Strategies,
                    )
                else:
                    univ_weight, bt = bt.with_pfo_ntl(
                        cap=ntl_cap,
                        prices=prices,
                        t=Strategies
                    )
                
                cap_ratio = DailyPfolioNotionalCap.new(
                    assert_is(bt.ntl), key="gross"
                )
                context.get().with_obj(
                    cap_ratio, force=True
                )

                mvar_cap = Strategy(
                    asset=AssetSynthetic(
                        bt.rx().path,
                    ),
                    params = StrategyParams(
                        scale=ScaleFactor(
                            cap_ratio.path,
                            key=f"cap-n{ntl_cap}",
                        )
                    ),
                    name=f"{mvar_key}_cap"
                )
                univ_cap, bt_cap = mvar_cap.bind(
                    univ_cap, Strategies
                )
                res_cap[f"{mvar_key}_cap"] = bt_cap

            

    if ntl_cap is not None and vol_norm is not None:
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
    elif ntl_cap is not None:
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

@dataclass(frozen=True)
class BasketSpec:
    symbols: tuple[str, ...]
    vol_norm: int | None = None
    not_cap: float | None = None
    params: tuple[t.frozen.dict, ...] | None = None
    
    @classmethod
    def new(
        cls,
        s: str | Iterable[str],
        *symbols: str,
        vol_norm: int | None = None,
        not_cap: float | None = None,
        params: tuple[dict, ...] | None = None,
    ):
        symbols = (
            (s,) if isinstance(s, str) else tuple(s)
        ) + symbols
        return cls(
            symbols,
            vol_norm=vol_norm,
            not_cap=not_cap,
            params=(
                params if params is None
                else tuple(map(frozendict, params))
            ),
        )
    
    def update(self, **kwargs):
        return replace(self, **kwargs)

    def arg_spec(self) -> tuple[
        int | None,
        float | None,
        tuple[frozendict, ...] | None,
    ]:
        return (
            self.vol_norm,
            self.not_cap,
            self.params
        )


@dataclass(frozen=True)
class WeightBasket(BasketSpec):
    
    @classmethod
    def build_params(
        cls,
        rxs: list[Daily],
        params: dict
    ):
        sig = params.get("signal", None)
        scale = params.get("scale", None)
        cap = params.get("cap", None)

        # TODO: rebal
        res = []
        for i, rx in enumerate(rxs):
            # TODO: some kind of callable that can use the rx?
            res.append(StrategyParams(
                scale=unpack_basket_param(
                    scale, i, 
                    TScale # type: ignore
                ),
                signal=unpack_basket_param(
                    sig, i, 
                    TSignal # type: ignore
                ),
                cap=unpack_basket_param(
                    cap, i, 
                    TCap # type: ignore
                ),
            ))
        return res

T = TypeVar("T")

def unpack_signal_param(
    param: (
        TSignal
        | TScale
        | TCap
        | list[TSignal]
        | list[TScale]
        | list[TCap]
        | None
    ),
    i: int,
):
    return unpack_basket_param(param, i, TSignal) # type: ignore
def unpack_scale_param(
    param: (
        TSignal
        | TScale
        | TCap
        | list[TSignal]
        | list[TScale]
        | list[TCap]
        | None
    ),
    i: int,
):
    return unpack_basket_param(param, i, TScale) # type: ignore
def unpack_cap_param(
    param: (
        TSignal
        | TScale
        | TCap
        | list[TSignal]
        | list[TScale]
        | list[TCap]
        | None
    ),
    i: int,
):
    return unpack_basket_param(param, i, TCap) # type: ignore

def unpack_basket_param(
    param: (
        TSignal
        | TScale
        | TCap
        | list[TSignal]
        | list[TScale]
        | list[TCap]
        | None
    ),
    i: int,
    t: Type[T]
) -> T | None:
    if param is None:
        return None
    elif isinstance(param, list):
        p = param[i]
        if p is None:
            return None
        return assert_is(p, t)
    else:
        return assert_is(param, t)
        

@dataclass(frozen=True)
class RankBasket(BasketSpec):
    pass

agg_rank_basket = RankBasket.new
agg_weight_basket = WeightBasket.new
    
def build_baskets(
    baskets: dict[str, RankBasket | WeightBasket],
    project: Project,
    univ_prefix: str,
    cv: dict[str, Strategy | Basket],
    prices: dict[str, Daily],
    rank_params_default: list[dict],
    weight_params_default: list[dict],
):
    
    required = set().union(*[
        b.symbols for b in baskets.values()
    ])
    strats: dict[
        str, Strategy | Basket
    ] = {
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

        deriv_indices: dict[
            str, list[int]
        ] = {}
        rank_arg_specs: dict[
            tuple[
                int | None,
                float | None,
                tuple[frozendict, ...] | None,
            ],
            list[str],
        ] = {}
        weight_arg_specs: dict[
            tuple[
                int | None,
                float | None,
                tuple[frozendict, ...] | None,
            ],
            list[str],
        ] = {}

        for b_key, b in baskets.items():
            if b_key in done:
                continue
            if any([
                symbol not in strats
                for symbol in b.symbols
            ]):
                continue
            indices: list[int] = []
            for symbol in b.symbols:
                univ, i = univ.with_object(
                    strats[symbol]
                    .rx(),
                    t=Strategies
                )
                indices.append(i)
            
            deriv_indices[b_key] = indices
            
            if isinstance(b, RankBasket):
                arg_spec = b.arg_spec()
                if arg_spec not in rank_arg_specs:
                    rank_arg_specs[arg_spec] = []
                rank_arg_specs[arg_spec].append(b_key)
            elif isinstance(b, WeightBasket):
                arg_spec = b.arg_spec()
                if arg_spec not in weight_arg_specs:
                    weight_arg_specs[arg_spec] = []
                weight_arg_specs[arg_spec].append(b_key)
            else:
                raise ValueError(b)
            
        for (
            vol_norm,
            ntl_cap,
            rank_params
        ), b_keys in rank_arg_specs.items():
            #     (
            #     basket_name,
            #     basket_var,
            # )
            if vol_norm is not None:
                if univ_r_norm is None:
                    univ_r_norm = Strategies.under_project( 
                        project,
                        f"{univ_prefix}-rw-norm",
                        depth=depth,
                    )
            if ntl_cap is not None:
                if univ_r_cap is None:
                    univ_r_cap = Strategies.under_project( 
                        project,
                        f"{univ_prefix}-rw-cap",
                        depth=depth,
                    )
            
            if rank_params is None:
                rank_params = rank_params_default

            # no prefix, entirely up to you in how you define basket names (to include project, proejct and page, nethier etc.)

            b_inds = {
                b_key: inds
                for b_key, inds
                in deriv_indices.items()
                if b_key in b_keys
            }

            if (
                vol_norm is None
                and ntl_cap is None
            ):
                (
                    univ,
                    univ_rank,
                    b_strats
                ) = rank_baskets(
                    univ,
                    univ_rank,
                    b_inds,
                    list(map(dict, rank_params)),
                    vol_norm=vol_norm,
                    ntl_cap=ntl_cap,
                )
                strats = {
                    **strats,
                    **b_strats,
                }
            elif (
                vol_norm is not None
                and ntl_cap is None
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
                    b_inds,
                    list(map(dict, rank_params)),
                    vol_norm=vol_norm,
                    ntl_cap=ntl_cap,
                    univ_norm=assert_is(
                        univ_r_norm
                    ),
                )
                strats = {
                    **strats,
                    **b_strats,
                    **b_strats_norm,
                }
            elif (
                vol_norm is None
                and ntl_cap is not None
            ):
                (
                    univ,
                    univ_rank,
                    univ_r_cap,
                    b_strats,
                    b_strats_cap
                ) = rank_baskets(
                    univ,
                    univ_rank,
                    b_inds,
                    list(map(dict, rank_params)),
                    vol_norm=vol_norm,
                    ntl_cap=ntl_cap,
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
                and ntl_cap is not None
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
                    b_inds,
                    list(map(dict, rank_params)),
                    vol_norm=vol_norm,
                    ntl_cap=ntl_cap,
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
                raise ValueError(rank_arg_specs)

            done = {
                **done,
                **{b_key: True for b_key in b_keys}
            }
        for (
            vol_norm,
            ntl_cap,
            weight_params
        ), b_keys in weight_arg_specs.items():
            #     (
            #     basket_name,
            #     basket_var,
            # )
            if vol_norm is not None:
                if univ_w_norm is None:
                    univ_w_norm = Strategies.under_project( 
                        project,
                        f"{univ_prefix}-w-norm",
                        depth=depth,
                    )
            if ntl_cap is not None:
                if univ_w_cap is None:
                    univ_w_cap = Strategies.under_project( 
                        project,
                        f"{univ_prefix}-w-cap",
                        depth=depth,
                    )
            
            if weight_params is None:
                weight_params = weight_params_default

            b_inds = {
                b_key: inds
                for b_key, inds
                in deriv_indices.items()
                if b_key in b_keys
            }

            if (
                vol_norm is None
                and ntl_cap is None
            ):
                (
                    univ,
                    univ_weight,
                    b_strats
                ) = weight_baskets(
                    univ,
                    univ_weight,
                    b_inds,
                    list(map(dict, weight_params)),
                    vol_norm=vol_norm,
                    ntl_cap=ntl_cap,
                )
                strats = {
                    **strats,
                    **b_strats,
                }
            elif (
                vol_norm is not None
                and ntl_cap is None
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
                    b_inds,
                    list(map(dict, weight_params)),
                    vol_norm=vol_norm,
                    ntl_cap=ntl_cap,
                    univ_norm=assert_is(
                        univ_w_norm
                    ),
                )
                strats = {
                    **strats,
                    **b_strats,
                    **b_strats_norm,
                }
            elif (
                vol_norm is None
                and ntl_cap is not None
            ):
                (
                    univ,
                    univ_weight,
                    univ_w_cap,
                    b_strats,
                    b_strats_cap
                ) = weight_baskets(
                    univ,
                    univ_weight,
                    b_inds,
                    list(map(dict, weight_params)),
                    vol_norm=vol_norm,
                    ntl_cap=ntl_cap,
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
                and ntl_cap is not None
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
                    b_inds,
                    list(map(dict, weight_params)),
                    vol_norm=vol_norm,
                    ntl_cap=ntl_cap,
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
                raise ValueError(weight_arg_specs)

            done = {
                **done,
                **{b_key: True for b_key in b_keys}
            }

        if n_curr == len(done):
            raise ValueError(dict(
                universe=list(strats.keys()),
                baskets_done=list(done.keys()),
                baskets_todo=[
                    k for k in baskets
                    if k not in done
                ],
                strats_found=[
                    k for k in required
                    if k in strats
                ],
                strats_missing = [
                    k for k in required
                    if not k in strats
                ]
            ))

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

def param_str(o):
    try:
        return o.label()
    except:
        return str(o)

__all__ = [
    # "rank_baskets",
    "RankBasket",
    "WeightBasket",
    "build_baskets",
    "rank_baskets_dot",
    "agg_rank_basket",
    "agg_weight_basket",
]