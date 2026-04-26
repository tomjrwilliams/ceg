from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)
import datetime as dt
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

import ceg

from ceg.engine import *
from ceg.objs.dev.rand import *
from ceg.objs.dev.bars import *
from ceg.objs.dev.eco import *
from ceg.objs.dev.calc.d import *

from ceg.objs.dev.pfo import *
from ceg.objs.dev.strat import *

@dataclass(frozen=True)
class ProjectGraph(Graph):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            Project
            | StorageConfig
            | DailyNoise
            #
            | FutDailyOrdinal
            | FutDailyRx
            | EqDailyEtf
            | EqDailyRx
            | FxDailyRx
            | FxDailyPx
            | EcoDailyObservation
            #
            | DailyHAggregate.Quantize
            | DailyHAggregate.Rank
            | DailyHAggregate.Sum
            | DailyHAggregate.TransitionProb
            | DailyTransitionProbSignal.Type

            | DailyEOsc
            | DailyERms
            | DailyEMA

            | DailyBtPos
            | DailyBtRx
            | DailyBtPerf
            | DailyBtMvar
            | DailyBtUvar
            | DailyBtPx
            | DailyBtGross

            # | Container
            | Strategies
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            DailyNoise.Snap
            #
            | FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | EqDailyEtf.Snap
            | EqDailyRx.Snap
            | FxDailyRx.Snap
            | FxDailyPx.Snap
            | EcoDailyObservation.Snap
            #
            | DailyHAggregate.Rank.Snap
            | DailyHAggregate.Quantize.Snap
            | DailyHAggregate.Sum.Snap
            | DailyHAggregate.TransitionProb.Snap
            | DailyTransitionProbSignal.Type.Snap

            | DailyEOsc.Snap
            | DailyERms.Snap
            | DailyEMA.Snap

            | DailyBtPos.Snap
            | DailyBtRx.Snap
            | DailyBtPerf.Snap
            | DailyBtMvar.Snap
            | DailyBtUvar.Snap
            | DailyBtPx.Snap
            | DailyBtGross.Snap

            | Strategies.Snap
        )


from ceg.objs.dev import strat
from functools import partial

def f_cv(
    container_id: str, *args, **kwargs
):
    s = strat.cv(*args, **kwargs)
    container = context.get().get(container_id, strat.Strategies)
    return s.bind(container, t=strat.Strategies)

def f_strategy(
    container_id: str, *args, **kwargs    
):
    s = Strategy(*args, **kwargs)
    container = context.get().get(container_id, strat.Strategies)
    container, s = s.bind(container)
    return container, s

def f_basket(
    container_id: str, *args, **kwargs
):
    basket = Basket(*args, **kwargs)
    container = context.get().get(container_id, strat.Strategies)
    container, basket = basket.bind(container, t=strat.Strategies)
    # TODO: wrap into some kind of shared object?
    return container, basket

def bind_basket(
    name: str,
    container: str,
    asset: tuple[Strategy],
    params: StrategyParams,
):
    return Bound(
        f_basket,
        name,
        container=bound(container),
        params=fdict(
            asset=asset,
            params=params,
            name=name,
        )
    ).unpack("asset", "params")
    
def bind_strategy(
    name: str,
    container: str,
    asset: TAsset,
    params: StrategyParams,
):
    return Bound(
        f_basket,
        name,
        container=bound(container),
        params=fdict(
            asset=asset,
            params=params,
            # name?
        )
    ).unpack("asset", "params")
    


def bind_cv(
    symbol: str,
    container: str,
    start: dt.date,
    end: dt.date,
    signal: TSignal | None = None,
):
    ac, sym = symbol.split("-", maxsplit=1)
    res = Bound(
        f_cv,
        f"cv-{symbol}",
        container=bound(container),
        params = fdict(
            asset=(ac, sym),
            span=48,
            cap=strat.CapNotional(10.0),
            signal=signal,
        )
    )
    if signal is not None:
        res = res.unpack("signal")
    return res

import json
from pathlib import Path

def next_config(app: Path, i: int | None = None):
    if not app.exists():
        app.mkdir(exist_ok=True, parents=True)
    if i is None:
        i = 0
        config_fp = app / f"config_{i}.json"
        while config_fp.exists():
            i += 1
            config_fp = app / f"config_{i}.json"
    config_fp = app / f"config_{i}.json"
    return i, config_fp

def write_config(
    config: dict,
    app: Path,
    i: int | None = None,
):
    i, config_fp = next_config(app, i=i)
    with config_fp.open("w+") as f:
        json.dump(config, f)

def render_px(
    df: pl.DataFrame,
    app: Path,
    t: str,
    i: int | None = None,
    **kwargs,
):
    i, config_fp = next_config(app, i)
    df_fp = app / f"df_{i}.parquet"
    df.write_parquet(df_fp)
    config = {
        "type": t,
        "fp": str(df_fp),
        **kwargs,
    }
    write_config(config, app, i = i)


render_line = partial(render_px, t = "line")
render_vline = partial(render_px, t = "vline")
render_scatter = partial(render_px, t = "scatter")
render_vscatter = partial(render_px, t = "vscatter")
render_area = partial(render_px, t = "area")
render_varea = partial(render_px, t = "varea")
render_bar = partial(render_px, t = "bar")
render_hbar = partial(render_px, t = "hbar")

def index_where(symbols, ranks, i):
    sym=[s for s, r in zip(symbols, ranks) if int(r) == i]
    if len(sym):
        assert len(sym) == 1, sym
        return sym[0]
    return None
    
from ceg.objs.dev.calc.d.h.trans_prob import *

# NOTE: spy down, tsy and gold up -> spy up, gold up
# ie. spy reversion, conditional on big risk off move
# gld keeps the momentum

# TODO: try integrating 5y5y (or some other breakeven?)
# and alterantively cpi (or equiv): ideally a daily measure for measurement sake?
# eg. with a flag for inflaton-market vs inflation-eco

# TODO as itself a state discriminator for the tlt vs gold relationship to spy

# TODO: plot 3d as well as 1d returns, seems a fair bit of reversion here

def get_state_names(symbols: list[str], method: str):
    if method == "rank":
        bwd_states = rank_permutations(len(symbols))
        return [
            f".".join([
                index_where(symbols, state, i)
                for i in range(len(symbols))
            ])
            for state in bwd_states
        ]
    elif method == "quantize":
        import itertools
        quantize_universe = list(itertools.product(*[
            [0, 1] for _ in symbols
        ]))
        labels = ["DN", "UP"]
        return [
            ".".join([
                f"{s}-{labels[q]}"
                for s, q in zip(symbols, qs)
            ])
            for qs in quantize_universe
        ]

def plot_conditional_mean(
    df: pl.DataFrame,
    symbols: list[str],
    method: str,
    i_chart: int,
    path: Path,
    prob_obj,
    rank_obj: str,
    rx_objs,
):
    e_probs = pl.col(prob_obj.id)

    bwd_states = rank_permutations(len(symbols))

    state_names = get_state_names(symbols, method)

    n_bwd = len(state_names)
    n_fwd = len(symbols)

    acc = []
    acc_fwd = []

    for i, state in enumerate(state_names):
        if method == "rank":
            df_state = df.select(
                # "date",
                get_state_indicator(
                    e_probs,
                    n_bwd=n_bwd,
                    n_fwd=n_fwd,
                    bwd=i,
                    fwd=None,
                ).alias(state),
                *[o.id for o in rx_objs],
            )
        elif method == "quantize":
            df_state = df.select(
                # "date",
                pl.col(rank_obj).eq(i).alias(state),
                *[o.id for o in rx_objs],
            )
        else:
            raise ValueError(method)
        
        r_mean = df_state.filter(
            pl.col(state).cast(pl.Boolean)
        ).drop(state).mean().with_columns(
            pl.lit(state).alias("state")
        ).unpivot(
            index="state"
        )
        r_mean_fwd = df_state.filter(
            pl.col(state).cast(pl.Boolean).shift(1)
        ).drop(state).mean().with_columns(
            pl.lit(state).alias("state")
        ).unpivot(
            index="state"
        )

        acc.append(r_mean)
        acc_fwd.append(r_mean_fwd)

    df_mean = pl.concat(acc)

    render_bar(
        df_mean,
        path,
        i=i_chart, group="variable", x="state", y="value", tall=True
    )
    i_chart += 1

    render_bar(
        pl.concat(acc_fwd),
        path,
        i=i_chart, group="variable", x="state", y="value", tall=True
    )
    i_chart += 1

    return i_chart

def plot_ranks(
    df: pl.DataFrame,
    symbols: list[str],
    window: str,
    agg: str,
    pred: str,
    agg_pred: str,
    path: Path,
    i_chart: int = 2,
):
    
    bwd_states = rank_permutations(len(symbols))
    state_symbols = [
        [
            index_where(symbols, state, i)
            for i in range(len(symbols))
        ]
        for state in bwd_states
    ]
    state_names = [
        ".".join(symbs) for symbs in state_symbols
    ]
    
    res: list[str] = []
    codes: set[str] = []
    ranks_bwd = [
        int(c) for c in df.get_column(f"rank-{window}-bwd").to_list()
    ]
    ranks_fwd = [
        int(c) for c in df.get_column(f"rank-{pred}-fwd").to_list()
    ]

    df = df.with_columns(
        *(
            pl.Series([
                state_symbols[code][i]
                for code in ranks_bwd
            ])
            .alias(f"{i}_{window}-bwd")
            for i in range(len(symbols))
        ),
        *(
            pl.Series([
                state_symbols[code][i]
                for code in ranks_bwd
            ])
            .alias(f"{i}_{pred}-fwd")
            for i in range(len(symbols))
        ),
        pl.Series([
            state_names[code] for code in ranks_bwd
        ])
        .alias(f"code_{window}-bwd"),
        pl.Series([
            state_names[code] for code in ranks_fwd
        ])
        .alias(f"code_{pred}-fwd"),
    )
    codes = list(
        df.get_column(f"code_{window}-bwd")
        .value_counts()
        .sort(by="count", descending=True)
        .get_column(f"code_{window}-bwd")
    )
    df = df.with_columns(*(
        (
            pl.col(f"code_{window}-bwd") == code
        )
        .cast(pl.Int32)
        .alias(f"{code}-{window}-bwd")
        for code in codes
    ), *(
        (
            pl.col(f"code_{window}-bwd")
            .shift(1)
            .eq(l)
            .and_(
                pl.col(f"code_{pred}-fwd")
                .eq(r)
            )
        )
        .cast(pl.Int32)
        .alias(f"{l}-{r}-{window}-{pred}")
        for l in codes
        for r in codes
    ),*(
        (
            pl.col(f"code_{window}-bwd")
            .shift(1)
            .eq(l)
            .and_(
                pl.col(f"0_{pred}-fwd")
                .eq(pl.lit(s))
            )
        )
        .cast(pl.Int32)
        .alias(f"{l}-{s}-{window}-{pred}")
        for l in codes
        for s in symbols
    ), *(
        (
            pl.col(f"code_{window}-bwd")
            .shift(1)
            .eq(l)
            .not_()
        )
        .cast(pl.Int32)
        .alias(f"{l}-off-{window}-{pred}")
        for l in codes
    ))
    
    res = []
    df = df.with_columns((
        (
            pl.col(f"{code}-{window}-bwd")
            .fill_null(0)
            .rolling_sum(
                window_days(agg),
                min_samples=1,
            )
            # .ewm_mean(
            #     alpha=2/ (window_days(w) + 1),
            #     min_samples=1,
            #     ignore_nulls=True,
            # )
        )
        .alias(f"{code}-{window}-{agg}")
        for code in codes
    ))
    for code in codes:
        col = f"{code}-{window}-{agg}"
        res.append(col)

    df_t = df.unpivot(
        index="date",
        on=res,
    ).drop("date", "variable")

    min, max = (
        df_t.quantile(0.1).row(0)[0],
        df_t.quantile(0.99).row(0)[0]
    )
    df = df.with_columns((
        pl.col(col).clip(min, max).alias(col)
        for col in res
    ))

    render_area(
        df.select("date", *res), 
        path,
        i=i_chart
    )
    i_expr = pl.int_range(0, len(df))

    e_window= pl.when(i_expr < window_days(agg_pred)).then(i_expr).otherwise(window_days(agg_pred))

    i_chart += 1
    for l in codes:
        probs = []
        for r in codes + ["off"]:
            col = f"{r}-{pred}-{agg_pred}"
            df = df.with_columns((
                (
                    pl.col(f"{l}-{r}-{window}-{pred}")
                    .fill_null(0)
                    .rolling_sum(
                        window_days(agg_pred),
                        min_samples=1,
                    )
                    .mul(1 / e_window)
                    # .ewm_mean(
                    #     alpha=2/ (window_days(w) + 1),
                    #     min_samples=1,
                    #     ignore_nulls=True,
                    # )
                )
                .alias(col)
            ))
            probs.append(col)

        render_area(
            df.pipe(clip_last_for_area, probs, window_days(agg_pred)),
            path,
            i=i_chart,
            title=l,
            # groupnorm="percent"
        )
        i_chart += 1

    for l in codes:
        probs = []
        for s in symbols + ["off"]:
            col = f"{s}-{pred}-{agg_pred}"
            df = df.with_columns((
                (
                    pl.col(f"{l}-{s}-{window}-{pred}")
                    .fill_null(0)
                    .rolling_sum(
                        window_days(agg_pred),
                        min_samples=1,
                    )
                    .mul(1 / e_window)
                    # .ewm_mean(
                    #     alpha=2/ (window_days(w) + 1),
                    #     min_samples=1,
                    #     ignore_nulls=True,
                    # )
                )
                .alias(col)
            ))
            probs.append(col)

        render_area(
            df.pipe(clip_last_for_area, probs, window_days(agg_pred)),
            path,
            i=i_chart,
            title=l,
            # groupnorm="percent"
        )
        i_chart += 1

def clip_last_for_area(
    df: pl.DataFrame,
    cols: list[str],
    window: int,
):
    v_max = 0
    for i, col in enumerate(cols[:-1]):
        if i == 0:
            df = df.with_columns(pl.col(col).alias("acc"))
        else:
            df = df.with_columns(
                pl.col(col)
                .add(pl.col("acc"))
                .alias("acc")
            )
    v_max = df.filter(
        pl.int_range(0, len(df)) > window
    ).get_column("acc").max()
    df = df.with_columns(pl.lit(0).alias("acc"))
    for i, col in enumerate(cols[:-1]):
        df = df.with_columns(
            pl.col(col)
            .clip(
                0, upper_bound=(v_max - pl.col("acc")).clip(0)
            )
            .alias(col)
        )
        if i == 0:
            df = df.with_columns(
                pl.col(col).alias("acc")
            )
        else:
            df = df.with_columns(
                pl.col(col)
                .add(pl.col("acc"))
                .alias("acc")
            )
    return df.select(
        "date",
        *cols[:-1],
        pl.col("acc")
        .add(pl.col(cols[-1]))
        .clip(upper_bound=(v_max - pl.col("acc")).clip(0))
        .alias(f"{col}-{round(v_max, 2)}")
    )

WEIGHTS = {
    "ETF-SPY.ETF-TLT.ETF.GLD": {

    }
}

def get_ranks(
    symbols: list[str],
    window: str,
    pred: str,
    agg_pred: str,
    start: dt.date,
    end: dt.date,
    window_method: str | None = None,
    pred_method: str | None = None,
    method: str = "rank"
):

    c = ceg.Collection("c")

    rx_containers: list[ceg.Bound] = [
        ceg.Bound(
            strat.Strategies.under_project,
            f"container-cv-{symbol}",
            params = bind_params(
                bound(PROJECT_CV),
                name=symbol,
            )
        )
        for symbol in symbols
    ]
    strat_container = ceg.Bound(
        strat.Strategies.under_project,
        "strats",
        params=bind_params(
            bound(PROJECT),
            name="strats"
        )
    )
    rx_binds = [
        bind_cv(
            symbol,
            f"container-cv-{symbol}",
            start,
            end
        )
        for symbol in symbols
    ]

    # TODO: rename code
    if method == "rank":
        rank_binds = [
            DailyHAggregate.bind_rank(
                f"rank-{win}-{suffix}",
                bound(PROJECT),
                params=tuple((f"{o.name}.rx()" for o in rx_binds)),
                func="rank",
                kwargs=fdict(
                    diff=False,
                    # norm="ewrms-48",
                    window=win,
                    reverse=reverse,
                    method=method,
                ),
                xid=f"rank-{win}-{suffix}",
            ).unpack("params")
            for win, reverse, suffix, method in [
                (window, False, "bwd", window_method),
                (pred, True, "fwd", pred_method),
            ]
        ]
    elif method == "quantize":
        rank_binds = [
            DailyHAggregate.bind_quantize(
                f"quantize-{win}-{suffix}",
                bound(PROJECT),
                params=tuple((f"{o.name}.rx()" for o in rx_binds)),
                func="quantize",
                kwargs=fdict(
                    diff=False,
                    # norm="ewrms-48",
                    window=win,
                    reverse=reverse,
                    # method=method,
                    boundaries=tuple([
                        (0.,) for o in rx_binds
                    ]),
                ),
                xid=f"quantize-{win}-{suffix}",
            ).unpack("params")
            for win, reverse, suffix, method in [
                (window, False, "bwd", window_method),
                (pred, True, "fwd", pred_method),
            ]
        ]
    else:
        raise ValueError(method)
    
    top_binds = [
        DailyHAggregate.bind_rank(
            f"top-{win}-{suffix}",
            bound(PROJECT),
            params=tuple((f"{o.name}.rx()" for o in rx_binds)),
            func="rank",
            kwargs=fdict(
                diff=False,
                # norm="ewrms-48",
                window=win,
                method=method,
                top=True,
            ),
            xid=f"{win}-{suffix}",
        ).unpack("params")
        for win, reverse, suffix, method in [
            (window, False, "bwd", window_method),
            (pred, True, "fwd", pred_method),
        ]
    ]

    bwd_states = rank_permutations(len(symbols))

    state_names = get_state_names(symbols, method)

    n_bwd = len(bwd_states)
    n_fwd = len(symbols)

    prob_bind = DailyHAggregate.bind_transition_prob(
        f"prob-{window}-{pred}",
        bound(PROJECT),
        params = tuple([
            bound(rb.name) for rb in [
                rank_binds[0],
                top_binds[-1],
            ]
        ]),
        func="transition_prob",
        kwargs = fdict(
            window_bwd=window,
            window_fwd=pred,
            n_bwd=n_bwd,
            n_fwd=n_fwd,
            # norm=f"winmean-{window_days(agg_pred)}",
            norm=f"ewmean-{window_days(agg_pred)}"
            # discretise=...
        ),
        xid=f"prob-{window}-{pred}",
    ).unpack("params")

    prob_signals = {
        i: [
            DailyTransitionProbSignal.bind(
                f"sig-{state_names[i]}-{i_s}",
                bound(prob_bind.name),
                n_bwd=n_bwd,
                n_fwd=n_fwd,
                bwd=i,
                fwd=i_s,
                xid=f"sig-{state_names[i]}-{i_s}"
            )
            for i_s, symbol in enumerate(symbols)
        ]
        for i in range(n_bwd)
    }
    symbol_sums = [
        DailyHAggregate.bind_sum(
            f"sig-tot-{i_s}",
            bound(PROJECT),
            params = tuple([
                prob_signals[i][i_s].name
                for i in range(n_bwd)
            ]),
            func="sum",

        ).unpack("params")
        for i_s, symbol in enumerate(symbols)
    ]
    basket_binds = [
        bind_basket(
            # TODO: name = actual state name?
            f"basket_{state_names[i]}",
            container = strat_container.name,
            # TODO: object prefix?
            asset = tuple([
                Strategy(
                    asset=AssetConcrete(*tuple(symbol.split("-"))),
                    params=StrategyParams(
                        signal=SignalFactor(
                            factors=bound(prob_sig.name),
                            key=f"{state_names[i]}-{symbol}"
                        ),
                        cap=strat.CapNotional(10.0),
                        scale=strat.ScaleVol(span=48),
                    )
                )
                for symbol, prob_sig in zip(
                    symbols,
                    prob_sigs,
                )
            ]),
            params = StrategyParams(
                # scale=strat.ScaleVol(span=756),
                # cap=strat.CapNotional(10.0),

                # TODO: how to do?
            )
        )
        for i, prob_sigs in prob_signals.items()
    ] + [
        bind_basket(
            # TODO: name = actual state name?
            f"basket_total",
            container = strat_container.name,
            # TODO: object prefix?
            asset = tuple([
                Strategy(
                    asset=AssetConcrete(*tuple(symbol.split("-"))),
                    params=StrategyParams(
                        signal=SignalFactor(
                            factors=bound(symbol_sig.name),
                            key=f"sum-{symbol}"
                        ),
                        cap=strat.CapNotional(10.0),
                        scale=strat.ScaleVol(span=48),
                    )
                )
                for i, (
                    symbol_sig,
                    symbol
                ) in enumerate(zip(symbol_sums, symbols))
            ]),
            params = StrategyParams(
                # scale=strat.ScaleVol(span=756),
                # cap=strat.CapNotional(10.0),

                # TODO: how to do?
            )
        )
    ]

    # TODO: start by flushing all keys for project?
    # but keep data, config, etc.

    with ceg.context.graph(ROOT, t=ProjectGraph):
        cv = Project.new(PROJECT_CV)
        project = Project.new(PROJECT)

        context.get().with_objs(cv, project)
        
        project.set_storage_defaults([
            "config",
            "eq",
            "fut",
            "fx",
            # --
            "cv",
            # --
            ROOT
        ])

        c = c.with_obj(project, PROJECT)
        c = c.with_obj(cv, "cv")

        for o in rx_containers:
            c = o.bind(c)

        c = strat_container.bind(c)

        for o in rx_binds:
            c = o.bind(c)

        for o in rank_binds:
            c = o.bind(c)

        for o in top_binds:
            c = o.bind(c)
        
        c = prob_bind.bind(c)

        for _, prob_sigs in prob_signals.items():
            for prob_sig in prob_sigs:
                c = prob_sig.bind(c)

        for o in symbol_sums:
            c = o.bind(c)

        for o in basket_binds:
            c = o.bind(c)

        # for each base state
        # create a basket by mapping asset to conditional prob, given that state
        # and none where that state is not active

        context.get().sync()

        print("Done sync")

        rx_objs = [
            assert_is(c.get(f"{o.name}.rx()"), Daily) for o in rx_binds
        ]
        basket_objs = {
            b.name: assert_is(c.get(f"{b.name}.rx()"), Daily)
            for b in basket_binds
        }
        prob_obj = assert_is(c.get(f"prob-{window}-{pred}"), Daily)

        # TODO: prob signals?

        rank_objs = {
            o.name: assert_is(c.get(o.name), Daily) for o in rank_binds
        }
        top_objs = {
            o.name: assert_is(c.get(o.name), Daily) for o in top_binds
        }

        df = daily_pl_df(
            [prob_obj] 
            + rx_objs
            + list(rank_objs.values())
            + list(basket_objs.values())
            + list(top_objs.values())
        ).rename({
            # prob_obj.id: prob_bind.name
        }).rename({
            o.id: k for k, o in basket_objs.items()
        }).rename({
            o.id: k for k, o in rank_objs.items()
        }).filter(
            pl.col("date").is_between(start, end)
        )

    return df, state_names, rx_objs, rank_objs, top_objs, prob_obj, basket_objs


ROOT = "rank_001"
PROJECT = f"{ROOT}-p"
PROJECT_CV = "cv"

# DEFAULT_D0 = dt.date(2000, 1, 1)
DEFAULT_D0 = dt.date(2005, 1, 1)

DEFAULT_D1 = dt.date(2026, 4, 3)

import os
import shutil
os.environ["FRD_DATA"] = "./__local__/data/frd"

if __name__ == "__main__":
    from redis import Redis

    conn = Redis("localhost", port =7777)

    for env in ["prod"]:
        for table in [
            "Events",
            "Objects",
            "Snaps",
            "Deps",
            "ValuesJSON",
            "ValuesDF",
            "ValuesSeries",
        ]:
            # ChunksPlSeriesF64/cv-ETF-GLD-comb-0-gross/1
            # etc.
            # keys = conn.keys(f"{env}/{table}/*")
            keys = conn.keys(f"{env}/{table}/{ROOT}-*")
            if len(keys):
                print("DELETE", keys)
                conn.delete(*keys)

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("window", type=str)
    parser.add_argument("agg", type=str)
    parser.add_argument("pred", type=str)
    parser.add_argument("agg_pred", type=str)
    parser.add_argument("symbols", type=str)
    parser.add_argument("page", type=str)
    parser.add_argument("method", type = str)
    parser.add_argument(
        "-s",
        type=str,
        default=DEFAULT_D0.isoformat().replace("-", "")
    )
    parser.add_argument(
        "-e",
        type=str,
        default=DEFAULT_D1.isoformat().replace("-", "")
    )

    args = parser.parse_args()

    method = args.method
    assert method in ["rank", "quantize"], args.method

    # TODO: parametrise what to clear

    path = Path(f"./__local__/st/rank/{args.page}_{args.method}")

    if path.exists():
        shutil.rmtree(path)

    symbols = args.symbols.split(".")

    window_method = pred_method = None

    window = args.window
    if "." in window:
        window, window_method = window.split(".")

    pred = args.pred
    if "." in pred:
        pred, pred_method = pred.split(".")

    (
        df,
        state_names,
        rx_objs, rank_objs, top_objs, prob_obj, basket_objs
    ) = get_ranks(
        symbols=symbols,
        window=window,
        pred=pred,
        agg_pred=args.agg_pred,
        window_method=window_method,
        pred_method=pred_method,
        start=dt.datetime.strptime(args.s, "%Y%m%d").date(),
        end=dt.datetime.strptime(args.e, "%Y%m%d").date(),
        method=method,
    )

    basket_keys = list(basket_objs.keys())

    render_line(
        df.select("date", *[
            rx_obj.id for rx_obj in rx_objs
        ], basket_keys[-1]), 
        path,
        i=0,
        cum=True,
    )

    # TODO: render when in each state?

    i_chart = 1

    i_chart = plot_conditional_mean(
        df,
        symbols,
        method,
        i_chart,
        path,
        prob_obj,
        list(rank_objs.keys())[0],
        rx_objs,
    )

    render_line(
        df.select("date", *basket_keys[:-1]),
        path,
        i=i_chart,
        cum=True,
    )
    i_chart += 1

    df = df.filter(
        pl.any_horizontal(*[
            pl.col(o.id).is_not_null() for o in rx_objs
        ])
    )

    # plot_ranks(
    #     df,
    #     symbols,
    #     window=window,
    #     pred=pred,
    #     agg=args.agg,
    #     agg_pred=args.agg_pred,
    #     path=path,
    #     i_chart=i_chart,
    # )