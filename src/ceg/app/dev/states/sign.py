from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)
from pathlib import Path
import argparse
import datetime as dt
from datetime import date
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...core import App, AppConfig, APPS
from ....engine import *

import ceg.apis.vis as vis

from ceg.engine import *

from ceg.objs.dev import strat

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
            | DailyBasket
            | DailyStrategy
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
            | DailyBasket.Snap
            | DailyStrategy.Snap
            | DailyBtPx.Snap
            | DailyBtGross.Snap

            | Strategies.Snap
        )

from ceg.objs.dev import strat
from functools import partial




ROOT = "states_sign"
# NOTE: given root storage, ahs to be a single word (no -)


def build(
    args: argparse.Namespace
):
    PROJECT = ROOT
    PROJECT_CV = "cv"
    PROJECTS = [PROJECT, PROJECT_CV]
    
    config = AppConfig(ProjectGraph, tuple(PROJECTS))

    acc = frozendict(dict(
        path=Path(args.path),
        i = 0,
    ))

    univ_fp = Path(__file__).parent / "models" / f'{str(args.universe).rjust(3, "0")}.csv'
    print(univ_fp)

    univ = pl.read_csv(univ_fp).with_columns(
        (pl.col(k) == "Y").fill_null(False)
        for k in [
            "rx_s",
            "rx_y",
            "rx_b",
            "eco_s",
        ]
    )

    symbols_rx_s = [
        f"{ac}-{symbol}"
        for ac, symbol in (
            univ.filter(pl.col("rx_s"))
            .select("ac", "symbol")
            .iter_rows()
        )
    ]
    symbols_rx_y = [
        f"{ac}-{symbol}"
        for ac, symbol in (
            univ.filter(pl.col("rx_y"))
            .select("ac", "symbol")
            .iter_rows()
        )
    ]
    symbols_rx_b = [
        f"{ac}-{symbol}"
        for ac, symbol in (
            univ.filter(pl.col("rx_b"))
            .select("ac", "symbol")
            .iter_rows()
        )
    ]

    symbols = list({
        **{s: None for s in symbols_rx_s},
        **{s: None for s in symbols_rx_y},
        **{s: None for s in symbols_rx_b},
    }.keys())
    

    config = config.with_configs(
        rx_containers=(
            select(name=symbols)
            .bind(strat.new_container, project=bound(PROJECT_CV))
            .under("strats-cv-{name}")
            .info("name")
        ),
        # NOTE rx used for state codes
        rx=(
            select(asset=symbols)
            .zip(container=where(CONFIG="rx_containers"))
            .bind(strat.cv, span=48, notional=10.)
            .into("container")
            .under("cv-{asset}")
            .info("asset")
        ),
        # NOTE: eco used for state codes
        # eco = (

        # ),
        # NOTE: rx to research given states
        # rx_y = (

        # ),
        # # NOTE: rx to take beta against, given states
        # rx_b = (

        # )
    )

    # TODO: twisties so easy to minimise particular graphs when on phone
    
    config = config.with_configs(
        rx_s=(
            select(
                objs=where(lambda _, d: d.get("asset") in symbols_rx_s).group()
            )
            .apply(unpack_daily_df, unpack="rx")
            .then(
                vis.write_data,
                name = "rx_s",
                pipe = "ceg.apis.pl.filter_dates_between",
                params = dict(
                    d0=date,
                    d1=date,
                )
            )
            .apply(unpack_daily_col_names, unpack="rx")
            .then(
                vis.write_transform,
                name="sign_code",
                pipe="ceg.apis.states.sign_code_product",
                data="data:rx_s",
                params=dict(
                    smoothed=bool,
                    span=float,
                    smooth=float,
                    d0="param:rx_s.d0",
                    d1="param:rx_s.d1",
                ),
                smoothed=False,
                span=16,
                smooth=4,
            )
            .apply(unpack_daily_col_names, unpack="rx")
            .then(
                vis.plot_transition_matrix,
                name="sign_code_t",
                data="data:sign_code"
            )
            .apply(unpack_daily_col_names, unpack="rx")
            .then(
                vis.plot_transition_corr,
                name="sign_code_corr",
                data="data:sign_code"
            )
            .apply(unpack_daily_col_names, unpack="rx")
            .then(
                vis.plot_transition_tree,
                name="sign_code_tree",
                data="data:sign_code"
            )
            # TODO: with conditional expectation
        ),
        rx_y = (
            select(
                objs=where(lambda _, d: d.get("asset") in symbols_rx_y).group()
            )
            .apply(unpack_daily_df, unpack="rx")
            .then(
                vis.write_data,
                name = "rx_y",
                pipe = "ceg.apis.pl.filter_dates_between",
                params = dict(
                    d0="param:rx_s.d0",
                    d1="param:rx_s.d1",
                )
            )
        ),
        rx_b = (
            select(
                objs=where(lambda _, d: d.get("asset") in symbols_rx_b).group()
            )
            .apply(unpack_daily_df, unpack="rx")
            .then(
                vis.write_data,
                name = "rx_b",
                pipe = "ceg.apis.pl.filter_dates_between",
                params = dict(
                    d0="param:rx_s.d0",
                    d1="param:rx_s.d1",
                )
            )
        ),
        rx_y_s_dist = (
            select(
                objs_s=where(lambda _, d: d.get("asset") in symbols_rx_s).group(),
                objs_y=where(lambda _, d: d.get("asset") in symbols_rx_y).group(),
                every=True,
            )
            .apply(
                unpack_daily_col_names,
                keys = ["objs_s", "objs_y"],
                unpack="rx"
            )
            .then(
                vis.plot_conditional_dist_w_fwd,
                name="rx_y_cond",
                # objs_s and objs_y from above (from s, derive state labels)
                data_s="data:sign_code",
                data_y="data:rx_y",
                # fwd = int, # default to none ie. 1 day
                # TODO: boolean toggle group by asset vs bwd fwd
            )

        ),
        rx_s_y_b = (
            select(
                objs_s=where(lambda _, d: d.get("asset") in symbols_rx_s).group(),
                objs_y=where(lambda _, d: d.get("asset") in symbols_rx_y).group(),
                objs_b=where(lambda _, d: d.get("asset") in symbols_rx_b).group(),
                every=True,
            )
            .apply(
                unpack_daily_col_names,
                keys = ["objs_s", "objs_y", "objs_b"],
                unpack="rx"
            )
            .then(
                vis.plot_conditional_beta_w_vol,
                name="rx_y_cond_beta",
                # objs_s and objs_y from above (from s, derive state labels)
                data_s="data:sign_code",
                data_y="data:rx_y",
                data_b="data:rx_b",
                
                # span = float,
            )
        ),
        rx_y_s_hist = (
            select(
                objs_s=where(lambda _, d: d.get("asset") in symbols_rx_s).group(),
                objs_y=where(lambda _, d: d.get("asset") in symbols_rx_y).group(),
                every=True,
            )
            .apply(
                unpack_daily_col_names,
                keys = ["objs_s", "objs_y"],
                unpack="rx"
            )
            .then(
                vis.plot_state_codes_w_y,
                name="state_codes_w_y",
                # objs_s and objs_y from above (from s, derive state labels)
                data_s="data:sign_code",
                data_y="data:rx_y",
                # TODO: discretisation cell length
            )
        )
    )

    # with the average rx, plot the median and iqr

    # for the betas, vol, could plot dist grouped by each state period
    # rx likewise? (if can plot two dist lines on one bar)

    # toggle for rx to be arith average, vs compound in state period, and decompound
    # ie. grouped by same signal

    return acc, config


APPS[ROOT] = (
    App(ROOT + "-p", build)
    .with_args(
        path=dict(type=str),
        universe=dict(type=int),
    )
)