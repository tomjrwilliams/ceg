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

from ceg.objs.dev import strat, calc

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


ROOT = "states_sign_strat"
# NOTE: given root storage, ahs to be a single word (no -)


def sign_code(
    assets: tuple[Object, ...],
    span: int,
    smooth: int | None = None,
):
    return calc.d.DailyHAggregate.sign_code(
        project,
        tuple(asset.id for asset in assets),
        kwargs = dict(
            span=span,
            smooth=smooth,
            smoothed=smooth is not None,
        )
    )

def transition_matrix(
    sign_code: Object,
    smooth: int,
    # lag?
):
    return 

def conditional_rx(
    sign_code: Object,
    rx: Object,
):
    # metric?
    return

def predicted_rx(
    sign_code: Object,
    # single col with all signs
    transition: Object,
    # presumably expanded to n_states * n_states
    rx_cond: tuple[Object, ...],
    # so this will actually produce n_states * n_rx
):
    
    # return n_rx
    return


def signal_bt(
    rx_pred: Object,
    rx: tuple[Object, ...],
    # binary vs not
):
    return


# TODO: state conditional corr adjusted vol


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

    # TODO: postfix universe to PROJECT?

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
        # TODO: signal container (per signal?)
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
        sign_code=(
            select(assets=where(CONFIG="rx").group())
            .product(
                select(every = dict(
                    span=[4, 16, 32, 64],
                    smooth=[None, 16, 32, 64]
                ))
            )
            .bind(sign_code, project = PROJECT)
            .under("sign_code_{span}_{smooth}")
        ),
        sign_code_trans = (
            select(sign_code=where(CONFIG="sign_code"))
            .bind(transition_matrix)
            # smooth / lag?
            .info("sign_code")
            .under("sign_code_prob_{sign_code.span}_{sign_code.smooth}")
        ),
        # transition corr? generalisation of matrix? 
        rx_cond = (
            select(asset=where(CONFIG="rx"))
            .product(code = where(CONFIG="sign_code"))
            .bind(conditional_rx)
            # NOTE: one sign code col has all signs
            # so this will actually produce n_signs * n_rx
            .under(
                "{asset.name}_state_cond_{sign_code.span}_{sign_code.smooth}"
            )
            # for metric in metrics?
            # for shift / drop first in ...?
        ),
        signal = (
            select(state = where(CONFIG="sign_code"))
            .zip(transition = where(CONFIG="sign_code_trans"))
            .zip(
                rx_cond=where(CONFIG="rx_cond").group("sign_code")
                # TODO: will this group properly?
            )
            .bind(predicted_rx)
        ),
        rx_bt = (
            select(signal=where(CONFIG="signal"))
            .product(rx=where(CONFIG="rx").group())
            # TODO: signal rx / bt container?
            .bind(signal_bt)
            # under?
            # TODO: one for l/s only, one with magnitude?
        )
    )
    
    # TODO: v2:
    # weights given signal l/s, determined by conditional corr matrix

