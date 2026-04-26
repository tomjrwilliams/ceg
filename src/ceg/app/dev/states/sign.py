from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)
import argparse
import datetime as dt
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...core import App, AppConfig, APPS
from ....engine import *

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

def plot_line(
    collection: Collection,
    acc: frozendict,
    objs: tuple[list[str], list[Daily]],
    cum: bool,
    rx: bool = False,
):
    names, objs = objs
    if not rx:
        df = daily_pl_df(objs)
    else:
        df = daily_pl_df([
            collection.get(f"{name}.rx()")
            for name in names
        ])
    # date filter?

    render_line(
        df, 
        acc["path"],
        i=acc["i"],
        cum=cum,
    )

    return acc.set("i", acc["i"]+1)

ROOT = "states_sign"
# NOTE: given root storage, ahs to be a single word (no -)

def strat_container_cv(symbol: str):
    return f"strats-cv-{symbol}"

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

    symbols = [
        "ETF-SPY",
        "ETF-TLT",
    ]

    config = config.with_configs(
        # -- CV containers
        cv_containers=(
            universe(name=symbols)
            .apply(strat.new_container, project=bound(PROJECT_CV))
            .alias("name", strat_container_cv)
        ),
        # -- CV rx
        cv_rx=(
            universe(dict(symbol=symbols))
            .zip(container=where(config="cv_containers"))
            .apply(strat.bind_cv)
            .alias("symbol", "cv-{symbol}")
            # TODO: span, notional cap?
        ),
    )
    
    config = config.with_configs(
        # -- CV rx line
        cv_rx_line=(
            # select(objs=where(FUNC=strat.bind_cv).group()) # equiv
            where(config="cv_rx")
            .group()
            .output(plot_line, cum=True, rx = True)
        )
    )

    return acc, config


APPS[ROOT] = (
    App(ROOT + "-p", build)
    .with_args(
        path=dict(type=str)
    )
)