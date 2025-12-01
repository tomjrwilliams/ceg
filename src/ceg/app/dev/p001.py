from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...objs.vis import explorer
from ...objs.dev import bars
from ...objs.dev import pfo

from ...objs.dev.strat.mom import eosc as strat

from .core import PROJECTS

PROJECT = "p001-mom"

# TODO: put all the bars under bars module

# d -> d/root.py
# then asset classes
# then the calc (moved up a dir, drop calc parent container, eg. straight m, em etc.)

# calc stays as calc, with just base and handlers

# asset classes can then in time have eg. chain
# but for now will 

# have to move the root in with the asset class, but its only one file

@dataclass(frozen=True)
class Project_001(explorer.Project):
    def with_mom_eosc_page(
        self, symbol: str
    ):
        bts: list[strat.MomEOscFutBtObjects]

        vol_spans = [
            # 16,
            48,
            # 48 * 4
        ]
        vols = [
            # 16,
            48, 
            # 48 * 4
        ]
        fasts = [4, 8, 16, 32, 64, 128, 256, 512]
        ratios = [
            1.5, 
            # 3
        ]

        signals = [None] + [
            strat.EOscSignalParams(
                fast, int(fast * ratio), vol
            )
            for fast in fasts
            for vol in vols
            for ratio in ratios
        ]
        scales = [
            strat.EOscScaleVolParams(span=span)
            for span in vol_spans
        ]
        d, bts = strat.StratMomEOscFut.new(
            *[
                strat.MomEOscFutBtParams(
                    symbol,
                    scale=scale,
                    signal=signal,
                )
                for scale in scales
                for signal in signals
            ]
        )
        self = self.with_domains(**{
            symbol: lambda sync: explorer.AppDomain(
                [d], sync
            )
        }).with_page(
            symbol,
            type="daily",
            domain=symbol,
        )
        for bt in bts:
            self = self.with_object(
                bt.rx.path,
                graph=0,
                label=bt.params.label()
            )
        self = self.with_graph(
            0,
            "perf"
        )
        return cast(Project_001, self)

PROJECTS[PROJECT] = (
    Project_001()
    .with_config(explorer.ExplorerConfig(PROJECT))
    .with_mom_eosc_page("ES")
)