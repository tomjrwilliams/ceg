import os
from typing import cast, TYPE_CHECKING, ClassVar
import itertools
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...vis import explorer

from ... import apis
from ... import engine

from ...objs.dev import bars
from ...objs.dev import pfo
from ...objs.dev import agg

from ...objs.dev import strat

from .core import PROJECTS
from .p001 import *

logger, log = apis.logs.logger(
    __file__,
    streams=dict(formatter=True),
)

PROJECT = "p002"

SPAN_VOL_SCALE = [
    # 16,
    48,
    # 48 * 4
]
SPAN_VOL_SIG = [
    # 16,
    48, 
    # 48 * 4
]

SPAN_RATIO = [
    1.5, 
    # 3
]

def eosc(fast: int, ratio: float = 1.5, vol: int = 48):
    return strat.SignalOsc(
        fast, int(fast * ratio), vol
    )

@dataclass(frozen=True)
class Project_002(strat.StratExplorer):
    
    def product_symbol_signals(self):
        ES_TN = [
            ("ETF", "SPY", None),
            ("FUT", "ES", eosc(32)),
            ("FUT", "ES", eosc(128)),
            ("FUT", "TN", eosc(16)),
            ("FUT", "TN", eosc(32)),
        ]
        GC = [
            ("FUT", "GC", None),
            ("FUT", "GC", eosc(32)),
        ]
        return [
            ("es-tn", ES_TN),
            ("es-tn-gc", ES_TN + GC),
        ]

    @classmethod
    def load(
        cls,
        target: str | None = None,
        root: str = "./__local__/apps",
        test: bool = False,
    ):
        """
        >>> root = "./__local__/test"
        >>> proj = Project_002.load(roott, test=True).reset()
        >>> nav = proj.prepare_app()
        >>> nav[0]()
        >>> nav[1]()
        """

        WITH_UI = bool(os.environ.get("STREAMLIT"))
        # TODO: root, test etc. from env vars

        self = (
            Project_002(root=root)
            .with_config(explorer.ExplorerConfig(PROJECT))
        )
        project = engine.Project.new(PROJECT)

        i = 0
        for (
            name, product_symbol_signals
        ), (
            suff, rebal
        ) in itertools.product(
            self.product_symbol_signals(),
            self.rebal_variants()
        ):
            page = f"{PROJECT}-{name}{suff}"
            if target is not None and page != target:
                log("INFO")( skipping=page)
                continue
            log("INFO")( loading=page)

            with engine.context.graph(
                page, t = strat.StrategyGraph
            ):
                project.set_storage_defaults([
                    
                ])
                univ = strat.Strategies.new(
                    project,
                    f"{page}-strats",
                )
                res = []
                for params in [
                    strat.Strategy(
                        asset = strat.AssetConcrete(
                            product,
                            symbol,
                        ),
                        params = strat.StrategyParams(
                            scale=strat.ScaleVol(
                                span=48
                            ),
                            signal=signal,
                            rebal=rebal,
                        )
                    )
                    for product, symbol, signal
                    in product_symbol_signals
                ]:
                    univ, res = params.bind(
                        univ, strat.Strategies, into=res
                    )
                
                engine.context.get().with_obj(
                    univ, force=True
                )
                # TODO: rescale?

                self = self.with_page(
                    page,
                    type="daily",
                    graph=True,
                    t=Project_002
                ).with_perf_table(
                    res, t=Project_002
                ).with_rx_graph(
                    res, t=Project_002
                # ).with_signal_graph(
                #     res, t=Project_002
                ).with_graph(
                    engine.context.get(),
                    t=Project_002
                )
                if test:
                    return self
                i += 1
        return self

PROJECTS[PROJECT] = Project_002