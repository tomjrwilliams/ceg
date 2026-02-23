from typing import cast, TYPE_CHECKING, ClassVar
import itertools
from dataclasses import replace
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

import polars as pl

from ... import apis
from ...apis import assert_is

from ... import engine
from ...vis import explorer
from ...objs.dev import bars
from ...objs.dev import pfo
from ...objs.dev import agg
from ...objs.dev import calc

from ...objs.dev import strat

from .core import PROJECTS

PROJECT = "p202"


@dataclass(frozen=True)
class Project_202(strat.StratExplorer):
    PROJECT: ClassVar[str] = PROJECT

    @classmethod
    def load_(cls, kw: dict):
        return cls.load(**{
            k: v for k, v in kw.items()
            if k in [
                "root", "with_ui", "with_files", "test"
            ]
        })
    
    @classmethod
    def load(
        cls,
        root: str = "./__local__/apps",
        test: bool = False,
        with_ui: bool = True,
        with_files: bool = True,
    ):
        """
        >>> root = "./__local__/test"
        >>> proj = Project_101.load(root, test=True).reset()
        >>> nav = proj.prepare_app()
        >>> nav[0]()
        >>> nav[1]()
        """

        WITH_UI = bool(os.environ.get("STREAMLIT"))
        # TODO: root, test etc. from env vars

        self = (
            Project_202(root=root)
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
                project.set_storage_defaults()
                univ = strat.Strategies.new(
                    project,
                    f"{page}-strats",
                )

        UNIVERSE = dict(
            # ES = ("FUT", "ES"),
            # TN = ("FUT", "TN"),
            # GC = ("FUT", "GC"),

            SPY = ("ETF", "SPY"),

            XLF = ("ETF", "XLF"),
            XLE = ("ETF", "XLE"),
            XLY = ("ETF", "XLY"),
            XLP = ("ETF", "XLP"),
            XLI = ("ETF", "XLI"),
            XLB = ("ETF", "XLB"),
            XLU = ("ETF", "XLU"),
            XLV = ("ETF", "XLV"),
            XLK = ("ETF", "XLK"),

            GLD = ("ETF", "GLD"),
            CMD = ("ETF", "DBC"), # commod

            TLT = ("ETF", "TLT"),
            IEF = ("ETF", "IEF"), # pretty similar to tlt
            BND = ("ETF", "BND"), # total bond, similar to ief
            TMT = ("ETF", "SCHR"),
            TST = ("ETF", "SCHO"),
            GOV = ("ETF", "GOVT"),

            # JPST = ("ETF", "JPST"), # massive returns, odd?

            # USHY = ("ETF", "USHY"), v similar to HYG
            HYG = ("ETF", "HYG"),
            LQD = ("ETF", "LQD"),
            VCIT = ("ETF", "VCIT"), # int corp
            VCSH = ("ETF", "VCSH"), # short corp

            # both similar to hyg, but less 2015 dd, more 2018, smoother return series, less 25 dd
            # ah, just much closer to lqd

            # which sector etfs is lqd dd closest to?
            # hyg?
        )
        BASKETS = [
            list(UNIVERSE.keys())
        ]
        SIGNALS = {
            "": None,
            **{
                f"_{sp}_{int(m*10)}_{var}": strat.SignalOsc(
                    sp, 
                    int(sp*m),
                    vol=48,
                    constraint=var
                )
                for sp in [
                    # 8, 16, 32, 64, 128, 256
                ]
                for m in [
                    # 3.
                ]
                for var in [
                    # "bls",
                    # "blo", 
                    # "bso"
                ]
            }
        }

        baskets = {
            "_".join(symbols): {
                symbol + suff: strat.Strategy.cv(
                    UNIVERSE[symbol],
                    span=48,
                    signal=sig,
                    cap=strat.CapNotional(10.),
                )
                for symbol in symbols
                for suff, sig in SIGNALS.items()
            }
            for symbols in BASKETS
        }
        basket_symbols = {
            basket: list(strats.keys())
            for basket, strats in baskets.items()
        }
        prices: dict[str, dict[str, engine.Daily]] = {
            "_".join(symbols): {
                sym: assert_is(
                    bars.EqDailyEtf.new(
                        eng,
                        symbol=sym,
                        field="close",
                        adjust="price",
                        source="frd"
                    )
                    if ac == "ETF"
                    else bars.FutDailyOrdinal.new(
                        eng,
                        symbol=sym,
                        ident = 1,
                        field="close",
                        adjust="raw",
                        source="frd"
                    )
                    if ac == "FUT"
                    else None,
                    engine.Daily,
                )
                for (ac, sym), symbol in zip(
                    map(lambda s: UNIVERSE[s], symbols),
                    symbols,
                )
            }
            for symbols in BASKETS
        }

        bt_lo = {}
        rx_lo = {}
        rx_panels = {}
        rx_sigs = {}

        for basket, strats in baskets.items():
            bt_lo[basket] = []
            rx_lo[basket] = []
            # or index by params?

            for params in strats.values():
                univ, bt = domain.with_backtest(
                    univ, params,
                )
                rx_lo[basket].append(context.get().get(
                    bt.assume_res().rx,
                    engine.Daily
                ))
                bt_lo[basket].append(bt)
 rx_panel = strat.rx_panel(
                rx_lo[basket],
                name,
                f"{basket}-cv-48"
            )

            rx_sigs[basket] = {}
            for params_dict in SIG_PARAMS:
                params = tuple(params_dict.values())
                rx_panel, sig_agg = rx_panel.with_agg()
                rx_sigs[basket][params] = (
                    agg.DailyAggregate.new(
                        rx_panel.coll,
                        index=sig_agg,
                        metric="metrics_ewm",
                        params=params,
                    )
                )

            rx_panels[basket] = rx_panel

            context.get().with_obj(
                rx_panel.coll,
                strat.StratDomain,
                force=True
            ).with_objs(
                *rx_sigs[basket].values(),
                t=strat.StratDomain,
                force=True
            )

        self = self.with_page(
            self.page_name(0, name),
            type="daily",
            graph=True,
            t=Project_202
        )

        wm_objs = []
        for basket, param_sigs in rx_sigs.items():
            symbols = basket_symbols[basket]
            for params, sigs in param_sigs.items():
                params_key = "-".join(map(str, params))
                for k in [
                    "rx_cum",
                    "rx_cmu",
                    "rx_max",
                    "rx_hwm",
                    "rx_min",
                    "rx_lwm",
                ]:
                    for i, symb in enumerate(symbols):
                        group = f"{symb}-{params_key}-{k}"
                        val = agg.DailyAggregateDeriv.new(
                            sigs, index=i, key=k
                        )
                        context.get().with_obj(
                            val, strat.StratDomain
                        )
                        wm_objs.append((
                            val.path,
                            f"{symb}-{params_key}",
                            dict(group=group)
                        ))

        self = self.with_objects_from_template(
            "wms",
            "watermarks",
            wm_objs,
            t=Project_202,
            # params=f"x=key:mu,y=label,v={value}"
        )

        wt_objs = []
        for basket, param_sigs in rx_sigs.items():
            symbols = basket_symbols[basket]
            for params, sigs in param_sigs.items():
                params_key = "-".join(map(str, params))
                for k in [
                    "rx",
                    "hwm_len",
                    "lwm_len",
                    # "hwm_net",
                    # "hwm_grs",
                    # "lwm_net",
                    # "lwm_grs",
                    "osc_15",
                    "osc_30",
                ]:
                    for i, symb in enumerate(symbols):
                        group = f"{symb}-{params_key}-{k}"
                        val = agg.DailyAggregateDeriv.new(
                            sigs, index=i, key=k
                        )
                        context.get().with_obj(
                            val, strat.StratDomain
                        )
                        wt_objs.append((
                            val.path,
                            f"{symb}-{params_key}",
                            dict(group=group)
                        ))

        self = self.with_objects_from_template(
            "wts",
            "watertrails",
            wt_objs,
            t=Project_202,
            # params=f"x=key:mu,y=label,v={value}"
        )

        self = self.with_graph(
            t=Project_202,
            name=self.page_name(0, name)
        )
        
        return self
    
SIG_PARAM_KEYS = [
    "span", 
    # "width", "re_grid"
]
SIG_PARAMS = [
    {
        k: v for k, v in 
        zip(SIG_PARAM_KEYS, (
            float(2 ** pow_2), # span
            # grid_width,
            # re_grid,
        ))
    }
    for pow_2 in range(
        # 3, #3=8, 
        4,
        # 6, # 6=64
        9, # 9=512
    ) # 
    # for grid_width in [
    #     0.1, 0.25, 0.5, 1
    # ]
    # for re_grid in [0, 1]
]

PROJECTS[PROJECT] = Project_202.load_

__all__ = [
    "Project_202",
]