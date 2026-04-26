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

PROJECT = "p201"

    # def load()...
        # for params_eq in strats_eq.values():
            # dd_objs2.append((
            #     bt.assume_res().rx,
            #     bt.params.label(),
            #     dict(group="rx")
            # ))
            # for span in [
            #     512
            # ]:
            #     dd_reg = calc.DailyDDRegime.new(
            #         context.get().get(
            #             bt.assume_res().rx, engine.Daily
            #         ),
            #         threshold=1.,
            #         span=span,
            #         continuation=0.,
            #         window=128 + 64,
            #     )
            #     context.get().with_obj(
            #         dd_reg, strat.StratDomain
            #     )
            #     dd_objs2.append((
            #         dd_reg.id,
            #         bt.params.label(),
            #         dict(
            #             group="regime",
            #             params=f"span={span}"
            #         )
            #     ))

@dataclass(frozen=True)
class Project_201(strat.StratExplorer):
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
        domain: strat.StratDomain

        self = (
            Project_201(root=root)
            .with_config(explorer.ExplorerConfig(PROJECT))
        )

        name = self.univ_name("es-tn")

        universes: dict[str, pfo.ContainerDaily] = {}
         univ, universes = (
            strat.StratDomain.new()
            .with_univ(
                name,
                depth=0,
                universes=universes,
            )
        )
        eng = univ.get_parent()

        ES = ("FUT", "ES")
        TN = ("FUT", "TN")
        GC = ("FUT", "GC")
        SPY = ("ETF", "SPY")
        XLF = ("ETF", "XLF")
        XLE = ("ETF", "XLE")
        XLY = ("ETF", "XLY")
        XLP = ("ETF", "XLP")
        XLI = ("ETF", "XLI")
        XLB = ("ETF", "XLB")
        XLU = ("ETF", "XLU")
        XLV = ("ETF", "XLV")
        XLK = ("ETF", "XLK")
        TLT = ("ETF", "TLT")
        TMT = ("ETF", "SCHR")
        TST = ("ETF", "SCHO")

        strats_fi = {
            # TLT: strat.Strategy.cv(
            #     TLT, span=48, cap=strat.CapNotional(20.),
            # ),
            # TMT: strat.Strategy.cv(
            #     TMT, span=48, cap=strat.CapNotional(20.),
            # ),
            # TST: strat.Strategy.cv(
            #     TST, span=48, cap=strat.CapNotional(20.),
            # ), # best
        }
        strats_eq = {
            # SPY: strat.Strategy.cv(
            #     SPY, span=48, cap=strat.CapNotional(10.),
            # ),
            # XLF: strat.Strategy.cv(
            #     XLF, span=48, cap=strat.CapNotional(10.),
            # ),
            # XLE: strat.Strategy.cv(
            #     XLE, span=48, cap=strat.CapNotional(10.),
            # ),
            # XLY: strat.Strategy.cv(
            #     XLY, span=48, cap=strat.CapNotional(10.),
            # ),
            XLP: strat.Strategy.cv(
                XLP, span=48, cap=strat.CapNotional(10.),
            ),
            # XLI: strat.Strategy.cv(
            #     XLI, span=48, cap=strat.CapNotional(10.),
            # ),
            # XLB: strat.Strategy.cv(
            #     XLB, span=48, cap=strat.CapNotional(10.),
            # ),
            XLU: strat.Strategy.cv(
                XLU, span=48, cap=strat.CapNotional(10.),
            ),
            # XLV: strat.Strategy.cv(
            #     XLV, span=48, cap=strat.CapNotional(10.),
            # ),
            XLK: strat.Strategy.cv(
                XLK, span=48, cap=strat.CapNotional(10.),
            ),
        }
        PRICES: dict[str, engine.Daily] = {
            t[1]: bars.EqDailyEtf.new(
                symbol=t[1],
                field="close",
                adjust="price",
                source="frd"
            )
            for t in strats_eq.keys()
        }

        strat_keys = (
            list(strats_fi.keys()) 
            + list(strats_eq.keys())
        )

        bt_eq = []
        bt_fi = []
        
        dd_objs = []
        # dd_objs2 = []

        rx_eq: list[engine.Daily] = []
        rx_fi: list[engine.Daily] = []

        for params_eq in strats_eq.values():
        univ, bt_eq = domain.with_backtest(
                univ, params_eq, into=bt_eq
            )
            bt = bt_eq[-1]
            rx_eq.append(context.get().get(
                bt.assume_res().rx,
                engine.Daily
            )) rx_eq_panel = strat.rx_panel(
            rx_eq,
            eng,
            name,
            "eq-cv-48"
        )
        for params_fi in strats_fi.values():
        univ, bt_fi = domain.with_backtest(
                univ, params_fi, into=bt_fi
            )
            bt = bt_fi[-1]
            rx_fi.append(context.get().get(
                bt.assume_res().rx,
                engine.Daily
            )) rx_fi_panel = strat.rx_panel(
            rx_fi,
            eng,
            name,
            "fi-cv-48"
        )

        regimes_eq = {}
        regime_labels_eq = {}
        regime_allocs_eq = {}
        regimes_fi = {}
        regime_labels_fi = {}
        regime_allocs_fi = {}

        for reg_params in REG_PARAMS:
            rx_eq_panel, agg_eq = rx_eq_panel.with_agg()
            rx_fi_panel, agg_fi = rx_fi_panel.with_agg()
            regime_eq = agg.DailyAggregate.new(
                rx_eq_panel.coll,
                index=agg_eq,
                metric="dd_regime",
                params = reg_params
            )
            labels_eq = [
                agg.DailyAggregateDeriv.new(
                    regime_eq, index=i, key="label"
                ) for i in range(len(rx_eq))
            ]
            allocs_eq = [
                agg.DailyAggregateDeriv.new(
                    regime_eq, index=i, key="alloc"
                ) for i in range(len(rx_eq))
            ]
            regimes_eq[reg_params] = regime_eq
            regime_labels_eq[reg_params] = labels_eq
            regime_allocs_eq[reg_params] = allocs_eq

            regime_fi = agg.DailyAggregate.new(
                rx_fi_panel.coll,
                index=agg_fi,
                metric="dd_regime",
                params = reg_params
            )
            labels_fi = [
                agg.DailyAggregateDeriv.new(
                    regime_fi, index=i, key="label"
                ) for i in range(len(rx_fi))
            ]
            allocs_fi = [
                agg.DailyAggregateDeriv.new(
                    regime_fi, index=i, key="alloc"
                ) for i in range(len(rx_fi))
            ]
            regimes_fi[reg_params] = regime_fi
            regime_labels_fi[reg_params] = labels_fi
            regime_allocs_fi[reg_params] = allocs_fi

        domain = (
            domain.with_obj(
                rx_eq_panel.coll,
                strat.StratDomain,
                force=True
            )
            .with_obj(
                rx_fi_panel.coll,
                strat.StratDomain,
                force=True
            )
            .with_objs(
                *regimes_eq.values(), 
                *regimes_fi.values(),                 
                t=strat.StratDomain
            )
        )


        context.get().with_objs(
            *sum(regime_labels_eq.values(), []),
            *sum(regime_allocs_eq.values(), []),
            *sum(regime_labels_fi.values(), []),
            *sum(regime_allocs_fi.values(), []),
            t=strat.StratDomain,
        )

        bt_all = []
        bt_signals = []
 univ, universes = universes.by_depth(
            name,
            depth=1,
            universes=universes,
        )
        for i, (bt, params_eq) in enumerate(zip(
            bt_eq,
            strats_eq.values(),
        )):
            bt_all.append(bt)
            bt_alloc = []
            for reg_params, allocs in regime_allocs_eq.items():
                alloc = allocs[i]
                label = regime_labels_eq[reg_params][i]
                reg_label = "-".join([
                    str(p) for p in reg_params
                ])
                if reg_params[-1] == 0:
                    dd_objs.append((
                        bt.assume_res().rx,
                        bt.params.label() + reg_label,
                        dict(group="rx",)
                    ))
                    dd_objs.append((
                        label.id,
                        bt.params.label() + reg_label,
                        dict(group="regime"),
                    ))
                params = replace(
                    params_eq,
                    signal=strat.SignalFactor(
                        factors=alloc.id,
                        key="eq-reg-" + reg_label
                    )
                )
                univ, bt_alloc = domain.with_backtest(
                    univ, params, into=bt_alloc
                )
            bt_all.extend(bt_alloc)
            bt_signals.extend(bt_alloc)
        
        for i, (bt, params_fi) in enumerate(zip(
            bt_fi,
            strats_fi.values(),
        )):
            bt_all.append(bt)
            bt_alloc = []
            for reg_params, allocs in regime_allocs_fi.items():
                alloc = allocs[i]
                params = replace(
                    params_fi,
                    signal=strat.SignalFactor(
                        factors=alloc.id,
                        key="fi-reg-" + "-".join([
                            str(p) for p in reg_params
                        ])
                    )
                )
                univ, bt_alloc = domain.with_backtest(
                    univ, params, into=bt_alloc
                )
            bt_all.extend(bt_alloc)
            bt_signals.extend(bt_alloc)

        n_vars = len(REG_PARAMS) + 1
        for i in range(n_vars):
            bt_i = [
                bt_all[
                    i + (j * n_vars)
                ]
                for j in range(len(rx_eq))
            ]
            assert len(bt_i) == len(set([
                bt_.params.asset for bt_ in bt_i
            ]))
            reg_ps = REG_PARAMS[i-1]
            ident = strat.PfolioIdent(
                name,
                (
                    "lo" if i == 0
                    else "-".join([
                        str(p)
                        for p in reg_ps
                    ])
                ),
                1
            ) universes, bt_raw, pf_raw = strat.basket_pfo(
                bt_i,
                universes,
                ident,
                "raw",
                rebal=None,
            )
 universes, ntl = strat.pfo_notionals(
                pf_raw,
                universes,
                ident,
                PRICES,
                cap=10.,
            )
            (
                universes,
                pf_ntl_cap_ratio,
                bt_capped,
                bt_basket,
                pf,
            ) = strat.cap_pfo_notionals(
                bt_i,
                ntl,
                universes,
                ident,
                10.,
                suffix="cap",
                rebal=None,
            )

            bt_all.append(bt_basket)

        self = self.with_page(
            self.page_name(0, name),
            type="daily",
            graph=True,
            t=Project_201
        ).with_perf_table(
            bt_all, t=Project_201
        ).with_rx_graph(
            bt_all, t=Project_201
        )
        # .with_signal_graph(
        #     bt_signals, t=Project_201
        # )

        # self = self.with_objects_from_template(
        #     name,
        #     "dd_regimes",
        #     dd_objs2,
        #     t=Project_201,
        #     # params=f"x=key:mu,y=label,v={value}"
        # )
        self = self.with_objects_from_template(
            name,
            "dd_regimes",
            dd_objs,
            t=Project_201,
            # params=f"x=key:mu,y=label,v={value}"
        )

        self = self.with_graph(
            t=Project_201,
            name=self.page_name(0, name)
        )
        
        return self

REG_PARAMS = [
    (
        threshold, # threshold
        span, # span
        window, # window,
        1 / threshold_div, # continuation
        share[0], # share threshold
        share[1], # share recovery
        continuous, # continuous
    )
    for span in [
        # 32, 
        # 64, 
        # 128, 
        512, 
        # 512,
    ][:]
    for window in [
        128,
        # int(span/4),
        # int(span/4) + int(span/8)
    ]
    for threshold in [1.][:]
    for threshold_div in [
        1000
    ][:]
    for share in [
        (0, 0),
        (1, 0),
        (0, 1),
        # (1, 1)
    ]
    for continuous in [0, 1]
]

PROJECTS[PROJECT] = Project_201.load_

__all__ = [
    "Project_201",
]