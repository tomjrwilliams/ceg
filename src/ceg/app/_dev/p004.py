from typing import cast, TYPE_CHECKING, ClassVar

import itertools
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...vis import explorer
from ...objs.dev import calc
from ...objs.dev import bars
from ...objs.dev import pfo
from ...objs.dev import agg
from ...objs.dev import strat

from .core import PROJECTS

from .p002 import eosc
from .p003 import eco

PROJECT = "p004"

ECO_SPEEDS: list[float] = [
    16,
    # 8, 
    32, 
    # 128, 
    512,
    # 1024, 
    2048
]
MOM_SPEEDS: list[float] = [
    16, 32, 128
]

def f_cmom(
    symb_sc: dict[str, float],
    eco_sc: dict[str, float],
    eco_tf_acts: dict[str, str | None],
    eco_speeds: list[float] = ECO_SPEEDS,
    mom_speeds: list[float] = MOM_SPEEDS,
):
    eco_sigs = [
        eco(
            ec,
            tf,
            speed=s,
            scale=sc,
            activation=act
        )
        for ec, sc in eco_sc.items()
        for s in eco_speeds
        for tf, act in eco_tf_acts.items()
    ]
    return [
        ("ETF", symb, None, None, None)
        for symb in symb_sc.keys()
    ] + [
        ("ETF", symb, eosc(int(sp)), None, None)
        for symb in symb_sc.keys()
        for sp in mom_speeds
    ] + [
        (
            "ETF",
            symb,
            eosc(int(sp)),
            eco_sig, 
            f"{symb}-{eosc(int(sp)).label()}"
        )
        for sp in mom_speeds
        for symb, sc in symb_sc.items()
        for eco_sig in eco_sigs
    ]


@dataclass(frozen=True)
class Project_004(strat.StratExplorer):
    
    def product_symbol_conds(self):

        spread_acts: dict[str, str | None] = {
            "ez": "unit_tanh"
            # unit_tanh = a=1,f=tanh,m=0.5
        }

        SPY_CMOM = f_cmom(
            {
                "SPY": 1.,
            },
            {
                "VIX": 1.,
                "D_US_BAA": 1.,
                "D_US_BAA_FF": 1.,
                "US_IG_OAS": 1.,
                "US_HY_OAS": 1.,
            },
            spread_acts
        )

        TLT_CMOM = f_cmom(
            {
                "TLT": 1.,
            },
            {
                "VIX": 1.,
                "D_US_BAA": 1.,
                "D_US_BAA_FF": 1.,
                "US_IG_OAS": 1.,
                "US_HY_OAS": 1.,
            },
            spread_acts
        )

        EUR_CMOM = f_cmom(
            {
                "VGK": 1.,
                "EZU": 1.,
            },
            {
                "EUR_HY_OAS": 1.,
            },
            spread_acts
        )
        return [
            ("spy", SPY_CMOM),
            ("tlt", TLT_CMOM),
            ("eur", EUR_CMOM),
        ]
        
    # SPY:
    # NOTE: fast (16) 16 mom, slow credit signals condition does well in crisis, flat elsewhere (so little contribution to total notional budget when not active)

    # TODO: try average z scores, vs average positions?
    # TODO: compare ES?

    # TODO: try replace TLT with TN (so only excess return)?

    # TODO: eur eq vol?
    # TODO: jp? 

    @classmethod
    def load(
        cls, root: str = "./__local__/apps", test: bool = False
    ):
        """
        >>> root = "./__local__/test"
        >>> proj = Project_004.load(root, test=True).reset()
        >>> nav = proj.prepare_app()
        >>> nav[0]()
        >>> nav[1]()
        """
        self = (
            Project_004(root=root)
            .with_config(explorer.ExplorerConfig(PROJECT))
        )

        i = 0
        for (
            (name, product_symbol_signal_cond_aliases),
            (suff, rebal)
        ) in itertools.product(
            self.product_symbol_conds(),
            self.rebal_variants(),
        ):
            configs_w_cond = [
                (product, symbol, signal, cond, alias)
                for product, symbol, signal, cond, alias
                in product_symbol_signal_cond_aliases
                if cond is not None
            ] univ = strat.StratDomain.new().with_univ(
                self.univ_name(name + "-raw")
            )
            res_raw = []
            for params in [
                strat.Strategy(
                    asset = strat.AssetConcrete(
                        product,
                        symbol,
                    ),
                    scale=strat.ScaleVol(
                        span=48
                    ),
                    signal=signal,
                    # rescale=strat.ScaleVol(
                    #     span=256
                    # ),
                    rebal=rebal,
                )
                for product, symbol, signal, cond, alias
                in configs_w_cond
            ]:
                univ, res_raw = domain.with_backtest(
                    univ, params, into = res_raw
                )
 univ = domain.with_univ(
                self.univ_name(name)
            )
            res = []
            for params in [
                strat.Strategy(
                    asset = strat.AssetConcrete(
                        product,
                        symbol,
                    ),
                    scale=strat.ScaleVol(
                        span=48
                    ),
                    signal=signal,
                    rescale=strat.ScaleVol(
                        span=256
                    ),
                    rebal=rebal,
                )
                for product, symbol, signal, cond, alias
                in product_symbol_signal_cond_aliases
                if cond is None
            ] + [
                strat.Strategy(
                    asset = strat.AssetSynthetic(
                        bt.assume_res().rx,
                        alias=alias,
                    ),
                    scale=strat.ScaleVol(
                        span=256
                        # TODO: just const?
                        # (given rescale)
                    ),
                    signal=cond,
                    # rescale=strat.ScaleVol(
                    #     span=256
                    # ),
                    rebal=rebal,
                )
                for bt, (
                    product, symbol, signal, cond, alias
                )
                in zip(res_raw, configs_w_cond)
            ]:
                univ, res = domain.with_backtest(
                    univ, params, into = res
                )
            self = self.with_page(
                self.page_name(i, name, suffix=suff),
                type="daily",
                graph=True,
                t=Project_004
            ).with_perf_table(
                res, t=Project_004
            ).with_rx_graph(
                res, t=Project_004
            ).with_signal_graph(
                res, t=Project_004
            ).with_graph(t=Project_004)
            if test:
                return self
            i += 1
        return self

PROJECTS[PROJECT] = Project_004.load