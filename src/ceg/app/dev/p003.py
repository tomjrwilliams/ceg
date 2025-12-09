from typing import cast, TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...objs.vis import explorer
from ...objs.dev import calc
from ...objs.dev import bars
from ...objs.dev import pfo

from ...objs.dev.strat.mom import eosc as strat

from .core import PROJECTS
from .p001 import *

PROJECT = "p003"

@dataclass(frozen=True)
class Project_003(strat.MomEOscProject):
    pass
    
def eco(
    symbol: str,
    method: str,
    **kwargs
):
    return strat.EcoSignalParams(symbol, method, **kwargs)
        
def load_p002():
    project = (
        Project_003()
        .with_config(explorer.ExplorerConfig(PROJECT))
    )
    
    activation=None
    speeds = [
        # 8, 
        32, 
        # 128, 
        512,
        # 1024, 
        2048
    ]
    f_sigs = lambda scale: [
        None,
    ] + [
        eco(
            "VIX",
            "ez",
            speed=s,
            scale=scale,
            activation=activation
        )
        for s in speeds
    ] + [
        eco(
            "D_US_BAA",
            "ez",
            speed=s,
            scale=scale,
            activation=activation
        )
        for s in speeds
    ] + [
        eco(
            "D_US_BAA_FF",
            "ez",
            speed=s,
            scale=scale,
            activation=activation
        )
        for s in speeds
    ] + [
        eco(
            "US_IG_OAS",
            "ez",
            speed=s,
            scale=scale,
            activation=activation
        )
        for s in speeds
    ] + [
        eco(
            "US_HY_OAS",
            "ez",
            speed=s,
            scale=scale,
            activation=activation
        )
        for s in speeds
    ]

    ES = [("ETF", "SPY", s) for s in f_sigs(-1.)]
    TN = [("ETF", "TLT", s) for s in f_sigs(1.)]
    GC = [("FUT", "GC", s) for s in f_sigs(1.)]

    # TODO: ignore null parametrisation to z
    # equivalent speed when monthly sample (eg. try fill forward values?)

    # cap util:

    # TODO: distance from min / max?
    # TODO: plot signal, check
    # TODO: industrial production?
    # TODO: sector specific IP to specific etf?
    # (construction, industry, etc.)

    activations = []
    speeds_r = [
        16, 32, 64
    ]

    f_sig_eco = lambda ind, scale, acts: [None] + [
        eco(
            # f"{ind}:L{lag}{asof}",
            f"{ind}{asof}",
            "demax" if act == "ng_tnlog_ng" else "zdemax",
            # speed=int(s / 16),
            speed = s,
            scale=scale,
            activation=act
        )
        for s in speeds_r
        # for s in speeds
        for act in activations + acts
        for asof in [
            # "",
            ":3W"
        ]
        # for lag in [90, 120, 150, 180]

        # diff vol on the z, TODO: parametrise
    ]

    ES_ECO = [
        ("ETF", "SPY", s)
        for s in f_sig_eco("TCU", 1, ["ng_tnlog_ng"])
    ]
    IND_ECO = [
        ("ETF", symbol, s)
        for symbol, sc in {
            "SPY": 1.,
            # "IWM": 1.,
            "IYJ": 1.,
            # "TLT": -1.,
        }.items()
        for s in f_sig_eco(
            "US_IP", sc, [
                None,
                "n_tanh_p_one",
                "ng_tnlog_ng"
            ]
        )
    ]
    TN_ECO = [
        ("ETF", "TLT", s)
        for s in f_sig_eco("TCU", -1., ["ng_tnlog_ng"])
    ]
    
    # spreads:

    # reasonable (with or without tanh, given post-scale):

    # cv 48,256 tlt, sign=+1.:
    # 32 span ez baa spread (or HY OAS)

    # cv 48,256 spy, sign=-1.:
    # 2048 span ez IG OAS
    # 2048 span ez BAA FF (and 32 span)

    # TODO: EUR_HY_OAS and euro etf (pnl needs dollarising)
    # TODO: vol of vol (vvol)

    # TODO: wider bullish / bearish universe?

    i = 0
    for name, product_symbol_signals in [
        ("es-tn", ES + TN),
        # ("es-tn-eco", ES_ECO + TN_ECO)
        ("ind", IND_ECO)
        # ("es-tn-gc", ES + TN + GC),
    ]:
        for suff, rebal in [
            ("", None),
            ("-w", "3D"),
            # ("-m", "3D1W"),
        ]:
            i_str = str(i).rjust(3, "0") + "-"
            params = [
                strat.MomEOscBtParams(
                    product,
                    symbol,
                    scale=strat.EOscScaleVolParams(
                        span=48
                    ),
                    signal=signal,
                    rescale=strat.EOscScaleVolParams(
                        span=256
                    ),
                    rebal=rebal,
                )
                for product, symbol, signal
                in product_symbol_signals
            ]
            project = (
                project.with_mom_eosc_page_from_params(
                    params,
                    i_str + name + "-v" + suff,
                    with_agg_rx=True
                )
            )
        i += 1
    return project

PROJECTS[PROJECT] = load_p002