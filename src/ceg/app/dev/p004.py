from typing import cast, TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...objs.vis import explorer
from ...objs.dev import calc
from ...objs.dev import bars
from ...objs.dev import pfo

from ...objs.dev.strat import uvar as strat

from .core import PROJECTS

from .p002 import eosc
from .p003 import eco

PROJECT = "p004"

@dataclass(frozen=True)
class Project_004(strat.UvariateBtProject):
    pass
    
ECO_SPEEDS: list[float] = [
    16,
    # 8, 
    32, 
    # 128, 
    512,
    # 1024, 
    2048
]
MOM_SPEEDS = ECO_SPEEDS

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
        (
            "ETF",
            symb,
            eosc(int(sp)),
            eco_sig, 
            f"{symb}-{int(sp)}-{eco_sig.label()}"
        )
        for sp in mom_speeds
        for symb, sc in symb_sc.items()
        for eco_sig in eco_sigs
    ]


def load_p004():
    project = (
        Project_004()
        .with_config(explorer.ExplorerConfig(PROJECT))
    )

    spread_acts: dict[str, str | None] = {
        "ez": "unit_tanh"
    }

    SPY_CMOM = f_cmom(
        {
            "SPY": 1.,
        },
        {
            "VIX": -1.,
            # "D_US_BAA": -1.,
            # "D_US_BAA_FF": -1.,
            # "US_IG_OAS": -1.,
            # "US_HY_OAS": -1.,
        },
        spread_acts
    )

    TLT_CMOM = f_cmom(
        {
            "TLT": 1.,
        },
        {
            "VIX": -1.,
            # "D_US_BAA": -1.,
            # "D_US_BAA_FF": -1.,
            # "US_IG_OAS": -1.,
            # "US_HY_OAS": -1.,
        },
        spread_acts
    )

    EUR_CMOM = f_cmom(
        {
            "VGK": 1.,
            "EZU": 1.,
        },
        {
            "EUR_HY_OAS": -1.,
        },
        spread_acts
    )

    i = 0
    for name, product_symbol_signal_cond_aliases in [
        ("spy", SPY_CMOM),
        ("tlt", TLT_CMOM),
        ("eur", EUR_CMOM),
    ]:
        for suff, rebal in [
            ("", None),
            ("-w", "3D"),
            # ("-m", "3D1W"),
        ]:
            i_str = str(i).rjust(3, "0") + "-"
            params = [
                strat.UvariateBtParams(
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
                in product_symbol_signal_cond_aliases
                if cond is not None
            ]
            domain, univ = strat.UvariateBtDomain.new(
                PROJECT + "-" + name,
            )
            domain, univ, bts = domain.with_bts(
                univ, *params
            )
            params = [
                strat.UvariateBtParams(
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
                if cond is None
                else strat.UvariateBtParams(
                    asset = strat.AssetSynthetic(
                        bt.assume_res().rx,
                        alias=alias,
                    ),
                    scale=strat.ScaleVol(
                        span=256
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
                in zip(
                    bts,
                    product_symbol_signal_cond_aliases
                )
            ]
            domain, univ, bts = domain.with_bts(
                univ, *params
            )
            domain, univ, rx_agg = domain.with_agg(
                univ, bts
            )
            
            # TODO: presumably the issue is we just listen for new rather than for the existence of upstream?
            # TODO: can maybe try ordering the sync, but better to just fix the dependency logic

            # idealyl also deal with changing defs so we dont have to keep deleting the db cache

            project = (
                project.with_bt_page(
                    i_str + name + "-v" + suff,
                    domain,
                    bts,
                    rx_agg=rx_agg
                )
            )
        i += 1
    return project

PROJECTS[PROJECT] = load_p004