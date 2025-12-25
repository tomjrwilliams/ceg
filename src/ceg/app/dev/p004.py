from typing import cast, TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...objs.vis import explorer
from ...objs.dev import calc
from ...objs.dev import bars
from ...objs.dev import pfo
from ...objs.dev import agg
from ...objs.dev import strat

from .core import PROJECTS

from .p002 import eosc
from .p003 import eco

PROJECT = "p004"

@dataclass(frozen=True)
class Project_004(strat.StratProject):
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


# SPY:
# NOTE: fast (16) 16 mom, slow credit signals condition does well in crisis, flat elsewhere (so little contribution to total notional budget when not active)

# TODO: try average z scores, vs average positions?
# TODO: compare ES?

# TODO: try replace TLT with TN (so only excess return)?

# TODO: eur eq vol?
# TODO: jp? 

def load_p004(root: str = "./__local__/apps"):
    """
    >>> root = "./__local__/test"
    >>> proj = load_p004(root=root).reset()
    >>> nav = explorer.prepare_app(PROJECT, root, proj.domains)
    >>> nav[0]()
    >>> nav[1]()
    """
    project = (
        Project_004(root=root)
        .with_config(explorer.ExplorerConfig(PROJECT))
    )

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
            configs_w_cond = [
                (product, symbol, signal, cond, alias)
                for product, symbol, signal, cond, alias
                in product_symbol_signal_cond_aliases
                if cond is not None
            ]
            params = [
                strat.StratParams(
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
            ]
            domain, univ = strat.StratDomain.new().with_univ(
                PROJECT + "-" + name + "-raw",
            )
            domain, univ, bts = domain.with_strats(
                univ, *params
            )
            params = [
                strat.StratParams(
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
                strat.StratParams(
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
                in zip(bts, configs_w_cond)
            ]
            domain, univ = domain.with_univ(
                PROJECT + "-" + name,
            )
            domain, univ, bts = domain.with_strats(
                univ, *params
            )
            project = (
                project.with_bt_page(
                    i_str + name + "-v" + suff,
                    domain,
                    bts,
                )
            )
        i += 1
    return project

PROJECTS[PROJECT] = load_p004