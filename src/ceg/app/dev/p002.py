from typing import cast, TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...objs.vis import explorer
from ...objs.dev import bars
from ...objs.dev import pfo

from ...objs.dev.strat import uvar as strat

from .core import PROJECTS
from .p001 import *

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

@dataclass(frozen=True)
class Project_002(strat.UvariateBtProject):
    SPAN_FAST: ClassVar[list[int]] = SPAN_POW_2
    SPAN_RATIO: ClassVar[list[float]] = SPAN_RATIO
    SPAN_VOL_SIG: ClassVar[list[int]] = SPAN_VOL_SIG
    SPAN_VOL_SCALE: ClassVar[list[int]] = SPAN_VOL_SCALE

def eosc(fast: int, ratio: float = 1.5, vol: int = 48):
    return strat.SignalOsc(
        fast, int(fast * ratio), vol
    )

def load_p002():
    project = (
        Project_002()
        .with_config(explorer.ExplorerConfig(PROJECT))
    )

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

    i = 0
    for name, product_symbol_signals in [
        ("es-tn", ES_TN),
        ("es-tn-gc", ES_TN + GC),
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
                    rescale=strat.ScaleVol(
                        span=256
                    ),
                    rebal=rebal,
                )
                for product, symbol, signal
                in product_symbol_signals
            ]
            domain, univ = strat.UvariateBtDomain.new(
                PROJECT + "-" + name,
            )
            domain, univ, bts = domain.with_bts(
                univ, *params
            )
            domain, univ, rx_agg = domain.with_agg(
                univ, bts
            )
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

PROJECTS[PROJECT] = load_p002