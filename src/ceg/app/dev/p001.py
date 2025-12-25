from typing import cast, TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...objs.vis import explorer
from ...objs.dev import bars
from ...objs.dev import pfo
from ...objs.dev import agg

from ...objs.dev import strat

from .core import PROJECTS

PROJECT = "p001"

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
SPAN_POW_2 = [4, 8, 16, 32, 64, 128, 256, 512]
SPAN_POW_2_EVEN = [4, 16, 64, 256]
SPAN_POW_2_ODD = [8, 32, 128, 512]

SPAN_RATIO = [
    1.5, 
    # 3
]

@dataclass(frozen=True)
class Project_001(strat.StratProject):
    SPAN_FAST: ClassVar[list[int]] = SPAN_POW_2
    SPAN_RATIO: ClassVar[list[float]] = SPAN_RATIO
    SPAN_VOL_SIG: ClassVar[list[int]] = SPAN_VOL_SIG
    SPAN_VOL_SCALE: ClassVar[list[int]] = SPAN_VOL_SCALE


def load_p001(root: str = "./__local__/apps"):
    """
    >>> root = "./__local__/test"
    >>> proj = load_p001(root=root).reset()
    >>> nav = explorer.prepare_app(PROJECT, root, proj.domains)
    >>> nav[0]()
    >>> nav[1]()
    """

    project = (
        Project_001(root=root)
        .with_config(explorer.ExplorerConfig(PROJECT))
    )

    i = 0
    for name, product_symbols, spans in [
        ("es", [("FUT", "ES")], SPAN_POW_2),
        ("tn", [("FUT", "TN")], SPAN_POW_2),
        ("gc", [("FUT", "GC")], SPAN_POW_2),
        ("spy", [("ETF", "SPY")], SPAN_POW_2),
        ("tlt", [("ETF", "TLT")], SPAN_POW_2),
        (
            "es-tn",
            [("FUT", "ES"), ("FUT", "TN")],
            SPAN_POW_2_ODD
        ),
        (
            "spy-tlt",
            [("ETF", "SPY"), ("ETF", "TLT")],
            SPAN_POW_2_ODD
        ),
        (
            "es-spy",
            [("FUT", "ES"), ("ETF", "SPY")],
            SPAN_POW_2_ODD
        ),
        (
            "tn-tlt",
            [("FUT", "TN"), ("ETF", "TLT")],
            SPAN_POW_2_ODD
        ),
    ]:
        for suff, rebal in [
            ("", None),
            ("-w", "3D"),
            # ("-m", "3D1W"),
        ]:
            i_str = str(i).rjust(3, "0") + "-"
            # project = project.with_mom_eosc_page(
            #     i_str + name + suff,
            #     product_symbols,
            #     spans,
            # )
            params = project.mom_params(
                product_symbols,
                spans,
                span_vol_rescale=[256],
                rebal=rebal,
            )
            domain, univ = strat.StratDomain.new().with_univ(
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

PROJECTS[PROJECT] = load_p001

__all__ = [
    "load_p001",
    "SPAN_POW_2",
    "SPAN_POW_2_EVEN",
    "SPAN_POW_2_ODD",
]