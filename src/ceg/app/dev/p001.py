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
class Project_001(explorer.Project):
    def with_mom_eosc_page(
        self,
        name: str,
        product_symbols: list[tuple[str, str]],
        span_fast: list[int]=SPAN_POW_2,
        span_ratio: list[float]=SPAN_RATIO,
        span_vol_sig: list[int]=SPAN_VOL_SIG,
        span_vol_scale: list[int]=SPAN_VOL_SCALE,
        span_vol_rescale: list[int] | None = None,
        rebal: str | None = None,
        with_agg_rx: bool = False,
        # TODO: rebal rules
    ):
        bts: list[strat.MomEOscFutBtObjects]

        signals = [None] + [
            strat.EOscSignalParams(
                fast, int(fast * ratio), vol
            )
            for fast in span_fast
            for vol in span_vol_sig
            for ratio in span_ratio
        ]
        scales = [
            strat.EOscScaleVolParams(span=span)
            for span in span_vol_scale
        ]
        rescales = [
            None
        ] if span_vol_rescale is None else [
            strat.EOscScaleVolParams(span=span)
            for span in span_vol_rescale
        ]

        d, bts, rx_agg = strat.StratMomEOscFut.new(
            *[
                strat.MomEOscFutBtParams(
                    product,
                    symbol,
                    scale=scale,
                    signal=signal,
                    rescale=rescale,
                    rebal=rebal,
                )
                for rescale in rescales
                for scale in scales
                for signal in signals
                for product, symbol in product_symbols
            ],
            name=f"mom-eosc-fut-{name}"
        )
        self = self.with_domains(**{
            name: lambda sync: explorer.AppDomain(
                [d], sync
            )
        }).with_page(
            name,
            type="daily",
            domain=name,
        )
        if with_agg_rx:
            self = self.with_object(
                rx_agg.path,
                graph=0,
                label=name,
                group="rx",
            ).with_object(
                rx_agg.path,
                graph=0,
                label=name,
                group="pos"
            )
        for bt in bts:
            self = self.with_object(
                bt.res.rx.path,
                graph=0,
                label=bt.params.label(),
                group="rx",
            ).with_object(
                bt.res.pos.path,
                graph=0,
                label=bt.params.label(),
                group="pos"
            )
        self = self.with_graph(
            0,
            "perf"
        )
        return cast(Project_001, self)


project = (
    Project_001()
    .with_config(explorer.ExplorerConfig(PROJECT))
)

i = 0
for name, product_symbols, spans in [
    ("es", [("FUT", "ES")], SPAN_POW_2),
    ("tn", [("FUT", "TN")], SPAN_POW_2),
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
        project = project.with_mom_eosc_page(
            i_str + name + "-v" + suff,
            product_symbols,
            spans,
            span_vol_rescale=[256],
            rebal=rebal,
        )
    i += 1

PROJECTS[PROJECT] = project