from typing import cast, TYPE_CHECKING, ClassVar
import itertools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...vis import explorer

from ...apis import assert_is

from ... import engine

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

notional_caps = {
    "TN": 20.0,
    "TLT": 20.0,
    "SCHP": 20.0,
    "SCHO": 20.0,
    "SCHR": 20.0,
    "SCHJ": 20.0,
    "SCHI": 20.0,
    "SCHZ": 20.0,
}


@dataclass(frozen=True)
class Project_001(strat.MomExplorer):
    PROJECT: ClassVar[str] = PROJECT

    # NOTE: defaults
    SPAN_FAST: ClassVar[list[int]] = SPAN_POW_2
    SPAN_RATIO: ClassVar[list[float]] = SPAN_RATIO
    SPAN_VOL_SIG: ClassVar[list[int]] = SPAN_VOL_SIG
    SPAN_VOL_SCALE: ClassVar[list[int]] = SPAN_VOL_SCALE

    def product_symbol_spans(self):
        return [
            ("es", [("FUT", "ES")], SPAN_POW_2),
            ("tn", [("FUT", "TN")], SPAN_POW_2),
            ("gc", [("FUT", "GC")], SPAN_POW_2),
            ("spy", [("ETF", "SPY")], SPAN_POW_2),
            ("xlf", [("ETF", "XLF")], SPAN_POW_2),
            # NOTE: since 2000
            # medium pos signal decent
            # fast eg. 8,16 MR
            # larger 08 drawdown?
            ("xle", [("ETF", "XLE")], SPAN_POW_2),
            # NOTE: since 2000
            # moderate fast neg mean reversion, outside of covid and 14/15
            # pos maybe best fast (eg. 8), or broader signal base (3rd and 4th qintile?)
            # chunky 00 drawdown, great 05 rally, limited 08 drawdown? 2015 bad, covid bad, then flat
            ("xly", [("ETF", "XLY")], SPAN_POW_2),
            # NOTE: start 2000
            # medium pos good, fast pos
            # fast neg MR (but, covid)
            # flat to 2005 (ie. little 2002 dd)
            ("xlp", [("ETF", "XLP")], SPAN_POW_2),
            # NOTE: same, arguably much better 2015, covid, 2025 performance than spy?
            # bit more 02 and 08 dd (than discretionary?)
            # fast neg MR arguably better than XLY
            ("xli", [("ETF", "XLI")], SPAN_POW_2),
            # NOTE: start 2000, similar, perhaps lower overall?
            ("xlb", [("ETF", "XLB")], SPAN_POW_2),
            # NOTE: start 2000
            # larger 2015, 2018 struggles flat ish post covid
            # generally weak ish positive, some fast neg MR potential
            ("xlu", [("ETF", "XLU")], SPAN_POW_2),
            # NOTE: start 2000, very little percetible 2015 or 2020 drop, 02 and 08 do dd
            # fast or med-slow positive good, some fast to med neg MR
            ("xlv", [("ETF", "XLV")], SPAN_POW_2),
            # generally muted, but relatively little 08 dd, 08 probably likewise
            # possibly some fast neg MR
            # seemingly only v pos signal (all speeds)? medium is weak
            ("xlk", [("ETF", "XLK")], SPAN_POW_2),
            # similar to discretionary, maybe better 2018? v material 02 dd, seemingly only strong positive, not med? some fast neg MR?
            ("xlre", [("ETF", "XLRE")], SPAN_POW_2),
            # NOTE: start 2015, poss fast neg signal mean rev, fast to med (16-64) pos signal good 2021 else flat
            ("xlc", [("ETF", "XLC")], SPAN_POW_2),
            # NOTE: starts 2018, medium pos signal best
            ("vonv", [("ETF", "VONV")], SPAN_POW_2),
            # NOTE: value, starts 2010
            ("vong", [("ETF", "VONG")], SPAN_POW_2),
            # NOTE: growth, starts 2010
            ("tlt", [("ETF", "TLT")], SPAN_POW_2),
            # NOTE: start 2002
            # fast-medium positive contributes, flat ish from 2020
            ("tint", [("ETF", "SCHR")], SPAN_POW_2),
            # NOTE: intermdiate term treasury, start 2010
            # faster long signals positive, average 4-32
            ("tst", [("ETF", "SCHO")], SPAN_POW_2),
            # NOTE: short term tsy, start 2010
            # fast pos great in covid, else flat ish
            # fast neg poss mean rev?
            ("cst", [("ETF", "SCHJ")], SPAN_POW_2),
            # NOTE: corp 1-5
            # start 2020, wrecked in 2020 - but more or less upward drift thereafter? carry?
            # TODO: post credit shock recovery signal?
            ("clt", [("ETF", "SCHI")], SPAN_POW_2),
            # NOTE: corp 5-10
            # start 2020, wrecked in 2020 - but more or less upward drift thereafter? carry?
            ("agg", [("ETF", "SCHZ")], SPAN_POW_2),
            # NOTE: us aggregate bond
            # start 2011
            # pos signal, fast-med decent, barring covid shock
            ("tip", [("ETF", "SCHP")], SPAN_POW_2),
            # NOTE: start 2010
            # slower pos signal adds? slower escapes covid crash?
            # fast neg signal MR?
            ("hyb", [("ETF", "SCYB")], SPAN_POW_2),
            # NOTE: high yield bond
            # only 2023 onward
            ("gld", [("ETF", "GLD")], SPAN_POW_2),
            # NOTE: start 2003
            # pos signal adds, else mixed
            # weak poss MR for neg signal?
        ]

    # TICKERS:

    # SP500 trackers:
    # US: SPY, VOO, and IVV
    # UK/EU: VUSA, CSPX, and SPXP.

    # ETFs:
    # XLY (Consumer Discretionary), XLE (Energy), XLU (Utilities), XLB (Materials).
    # VONV (Russell 1000 Value ETF), VTWO (Russell 2000 ETF).
    # VONG (Russell 1000 Growth ETF).
    # SCHD (U.S. Dividend Equity ETF), SCHM (U.S. Mid-Cap ETF), SCHV (U.S. Large-Cap Value ETF), FNDX (Fundamental U.S. Large Company ETF), FNDB (Fundamental U.S. Broad Market ETF), SCHX (U.S. Large-Cap ETF), SCHB (U.S. Broad Market ETF).
    # SCHG (U.S. Large-Cap Growth ETF).
    # FNDA (Fundamental U.S. Small Company ETF), SCHK (1000 Index® ETF), SCHP (U.S. TIPS ETF), SCYB (High Yield Bond ETF), SCMB (Municipal Bond ETF), SCHA (U.S. Small-Cap ETF), SCHR (Intermediate-Term U.S. Treasury ETF), SCHJ (1-5 Year Corporate Bond ETF), SCHO (Short-Term U.S. Treasury ETF), SCHZ (U.S. Aggregate Bond ETF), SCHI (5-10 Year Corporate Bond ETF), SCHF (International Equity ETF).
    # GLL (UltraShort Gold), EEV (UltraShort Emerging Markets), YXI (UltraShort Japan), BZQ (UltraShort Basic Materials).
    # Vanguard High Dividend Yield ETF (VYM)

    # TODO: try long-spy short-tlt basket, run momentum
    # essentially, more volatile when normal corr, otherwise flat-ish?

    @classmethod
    def load_(cls, kw: dict):
        return cls.load(
            **{
                k: v
                for k, v in kw.items()
                if k in ["root", "test"]
            }
        )

    @classmethod
    def load(
        cls,
        root: str = "./__local__/apps",
        test: bool = False,
    ):
        """
        >>> root = "./__local__/test"
        >>> proj = Project_001.load(root, test=True).reset()
        >>> nav = proj.prepare_app()
        >>> nav[0]()
        >>> nav[1]()
        """

        self = Project_001(root=root).with_config(
            explorer.ExplorerConfig(PROJECT)
        )

        project = engine.Project.new(PROJECT)

        qiles = {
            **{
                (p, sym, span_f, span_vol, span_ratio): [
                    None,
                    cls.qile(5, 0, m=-1, f="z"),
                    cls.qile(5, 4, f="z"),
                ]
                for span_vol in cls.SPAN_VOL_SIG
                for span_ratio in cls.SPAN_RATIO
                for span_f in [4.0, 8]
                for p, sym in [
                    ("FUT", "ES"),
                    ("ETF", "SPY"),
                ]
            },
            **{
                (p, sym, span_f, span_vol, span_ratio): [
                    None,
                    cls.qile(5, 3, 5, f="z"),
                ]
                for span_vol in cls.SPAN_VOL_SIG
                for span_ratio in cls.SPAN_RATIO
                for span_f in [16, 32.0, 64.0, 128]
                for p, sym in [
                    ("ETF", "SPY"),
                    ("FUT", "ES"),
                ]
            },
            **{
                (p, sym, span_f, span_vol, span_ratio): [
                    None,
                    cls.qile(5, 2, 5, f="z"),
                ]
                for span_vol in cls.SPAN_VOL_SIG
                for span_ratio in cls.SPAN_RATIO
                for span_f in [8.0, 16, 32.0, 64.0]
                for p, sym in [
                    ("ETF", "TLT"),
                ]
            },
            **{
                (p, sym, span_f, span_vol, span_ratio): [
                    None,
                    cls.qile(5, 4, 1, f="z"),
                ]
                for span_vol in cls.SPAN_VOL_SIG
                for span_ratio in cls.SPAN_RATIO
                for span_f in [8.0, 16, 32.0]
                for p, sym in [
                    ("FUT", "TN"),
                ]
            },
            **{
                (p, sym, span_f, span_vol, span_ratio): [
                    None,
                    cls.qile(5, 3, 5, f="z"),
                ]
                for span_vol in cls.SPAN_VOL_SIG
                for span_ratio in cls.SPAN_RATIO
                for span_f in [
                    4,
                    8.0,
                    16,
                    32.0,
                    64.0,
                    128.0,
                ]
                for p, sym in [
                    ("FUT", "GC"),
                ]
            },
        }
        # TODO: test if rebal day for weekly matters?

        # NOTE: flip 512,0.5, log-exp unless otherwise stated

        # NOTE: FUT-GC (2008): long only, 4,8,16 best
        # vol clip / flip marginal

        # NOTE: ETF-TLT (2002): long only, mid speed up to 2020, best 64, faster okay. then bleed (slower better).
        # NOTE: l/s: faster better.
        # NOTE: vol clip / flip no major impact.
        # TODO: s vs l only TLT conditional on rate / inflation regime?

        # NOTE: FUT-TN (2016): 16 l/s, extreme sig only better (8,32 okay, trade off 2015 v 2022)
        # infl / rate regime condition l vs s only?
        # z flip plausibly slightly improves

        # NOTE: FUT-ES (2008): fast (4,8) v short mean reverts (but less than spy), any v long, best 64,128
        # z flip marginal? try with wider long zile

        # NOTE: ETF-SPY (2000): fast v short mean reverts, fast v long good (4,8), medium long good (c. 64)
        # vol flip 0.5 improves ls and lo (med, fast unaffacted)

        z_clip: list[tuple[str, int, float] | None] = [
            None,
            # ("f", 512, .5),
        ]
        i = 0
        for (name, product_symbols, spans), (
            suff,
            rebal,
        ) in itertools.product(
            self.product_symbol_spans(),
            self.rebal_variants(),
        ):
            with engine.context.graph(
                f"{PROJECT}-{name}", strat.StrategyGraph
            ):
                project.set_storage_defaults()
                univ_0 = strat.Strategies.under_project(
                    project,
                    name,
                    depth=0,
                )
                bt_raw: dict[
                    tuple, strat.Strategy | strat.Basket
                ] = {}
                bt_norm = {}
                for params in self.mom_params(
                    product_symbols,
                    spans,
                    rebal=rebal,
                    z_clip=z_clip
                    + [
                        # ("z", 512, .5),
                        # ("z", 512, 1.0),
                        # ("z", 512, 1.5),
                    ],
                    notional_caps=notional_caps,
                    notional_caps_default=10.0,
                    # qiles = qiles
                ):
                    univ_0, params = params.bind(
                        univ_0,
                        strat.Strategies,
                    )
                    r = params
                    bt_raw[
                        (
                            r.asset,
                            r.params.signal,
                            r.params.scale,
                            r.params.cap
                            if r.params.signal is None
                            else None,
                        )
                    ] = params
                univ_1 = strat.Strategies.under_project(
                    project,
                    name,
                    depth=1,
                )
                for params in self.mom_params(
                    product_symbols,
                    spans,
                    rebal=rebal,
                    qiles=qiles,
                    z_clip=z_clip,
                    with_no_signal=False,
                    notional_caps=notional_caps,
                    notional_caps_default=10.0,
                    qiles_default=[
                        None,
                        # cls.qile(5, 1, 4, f="z"),
                        cls.qile(5, 0, f="z"),
                        cls.qile(5, 4, f="z"),
                    ],
                ):
                    r = params
                    bt_raw[
                        (
                            r.asset,
                            r.params.signal,
                            r.params.scale,
                            r.params.cap
                            if r.params.signal is None
                            else None,
                        )
                    ]
                    univ_1, params = params.bind(
                        univ_1,
                        strat.Strategies,
                    )
                    bt_norm[
                        (
                            r.asset,
                            r.params.signal,
                            r.params.scale,
                            r.params.cap,
                        )
                    ] = params
                bt_no_signal = {
                    (asset, sig, sc, c): bt
                    for (
                        asset,
                        sig,
                        sc,
                        c,
                    ), bt in bt_raw.items()
                    if sig is None
                }
                # bt_no_qile = {
                #     (asset, sig): bt
                #     for (asset, sig), bt
                #     in bt_norm.items()
                #     if sig is not None
                #     and sig.qile is None
                #     and sig.zile is None
                # }
                # bt_w_qile = {
                #     (asset, sig): bt
                #     for (asset, sig), bt
                #     in bt_norm.items()
                #     if sig is not None
                #     and (
                #         sig.qile is not None
                #         sig.zile is not None
                #     )
                # }
                bts = (
                    list(bt_no_signal.values())
                    + list(bt_norm.values())
                    # + [
                    #     bt
                    #     for (asset, sig, sc, c), bt
                    #     in bt_norm.values()
                    #     if c is None
                    # ]
                    # + list(bt_w_qile.values())
                )
                self = self.with_page(
                    self.page_name(i, name, suffix=suff),
                    type="daily",
                    graph=True,
                    t=Project_001,
                ).with_perf_table(bts, t=Project_001)
                (self, bts) = self.with_partn_heatmap(
                    bts,
                    by="period",
                    ref="y",
                    value="shrp",
                    t=Project_001,
                    # include=lambda r: r.params.cap is None
                )

                bt_ref = list(bt_no_signal.values())[0]
                # long only?

                (self, bts) = self.with_partn_heatmap(
                    bts,
                    by="dd",
                    ref=f"{name}={assert_is(bt_ref.bt).rx().id}",
                    threshold=(1.0, 756),
                    value="shrp",
                    t=Project_001,
                    # include=lambda r: r.params.cap is None
                )
                res_no_signal = [
                    r
                    for r in bts
                    if r.params.signal is None
                ]
                res_no_qile = [
                    r
                    for r in bts
                    if isinstance(
                        r.params.signal, strat.SignalOsc
                    )
                    and r.params.signal.qile is None
                    and r.params.signal.zile is None
                ]
                res_qile = [
                    r
                    for r in bts
                    if isinstance(
                        r.params.signal, strat.SignalOsc
                    )
                    and (
                        r.params.signal.qile is not None
                        or r.params.signal.zile is not None
                    )
                ]
                (self, res_no_qile) = (
                    self.with_mutn_heatmap(
                        res_no_qile,
                        target="sig",
                        change="sdev",
                        params=(
                            5,
                            0.5,
                        ),
                        value="shrp",
                        t=Project_001,
                        # include=lambda r: r.params.cap is None
                    )
                )
                # self,res_no_qile = self.with_mutn_heatmap(
                #     res_no_qile,
                #     domain,
                #     target="sig",
                #     change="sdev",
                #     params=(5, 0.5),
                #     value="shrp",
                #     t=Project_001
                # )
                res = res_no_signal + res_no_qile + res_qile
                self = (
                    self.with_rx_graph(res, t=Project_001)
                    .with_signal_graph(res, t=Project_001)
                    .with_graph(
                        engine.context.get(), t=Project_001
                    )
                )
                if test:
                    return self
                i += 1
        return self


PROJECTS[PROJECT] = Project_001

__all__ = [
    "Project_001",
    "SPAN_POW_2",
    "SPAN_POW_2_EVEN",
    "SPAN_POW_2_ODD",
]
