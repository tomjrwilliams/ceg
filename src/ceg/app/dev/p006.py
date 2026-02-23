import os
from typing import overload, cast, TYPE_CHECKING, ClassVar
import itertools
from dataclasses import replace
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

import polars as pl

from ... import apis
from ...apis import assert_is, st

from ... import engine
from ...vis import explorer, TTemplateObject
from ...objs.dev import bars
from ...objs.dev import pfo
from ...objs.dev import agg
from ...objs.dev import calc

from ...objs.dev import strat
from ...objs.dev.strat.baskets.agg import (
    agg_rank_basket,
    agg_weight_basket,
)
from ...objs.dev.strat.baskets.comb import (
    comb_rank_basket,
    comb_weight_basket,
)

from .core import PROJECTS

logger, log = apis.logs.logger(
    __file__, streams=dict(formatter=True)
)
PROJECT = "p006"

@dataclass(frozen=True)
class Project_006(strat.StratExplorer):
    PROJECT: ClassVar[str] = PROJECT

    def build_base_strategies(
        self,
        basket: str | None = None,
        keep: list[str] | None = None,
        # symbols in cv
        page_symbols: list[str] | None = None,
        keep_all: bool = False,
    ) -> dict[str, strat.Strategy | strat.Basket]:
        cv_keep = {}
        cv_rank = {}
        cv_c_rank = {}
        cv_agg = {}

        project_cv = engine.Project.new("cv")
        engine.context.get().with_obj(
            project_cv, force=True
        )

        # rw_project = engine.Project.new("rw-agg")
        # engine.context.get().with_obj(
        #     rw_project, force=True
        # )

        rw_project_clip = engine.Project.new("rw-clip")
        engine.context.get().with_obj(
            rw_project_clip, force=True
        )
        rw_project_ls = engine.Project.new("rw-ls")
        engine.context.get().with_obj(
            rw_project_ls, force=True
        )

        for SYMBOL in UNIVERSE.keys():
            if (
                page_symbols is not None
                and SYMBOL not in page_symbols
            ):
                continue
            log("INFO")(building_strategies=SYMBOL)

            universe = {
                f"{SYMBOL}{suff}": strat.cv(
                    UNIVERSE[SYMBOL],
                    span=48,
                    signal=sig,
                    cap=strat.CapNotional(10.),
                )
                for suff, sig in SIGNALS.items()
            }

            cv_no_sig: dict[
                strat.TAsset,
                tuple[str, strat.Strategy]
            ] = {} # key = asset

            univs_cv = {}

            for symbol, params in universe.items():
                univ_cv = univs_cv.get(
                    "symbol",
                    strat.Strategies.under_project(
                        project_cv,
                        f"{symbol}",
                    )
                )
                if params.params.signal is None:
                    if params.asset in cv_no_sig:
                        continue
                    univ_cv, params = params.bind(
                        univ_cv,
                        strat.Strategies
                    )
                    cv_no_sig[
                        params.asset
                    ] = (symbol, params)

                engine.context.get().with_obj(
                    univ_cv, force=True
                ) # can be moved outsdie loop? TODO
                univs_cv[symbol] = univ_cv
            
            cv: dict[
                str,
                strat.Strategy | strat.Basket
            ] = {
                symbol: p
                for _, (symbol, p)
                in cv_no_sig.items()
            }

            # --

            univs_cv_sig = {}

            for symbol, params in universe.items():
                univ_cv_sig = univs_cv_sig.get(
                    symbol, 
                    strat.Strategies.under_project(
                        project_cv,
                        f"{symbol}-sig",
                    )
                )
                if params.params.signal is None:
                    # bt_pre_norm[symbol] = bt
                    continue
                if symbol in cv:
                    continue
                params_norm = replace(
                    params,
                    asset=strat.AssetSynthetic(
                        cv_no_sig[params.asset][1]
                        .rx().path,
                    ),
                    params=replace(
                        params.params,
                        scale=None,
                        cap=None,
                    ),
                )
                univ_cv_sig, params_norm = params_norm.bind(
                    univ_cv_sig,
                    strat.Strategies,
                )
                cv[symbol] = params_norm
                engine.context.get().with_obj(
                    univ_cv_sig, force=True
                ) # can be moved outsdie loop? TODO
                univs_cv_sig[symbol] = univ_cv_sig
            
            # --

            rw_symbols_clip = {
                f"{SYMBOL}-{sig}-{var[1:]}": [
                    k for k in universe.keys()
                    if k.startswith(
                        f"{SYMBOL}_{sig}_"
                    )
                    and k.endswith(var)
                ]
                for sig in ["osc", "wm", "shrp"]
                for var in ["_blo", "_bso"]
            }
            rw_symbols_ls = {
                f"{SYMBOL}-{sig}-{var[1:]}": [
                    k for k in universe.keys()
                    if k.startswith(
                        f"{SYMBOL}_{sig}_"
                    )
                    and k.endswith(var)
                ]
                for sig in ["osc", "wm", "shrp"]
                for var in ["_ls", "_bls"]
            }

            # TODO: put back maybe once we have dot 
            # so can do best sig per sp, then re-agg
            # rw_symbols_sp = {
            #     f"{SYMBOL}-{sp}": [
            #         k for k in universe.keys()
            #         if any([
            #             k.startswith(
            #                 f"{SYMBOL}_{sig}_{sp}_"
            #             )
            #             for sig in ["osc", "wm", "shrp"]
            #         ])
            #     ]
            #     for sp in [4, 5, 6, 7, 8]
            # }
            rw_strats_clip = {
                symbol_sig_var: [
                    assert_is(cv[k], strat.Strategy)
                    for k in ks
                ]
                for symbol_sig_var, ks
                in {
                    **rw_symbols_clip,
                    # **rw_symbols_sp,
                }.items()
                if len(ks)
            }
            rw_strats_ls = {
                symbol_sig_var: [
                    assert_is(cv[k], strat.Strategy)
                    for k in ks
                ]
                for symbol_sig_var, ks
                in {
                    **rw_symbols_ls,
                    # **rw_symbols_sp,
                }.items()
                if len(ks)
            }

            # (
            #     univs_rw,
            #     univs_rw_rank,
            #     # univs_rw_norm,
            #     strats_rank,
            #     # strats_rank_norm
            # ) = strat.baskets.agg.rank_baskets_dot(
            #     rw_project,
            #     rw_strats,
            #     RANK_PARAMS,
            #     scale=None,
            #     cap=strat.CapNotional(10.),
            #     vol_norm=None,
            #     # vol_norm=256,
            #     # TODO: too quick vol norm, gets massively over levered if out for long periods
            # )
            strats_rank = {}

            # pfo_params_clip = (
            #     strat.baskets.comb.PfoWeightParams(
            #         metrics=("sharpe", "calmar"),
            #         samples=RANK_SAMPLE,
            #         windows=RANK_WINDOW,
            #         smoothings=RANK_SMOOTH,
            #         tops=(1, 2, 3, 5),
            #         norms=("eq",)
            #     )
            # )
            # (
            #     univs_c_rw,
            #     univs_c_rw_rank,
            #     # univs_rw_norm,
            #     strats_rank_clip,
            #     # strats_rank_norm
            # ) = strat.baskets.comb.rank_baskets_dot(
            #     rw_project_clip,
            #     rw_strats_clip,
            #     comb_rank_basket(
            #         (),
            #         params=pfo_params_clip,
            #         not_cap=10.,
            #     ),
            #     # vol_norm=256,
            #     # TODO: too quick vol norm, gets massively over levered if out for long periods
            # )
            strats_rank_clip = {}
            # TODO: put back once hrp works...

            pfo_params_ls = (
                strat.baskets.comb.PfoWeightParams(
                    metrics=("sharpe", "calmar"),
                    samples=RANK_SAMPLE,
                    windows=RANK_WINDOW,
                    smoothings=RANK_SMOOTH,
                    tops=(1, 2, 3, 5),
                    norms=(
                        "eq",
                        "hrp_48_15",
                        "hrp_192_15",
                        # TODO: corr and vol sep speeds
                    )
                )
            )
            (
                univs_c_rw,
                univs_c_rw_rank,
                univs_rw_norm,
                strats_rank_ls,
                strats_rank_ls_norm
            ) = strat.baskets.comb.rank_baskets_dot(
                rw_project_ls,
                rw_strats_ls,
                comb_rank_basket(
                    (),
                    params=pfo_params_ls,
                    not_cap=10.,
                ),
                vol_norm=256,
            )

            strats_c_rank = {
                # **strats_rank_clip,
                **strats_rank_ls,
                **strats_rank_ls_norm,
            }

            weight_keys = [
                "_".join(map(
                    str, w_params.values()
                ))
                for w_params in RANK_PARAMS
            ]

            # rw_strats_agg = {
            #     f"{SYMBOL}-sp": {
            #         weight_key: [
            #             assert_is(
            #                 s, strat.Strategy
            #             ) for k, s in strats_rank.items()
            #             if k.endswith(weight_key)
            #             and any([
            #                 k.startswith(
            #                     f"rw-{SYMBOL}-{sp}"
            #                     # NOTE: rw = univ_prefix
            #                     # name={symbol}-{sp}
            #                 )
            #                 for sp in [4, 5, 6, 7, 8]
            #             ])
            #         ]
            #         for weight_key in weight_keys
            #     }
            # }

            # (
            #     univs_rw_agg,
            #     univs_rw_agg_rank,
            #     # univs_rw_norm,
            #     strats_agg,
            #     # strats_rank_norm
            # ) = strat.rank_baskets_dot(
            #     rw_project,
            #     rw_strats_agg,
            #     RANK_PARAMS,
            #     scale=None,
            #     cap=strat.CapNotional(10.),
            #     vol_norm=None,
            #     # vol_norm=256,
            #     # TODO: too quick vol norm, gets massively over levered if out for long periods
            # )
            strats_agg = {}

            # TODO: need a dot node so we have a single signal path to pass into dot

            cv_keep = {
                **cv_keep,
                **{
                    k: v for k, v in cv.items()
                    if (
                        keep is None
                        or k in keep
                        or keep_all
                    )
                }
            }
            cv_rank = {
                **cv_rank,
                **{
                    k: v for k, v in strats_rank.items()
                    if keep_dot(k, basket)
                    or keep_all
                }
            }
            cv_agg = {
                **cv_agg,
                **{
                    k: v for k, v in strats_agg.items()
                    if keep_dot(k, basket)
                    or keep_all
                },
            }
            cv_c_rank = {
                **cv_c_rank,
                **{
                    k: v for k, v in strats_c_rank.items()
                    if keep_dot(k, basket)
                    or keep_all
                },
            }
        return {
            **cv_keep,
            **cv_rank,
            **cv_agg,
            **cv_c_rank,
        }
    
    def build_base_page(
        self,
        project: engine.Project,
        SYMBOL: str,
    ):
        # TODO: possibly one per symbol?
        # then can just run the domain per symbol when needed for current baskets
        # with p006 often in regen, but not cv / rw

        cv = self.build_base_strategies(
            keep=None,
            basket=None,
            page_symbols = [SYMBOL],
            keep_all=True
        )

        # TODO: only if reset
        # cv = {}

        self = self.with_page(
            f"{project.name}-{SYMBOL}",
            type="daily",
            graph=True,
            t=Project_006
        )
        
        perf_objs = []
        rx_objs = []

        for symbol, bt in cv.items():
            perf_objs.append((
                assert_is(bt.perf).path,
                symbol
            ))
            rx_objs.append((
                assert_is(bt.rx()).path,
                symbol
            ))

        # self = self.with_objects_from_template(
        #     "variants",
        #     "perf_variants",
        #     rx_objs,
        #     t=Project_006,
        #     kws=dict(group="rx")
        # )
        self = self.with_objects_from_template(
            "perf",
            "perf_table",
            perf_objs,
            t=Project_006,
            kws=dict(group="perf")
        )
        self = self.with_objects_from_template(
            "rx_cum",
            "line_cum",
            rx_objs,
            t=Project_006,
            kws=dict(group="rx")
        )
        self = self.with_graph(
            engine.context.get(),
            t=Project_006,
        )

        return self
    
    def build_page(
        self,
        name: str,
        page: str, #  = {project}.{name}, bit redundant
        page_symbols: list[str],
        project: engine.Project,
    ):
    
        universe = {
            f"{symbol}{suff}": strat.cv(
                UNIVERSE[symbol],
                span=48,
                signal=sig,
                cap=strat.CapNotional(10.),
            )
            for symbol in page_symbols
            for suff, sig in SIGNALS.items()
            if keep_signal(
                name, symbol, sig
            )
        }
        baskets_agg = BASKETS[name]
        baskets_comb = BASKETS_COMB[name]

        prices: dict[
            str, engine.Daily
        ] = {
            sym: assert_is(
                bars.EqDailyEtf.new(
                    symbol=sym,
                    field="close",
                    adjust="price",
                    source="frd"
                )
                if ac == "ETF"
                else bars.FutDailyOrdinal.new(
                    symbol=sym,
                    ident = 1,
                    field="close",
                    adjust="raw",
                    source="frd"
                )
                if ac == "FUT"
                else bars.FxDailyPx.new(
                    symbol=sym,
                    # ident = 1,
                    field="close",
                    # adjust="raw",
                    source="frd"
                )
                if ac == "FX"
                else None,
                engine.Daily,
            )
            for (ac, sym) in [
                UNIVERSE[symbol]
                for symbol in page_symbols
            ]
        }
        
        cv = self.build_base_strategies(
            keep=list(universe.keys()),
            basket=name,
            page_symbols=page_symbols,
        )

        # --

        strats = strat.baskets.agg.build_baskets(
            baskets_agg,
            project,
            f"{name}-strat-agg",
            cv,
            prices,
            RANK_PARAMS,
            WEIGHT_PARAMS,
        )
        strats_comb = strat.baskets.comb.build_baskets(
            baskets_comb,
            project,
            f"{name}-strat-comb",
            cv,
            prices,
        )
        strats = {**strats, **strats_comb}

        # TODO: now extract the bso signal we want to use, eg. spx

        # and run as the signal for spx
        # and then basket with that and the pkf?

        # same loop model? but more general?

        # --

        self = self.with_page(
            page,
            type="daily",
            graph=True,
            t=Project_006
        )
        
        perf_objs = []
        rx_objs = []

        for symbol, bt in strats.items():
            perf_objs.append((
                assert_is(bt.perf).path,
                symbol
            ))
            rx_objs.append((
                assert_is(bt.rx()).path,
                symbol
            ))

        # self = self.with_objects_from_template(
        #     "variants",
        #     "perf_variants",
        #     rx_objs,
        #     t=Project_006,
        #     kws=dict(group="rx")
        # )
        self = self.with_objects_from_template(
            "perf",
            "perf_table",
            perf_objs,
            t=Project_006,
            kws=dict(group="perf")
        )
        self = self.with_objects_from_template(
            "rx_cum",
            "line_cum",
            rx_objs,
            t=Project_006,
            kws=dict(group="rx")
        )
        sig_objs: list[TTemplateObject] = [
            (
                bt.params.signal_path,
                symbol,
                {}
            )
            for symbol, bt in strats.items()
            if isinstance(bt.params.signal_path, str)
        ]
        sig_objs_full = sig_objs + sum([
            [
                (p, p)
                for p in bt.params.signal_path
                # TODO: save the params as well, so we can re-agg user side
            ]
            for symbol, bt in strats.items()
            if isinstance(bt.params.signal_path, tuple)
        ], [])
        scale_objs: list[TTemplateObject] = [
            ((
                bt.params.scale_path,
                symbol,
                {}
            ))
            for symbol, bt in strats.items()
            if isinstance(bt.params.scale_path, str)
        ]
        scale_objs_full = scale_objs + sum([
            [
                (p, p)
                for p in bt.params.scale_path
                # TODO: save the params as well, so we can re-agg user side
            ]
            for symbol, bt in strats.items()
            if isinstance(bt.params.scale_path, tuple)
        ], [])

        self = self.with_objects_from_template(
            "sigs",
            "line",
            sig_objs,
            t=Project_006,
            kws=dict(group="sig")
        )
        self = self.with_objects_from_template(
            "scales",
            "line",
            scale_objs,
            t=Project_006,
            kws=dict(group="sig")
        )
        
        # for k, weight_objs in weight_objs.items():
        #     self = self.with_objects_from_template(
        #         f"{k}_line",
        #         "line",
        #         weight_objs,
        #         t=Project_006,
        #         kws=dict(group="rx")
        #     )

        self = self.with_graph(
            engine.context.get(),
            t=Project_006,
        )
        return self

    @classmethod
    def load(
        cls,
        target: str | None = None,
        root: str = "./__local__/apps",
        test: bool = False,
        with_ui: bool = True,
        with_files: bool = True,
    ):
        """
        >>> root = "./__local__/test"
        >>> proj = Project_006.load(root, test=True).reset()
        >>> nav = proj.prepare_app()
        >>> nav[0]()
        >>> nav[1]()
        """

        WITH_UI = bool(os.environ.get("STREAMLIT"))
        # TODO: root, test etc. from env vars

        self = (
            Project_006(root=root)
            .with_config(explorer.ExplorerConfig(PROJECT))
        )
        project = engine.Project.new(PROJECT)

        for SYMBOL in UNIVERSE.keys():
            page = f"{PROJECT}-{SYMBOL}"
            if target is not None and page != target:
                log("INFO")(skipping=page)
                continue

            with engine.context.graph(
                page, t = strat.StrategyGraph
            ):
                project.set_storage_defaults([
                    "config",
                    "eq",
                    "fut",
                    "fx",
                    "cv",
                    "rw",
                    "p006"
                ])
                self = self.build_base_page(
                    project,
                    SYMBOL,
                )

        for i_page, (name, page_symbols) in enumerate(
            UNIVERSES.items()
        ):
            page = f"{PROJECT}-{name}"

            if target is not None and page != target:
                log("INFO")(skipping=page)
                continue

            log("INFO")( loading=page)

            with engine.context.graph(
                page, t = strat.StrategyGraph
            ):
                project.set_storage_defaults([
                    "config",
                    "eq",
                    "fut",
                    "fx",
                    "cv",
                    "rw",
                    "p006"
                ])

                self = self.build_page(
                    name,
                    page,
                    page_symbols,
                    project,
                )
                
                # o = engine.context.get().get(
                #     "p006-osc-strat-w-cap-1",
                #     agg.Collection
                # )
                # raise ValueError(
                #     o.get_snap(engine.Snap)
                # )

        return self


SECTORS = dict(
    XLF = ("ETF", "XLF"),
    XLE = ("ETF", "XLE"),
    XLY = ("ETF", "XLY"),
    XLP = ("ETF", "XLP"),
    XLI = ("ETF", "XLI"),
    XLB = ("ETF", "XLB"),
    XLU = ("ETF", "XLU"),
    XLV = ("ETF", "XLV"),
    XLK = ("ETF", "XLK"),
)
COMDS = dict(
    GLD = ("ETF", "GLD"),
    DBC = ("ETF", "DBC"), # commod
)
# usdjpy = 153
FX = dict(
    USDJPY= ("FX", "USDJPY"),
    USDCHF= ("FX", "USDCHF"),
    EURUSD= ("FX", "EURUSD"),
)

BONDS = dict(
    TLT = ("ETF", "TLT"),
    IEF = ("ETF", "IEF"), # pretty similar to tlt
    BND = ("ETF", "BND"), # total bond, similar to ief
    SCHR = ("ETF", "SCHR"),
    SCHO = ("ETF", "SCHO"),
    GOVT = ("ETF", "GOVT"),
    # JPST = ("ETF", "JPST"), # massive returns, odd?
)
CREDIT = dict(
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
@dataclass(frozen=True)
class SignalOsc(strat.SignalOsc):
    pass

@dataclass(frozen=True)
class SignalSharpe(strat.SignalOsc):
    pass

SIGNALS = {
    "": None,
    **{
        f"_osc_{sp}_{var}": SignalOsc(
            int(2 ** sp), 
            int((2 ** sp)*m),
            vol=48,
            constraint=var
        )
        for sp in [
            4, 5, 6, 7, 8
        ]
        for m in [3.]
        for var in ["ls", "bls", "blo", "bso"]
    },
    **{
        f"_shrp_{sp}_{var}": SignalSharpe(
            int(2 ** sp), 
            slow=None,
            vol=48,
            constraint=var
        )
        for sp in [
            4, 5, 6, 7, 8
        ]
        for var in ["ls", "bls", "blo", "bso"]
    },
    **{
        f"_wm_{sp}_{int(threshold)}{reen_k}_{var}"
        .replace("-", "n")
        .replace("_None", ""): strat.SignalWmThold(
            int(2 ** sp) * 2, 
            "len",
            threshold=threshold,
            reentry=reentry,
            constraint=var
        )
        for sp in [
            4, 5, 6, 7, 8
        ]
        for threshold in [-2.]
        for reen_k, reentry in {
            # "_re": True, # TODO: fix
            "": None
        }.items()
        for var in ["ls", "bls", "blo", "bso"]
    }
}

RANK_WINDOW = round(2 / (30 + 1), 3)
RANK_SMOOTH = round(2 / (180 + 1), 3)
RANK_SAMPLE = 5

RANK_PARAMS = [
    dict(
        metric=metric,
        sample=int(sample),
        window=round(window, 3),
        smooth=round(smooth, 3),
        norm=method,
        top=top
    )
    for top in [
        1, 
        2, 
        3, 
        5
    ]
    for metric in [
        "sharpe",
        # "netsharpe",
        "calmar",
        # "netcalmar",
        # "sharpe_calmar",
        # "netsharpe_netcalmar"
    ]
    for sample in [
        RANK_SAMPLE,
        # 10,
        # 30
    ]
    for window in [
        RANK_WINDOW,
        # 0, # ie. expanding
        # 2 / (90 + 1),
        # 2 / (180 + 1),
        # 2 / (360 + 1),
    ]
    for smooth in [
        # 2 / (60 + 1),
        # 2 / (90 + 1),
        RANK_SMOOTH,
    ]
    for method in [
        "eq",
        # "pow_rank_15",
        "pr2",
        # "pow_rank_3",
        # "sm",
    ]
    if (
        top > 1 
    ) or (
        top == 1
        and method == "eq"
    )
]

WEIGHT_PARAMS = [
    dict()
]

# TODO: more aggressive equity basket
# eg. add XLK (for instance)
# or just more diverisifed - genuinely different rx streams to expose to? (XLF, XLE, etc.)

# then try replacing the spx in the rp with the eq

T_KEYS = [
    # None,
    SignalOsc,
    SignalSharpe, # doesnt work for eq?
    strat.SignalWmThold
]
T_NAMES = {
    None: "all",
    SignalOsc: "osc",
    SignalSharpe: "shrp",
    strat.SignalWmThold: "wm"
}
SIG_TYPES = {
    None: "",
    "osc": SignalOsc,
    "shrp": SignalSharpe,
    "wm": strat.SignalWmThold,
}

W_POST = False
W_PRE = False

UNIVERSE = dict(
    # ES = ("FUT", "ES"),
    # TN = ("FUT", "TN"),
    # GC = ("FUT", "GC"),
    SPY = ("ETF", "SPY"),
    **SECTORS,
    **BONDS,
    **CREDIT,
    **COMDS,
    **FX,
)

def keep_signal(
    basket: str, # page name
    symbol: str,
    sig: str | None,
):
    if sig is None:
        return True # always keep LO
    elif basket not in W_SIGNALS:
        return False
    basket_sigs = W_SIGNALS[basket]
    if symbol in basket_sigs:
        return type(sig) in basket_sigs[symbol]
    return False

def keep_dot(
    k: str,
    basket: str | None = None,
):
    if basket is None:
        return False
    _, symbol, k, res = k.split("-", maxsplit=3)
    basket_sigs = W_SIGNALS[basket]
    if symbol in basket_sigs:
        if k == "sp":
            return True
        try:
            sp = int(k)
            return True
        except:
            pass
        if k in SIG_TYPES:
            if res.endswith("2"):
                return SIG_TYPES[k] in basket_sigs[symbol]
            return False
        return False

def combine_dicts(ds: list[dict]):
    d, *ds = ds
    for dd in ds:
        d = {**d, **dd}
    return d


def rank_key(
    prefix: str,
    metric: str,
    top: int,
    suffix: str = "",
    norm: str = "",
    sample: int=RANK_SAMPLE,
    window: float=RANK_WINDOW,
    smooth: float=RANK_SMOOTH,
):
    assert norm != "", (
        prefix,
        metric,
        top,
        suffix,
        norm,
    )
    w_key = "_".join(map(str, [
        metric,
        sample,
        window,
        smooth,
        norm,
        top,
    ]))
    return f"{prefix}-{w_key}{suffix}"

@overload
def rank_params(
    metric: str,
    top: int,
    norm: str,
) -> dict: ...

@overload
def rank_params(
    metric: str,
    top: list[int],
    norm: str,
) -> list[dict]: ...

def rank_params(
    metric: str,
    top: int | list[int],
    norm: str,
):
    if isinstance(top, list):
        return [
            dict(
                metric=metric,
                sample=RANK_SAMPLE,
                window=RANK_WINDOW,
                smooth=RANK_SMOOTH,
                norm=norm,
                top=t
            )
            for t in top
        ]
    return dict(
        metric=metric,
        sample=RANK_SAMPLE,
        window=RANK_WINDOW,
        smooth=RANK_SMOOTH,
        norm=norm,
        top=top
    )
    
def rank_eq(*args, **kwargs):
    return rank_key(*args, norm="eq", **kwargs)

def rank_pr2(*args, **kwargs):
    return rank_key(*args, norm="pr2", **kwargs)

# TODO: calmar for equities, sharpe x-asset?
# TODO: capping presumably reduces diversification if eg. bonds are more likely to scale up to max not?
# or reduce return its spx thats the issue - which is it?

OSC_MAP = {
    "SPY": rank_pr2("rw-SPY-osc-blo", "calmar", 2),
    "GLD": rank_pr2("rw-GLD-osc-blo", "calmar", 2),
    "DBC": rank_pr2("rw-DBC-osc-blo", "calmar", 2),
    
    "USDJPY": rank_pr2("rw-USDJPY-osc-blo", "calmar", 2),
    "USDCHF": rank_pr2("rw-USDCHF-osc-blo", "calmar", 2),
    "EURUSD": rank_pr2("rw-EURUSD-osc-blo", "calmar", 2),

    "XLF": rank_pr2("rw-XLF-osc-blo", "calmar", 2),
    "XLE": rank_pr2("rw-XLE-osc-blo", "calmar", 2),
    "XLY": rank_pr2("rw-XLY-osc-blo", "calmar", 2),
    "XLP": rank_pr2("rw-XLP-osc-blo", "calmar", 2),
    "XLB": rank_pr2("rw-XLB-osc-blo", "calmar", 2),
    "XLU": rank_pr2("rw-XLU-osc-blo", "calmar", 2),
    "XLV": rank_pr2("rw-XLV-osc-blo", "calmar", 2),
    "XLK": rank_pr2("rw-XLK-osc-blo", "calmar", 2),

    # TODO: osc1, osc 2 etc. for diff n?
}
SHRP_MAP = {
    k: v.replace("osc", "shrp")
    for k, v in OSC_MAP.items()
}

def osc_map(s: str):
    return OSC_MAP.get(s, s)
def shrp_map(s: str):
    return SHRP_MAP.get(s, s)

def map_symbol(
    sym: str,
    sym_map: dict[str, str],
    page: str,
    project: str = "p006"
):
    if sym not in sym_map:
        return sym
    return f"{project}-{page}_{sym_map[sym]}"



SPY_TLT = ["SPY", "TLT"]
PKF = ["XLP", "XLK", "XLF"]

# TODO: osc on pkf? on spy_tlt? etc.

# then once we have a good osc system, try substitute the short leg with spy


OSC_SPEEDS = [
    4, 5, 6, 7, 8
]

def osc_params(
    speeds: list[int] = OSC_SPEEDS,
    vars: list[str] = [
        "blo",
        # "bso"
    ]
):
    return tuple((
        dict(signal=SignalOsc(
            int(2 ** sp), 
            int((2 ** sp)*m),
            vol=48,
            constraint=var
        ))
        for sp in speeds
        for m in [3.]
        for var in vars
    ))

def osc_key(
    prefix: str,
    speed: int,
    variant: str,
    suffix: str = ""
):
    fast = int(2 ** speed)
    slow = int(fast * 3)
    vol = 48
    return f"{prefix}-osc-{fast}-{slow}-{vol}-tanh-{variant}{suffix}"

BASKETS: dict[str, dict[
    str,
    strat.baskets.agg.RankBasket
    | strat.baskets.agg.WeightBasket
]] = {
    "ex": {
        "agg_rank": agg_rank_basket(
            "SPY",
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
    }
}
BASKETS_COMB: dict[str, dict[
    str,
    strat.baskets.comb.WeightBasket
    | strat.baskets.comb.RankBasket
]] = {
    "ex": {
        "agg_comb": comb_rank_basket(
            "SPY",
            "TLT",
            params=strat.baskets.comb.PfoWeightParams(
                metrics=("sharpe", "calmar"),
                samples=RANK_SAMPLE,
                windows=RANK_WINDOW,
                smoothings=RANK_SMOOTH,
                tops=(1, 2),
                norms="eq"
            ),
            vol_norm=256,
            not_cap=10.,
        ),
    }
}
BASKETS_FX: dict[str, dict[
    str,
    strat.baskets.agg.RankBasket
    | strat.baskets.agg.WeightBasket
]] = {
    # actually want to sell jpy?

    "def": {
        "fx": agg_rank_basket(
            "USDJPY",
            "USDCHF",
            "EURUSD",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "fx_mo": agg_weight_basket(
            rank_eq("fx", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "fx_rw": agg_rank_basket(*[
            osc_key("fx_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),
        "fx_o": agg_rank_basket(
            osc_map("USDJPY"),
            osc_map("USDCHF"),
            osc_map("EURUSD"),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            # vol_norm=256,
            not_cap=10.,
        ),
        "def": agg_rank_basket(
            rank_eq("fx_rw", "sharpe", 2),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "def_mo": agg_weight_basket(
            rank_eq("def", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "def_rw": agg_rank_basket(*[
            osc_key("def_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),

        # def_rw is pretty good
        # xp_def therefore worth pursuing

        "def_o": agg_rank_basket(
            rank_eq("fx_o", "sharpe", 2),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),

        # --
        "xp_osc": agg_rank_basket(
            osc_map("SPY"),
            osc_map("GLD"),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        "xp_def": agg_rank_basket(
            osc_map("SPY"),
            rank_eq("def_rw", "sharpe", 2),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        "xp_fx": agg_rank_basket(
            osc_map("SPY"),
            rank_eq("fx_rw", "sharpe", 2),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        
        "xp_def_cmd": agg_rank_basket(
            osc_map("SPY"),
            rank_eq("def_rw", "sharpe", 2),
            rank_eq("fx_rw", "sharpe", 2),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
                rank_params("sharpe", 3, "eq"),
            ),
            not_cap=10.,
        ),
        # --

        # xp fx isnt awful?
        # possibly smoothign out 18/19?
    },
}

BASKETS_DEF: dict[str, dict[
    str, 
    strat.baskets.agg.RankBasket
    | strat.baskets.agg.WeightBasket
]] = {
    "def": {
        "cmd": agg_rank_basket(
            "GLD",
            "DBC",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "cmd_mo": agg_weight_basket(
            rank_eq("cmd", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "cmd_rw": agg_rank_basket(*[
            osc_key("cmd_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),
        "cmd_o": agg_rank_basket(
            osc_map("GLD"),
            osc_map("DBC"),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            # vol_norm=256,
            not_cap=10.,
        ),
        "def": agg_rank_basket(
            "GLD",
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "def_mo": agg_weight_basket(
            rank_eq("def", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "def_rw": agg_rank_basket(*[
            osc_key("def_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),

        # def_rw is pretty good
        # xp_def therefore worth pursuing

        "def_o": agg_rank_basket(
            osc_map("GLD"),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        # --
        "xp_osc": agg_rank_basket(
            osc_map("SPY"),
            osc_map("GLD"),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        "xp_def": agg_rank_basket(
            osc_map("SPY"),
            rank_eq("def_rw", "sharpe", 2),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        "xp_cmd": agg_rank_basket(
            osc_map("SPY"),
            rank_eq("cmd_rw", "sharpe", 2),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        
        "xp_def_cmd": agg_rank_basket(
            osc_map("SPY"),
            rank_eq("def_rw", "sharpe", 2),
            rank_eq("cmd_rw", "sharpe", 2),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
                rank_params("sharpe", 3, "eq"),
            ),
            not_cap=10.,
        ),
        # --

        # TLT vs best of DBC and GLD

        # generalised inflation vs defensive basket?
        # inflation = dbc, gld

        # dbc poss helped with 2022
        # poss better poss financial crisis (10 etc.)


        # TLT vs best of DBC and GLD
    },
}

# good = xp_osc_cap
# best = sec_xpo_rw (eq (cap))
BASKET_XRP: dict[str, dict[ 
    str, 
    strat.baskets.agg.RankBasket
    | strat.baskets.agg.WeightBasket
]] = {
    "osc": {
        # "pkf": agg_rank_basket(
        #     *PKF,
        #     params = (
        #         rank_params("sharpe", 2, "eq"),
        #         rank_params("calmar", 2, "eq"),
        #     ),
        #     vol_norm=256,
        #     not_cap=10.,
        # ),
        # "pkf_mo": agg_weight_basket(
        #     rank_eq("pkf", "sharpe", 2, "_cap"),
        #     params=osc_params(),
        #     not_cap=10.,
        #     # vol_norm=256,
        # ),
        # "pkf_rw": agg_rank_basket(*[
        #     osc_key("pkf_mo", sp, "blo")
        #     for sp in OSC_SPEEDS
        # ], params=tuple(rank_params(
        #     "sharpe",
        #     [1, 2, 3],
        #     "eq"
        # ))),
        "xp_osc": agg_rank_basket(
            osc_map("SPY"),
            osc_map("GLD"),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        "xp": agg_rank_basket(
            "SPY",
            "GLD",
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "xp_mo": agg_weight_basket(
            rank_eq("xp", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "xp_rw": agg_rank_basket(*[
            osc_key("xp_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),
        "rp": agg_rank_basket(
            *SPY_TLT,
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "rp_mo": agg_weight_basket(
            rank_eq("rp", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "rp_rw": agg_rank_basket(*[
            osc_key("rp_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),
        "sec": agg_rank_basket(
            *SECTORS.keys(),
            params = (
                rank_params("sharpe", 3, "eq"),
                rank_params("sharpe", 5, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "sec_mo": agg_weight_basket(
            rank_eq("sec", "sharpe", 5, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "sec_rw": agg_rank_basket(*[
            osc_key("sec_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),
        "sec_rw_p": agg_rank_basket(
            rank_eq("sec_rw", "sharpe", 2),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        "sec_rw_xp": agg_rank_basket(
            rank_eq("sec_rw", "sharpe", 2),
            osc_map("GLD"),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            not_cap=10.,
        ),
        
        "sec_p": agg_rank_basket(
            rank_eq("sec", "sharpe", 5, "_cap"),
            "TLT",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "sec_p_mo": agg_weight_basket(
            rank_eq("sec_p", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "sec_p_rw": agg_rank_basket(*[
            osc_key("sec_p_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),
        
        "sec_xp": agg_rank_basket(
            rank_eq("sec", "sharpe", 5, "_cap"),
            "TLT",
            "GLD",
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        "sec_xp_mo": agg_weight_basket(
            rank_eq("sec_xp", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "sec_xp_rw": agg_rank_basket(*[
            osc_key("sec_xp_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),

        "sec_xpo": agg_rank_basket(
            rank_eq("sec", "sharpe", 5, "_cap"),
            "TLT",
            osc_map("GLD"),
            params = (
                rank_params("sharpe", 1, "eq"),
                rank_params("calmar", 1, "eq"),
                rank_params("sharpe", 2, "eq"),
            ),
            vol_norm=256,
            not_cap=10.,
        ),
        
        "sec_xpo_mo": agg_weight_basket(
            rank_eq("sec_xpo", "sharpe", 2, "_cap"),
            params=osc_params(),
            not_cap=10.,
            # vol_norm=256,
        ),
        "sec_xpo_rw": agg_rank_basket(*[
            osc_key("sec_xpo_mo", sp, "blo")
            for sp in OSC_SPEEDS
        ], params=tuple(rank_params(
            "sharpe",
            [1, 2, 3],
            "eq"
        ))),
        
        # TODO: then look at wm/shrp/osc signals on the tlt (/scho etc.)


        # TODO: does the reverse signal eg. mo on spy-tlt vol weight (or best weight?) better predict spx alone return, than spx mo itself?
    },
}

# NOTE: this is effectively the universe: but specifies both what symbols and what signal classes to load
W_SIGNALS = {
    "ex": {
        "SPY": [SignalOsc,],
        "TLT": [],
    },
    "def": {
        **{
            k: []
            for k in [
                "TLT",
                # "IEF",
                # "SCHR",
                "SCHO",
            ]
        },
        **{
            k: [SignalOsc]
            for k in [
                "SPY",
                "USDJPY",
                "USDCHF",
                "EURUSD",
                "GLD"
            ]
        }
    },
    "cmd": {
        **{
            k: []
            for k in [
                "TLT",
                # "IEF",
                # "SCHR",
                "SCHO",
            ]
        },
        **{
            k: [SignalOsc]
            for k in [
                "SPY",
                "DBC",
                "GLD"
            ]
        }
    },
    "osc": {
        **{
            k: [SignalOsc]
            for k in [
                "SPY",
                "GLD",
            ]
        },
        **{
            k: []
            for k in [
                "TLT",
            ] + list(SECTORS.keys())
        }
    },
    "pkf": {
        # k: [
        #     SignalOsc, SignalSharpe
        # ]
        # for k in ["SPY", "XLP", "XLF", "XLK"]
    },
    "rp": {
    #     "SPY": [SignalOsc,],
    #     "TLT": [],
    #     "GLD": [SignalOsc,],
    },
    "credit": {
        # k: [SignalOsc] for k in ["SPY"] + list(
        #     CREDIT.keys()
        # )
    },
    "sectors": {
        # k: [SignalOsc] for k in ["SPY"] + list(
        #     SECTORS.keys()
        # )
    },
    "bonds": {
        # k: [SignalOsc] for k in ["SPY"] + list(
        #     BONDS.keys()
        # )
    },
}
UNIVERSES = {
    k: list(k_sigs.keys())
    for k, k_sigs in W_SIGNALS.items()
    if k in BASKETS
}
W_SIGNALS = {
    k: k_sigs for k, k_sigs in W_SIGNALS.items()
    if k in BASKETS
}

GROUP_TYPES = {
    "div": "div", # diversified 
    "osc": "osc",
    # mo, sig, etc.
}

PROJECTS[PROJECT] = Project_006

__all__ = [
    "Project_006",
]