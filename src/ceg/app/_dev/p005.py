from typing import cast, TYPE_CHECKING, ClassVar
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
from ...vis import explorer
from ...objs.dev import bars
from ...objs.dev import pfo
from ...objs.dev import agg
from ...objs.dev import calc

from ...objs.dev import strat

from .core import PROJECTS

logger, log = apis.logs.logger(
    __file__, streams=dict(formatter=True)
)
PROJECT = "p005"

@dataclass(frozen=True)
class Project_005(strat.StratExplorer):
    PROJECT: ClassVar[str] = PROJECT
    
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
            Project_005(root=root)
            .with_config(explorer.ExplorerConfig(PROJECT))
        )
        for i_page, (name, page_baskets) in enumerate(
            BASKETS.items()
        ):
            log("INFO")( building_page=name)

            universes: dict[str, pfo.ContainerDaily] = {} univ, universes = (
                strat.StratDomain.new()
                .with_univ(
                    self.univ_name(name),
                    depth=0,
                    universes=universes,
                )
            )
            eng = univ.get_parent()

            baskets = {
                f"basket-{i}": {
                    f"{symbol}{suff}": strat.Strategy.cv(
                        UNIVERSE[symbol],
                        span=48,
                        signal=sig,
                        cap=strat.CapNotional(10.),
                    )
                    for symbol in symbols
                    for suff, sig in SIGNALS.items()

                    # TODO: if symbol has signal, eg. equities dont seem to like sharpe signal
                }
                for i, symbols in enumerate(page_baskets)
            }
            basket_symbols = {
                basket: list(strats.keys())
                for basket, strats in baskets.items()
            }
            prices: dict[str, dict[str, engine.Daily]] = {
                "_".join(symbols): {
                    sym: assert_is(
                        bars.EqDailyEtf.new(
                            eng,
                            symbol=sym,
                            field="close",
                            adjust="price",
                            source="frd"
                        )
                        if ac == "ETF"
                        else bars.FutDailyOrdinal.new(
                            eng,
                            symbol=sym,
                            ident = 1,
                            field="close",
                            adjust="raw",
                            source="frd"
                        )
                        if ac == "FUT"
                        else None,
                        engine.Daily,
                    )
                    for (ac, sym), symbol in zip(
                        map(lambda s: UNIVERSE[s], symbols),
                        symbols,
                    )
                }
                for symbols in page_baskets
            }

            bt_no_norm = {}

            rx_lo = {}

            rx_indices = {}

            bt_pre_norm = {}
            bt_post_norm = {}

            for basket, strats in baskets.items():

                for symbol, params in strats.items():
                    if symbol in bt_no_norm:
                        continue
                    univ, bt = domain.with_backtest(
                        univ, params,
                    )
                    bt_no_norm[symbol] = bt
                    if params.signal is None:
                        rx_lo[params.asset] = (
                            bt.assume_res().rx,
                            params.label()
                        )
                    
                        rx_indices[params] = (
                            len(rx_indices),
                            bt,
                            assert_is(
                                params.asset,
                                strat.AssetConcrete
                            ).symbol,
                        )
            
            # TODO: universes as own type
            if W_POST:
                univ, universes = universes.by_depth(
                    self.univ_name(name),
                    depth=1,
                    universes=universes,
                )

                for basket, strats in baskets.items():

                    for symbol, params in strats.items():
                        if symbol in bt_post_norm:
                            continue
                        elif params.signal is None:
                            # bt_post_norm[symbol] = bt
                            continue
                        params_norm = strat.rescale_cv(
                            params,
                            bt_no_norm[symbol],
                        )
                        univ, bt = domain.with_backtest(
                            univ, params_norm,
                        )
                        bt_post_norm[symbol] = bt
            univ, universes = universes.by_depth(
                self.univ_name(name),
                depth=2,
                universes=universes,
            )

            for basket, strats in baskets.items():

                for symbol, params in strats.items():
                    if symbol in bt_pre_norm:
                        continue
                    elif params.signal is None:
                        # bt_pre_norm[symbol] = bt
                        continue
                    
                    params_norm = replace(
                        params,
                        asset=strat.AssetSynthetic(
                            *rx_lo[params.asset],
                        ),
                        scale=None,
                        cap=None,
                    )
                    univ, bt = domain.with_backtest(
                        univ, params_norm,
                    )

                    bt_pre_norm[symbol] = bt
                    rx_indices[params_norm] = (
                        len(rx_indices),
                        bt, 
                        assert_is(
                            params.asset,
                            strat.AssetConcrete
                        ).symbol,
                    )
            univ, universes = universes.by_depth(
                self.univ_name(name),
                depth=3,
                universes=universes,
            )

            rx_panel_bts = [
                bt for _, bt, _ in sorted(
                    rx_indices.values(),
                    key=lambda _bt_: _bt_[0]
                )
            ]
            
            rx_panel: strat.RxPanel rx_panel = strat.rx_panel(
                [
                    context.get().get(
                        bt.assume_res().rx, 
                        engine.Daily
                    )
                    for bt in rx_panel_bts
                ],
                name,
                f"0"
            )

            rx_i_asset_sig = {}

            for params, (i, _, sym) in rx_indices.items():
                for t in T_KEYS:
                    k = (sym, t)
                    if k not in rx_i_asset_sig:
                        rx_i_asset_sig[k] = []
                    if params.signal is None:
                        continue
                    if t is None:
                        rx_i_asset_sig[k].append(i)
                    elif isinstance(params.signal, t):
                        rx_i_asset_sig[k].append(i)

            i_rx_to_params = {
                i: params
                for params, (i, bt, sym) 
                in rx_indices.items()
            }

            (
                univ,
                bt_asset_sig, 
                weight_objs
            ) = strat.build_baskets(
                univ,
                self.univ_name(name),
                rx_panel,
                rx_panel_bts,
                # i_rx_to_params,
                rx_i_asset_sig,
                T_NAMES,
                WEIGHT_PARAMS,
            )
            univ, universes = universes.by_depth(
                self.univ_name(name),
                depth=4,
                universes=universes,
            )

            mvar_bts = list(bt_asset_sig.values()) mvar_panel = strat.rx_panel(
                [
                    context.get().get(
                        bt.assume_res().rx,
                        engine.Daily
                    )
                    for bt in mvar_bts
                ],
                name,
                f"1"
            )

            mvar_sym_baskets = {}
            mvar_params = {}

            for i, (
                (sym, t, w_params), bt
            ) in enumerate(bt_asset_sig.items()):
                if t is None:
                    continue

                group = (sym, "final")
                if group not in mvar_sym_baskets:
                    mvar_sym_baskets[group] = []

                mvar_sym_baskets[group].append(i)
                mvar_params[i] = bt.params

            (
                univ,
                bt_mvar_final, 
                mvar_weight_objs,
            ) = strat.build_baskets(
                univ,
                self.univ_name(name),
                mvar_panel,
                # mvar_params,
                mvar_bts,
                mvar_sym_baskets,
                {"final": "final"},
                WEIGHTS_FINAL,
            )

            # TODO: same deal again, but agg over the weight methodologies per symbol
                    
            # TODO: extra depth, build basket returns

            self = self.with_page(
                self.page_name(i_page, name),
                type="daily",
                graph=True,
                t=Project_005
            )
            
            bt_objs = []
            rx_objs = []

            perf = dict(group="perf")

            for symbol, bt in bt_no_norm.items():
                if bt.params.signal is not None:
                    continue
                res = bt.assume_res()
                label = f"{symbol}-cv"
                bt_objs.append((res.perf, label))
                rx_objs.append((res.rx, label))

            if W_PRE:
                for symbol, bt in bt_pre_norm.items():
                    if bt.params.signal is None:
                        continue
                    res = bt.assume_res()
                    label = f"{symbol}-pre"
                    bt_objs.append((res.perf, label))
                    rx_objs.append((res.rx, label))

            # if W_POST:
            #     for symbol, bt in bt_post_norm.items():
            #         if bt.params.signal is None:
            #             continue
            #         res = bt.assume_res()
            #         label = f"{symbol}-post"
            #         bt_objs.append((res.perf, label))
            #         rx_objs.append((res.rx, label))
            
            for (
                sym, sig, w_params
            ), bt in bt_asset_sig.items():
                res = bt.assume_res()
                label = bt.params.name
                bt_objs.append((res.perf, label))
                rx_objs.append((res.rx, label))

            for (
                sym, sig, w_params
            ), bt in bt_mvar_final.items():
                res = bt.assume_res()
                label = bt.params.name
                
                bt_objs.append((res.perf, label))
                rx_objs.append((res.rx, label))

            self = self.with_objects_from_template(
                "variants",
                "perf_variants",
                rx_objs,
                t=Project_005,
                kws=dict(group="rx")
            )
            self = self.with_objects_from_template(
                "perf",
                "perf_table",
                bt_objs,
                t=Project_005,
                kws=dict(group="perf")
            )
            self = self.with_objects_from_template(
                "rx_cum",
                "line_cum",
                rx_objs,
                t=Project_005,
                kws=dict(group="rx")
            )
            for k, weight_objs in weight_objs.items():
                self = self.with_objects_from_template(
                    f"{k}_line",
                    "line",
                    weight_objs,
                    t=Project_005,
                    kws=dict(group="rx")
                )

            self = self.with_graph(
                t=Project_005,
                name=self.page_name(i_page, name)
            )
        
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
    CMD = ("ETF", "DBC"), # commod
)
BONDS = dict(
    TLT = ("ETF", "TLT"),
    IEF = ("ETF", "IEF"), # pretty similar to tlt
    BND = ("ETF", "BND"), # total bond, similar to ief
    TMT = ("ETF", "SCHR"),
    TST = ("ETF", "SCHO"),
    GOV = ("ETF", "GOVT"),
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
            sp, 
            int(sp*m),
            vol=48,
            constraint=var
        )
        for sp in [
            16, 32, 64, 128, 256
        ]
        for m in [3.]
        for var in ["blo"]
    },
    **{
        f"_shrp_{sp}_{var}": SignalSharpe(
            sp, 
            slow=None,
            vol=48,
            constraint=var
        )
        for sp in [
            16, 32, 64, 128, 256
        ]
        for var in ["blo"]
    },
    **{
        f"_wm_{sp}_{int(threshold)}{reen_k}_{var}"
        .replace("-", "n")
        .replace("_None", ""): strat.SignalWmThold(
            sp, 
            "len",
            threshold=threshold,
            reentry=reentry,
            constraint=var
        )
        for sp in [
            32, 64, 128, 256, 512
        ]
        for threshold in [-2.]
        for reen_k, reentry in {
            # "_re": True, # TODO: fix
            "": None
        }.items()
        for var in ["blo"]
    }
}

WEIGHTS_FINAL = [(
    metric,
    5,
    round(2 / (30 + 1), 3),
    round(2 / (180 + 1), 3),
    agg,
    3,
) for metric in [
    "sharpe", "calmar"
] for agg in [
    "sm", "pow_rank_2"
]]

WEIGHT_PARAMS = [
    (
        metric,
        round(sample, 2),
        round(window, 3),
        round(smooth, 3),
        method,
        top
    )
    for top in [
        3
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
        5,
        # 10,
        # 30
    ]
    for window in [
        2 / (30 + 1),
        # 0, # ie. expanding
        # 2 / (90 + 1),
        # 2 / (180 + 1),
        # 2 / (360 + 1),
    ]
    for smooth in [
        # 2 / (60 + 1),
        # 2 / (90 + 1),
        2 / (180 + 1),
    ]
    for method in [
        # "pow_rank_15",
        "pow_rank_2",
        # "pow_rank_3",
        "sm",
    ]
]
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
)
BASKETS = {
    "spy": [
        ["SPY"],
    ],
    "forever": [
        ["SPY", "TLT", "LQD", "GLD", "XLP", "XLU"]
    ],
    # "all": [
    #     list(UNIVERSE.keys()),
    # ],
    # "sectors": [
    #     ["SPY"] + list(SECTORS.keys())
    # ],
    # "bonds": [
    #     ["SPY"] + list(BONDS.keys())
    # ],
    # "credit": [
    #     ["SPY"] + list(CREDIT.keys())
    # ],
}

# TODO: interest rate momentum rather than on the etf itself for TLT etc?


# TODO: is there a correlation between aggressive vs defensive
# and slow vs faster signals working
# and osc vs hwm?

# 32 short not that useful for any



# TODO: fix wm re entry


# TODO: maybe remove the per strat cap 
# and reinstitute at portfolio level?

# eg. if that's under-vol-ing us

# or re-vol and then cap, try both


# TODO: try weighted average of weight and signal, then binary, then pass back to the signal weights (to make binary again)
# as a new signal

# TODO: compare to just re-vol scale and re-cap - similar effect but will have smoother weights?


# TODO: try instead of softmax, norm relative to eg. some max theoretical sharpe




# TODO: try calmar ranking, skew?

# TODO: change the weight plot to stack (cum vertically not in time, to see better)

# TODO: regressions eg. of the weight (relative to equal weight, say) to forward returns (of strat)?

@st.cache(streamlit=False)
def load(kw: tuple):
    return Project_005.load(**{
        k: v for k, v in kw
        if k in [
            "root", "with_ui", "with_files", "test"
        ]
    })

PROJECTS[PROJECT] = load

__all__ = [
    "Project_005",
]