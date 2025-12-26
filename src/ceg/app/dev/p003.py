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
from .p001 import *

PROJECT = "p003"

@dataclass(frozen=True)
class Project_003(strat.StratProject):
    pass
    
def eco(
    symbol: str,
    method: str,
    **kwargs
):
    return strat.SignalEco(symbol, method, **kwargs)
        
def load_p003(
    root: str = "./__local__/apps", test: bool = False
):
    """
    >>> root = "./__local__/test"
    >>> proj = load_p003(root=root, test=True).reset()
    >>> nav = explorer.prepare_app(PROJECT, root, proj.domains)
    >>> nav[0]()
    >>> nav[1]()
    """
    project = (
        Project_003(root=root)
        .with_config(explorer.ExplorerConfig(PROJECT))
    )
    
    speeds = [
        16,
        # 8, 
        32, 
        # 128, 
        512,
        # 1024, 
        2048
    ]

    f_sim_spread = lambda symb_sc, ecos, tf_acts: [
        ("ETF", symb, None)
        for symb in symb_sc.keys()
    ] + [
        ("ETF", symb, eco(
            ec,
            tf,
            speed=s,
            scale=sc,
            activation=act

            # TODO: log?
        ))
        for s in speeds
        for symb, sc in symb_sc.items()
        for ec in ecos
        for tf, act in tf_acts.items()
    ]

    spread_acts = {
        "ez": None,
        # "ezw": None,
        # "zwdemin": None,
        # all quite similar
    }

    SPY_SPREAD = f_sim_spread(
        {
            "SPY": -1.,
        },
        [
            "VIX",
            "D_US_BAA",
            "D_US_BAA_FF",
            "US_IG_OAS",
            "US_HY_OAS",
        ],
        spread_acts
    )
    # SPY, ez:
    # cv 48,256, sign=-1.:
    # 2048 span ez IG OAS
    # 2048 span ez BAA FF (and 32 span)
    TLT_SPREAD = f_sim_spread(
        {
            "TLT": 1.,
        },
        [
            "VIX",
            "D_US_BAA",
            "D_US_BAA_FF",
            "US_IG_OAS",
            "US_HY_OAS",
        ],
        spread_acts
    )
    # TLT, ez:
    # cv 48,256, sign=+1.:
    # 32 span ez baa spread (or HY OAS)

    EUR_SPREAD = f_sim_spread(
        {
            "VGK": -1.,
            "EZU": -1.,
        },
        [
            "EUR_HY_OAS"
        ],
        spread_acts
        # ez: 32 bad, 512 good, 2048 okay
        # zdewmax bad
    )
    EM_SPREAD = f_sim_spread(
        {
            "IEMG": -1.,
            "VWO": -1.,
            "EEM": -1.,
        },
        [
            "EM_IG_OAS",
            "EM_HY_OAS",
        ],
        spread_acts
        # ez: IG better than HY, fast better than slow
        # but none fantastic
    )
    EM_ASIA_SPREAD = f_sim_spread(
        {
            "EEMA": -1.,
            # "CEMA": -1.,
            # "CEA1": -1.,
        },
        [
            "EM_IG_OAS",
            "ASIA_EM_IG_OAS",
            # ez: EM_IG MUCH better than asia, asia terrible
            # none fantastic 
        ],
        spread_acts
    )
    EM_LATAM_SPREAD = f_sim_spread(
        {
            "ILF": -1.,
        },
        [
            "EM_IG_OAS",
            "LATAM_EM_IG_OAS",
        ],
        spread_acts
        # ez: similar to em_asia
    )

    activations = []
    speeds_r = [
        16, 32, 64
        # 16 always best (monthly release, no ffill)
    ]

    f_sig_eco = lambda ind, scale, acts, asofs: [
        eco(
            f"{ind}{asof}",
            "raw"
            if act == "raw"
            else "demax" 
            if act == "ng_tnlog_ng" 
            # b=-1,m=-1,f=logtanh
            else "zwdemax",
            speed = s,
            scale=scale,
            activation=(
                None if act == "raw" else act
            )
        )
        for s in speeds_r
        # for s in speeds
        for act in activations + acts
        for asof in asofs
    ]

    f_sim_eco = lambda univ_w, symbs, acts, asofs: [
        ("ETF", symbol, None) for symbol in univ_w.keys()
    ] + [
        ("ETF", symbol, s)
        for symb in symbs
        for symbol, sc in univ_w.items()
        for s in f_sig_eco(
            symb, sc, acts, asofs
        )
    ]

    eco_acts = [
        # None,
        "n_tanh_p_one", # f=tanhrsign,params=(None,1)
        # "ng_tnlog_ng"
    ]

    IND_ECO = f_sim_eco(
        {
            # "SPY": 1.,
            # "IWM": 1.,
            "XLI": 1.,
            "IYJ": 1.,
            "VIS": 1.,
            # "TLT": -1.,
        },
        [
            "US_IP",
            "US_IP_MAN",
        ],
        eco_acts, [
            # "",
            ":3W"
        ]
    )
    OIL_ECO = f_sim_eco(
        {
            # "SPY": 1.,
            "XLE": 1.,
            "IYE": 1.,
            "VDE": 1.,
            # "TLT": -1.,
        },
        [
            "US_IP_DRILL",
            "US_CU_OIL_GAS",
        ],
        eco_acts, [
            # "",
            ":3W"
        ]
        # OWN RELEASE
        # 2015 release hist (ip), IP improves 2015 and 2020
        # cu similar start, doesnt help 2020
    )
    MAT_ECO = f_sim_eco(
        {
            # "SPY": 1.,
            "XLB": 1.,
            "IYM": 1.,
            "VAW": 1.,
        },
        ["US_IP_MATRL"],
        eco_acts, [
            # "",
            ":3W"
        ]
        # OWN RELEASE
        # improves 2009, 2015, 2020
        # starts 08 (with own release hist)
    )
    CONST_ECO = f_sim_eco(
        {
            # "SPY": 1.,
            "ITB": 1.,
            "XHB": 1,
        },
        [
            "US_IP_CONSTR", # (supplies not building)
            "US_HOUSE_PERMIT",
        ],
        eco_acts, [
            # "",
            ":3W"
        ]
        # OWN RELEASE
        # permits work (w long hist)
        # IP supplies not really
    )
    MINE_ECO = f_sim_eco(
        {
            # "SPY": 1.,
            "XME": 1.,
            "PICK": 1.,
        },
        [
            "US_IP_MINE",
            "US_CU_MINE"
            # both get out of 2016 drawdown, CU poss better
        ],
        eco_acts, [
            # "",
            ":3W"
        ]
    )
    DEFN_ECO = f_sim_eco(
        {
            # "SPY": 1.,
            "ITA": 1.,
            "XAR": 1.,
            "PPA": 1.,
        },
        [
            "US_IP_DEFN",
            # smooths 20, but only 2015 release hist
        ],
        eco_acts, [
            # "",
            ":3W"
        ]
    )
    JP_ECO = f_sim_eco(
        {
            # "SPY": 1.,
            "EWJ": 1.,
        },
        [
            "JP_MANU:IDX",
        ],
        eco_acts, [
            # "",
            ":3W"
            ":L90"
        ]
        # ["raw"]
    )
    
    # TODO: log transform? (generic tf by string for log, tanh, exp, arithmetic, etc. etc.)

    # TODO: vol of vol (vvol)?
    # TODO: wider bullish / bearish universe?

    i = 0
    for name, product_symbol_signals in [
        ("spy", SPY_SPREAD),
        ("tlt", TLT_SPREAD),
        ("eu", EUR_SPREAD),
        ("em", EM_SPREAD),
        ("asia", EM_ASIA_SPREAD),
        ("latam", EM_LATAM_SPREAD),
        ("ind", IND_ECO),
        ("oil", OIL_ECO),
        ("mat", MAT_ECO),
        ("const", CONST_ECO),
        ("mine", MINE_ECO),
        ("def", DEFN_ECO),
        ("jp", JP_ECO),
    ]:
        for suff, rebal in [
            ("", None),
            ("-w", "3D"),
            # ("-m", "3D1W"),
        ]:
            i_str = str(i).rjust(3, "0") + "-"
            domain, univ = strat.StratDomain.new().with_univ(
                PROJECT + "-" + name,
            )
            res = []
            for params in [
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
                for product, symbol, signal
                in product_symbol_signals
            ]:
                domain, univ, res = domain.with_backtest(
                    univ, params, into = res
                )
            project = project.with_page_from_domain(
                i_str + name + "-v" + suff,
                domain,
                {
                    "perf:perf_table": [
                        bt.vis_spec_perf()
                        for bt in res
                    ],
                    "rx:line_cum": [
                        bt.vis_spec_rx()
                        for bt in res
                    ],
                    "signal:line": [
                        bt.vis_spec_signal()
                        for bt in res
                        if bt.signal is not None
                    ],
                },
                t=Project_003
            )
            if test:
                return project
        i += 1
    return project

PROJECTS[PROJECT] = load_p003