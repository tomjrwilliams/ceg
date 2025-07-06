
from typing import Any, cast, Callable
from functools import wraps

from frozendict import frozendict

import ceg.fs as fs
import ceg.data as data
import ceg.app as app

def lines(
    symbol: str,
    product: str = "FUT",
    start: str = "2008",
    end: str = "2025",
    steps: int = 365,
    shared: app.Shared = cast(app.Shared, frozendict()),
    span: list[int] = [
        # 4,
        8, 
        16, 
        32, 
        64,
        128, 
        256, 
        512,
    ],
    span_mu: list[int] | None = None,
    truncate: float | None = None,
    norm: bool = True,
    relative: bool | str = False,
):
    consts = {"long": 1, "short": -1}
    factors = [0, 1, 2][:1]
    ident = f"{product}-{symbol}"
    fields = {
        "high": "max_ew",
        "low": "min_ew"
    }
    if span_mu is None:
        span_mu = span

    labels_spectrum = [
        f"H-max-{sp}" for sp in reversed(span)
    ] + [
        f"L-min-{sp}" for sp in span
    ]
    indices_spectrum = list(range(len(labels_spectrum)))

    n_spreads = len(span) - 1

    # truncate rel to rolling min / max as kwarg
    # separate pages for each

    b = [
        -1 
        for _ in range(n_spreads)
    ] + [1] + [1 for _ in range(n_spreads)]

    if relative:
        spreads = [
            dict(
                label=f"{ident}-{l0}-{l1}",
                func="spread_vec",
                l=f"l:ref={ident}-vec-unit",
                r=f"r:ref={ident}-vec-unit",
                il=f"il:int={i0}",
                ir=f"ir:int={i1}",
                b=f"b:float={b}",
            )
            for l0, l1, i0, i1, b in zip(
                labels_spectrum[:-1],
                labels_spectrum[1:],
                indices_spectrum[:-1],
                indices_spectrum[1:],
                b,
            )
        ]
    else:
        spreads = [
            dict(
                label=f"{ident}-{l0}-{l1}",
                func="spread",
                l=f"l:ref={ident}-{l0}",
                r=f"r:ref={ident}-{l1}",
                b=f"b:float={b}"
            )
            for l0, l1, b in zip(
                labels_spectrum[:-1],
                labels_spectrum[1:],
                b,
            )
        ]

    spread_high = spreads[:len(span) - 1]
    spread_low = spreads[-(len(span) - 1):]
    spread_suffixes = [
        s["label"].replace(f"{ident}-", "") for s in spreads
    ]

    spread_high_suffixes = [
        s["label"].replace(f"{ident}-", "") for s in spread_high
    ]
    spread_low_suffixes = [
        s["label"].replace(f"{ident}-", "") for s in spread_low
    ]

    spread_sums = [
        dict(
            label=f"{ident}-{l0}-{l1}",
            func="add",
            l=f"l:ref={ident}-{l0}",
            r=f"r:ref={ident}-{l1}",
        )
        for l0, l1 in zip(
            reversed(spread_low_suffixes),
            spread_high_suffixes, 
        )
    ]
    rms_span = 512
    spread_sum_rms = [
        dict(
            label=f"{ident}-{l0}-{l1}-rms",
            func="rms_ew",
            v=f"v:ref={ident}-{l0}-{l1}",
            span=f"span:float={rms_span}",
        )
        for l0, l1 in zip(
            reversed(spread_low_suffixes),
            spread_high_suffixes, 
        )
    ]
    # 256 512 first

    if truncate:
        spread_labels = [spread["label"] for spread in spreads]
        truncate_span = 1024
        spread_abs = [
            dict(label=f"{spread}-abs", func="abs", v=f"v:ref={spread}")
            for spread in spread_labels
        ]
        spread_mu = [
            dict(
                label=f"{spread}-abs-mu",
                func="mean_ew",
                v=f"v:ref={spread}-abs",
                span=1024, # type: ignore
            )
            for spread in spread_labels

        ]
        spread_max = [
            dict(
                label=f"{spread}-max",
                func="max_ew",
                v=f"v:ref={spread}-abs",
                mu=f"mu:ref={spread}-abs-mu",
                span=truncate_span, # type: ignore
            )
            for spread in spread_labels
        ]
        spread_truncate = [
            dict(
                label=f"{spread}-truncate",
                func="truncate",
                v=f"v:ref={spread}",
                bound=f"bound:ref={spread}-max",
                b=f"b:float={truncate}"
            )
            for spread in spread_labels
        ]
        spreads = ( # type: ignore
            spreads
            + spread_abs # type: ignore
            + spread_mu # type: ignore
            + spread_max
            + spread_truncate
        )
        spread_high = spread_truncate[:len(span) - 1]
        spread_low = spread_truncate[-(len(span) - 1):]
        spread_suffixes = [
            s["label"].replace(f"{ident}-", "") for s in spread_truncate
        ]

    spread_high_suffixes = [
        s["label"].replace(f"{ident}-", "") for s in spread_high
    ]
    spread_low_suffixes = [
        s["label"].replace(f"{ident}-", "") for s in spread_low
    ]

    spreads = (
        spreads 
        + spread_sums
        + spread_sum_rms
    )
    
    if norm:
        spreads_normed = [
            dict(
                label=f"{ident}-{spread}-norm",
                func="ratio",
                l=f"l:ref={ident}-{spread}",
                r=f"r:ref={rms['label']}",
            )
            for spread, rms in zip(
                spread_high_suffixes,
                spread_sum_rms,
            )
        ] + [
            dict(
                label=f"{ident}-{spread}-norm",
                func="ratio",
                l=f"l:ref={ident}-{spread}",
                r=f"r:ref={rms['label']}",
            )
            for spread, rms in zip(
                spread_low_suffixes,
                reversed(spread_sum_rms),
            )
        ]
        spread_suffixes = [
            s["label"].replace(f"{ident}-", "") for s in spreads_normed
        ]
    
        spreads = spreads + spreads_normed
    # spread_suffixes = spread_suffixes + [
    #     s["label"].replace(f"{ident}-", "") for s in spread_sums
    # ]

    name = f"range-spread"
    if relative:
        name += f"-{relative}"
    if truncate:
        name += f"-trunc({truncate})"
    if norm:
        name += f"-norm"
    name += f": {ident}"

    page = (
        app.model.Model.new(
            name,
            shared.set("steps", steps)
        )
        .with_universe(data.bars.UNIVERSE)
        .with_functions(cast(app.model.Universe, frozendict({
            "date": fs.dates.daily.loop,
            "spread": fs.binary.subtract.bind,
            "spread_vec": fs.binary.subtract_vec_i.bind,
            "add": fs.binary.add.bind,
            "abs": fs.unary.abs.bind,
            "ratio": fs.binary.ratio.bind,
            "truncate": fs.binary.truncate.bind,
            "const": fs.consts.const_float.bind,
            "pct_chg": fs.unary.pct_change.bind,  
            "abs_chg": fs.unary.abs_change.bind,  
            "mean_ew": fs.rolling.mean_ew.bind,
            "max_ew": fs.rolling.max_ew.bind,
            "min_ew": fs.rolling.min_ew.bind,
            "rms_ew": fs.rolling.rms_ew.bind,
            "high": data.bars.daily_high.bind,
            "low": data.bars.daily_low.bind,
            "close": data.bars.daily_close.bind,
            # "norm": fs.norm.norm_range_pct.bind,
            "norm_mid": (
                fs.norm.norm_mid_inner_vec.bind
                if relative == "mid-inner"
                else fs.norm.norm_mid_pct_vec.bind
                if relative == "mid-pct"
                else fs.norm.norm_mid_vec.bind
            ),
            "sum_mat": fs.agg.sum_mat_i.bind,
            "mean_vec": fs.agg.mean_vec.bind,
            "pos": fs.risk.pos_linear.bind,
            "pnl": fs.risk.pnl_linear.bind,
            "cum": fs.unary.cum_sum.bind,
            "stack": fs.shapes.v_args_to_vec.bind,
            "pca": fs.rolling.pca_v.month_end,
            "dot": fs.mm.vec_x_mat_i.bind,
            "loading": fs.shapes.mat_tup_to_v.bind,
        })))
        .with_model(init=[
            dict(
                label="date",
                I=True,# type: ignore
                func="date",
                start=start,
                end=end,
            )
        ] + [
            dict(
                label=f"{ident}-{field[0].upper()}",
                func=field,
                d="d:ref=date",
                product=product,
                symbol=symbol,
            )
            for field in list(fields.keys()) + ["close"]
        ] + [
            dict(
                label=f"{ident}-C-rx",
                func="abs_chg",
                v=f"v:ref={ident}-C"
            ),
            dict(
                label=f"{ident}-C-vol",
                func="rms_ew",
                v=f"v:ref={ident}-C-rx",
                span=64,
            ),
        ] + [
            dict(
                label=f"{ident}-{field[0].upper()}-mu-{sp}",
                func="mean_ew",
                v=f"v:ref={ident}-{field[0].upper()}",
                span=sp, # type: ignore
            )
            for field in fields
            for sp in span_mu
        ] + [
            dict(
                label=f"{ident}-{field[0].upper()}-{func[:-3]}-{sp}",
                func=func,
                v=f"v:ref={ident}-{field[0].upper()}",
                mu=f"mu:ref={ident}-{field[0].upper()}-mu-{sp}",
                span=sp, # type: ignore
            )
            for field, func in fields.items()
            for sp in span
        ] + [
            dict(
                label=f"{ident}-vec",
                func="stack",
                **{
                    f"v{i}": f"v{i}:ref={ident}-H-max-{sp}"
                    for i, sp in enumerate(reversed(span))
                },
                **{
                    f"v{i+len(span)}": f"v{i+len(span)}:ref={ident}-L-min-{sp}"
                    for i, sp in enumerate(span)
                }
            ),
            dict(
                label=f"{ident}-vec-unit",
                func="norm_mid",
                vec=f"vec:ref={ident}-vec",
            ),
        ]
        + spreads
        # + sum([
        #     [
        #         dict(
        #             label=f"{ident}-{const}-pos",
        #             func="pos",
        #             # signal=f"signal:ref={const}",
        #             scale=f"scale:ref={ident}-C-vol",
        #             d=f"d:ref=date",
        #             const=f"const:float={const_v}"
        #             # delta=f"delta:float=0.5",
        #             # lower=f"lower:float=-0.5",
        #             # upper=f"upper:float=0.5",
        #             # freq=f"freq:str=D15",
        #         ),
        #         dict(
        #             label=f"{ident}-{const}-pnl",
        #             func="pnl",
        #             pos=f"pos:ref={ident}-{const}-pos",
        #             px=f"px:ref={ident}-C",
        #         )
        #     ]
        #     for const, const_v in consts.items()
        # ], [])
        + sum([
            [
                dict(
                    label=f"{ident}-{spread}-pos",
                    func="pos",
                    signal=f"signal:ref={ident}-{spread}",
                    scale=f"scale:ref={ident}-C-vol",
                    d=f"d:ref=date",
                    # delta=f"delta:float=0.5",
                    # lower=f"lower:float=-0.5",
                    # upper=f"upper:float=0.5",
                    # freq=f"freq:str=D15",
                ),
                dict(
                    label=f"{ident}-{spread}-pnl",
                    func="pnl",
                    pos=f"pos:ref={ident}-{spread}-pos",
                    px=f"px:ref={ident}-C",
                )
            ]
            for spread in spread_suffixes
        ], [])
    ))

    # TODO: rolling max for pos, min for neg
    # truncate to zero at say < 0.2 rolling min / max (relatively slow)

    page = (
        page
        # .with_plot(init=[
        #     dict(label="date", x=True),
        # ] + [
        #     dict(label=f"{ident}-f{fac}", y=True, align=f"{ident}-C")
        #     for fac in factors
        # ], name = "factors")
        .with_plot(init=[
            dict(label="date", x=True),
        ] + [
            dict(label=f"{ident}-H-max-{sp}", y=True, align=f"{ident}-H")
            for sp in reversed(span)
        ] + [
            dict(label=f"{ident}-L-min-{sp}", y=True, align=f"{ident}-H")
            for sp in span
        ])
        .with_plot(init=[
            dict(label="date", x=True),
        # ] + [
        #     dict(
        #         label=f"{ident}-{const}-pnl",
        #         y=True,
        #         align=f"{ident}-C",
        #         expr="cumsum"
        #     )
        #     for const in consts
        ] + [
            dict(
                label=f"{ident}-{spread}-pnl",
                y=True,
                align=f"{ident}-C",
                expr="cumsum:trim=210"
            )
            for spread in spread_suffixes
        ], name = "pnl")
        .with_plot(init = [
            dict(label="date", x=True)
        ] + [
            dict(
                label=spread_sum["label"],
                y=True,
                align=f"{ident}-C",
                expr="identity:trim=210"
            )
            for spread_sum in spread_sum_rms
        ], name = "scaling")
    )

    page = (
        page
        .with_plot(init=[
            dict(label="date", x=True),
        ] + [
            dict(
                label=f"{ident}-{spread}", y=True, align=f"{ident}-C",
                expr="identity:trim=210"
            )
            for spread in spread_suffixes
        ], name = "signal")
        .with_plot(init=[
            dict(label="date", x=True),
        ] + [
            dict(
                label=f"{ident}-{spread}-pos", 
                slot= 0, # type: ignore
                y=True, 
                align=f"{ident}-C",
                expr="identity:trim=210"
            )
            for spread in spread_suffixes
        ], name = "pos")
    )
    return page

# TODO: histogram

# TODO: compare to just constant weights

# signal in the compression / spread of the pc1?
# signal in the weight sum of the pos vs negative ocmponents - more skew up or down (ie. that's the "base" you're "relative" to)


# interesting idea to do a rolling regression against teh backward factor path
# so you get updated poss exponentail weights each time, given the seed pca path

# is there a way to add an orthogonality contsraints?
# or is a linalg solution if multi variate, orthogonal?

# is the weight in weighted reg on the features or the observations?