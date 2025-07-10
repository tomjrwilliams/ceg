
from typing import Any, cast, Callable
from functools import wraps

from frozendict import frozendict

import ceg.fs as fs
import ceg.data as data
import ceg.app as app

def lines(
    symbol: str,
    product: str = "FUT",
    start: str = "2014",
    end: str = "2025",
    steps: int = 365,
    shared: app.Shared = cast(app.Shared, frozendict()),
    span: list[int] = [
        # 4,
        8, 
        # 16, 
        32, 
        # 64,
        128, 
        # 256, 
        # 512
    ],
    span_mu: list[int] | None = None,
):
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
    page = (
        app.model.Model.new(
            f"range-pca: {ident}",
            shared.set("steps", steps)
        )
        .with_universe(data.bars.UNIVERSE)
        .with_functions(cast(app.model.Universe, frozendict({
            "date": fs.dates.daily.loop,
            "pct_chg": fs.unary.pct_change.bind,  
            "abs_chg": fs.unary.abs_change.bind,  
            "mean_ew": fs.rolling.mean_ew.bind,
            "max_ew": fs.rolling.max_ew.bind,
            "min_ew": fs.rolling.min_ew.bind,
            "rms_ew": fs.rolling.rms_ew.bind,
            "high": data.bars.daily_high.bind,
            "low": data.bars.daily_low.bind,
            "close": data.bars.daily_close.bind,
            "norm": fs.norm.norm_range_pct.bind,
            # "norm_mid_pct": fs.norm.norm_mid_pct_vec.bind,
            "norm_mid": (
                fs.norm.norm_mid_vec.bind
                if symbol == "CL"
                else fs.norm.norm_mid_pct_vec.bind
            ),
            "sum_mat": fs.agg.sum_mat_i.bind,
            "mean_vec": fs.agg.mean_vec.bind,
            "pos": fs.risk.pos_linear.bind,
            "pnl": fs.risk.pnl_linear.bind,
            "cum": fs.unary.cum_sum.bind,
            "stack": fs.shapes.v_args_to_vec.bind,
            "pca": fs.factors.pca_v.month_end,
            "dot": fs.mm.vec_x_mat_i.bind,
            "loading": fs.shapes.mat_tup_to_v.bind,
        })))
        .with_graph(nodes=[
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
            dict(
                label=f"{ident}-pca",
                func="pca",
                vs=f"vs:ref={ident}-vec-unit",
                d=f"d:ref=date",
                window=365*3,
                factors=3,
            ),
            # TODO: monthly pca weights
            # daily pca factor path s

            # plot the pnl of first three factors, say
        ] + [
            dict(
                label=f"{ident}-f{fac}",
                func="dot",
                v=f"v:ref={ident}-vec-unit",
                vec=f"vec:ref={ident}-pca",
                slot=f"slot:int=1",
                f=f"f:int={fac}"
            )
            for fac in factors
        ] + [
            dict(
                label=f"{ident}-f{fac}-{l}",
                func="loading",
                vec=f"vec:ref={ident}-pca",
                i0=f"i0:int={i0}",
                i1=f"i1:int={fac}",
                slot=f"slot:int=1",
            )
            for fac in factors
            for i0, l in enumerate(labels_spectrum)
        ] 
        # + sum([
        #     [
        #         dict(
        #             label=f"{ident}-f{fac}-sig",
        #             func="sum_mat",
        #             mat=f"mat:ref={ident}-pca",
        #             i=f"i:int={fac}",
        #             t="t:bool=True",
        #             slot=f"slot:int=1",
        #         ),
        #         dict(
        #             label=f"{ident}-f{fac}-pos",
        #             func="pos",
        #             signal=f"signal:ref={ident}-f{fac}-sig",
        #             scale=f"scale:ref={ident}-C-vol",
        #             d=f"d:ref=date",
        #             # delta=f"delta:float=0.5",
        #             # lower=f"lower:float=-0.5",
        #             # upper=f"upper:float=0.5",
        #             # freq=f"freq:str=D15",
        #         ),
        #         dict(
        #             label=f"{ident}-f{fac}-pnl",
        #             func="pnl",
        #             pos=f"pos:ref={ident}-f{fac}-pos",
        #             px=f"px:ref={ident}-C",
        #         )
        #     ]
        #     for fac in factors
        # TODO: scale relative to rolling mean / median as eg. for es mostly positive mean (upward drift)
        # ], []) 
        + sum([
            [
                dict(
                    label=f"{ident}-f{fac}-sig",
                    func="mean_vec",
                    vec=f"vec:ref={ident}-vec-unit",
                    b=f"b:float=-1",
                ),
                dict(
                    label=f"{ident}-f{fac}-pos",
                    func="pos",
                    signal=f"signal:ref={ident}-f{fac}-sig",
                    scale=f"scale:ref={ident}-C-vol",
                    d=f"d:ref=date",
                    # delta=f"delta:float=0.5",
                    # lower=f"lower:float=-0.5",
                    # upper=f"upper:float=0.5",
                    # freq=f"freq:str=D15",
                ),
                dict(
                    label=f"{ident}-f{fac}-pnl",
                    func="pnl",
                    pos=f"pos:ref={ident}-f{fac}-pos",
                    px=f"px:ref={ident}-C",
                )
            ]
            for fac in factors
        ], [])
    ))
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
        ] + [
            dict(
                label=f"{ident}-f{fac}-pnl",
                y=True,
                align=f"{ident}-C",
                expr="cumsum"
            )
            for fac in factors
        ], name = "pnl")
    )

    for fac in factors:
        page = page.with_plot(init = [
            dict(label="date", x=True),
        ] + [
            dict(
                label=f"{ident}-f{fac}-{l}",
                y=True,
                align=f"{ident}-C"
            ) for i0, l in enumerate(labels_spectrum)
        ], name = f"f{fac}")

    page = (
        page
        .with_plot(init=[
            dict(label="date", x=True),
        ] + [
            dict(label=f"{ident}-f{fac}-sig", y=True, align=f"{ident}-C")
            for fac in factors
        ], name = "signal")
        .with_plot(init=[
            dict(label="date", x=True),
        ] + [
            dict(
                label=f"{ident}-f{fac}-pos", 
                slot= 0, # type: ignore
                y=True, 
                align=f"{ident}-C"
            )
            for fac in factors
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