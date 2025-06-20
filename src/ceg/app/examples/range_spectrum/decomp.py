
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
        16, 
        32, 
        64,
        128, 
        # 256, 
        # 512
    ],
    span_mu: list[int] | None = None,
):
    factors = [0, 1, 2]
    ident = f"{product}-{symbol}"
    fields = {
        "high": "max_ew",
        "low": "min_ew"
    }
    if span_mu is None:
        span_mu = span
    return (
        app.model.Model.new(
            f"minmax-pca: {ident}",
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
            "norm_vec": fs.norm.norm_range_pct_vec.bind,
            "pos": fs.risk.pos_linear.bind,
            "pnl": fs.risk.pnl_linear.bind,
            "cum": fs.unary.cum_sum.bind,
            "stack": fs.arrays.v_args_to_vec.bind,
            "pca": fs.rolling.pca_v.month_end,
            "dot": fs.mm.vec_x_mat_i.bind,
        })))
        .with_model(init=[
            dict(
                label="date",
                i=True,# type: ignore
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
                    for i, sp in enumerate(span)
                },
                **{
                    f"v{i+len(span)}": f"v{i+len(span)}:ref={ident}-L-min-{sp}"
                    for i, sp in enumerate(span)
                }
            ),
            dict(
                label=f"{ident}-vec-unit",
                func="norm_vec",
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
            # dict(
            #     label=f"{ident}-pos",
            #     func="pos",
            #     signal=f"signal:ref={ident}-sig",
            #     scale=f"scale:ref={ident}-C-vol",
            #     d=f"d:ref=date",
            #     # delta=f"delta:float=0.5",
            #     lower=f"lower:float=-0.5",
            #     upper=f"upper:float=0.5",
            #     freq=f"freq:str=D15",
            # ),
            # dict(
            #     label=f"{ident}-pnl-",
            #     func="pnl",
            #     pos=f"pos:ref={ident}-pos",
            #     px=f"px:ref={ident}-C",
            # )
            for fac in factors
        ])
        .with_plot(init=[
            dict(label="date", x=True),
        ] + [
            dict(label=f"{ident}-f{fac}", y=True, align=f"{ident}-C")
            for fac in factors
        ], name = "factors")
    )

# TODO: histogram