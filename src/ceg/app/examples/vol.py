
from typing import Any, cast, Callable
from functools import wraps

from frozendict import frozendict

import ceg.fs as fs
import ceg.data as data
import ceg.app as app

def lines(
    symbol: str,
    product: str = "FUT",
    start: str = "2024",
    end: str = "2025",
    steps: int = 365,
    shared: app.Shared = cast(app.Shared, frozendict())
):
    ident = f"{product}-{symbol}"
    fields = ["open", "high", "low", "close"]
    return (
        app.model.Model.new(
            f"vol: {ident}",
            shared.set("steps", steps)
        )
        .with_universe(cast(app.model.Universe, frozendict({
            "date": fs.dates.daily.loop,
            "pct_chg": fs.unary.pct_change.bind, 
            "std": fs.rolling.std.bind, 
            "rms": fs.rolling.rms.bind, 
            "std_ew": fs.rolling.std_ew.bind, 
            "rms_ew": fs.rolling.rms_ew.bind, 
            "open": data.bars.daily_open.bind,
            "high": data.bars.daily_high.bind,
            "low": data.bars.daily_low.bind,
            "close": data.bars.daily_close.bind,
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
            for field in fields
        ] + [
            dict(
                label=f"{ident}-{field[0].upper()}-pct",
                func="pct_chg",
                v=f"v:ref={ident}-{field[0].upper()}"
            )
            for field in fields
        ] + [
            dict(
                label=f"{ident}-{field[0].upper()}-vol",
                func="rms_ew",
                v=f"v:ref={ident}-{field[0].upper()}-pct",
                span=16, # type: ignore
            )
            for field in fields
        ])
        .with_plot(init=[
            dict(label="date", x=True),
            dict(label=f"{ident}-O-vol", y=True, align=f"{ident}-O"),
            dict(label=f"{ident}-H-vol", y=True, align=f"{ident}-H"),
            dict(label=f"{ident}-L-vol", y=True, align=f"{ident}-L"),
            dict(label=f"{ident}-C-vol", y=True, align=f"{ident}-C"),
        ])
    )

# TODO: histogram