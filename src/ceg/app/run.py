import sys
sys.path.append("./src")

from typing import Any, cast, Callable
from functools import wraps

from frozendict import frozendict

import ceg
import ceg.fs as fs
import ceg.data as data
import ceg.app as app

shared: frozendict[
    str, Any
] = frozendict() # type: ignore

pages: frozendict[
    str, tuple[app.nav.Page, ...]
] = frozendict() # type: ignore

def example_bars(
    symbol: str,
    product: str = "FUT",
    start: str = "2024",
    end: str = "2025",
    steps: int = 100
):
    ident = f"{product}-{symbol}"
    return (
        app.model.Model.new(
            ident,
            shared.set("steps", steps)
        )
        .with_universe(cast(app.model.Universe, frozendict({
            "date": fs.dates.daily.loop,
            "open": data.bars.daily_open.bind,
            "high": data.bars.daily_high.bind,
            "low": data.bars.daily_low.bind,
            "close": data.bars.daily_close.bind,
        })))
        .with_model(init=[
            dict(
                label="date",
                i=True,
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
            for field in ["open", "high", "low", "close"]
        ])
        .with_plot(init=[
            dict(label="date", x=True),
            dict(label=f"{ident}-O", y=True),
            dict(label=f"{ident}-H", y=True),
            dict(label=f"{ident}-L", y=True),
            dict(label=f"{ident}-C", y=True),
        ])
    )

pages = (
    pages.set("bars", (
        example_bars("ES"),
        example_bars("CL"),
    ))
)

app.nav.page(pages, page_config=dict(
    page_title="ceg",
    layout="wide"
)).run()