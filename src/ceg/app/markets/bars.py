
from typing import Any, cast, Callable
import datetime as dt

from frozendict import frozendict

import ceg
import ceg.fs as fs
import ceg.data as data
import ceg.app as app

FIELDS = {
    "open": data.bars.daily_open,
    "high": data.bars.daily_high,
    "low": data.bars.daily_low,
    "close": data.bars.daily_close,
}

def lines(
    symbol: str,
    product: str = "FUT",
    start: str = "2024",
    end: str = "2025",
    steps: int = 365,
    shared: app.Shared = cast(app.Shared, frozendict())
):
    ident = f"{product}-{symbol}"
    g = ceg.Graph.new()
    g, d = fs.dates.daily.loop(
        g, 
        app.model.parse_date(start), 
        app.model.parse_date(end),
        alias="date"
    )
    for field, node in FIELDS.items():
        g, n = node.bind(
            g,
            d, 
            product=product, 
            symbol=symbol, 
            alias=f"{ident}-{field[0].upper()}"
        )
    return (
        app.model.Model.new(
            f"bars: {ident}",
            shared.set("steps", steps)
        )
        .with_universe(data.bars.UNIVERSE)
        .with_graph(
            nodes=g,
            init={d: True},
            using = {d: fs.dates.daily.loop},
            aliasing = {
                fs.dates.daily.loop: "date",
                **{n: k for k, n in FIELDS.items()}
            }
        )
        .with_plot(init=[
            dict(label="date", x=True),
            dict(label=f"{ident}-O", y=True),
            dict(label=f"{ident}-H", y=True),
            dict(label=f"{ident}-L", y=True),
            dict(label=f"{ident}-C", y=True),
        ])
        # TODO: plot also take the refs instead of labels, pass g, resolve to alias
    )