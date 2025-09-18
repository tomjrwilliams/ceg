
from typing import Any, cast, Callable
import datetime as dt

from frozendict import frozendict

import ceg
import ceg.fs as fs
import ceg.data as data
import ceg.app as app

OHLC = {
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
    ohlc = {}
    for field, node in OHLC.items():
        g, ohlc[field] = node.bind(
            g,
            d, 
            product=product, 
            symbol=symbol, 
            alias=f"{ident}-{field[0].upper()}"
        )
    return (
        app.model.Model.new(f"bars: {ident}", shared)
        .with_universe(data.bars.UNIVERSE)
        .with_graph(
            nodes=g,
            init={d: True},
            using = {d: fs.dates.daily.loop},
            aliasing = {
                fs.dates.daily.loop: "date",
                **{n: k for k, n in OHLC.items()}
            }
        )
        .with_plot(init=[
            dict(label="date", x=True)
        ] + [
            dict(label=ohlc[field], y=True)
            for field in OHLC.keys()
        ])
    )