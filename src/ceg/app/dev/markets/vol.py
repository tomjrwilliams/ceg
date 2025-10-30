from typing import Any, cast, Callable
from functools import wraps
from dataclasses import dataclass

from frozendict import frozendict

import ceg
import ceg.func as func
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
    span: int = 16,
    shared: app.Shared = cast(app.Shared, frozendict()),
):

    ident = f"{product}-{symbol}"
    g = ceg.Graph.new()
    g, d = func.dates.daily.loop(
        g,
        app.model.parse_date(start),
        app.model.parse_date(end),
        alias="date",
    )
    ohlc = {}
    ohlc_pct = {}
    ohlc_vol = {}
    for field, node in OHLC.items():
        g, ohlc[field] = node.bind(
            g,
            d,
            product=product,
            symbol=symbol,
            alias=f"{ident}-{field[0].upper()}",
        )
        g, ohlc_pct[field] = func.unary.pct_change.bind(
            g,
            ohlc[field],
            alias=f"{ident}-{field[0].upper()}-pct",
        )
        g, ohlc_vol[field] = func.rolling.rms_ew.bind(
            g,
            ohlc_pct[field],
            span=span,
            alias=f"{ident}-{field[0].upper()}-vol",
        )
    return (
        app.model.Model.new(f"vol: {ident}", shared)
        .with_universe(data.bars.UNIVERSE)
        .with_graph(
            nodes=g,
            init={d: True},
            using={d: func.dates.daily.loop},
            aliasing={
                func.dates.daily.loop: "date",
                **{n: k for k, n in OHLC.items()},
                **{
                    func.unary.pct_change: "pct_chg",
                    func.rolling.std: "std",
                    func.rolling.rms: "rms",
                    func.rolling.std_ew: "std_ew",
                    func.rolling.rms_ew: "rms_ew",
                },
            },
        )
        .with_plot(
            init=[dict(label="date", x=True)]
            + [
                dict(
                    label=ohlc_vol[field],
                    y=True,
                    align=ohlc[field],
                )
                for field in OHLC.keys()
            ]
        )
    )


# TODO: histogram
