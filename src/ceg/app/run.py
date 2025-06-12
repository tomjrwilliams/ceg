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

pages = (
    pages.set("examples", (
        app.model.Model.new("ES - Close", shared.set("steps", 100))
        .with_universe(cast(app.model.Universe, frozendict({
            "date": fs.dates.daily.loop,
            "close": data.bars.daily_close.bind
        })))
        .with_model(init=[
            dict(
                label="date",
                i=True,
                func="date",
                start="2024",
                end="2025"
            ),
            dict(
                label="ES-Close",
                func="close",
                d="d:ref=date",
                product="FUT",
                symbol="ES",
            )
        ])
        .with_plot(init=[
            dict(label="date", x=True),
            dict(label="ES-Close", y=True),
        ]),
        app.model.Model.new("CL - Close", shared.set("steps", 100))
        .with_universe(cast(app.model.Universe, frozendict({
            "date": fs.dates.daily.loop,
            "close": data.bars.daily_close.bind
        })))
        .with_model(init=[
            dict(
                label="date",
                i=True,
                func="date",
                start="2024",
                end="2025"
            ),
            dict(
                label="CL-Close",
                func="close",
                d="d:ref=date",
                product="FUT",
                symbol="CL",
            )
        ])
        .with_plot(init=[
            dict(label="date", x=True),
            dict(label="CL-Close", y=True),
        ]),
    ))
)

app.nav.page(pages, page_config=dict(
    page_title="ceg",
    layout="wide"
)).run()