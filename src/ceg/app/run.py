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
    pages.set("rand", (
        app.rand.Gaussian("gaussian", shared),
        app.page.Dynamic(
            "dynamic",
            shared, 
            cast(app.page.Universe, frozendict({
                "gaussian.walk": fs.rand.gaussian.fs().walk,
                "days": fs.dates.daily.fs().loop,
                "close": (
                    data.bars.daily_close.bind,
                    data.bars.daily_close.new,
                )
            }))
        )
    ))
)

app.nav.page(pages, page_config=dict(page_title="ceg")).run()