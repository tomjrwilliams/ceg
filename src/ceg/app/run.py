import sys
sys.path.append("./src")

from typing import Any, cast
from frozendict import frozendict

import ceg.fs as fs
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
            }))
        )
    ))
)

app.nav.page(pages, page_config=dict(page_title="ceg")).run()