import sys
sys.path.append("./src")

from typing import Any
from frozendict import frozendict

import ceg.app as app

shared: frozendict[
    str, Any
] = frozendict() # type: ignore

pages: frozendict[
    str, tuple[app.nav.Page, ...]
] = frozendict() # type: ignore

pages = pages.set("rand", (
    app.rand.Gaussian(shared),
))

app.nav.page(pages, page_config=dict(page_title="ceg")).run()