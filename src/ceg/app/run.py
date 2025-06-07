import sys
sys.path.append("./src")

from typing import Any
from frozendict import frozendict

from ceg.app.examples import ExamplePage
from ceg.app import nav

shared: frozendict[
    str, Any
] = frozendict() # type: ignore

pages: frozendict[
    str, tuple[nav.Page, ...]
] = frozendict() # type: ignore

pages = pages.set("example", (
    ExamplePage(shared),
))

nav.page(pages).run()