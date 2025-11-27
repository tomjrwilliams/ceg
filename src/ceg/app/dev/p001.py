
from ...objs.vis import explorer
from ...objs.dev import fut

from .core import PROJECTS

PROJECT = "001-trend"

PROJECTS[PROJECT] = (
    explorer.Project()
    .with_domains()
    .with_config(
        explorer.ExplorerConfig(PROJECT)
        .with_page(
            "example",
            type="daily",
            domain="es",
            # TODO: ...
        )
    )
)