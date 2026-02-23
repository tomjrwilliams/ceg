from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Type,
    Callable,
)
from ...vis import explorer
from ...objs.dev.bars import *

from dataclasses import replace, field
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

PROJECTS: dict[str, Type[explorer.Explorer]] = {}