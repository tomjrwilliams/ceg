from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Callable,
)
from ...objs.vis import explorer
from ...objs.dev.bars import *

from dataclasses import replace, field
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


PROJECTS = {
    "": lambda: explorer.Project().with_domains(
        es= lambda sync: explorer.AppDomain([
            FutDailyOrdinalDomain.new(),
            FutDailyRxDomain.new(),
        ], sync),
        tn= lambda sync: explorer.AppDomain([
            FutDailyOrdinalDomain.new("TN"),
            FutDailyRxDomain.new("TN"),
        ], sync),
    )
}