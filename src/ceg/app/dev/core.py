from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Callable,
)
from ...objs.vis import explorer
from ...objs.dev import fut

from dataclasses import replace, field
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


PROJECTS = {
    "": explorer.Project().with_domains(
        es= lambda sync: explorer.AppDomain([
            fut.d.bars.ord.FutDailyOrdinalDomain.new(),
            fut.d.bars.rx.FutDailyRxDomain.new(),
        ], sync),
        tn= lambda sync: explorer.AppDomain([
            fut.d.bars.ord.FutDailyOrdinalDomain.new("TN"),
            fut.d.bars.rx.FutDailyRxDomain.new("TN"),
        ], sync),
    )
}