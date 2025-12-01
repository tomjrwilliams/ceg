
from .fut import (
    FutDailyOrdinal,
    FutDailyRx,
)
from .eq import (
    EqDailyEtf,
    EqDailyStock,
    EqDailyRx,
)
from .index import (
    IndexDailyPx,
    IndexDailyRx,
)
from .fx import (
    FxDailyPx,
    FxDailyRx,
)

type TBar = (
    FutDailyOrdinal
    | FutDailyRx
    | EqDailyEtf
    | EqDailyStock
    | EqDailyRx
    | IndexDailyPx
    | IndexDailyRx
    | FxDailyPx
    | FxDailyRx
)
type TBarSnap = (
    FutDailyOrdinal.Snap
    | FutDailyRx.Snap
    | EqDailyEtf.Snap
    | EqDailyStock.Snap
    | EqDailyRx.Snap
    | IndexDailyPx.Snap
    | IndexDailyRx.Snap
    | FxDailyPx.Snap
    | FxDailyRx.Snap
)

__all__ = [
    "FutDailyOrdinal",
    "FutDailyRx",
    "EqDailyEtf",
    "EqDailyStock",
    "EqDailyRx",
    "IndexDailyPx",
    "IndexDailyRx",
    "FxDailyPx",
    "FxDailyRx",
    "TBar",
    "TBarSnap",
]