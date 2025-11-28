
from ....fut.d.bars import (
    FutDailyOrdinal,
    FutDailyRx,
)
from ....eq.d.bars import (
    EqDailyEtf,
    EqDailyStock,
    EqDailyRx,
)
from ....index.d.bars import (
    IndexDailyPx,
    IndexDailyRx,
)
from ....fx.d.bars import (
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