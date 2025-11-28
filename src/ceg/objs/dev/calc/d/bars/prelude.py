
from ....fut.d.bars import *
from ....eq.d.bars import *
from ....index.d.bars import *
from ....fx.d.bars import *

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