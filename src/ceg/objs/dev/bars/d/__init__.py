from . import ident

from . import eq
from . import fut
from . import fx
from . import index

from .ident import *

from .eq import (
    EqDailyEtf,
    EqDailyRx,
    eq_daily_etf,
    eq_daily_rx
)
from .fut import (
    FutDailyOrdinal,
    FutDailyRx,
    fut_daily_ordinal,
    fut_daily_rx
)
from .fx import (
    FxDailyPx,
    FxDailyRx,
    fx_daily_px,
    fx_daily_rx,
)
from .index import (
    IndexDailyPx,
    IndexDailyRx,
    index_daily_px,
    index_daily_rx,
)

class Px:
    Etf = EqDailyEtf
    etf = eq_daily_etf

    class Fut:
        Ordinal = FutDailyOrdinal
        ordinal = fut_daily_ordinal

    Fx = FxDailyPx
    fx = fx_daily_px

    Index = IndexDailyPx
    index = index_daily_px

    type Type = (
        EqDailyEtf
        | FutDailyOrdinal
        | FxDailyPx
        | IndexDailyPx
    )
    type Snap = (
        EqDailyEtf.Snap
        | FutDailyOrdinal.Snap
        | FxDailyPx.Snap
        | IndexDailyPx.Snap
    )


class Rx:
    Eq = EqDailyRx
    eq = eq_daily_rx

    Fut = FutDailyRx
    fut = fut_daily_rx

    Fx = FxDailyRx
    fx = fx_daily_rx

    Index = IndexDailyRx
    index = index_daily_rx

    type Type = (
        EqDailyRx
        | FutDailyRx
        | FxDailyRx
        | IndexDailyRx
    )
    type Snap = (
        EqDailyRx.Snap
        | FutDailyRx.Snap
        | FxDailyRx.Snap
        | IndexDailyRx.Snap
    )
