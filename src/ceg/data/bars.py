from __future__ import annotations

from typing import ClassVar, NamedTuple, Literal, overload, Annotated

import os
import datetime as dt
from functools import partial, wraps

import polars as pl
import numpy as np

from ..apis import frd

from .. import core
from ..core import define, dataclass

from ..fs import dates

#  ------------------

class Products:
    FUT = "FUT"
    FUTC = "FUTC"
    ETF = "ETF"

#  ------------------

ONE_DAY = dt.timedelta(days=1)

IB_BAR_METHOD = "IB_BAR_METHOD"
IB_DATABASE = "IB_DATABASE"

FRD_DIRECTORY = "FRD_DIRECTORY"

def env(
    key: str,
    value: str | None = None,
):
    if value is not None:
        return value
    if key == IB_BAR_METHOD:
        return os.environ.get(IB_BAR_METHOD, "MIDPOINT")
    elif key == IB_DATABASE:
        # todo: mode prefixes
        return os.environ.get(
            IB_DATABASE, "./__local__/historic.db"
        )
    elif key == FRD_DIRECTORY:
        # todo: mode prefixes
        return os.environ.get(
            FRD_DIRECTORY, "./__local__/frd"
        )
    raise ValueError(key)

#  ------------------

ADJUST_DEFAULTS = {
    Products.FUT: frd.data.Suffix.abs,
    # Products.FUT: frd.data.Suffix.ratio,
}

class Bar(NamedTuple):
    product: str
    symbol: str
    adjust: str | None
    expiry: str | None
    fx: str | None # ccy to convert to
    source: str | None
    description: str | None

    @classmethod
    def new(
        cls, 
        product: str,
        symbol: str,
        adjust: str | None=None,
        expiry: str | None=None,
        fx: str | None=None,
        source: str | None=None,
        description: str | None=None,
    ) -> Bar:
        return Bar(
            product=product,
            symbol=symbol,
            adjust=ADJUST_DEFAULTS.get(product),
            expiry=expiry,
            fx=fx,
            source=source,
            description=description,
        )

# def bar_to_ib_contract(
#     bar: Bar,
#     currency: str | None=None,
#     exchange: str | None=None,
#     primary_exchange: str | None=None,
#     last_trade: str | None=None,
#     local_symbol: str | None=None,
#     strike: float | None=None,
#     right: str | None=None,
#     multiplier: str | None=None,
#     sec_id: str | None=None,
#     sec_id_type: str | None=None,
#     description: str | None=None,
#     include_expired: bool | None=None,
# ) -> Contract:
#     try:
#         t, s = IB_IDENTIFIERS[
#             (bar.product, bar.symbol)
#         ]
#     except:
#         raise ValueError(bar)
#     kwargs = dict(
#         currency=currency,
#         exchange=exchange,
#         primary_exchange=primary_exchange,
#         last_trade=last_trade,
#         local_symbol=local_symbol,
#         strike=strike,
#         right=right,
#         multiplier=multiplier,
#         sec_id=sec_id,
#         sec_id_type=sec_id_type,
#         description=description,
#         include_expired=include_expired,
#     )
#     try:
#         con = universes.CONTRACTS[(t, s)]
#         con_d = con._asdict()
#     except:
#         raise ValueError(bar, t, s)
#     kwargs = {
#         k: (
#             v if v is not None
#             else con_d[k]
#         ) for k, v in kwargs.items()
#     }
#     return Contract.new(
#         t,
#         s, 
#         **{
#             k: v for k, v 
#             in kwargs.items() 
#             if v is not None
#         } # type: ignore
#     )

#  ------------------

BARS_SCHEMA = {
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "open_interest": pl.Float64,
}

BAR_FIELD_INDICES = {
    k: i - 1
    for i, k in enumerate(BARS_SCHEMA)
    if k != "date"
}

class Fields:
    OPEN = BAR_FIELD_INDICES["open"]
    HIGH = BAR_FIELD_INDICES["high"]
    LOW = BAR_FIELD_INDICES["low"]
    CLOSE = BAR_FIELD_INDICES["close"]
    VOLUME = BAR_FIELD_INDICES["volume"]
    OPEN_INTEREST = BAR_FIELD_INDICES["open_interest"]

#  ------------------

def get_daily_level(
    bar: Bar,
    start: dt.date,
    end: dt.date,
    field: str | int,
    df: bool = False,
    at: dt.date | None = None,
):
    if df:
        assert isinstance(field, str), field
        return get_daily_bars(
            bar,
            start,
            end,
            df=True,
            at=at,
        ).select("date", field)
    i = (
        field if isinstance(field, int)
        else BAR_FIELD_INDICES[field]
    )
    return get_daily_bars(
        bar,
        start,
        end,
        df=df,
        at=at,
    )[:, i]

get_daily_open = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["open"]
)
get_daily_high = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["high"]
)
get_daily_low = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["low"]
)
get_daily_close = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["close"]
)
get_daily_volume = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["volume"]
)
get_daily_open_interest = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["open_interest"]
)

#  ------------------
    
CACHE: dict[
    Bar, 
    tuple[
        dt.date, 
        dt.date,
        np.ndarray # 2d
    ]
] = {}

FRD_SCHEMA = {
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "open_interest": pl.Float64,
}

@overload
def get_daily_bars(
    bar: Bar,
    start: dt.date,
    end: dt.date,
    at: Literal[None] = None,
    df: Literal[False] = False,
) -> np.ndarray: ... # 2d

@overload
def get_daily_bars(
    bar: Bar,
    start: dt.date,
    end: dt.date,
    at: dt.date,
    df: Literal[False] = False,
) -> np.ndarray: ... # 1d

@overload
def get_daily_bars(
    bar: Bar,
    start: dt.date,
    end: dt.date,
    at: dt.date | None = None,
    df: bool = True,
) -> pl.DataFrame: ...

def get_daily_bars(
    bar: Bar,
    start: dt.date,
    end: dt.date,
    at: dt.date | None = None,
    df: bool = False,
):
    key = bar
    if key not in CACHE:
        cache_end = end
        cache_start = end + ONE_DAY
        res = np.empty((0, 6))
    else:
        cache_start, cache_end, res = CACHE[key]

    if bar.source == "FRD" or bar.source is None:
        f_get = get_daily_bars_frd
        schema = FRD_SCHEMA
    else:
        f_get = get_daily_bars_ib
        schema = BARS_SCHEMA

    if start < cache_start:
        bars: pl.DataFrame = f_get(
            bar,
            start,
            cache_start - ONE_DAY,
        )
        res = np.vstack((
            bars.select(*list(schema.keys())[1:])
            .to_numpy(),
            res,
        ))
        cache_start = start
    if end > cache_end:
        bars: pl.DataFrame = f_get(
            bar,
            end + ONE_DAY,
            cache_end,
        )
        res = np.vstack((
            res,
            bars.select(*list(schema.keys())[1:])
            .to_numpy(),
        ))
        cache_end = end
    if (
        cache_start == start
        or cache_end == end
    ):
        CACHE[key] = (
            cache_start,
            cache_end,
            res,
        )
    if at is not None:
        return res[(at - cache_start).days]
    else:
        i_l = (start - cache_start).days
        i_r = (cache_end - end).days

    # TODO: pass back null mask as well, so dont have to infer by all nan

    if i_r > 0:
        res = res[i_l:-i_r,:]
    else:
        res = res[i_l:]
        
    if df:
        return pl.DataFrame(
            res.T,
            schema = {
                k: v for k, v in 
                schema.items()
                if k != "date"
            }
        ).with_columns(
            pl.date_range(
                start, end
            ).alias("date")
        ).select(*BARS_SCHEMA.keys())
    return res

PRODUCT_FOLDERS = {
    Products.ETF: frd.data.Folders.ETF,
    Products.FUT: frd.data.Folders.FUT,
    Products.FUTC: frd.data.Folders.FUTC,
    # etc.
}
ADJUSTMENT_SUFFIXES = {
    None: None,
    #
    "adjsplitdiv": "adjsplitdiv",
    # etc.
    #
    "ratio": "ratio",
    "abs": "abs",
    "stitch": "stitch"
}

def get_daily_bars_frd(
    bar: Bar,
    start: dt.date,
    end: dt.date,
    fp: str | None = None,
) -> pl.DataFrame:
    return frd.data.read_file(
        parent=env(FRD_DIRECTORY, fp),
        folder=PRODUCT_FOLDERS[bar.product],
        symbol=bar.symbol,
        suffix=ADJUSTMENT_SUFFIXES[bar.adjust],
        snap="full",
        freq="1day",
        start=start,
        end=end,
    )

def get_daily_bars_ib(
    bar: Bar,
    start: dt.date,
    end: dt.date,
    fp: str | None = None,
) -> pl.DataFrame:
    # # TODO: other kwrags as on the bar?
    # # have to ignore from hash?
    # con: Contract = bar_to_ib_contract(
    #     bar
    # )
    # return historic.req_daily_bars(
    #     env("IB_DATABASE", fp),
    #     con,
    #     start,
    #     end,
    #     bar_method=env(
    #         "IB_BAR_METHOD",
    #         "MIDPOINT"
    #         # or trades if index
    #     ),
    #     use_rth=True,
    #     df=True,
    # )
    raise ValueError(bar)

#  ------------------

TBar = Annotated[Bar | None, define.annotation("internal")]
TField = Annotated[str | int, define.annotation("internal")]

def f_daily_bar(
    self: daily_bar | daily_level,
    event: core.Event,
    graph: core.Graph
):
    assert self.bar is not None, self
    dx = self.d.history(graph).last_before(event.t)
    if not isinstance(dx, dt.date):
        return None
    res = get_daily_bars(
        self.bar,
        dx,
        dt.date.today(),
        df=False,
        at=dx,
    )
    if np.all(np.isnan(res)):
        return None
    return res

def new_bar(
    bar: Bar | None,
    product: str | None,
    symbol: str | None,
):
    if bar is None:
        assert product is not None, product
        assert symbol is not None, symbol
        bar = Bar.new(
            product=product, symbol=symbol
        )
    return bar, product, symbol

@dataclass(frozen=True)
class daily_bar(core.Node.D1_F64):
    """
    >>> start = dt.date(2025, 1, 1)
    >>> end = dt.date(2025, 1, 6)
    >>> g, d = core.Graph.new().pipe(
    ...     dates.daily.loop, start, end
    ... )
    >>> g, r = g.pipe(daily_bar.bind, d, product="FUT", symbol="ES")
    >>> for g, es, t in core.batches(
    ...     g, core.Event.zero(d), n=5, g=2, iter=True
    ... )():
    ...     dx = d.history(g).last_before(t)
    ...     v = r.history(g).last_before(t)
    ...     print(dx, list(v[:4]))
    2025-01-01 [nan, nan, nan, nan]
    2025-01-02 [6016.77, 6063.3, 5941.43, 5983.65]
    2025-01-03 [5988.2, 6064.81, 5978.34, 6057.48]
    2025-01-04 [nan, nan, nan, nan]
    2025-01-05 [nan, nan, nan, nan]
    """

    type: str
    #
    d: core.Ref.Scalar_Date
    product: str | None
    symbol: str | None
    bar: TBar

    @classmethod
    def new(
        cls,
        d: core.Ref.Scalar_Date,
        product: str | None=None,
        symbol: str | None=None,
        bar: TBar=None,
    ):
        bar, product, symbol = new_bar(bar, product, symbol)
        return daily_bar(
            "daily_bar",
            d=d,
            bar = bar,
            product=product,
            symbol=symbol,
        )

    bind = define.bind_from_new(
        new,
        core.Node.D1_F64.ref,
    )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        return f_daily_bar(self, event, graph)

#  ------------------

@dataclass(frozen=True)
class daily_level(core.Node.D0_F64):
    type: str
    #
    d: core.Ref.Scalar_Date
    bar: TBar
    product: str | None
    symbol: str | None
    field: TField

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        vs = f_daily_bar(self, event, graph)
        
        if isinstance(self.field, str):
            fx = BAR_FIELD_INDICES[self.field]
        else:
            fx = self.field
        
        if vs is None:
            return vs

        v = vs[fx]

        # if np.isnan(v):
        #     return None
        return v
    
#  ------------------

@dataclass(frozen=True)
class daily_open(daily_level):
    """
    >>> start = dt.date(2025, 1, 1)
    >>> end = dt.date(2025, 1, 6)
    >>> g, d = core.Graph.new().pipe(
    ...     dates.daily.loop, start, end
    ... )
    >>> g, r = g.pipe(daily_open.bind, d, product="FUT", symbol="ES")
    >>> for g, es, t in core.batches(
    ...     g, core.Event.zero(d), n=5, g=2, iter=True
    ... )():
    ...     dx = d.history(g).last_before(t)
    ...     v = r.history(g).last_before(t)
    ...     print(dx, v)
    2025-01-01 None
    2025-01-02 6016.77
    2025-01-03 5988.2
    2025-01-04 None
    2025-01-05 None
    """

    @staticmethod
    def new(
        # cls,
        d: core.Ref.Scalar_Date,
        product: str | None = None,
        symbol: str | None = None,
        bar: TBar = None,
        field: TField = "open",
    ):
        assert field == "open", field
        bar, product, symbol = new_bar(bar, product, symbol)
        return daily_open(
            "daily_open",
            d=d,
            bar=bar,
            product=product,
            symbol=symbol,
            field=field,
        )

    bind = define.bind_from_new(
        new,
        core.Node.D0_F64.ref,
    )

#  ------------------


@dataclass(frozen=True)
class daily_high(daily_level):
    """
    >>> start = dt.date(2025, 1, 1)
    >>> end = dt.date(2025, 1, 6)
    >>> g, d = core.Graph.new().pipe(
    ...     dates.daily.loop, start, end
    ... )
    >>> g, r = g.pipe(daily_high.bind, d, product="FUT", symbol="ES")
    >>> for g, es, t in core.batches(
    ...     g, core.Event.zero(d), n=5, g=2, iter=True
    ... )():
    ...     dx = d.history(g).last_before(t)
    ...     v = r.history(g).last_before(t)
    ...     print(dx, v)
    2025-01-01 None
    2025-01-02 6063.3
    2025-01-03 6064.81
    2025-01-04 None
    2025-01-05 None
    """

    @staticmethod
    def new(
        # cls,
        d: core.Ref.Scalar_Date,
        product: str | None = None,
        symbol: str | None = None,
        bar: TBar = None,
        field: TField = "high",
    ):
        assert field == "high", field
        bar, product, symbol = new_bar(bar, product, symbol)
        return daily_high(
            "daily_high",
            d=d,
            bar=bar,
            product=product,
            symbol=symbol,
            field=field,
        )
    bind = define.bind_from_new(
        new,
        core.Node.D0_F64.ref,
    )
    
#  ------------------


@dataclass(frozen=True)
class daily_low(daily_level):
    """
    >>> start = dt.date(2025, 1, 1)
    >>> end = dt.date(2025, 1, 6)
    >>> g, d = core.Graph.new().pipe(
    ...     dates.daily.loop, start, end
    ... )
    >>> g, r = g.pipe(daily_low.bind, d, product="FUT", symbol="ES")
    >>> for g, es, t in core.batches(
    ...     g, core.Event.zero(d), n=5, g=2, iter=True
    ... )():
    ...     dx = d.history(g).last_before(t)
    ...     v = r.history(g).last_before(t)
    ...     print(dx, v)
    2025-01-01 None
    2025-01-02 5941.43
    2025-01-03 5978.34
    2025-01-04 None
    2025-01-05 None
    """

    @staticmethod
    def new(
        # cls,
        d: core.Ref.Scalar_Date,
        product: str | None = None,
        symbol: str | None = None,
        bar: TBar = None,
        field: TField = "low",
    ):
        assert field == "low", field
        bar, product, symbol = new_bar(bar, product, symbol)
        return daily_low(
            "daily_low",
            d=d,
            bar=bar,
            product=product,
            symbol=symbol,
            field=field,
        )
    bind = define.bind_from_new(
        new,
        core.Node.D0_F64.ref,
    )
    
#  ------------------




@dataclass(frozen=True)
class daily_close(daily_level):
    """
    >>> start = dt.date(2025, 1, 1)
    >>> end = dt.date(2025, 1, 6)
    >>> g, d = core.Graph.new().pipe(
    ...     dates.daily.loop, start, end
    ... )
    >>> g, r = g.pipe(daily_close.bind, d, product="FUT", symbol="ES")
    >>> for g, es, t in core.batches(
    ...     g, core.Event.zero(d), n=5, g=2, iter=True
    ... )():
    ...     dx = d.history(g).last_before(t)
    ...     v = r.history(g).last_before(t)
    ...     print(dx, v)
    2025-01-01 None
    2025-01-02 5983.65
    2025-01-03 6057.48
    2025-01-04 None
    2025-01-05 None
    """

    @staticmethod
    def new(
        # cls,
        d: core.Ref.Scalar_Date,
        product: str | None = None,
        symbol: str | None = None,
        bar: TBar = None,
        field: TField = "close",
    ):
        assert field == "close", field
        bar, product, symbol = new_bar(bar, product, symbol)
        return daily_close(
            "daily_close",
            d=d,
            bar=bar,
            product=product,
            symbol=symbol,
            field=field,
        )

    bind = define.bind_from_new(
        new,
        core.Node.D0_F64.ref,
    )

#  ------------------


@dataclass(frozen=True)
class daily_volume(daily_level):
    """
    >>> start = dt.date(2025, 1, 1)
    >>> end = dt.date(2025, 1, 6)
    >>> g, d = core.Graph.new().pipe(
    ...     dates.daily.loop, start, end
    ... )
    >>> g, r = g.pipe(daily_volume.bind, d, product="FUT", symbol="ES")
    >>> for g, es, t in core.batches(
    ...     g, core.Event.zero(d), n=5, g=2, iter=True
    ... )():
    ...     dx = d.history(g).last_before(t)
    ...     v = r.history(g).last_before(t)
    ...     print(dx, v)
    2025-01-01 None
    2025-01-02 1826031.0
    2025-01-03 1206570.0
    2025-01-04 None
    2025-01-05 None
    """

    @staticmethod
    def new(
        # cls,
        d: core.Ref.Scalar_Date,
        product: str | None=None,
        symbol: str | None=None,
        bar: TBar=None,
        field: TField = "volume",
    ):
        assert field == "volume", field
        bar, product, symbol = new_bar(bar, product, symbol)
        return daily_volume(
            "daily_volume",
            d=d,
            bar=bar,
            product=product,
            symbol=symbol,
            field=field,
        )
    bind = define.bind_from_new(
        new,
        core.Node.D0_F64.ref,
    )

#  ------------------


@dataclass(frozen=True)
class daily_open_interest(daily_level):
    """
    >>> start = dt.date(2025, 1, 1)
    >>> end = dt.date(2025, 1, 6)
    >>> g, d = core.Graph.new().pipe(
    ...     dates.daily.loop, start, end
    ... )
    >>> g, r = g.pipe(daily_open_interest.bind, d, product="FUT", symbol="ES")
    >>> for g, es, t in core.batches(
    ...     g, core.Event.zero(d), n=5, g=2, iter=True
    ... )():
    ...     dx = d.history(g).last_before(t)
    ...     v = r.history(g).last_before(t)
    ...     print(dx, v)
    2025-01-01 None
    2025-01-02 2068557.0
    2025-01-03 2061748.0
    2025-01-04 None
    2025-01-05 None
    """

    @staticmethod
    def new(
        # cls,
        d: core.Ref.Scalar_Date,
        product: str | None = None,
        symbol: str | None = None,
        bar: TBar = None,
        field: TField = "open_interest",
    ):
        assert field == "open_interest", field
        bar, product, symbol = new_bar(bar, product, symbol)
        return daily_open_interest(
            "daily_open_interest",
            d=d,
            bar=bar,
            product=product,
            symbol=symbol,
            field=field,
        )

    bind = define.bind_from_new(
        new,
        core.Node.D0_F64.ref,
    )
    

#  ------------------

UNIVERSE = pl.DataFrame([
    dict(product="FUT", symbol="ES"),
    dict(product="FUT", symbol="CL"),
])