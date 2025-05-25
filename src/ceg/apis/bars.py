from __future__ import annotations

#  ------------------

from typing import ClassVar, NamedTuple, Literal, overload

import os
import datetime
from functools import partial

import polars
import numpy

from .ib.db import DB, Contract, Query, T, E, C
from .ib.api import Requests, connect
from .ib.contracts import contract

from .ib.historic import api, universe

from .frd import raw
from .frd.raw import StringNamespace, NestedStringNamespace

from .. import core

#  ------------------

ONE_DAY = datetime.timedelta(days=1)

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

PRODUCTS = StringNamespace(
    GENERIC="GENERIC",
    GEN="GENERIC",
    ETF="ETF",
)

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
            adjust=adjust,
            expiry=expiry,
            fx=fx,
            source=source,
            description=description,
        )

IB_IDENTIFIERS: dict[
    tuple[str, str], tuple[str, str]
] = {
    (PRODUCTS.GEN, "ES"): (T.EQ, "ES"),
}

def bar_to_ib_contract(
    bar: Bar,
    currency: str | None=None,
    exchange: str | None=None,
    primary_exchange: str | None=None,
    last_trade: str | None=None,
    local_symbol: str | None=None,
    strike: float | None=None,
    right: str | None=None,
    multiplier: str | None=None,
    sec_id: str | None=None,
    sec_id_type: str | None=None,
    description: str | None=None,
    include_expired: bool | None=None,
) -> Contract:
    try:
        t, s = IB_IDENTIFIERS[
            (bar.product, bar.symbol)
        ]
    except:
        raise ValueError(bar)
    kwargs = dict(
        currency=currency,
        exchange=exchange,
        primary_exchange=primary_exchange,
        last_trade=last_trade,
        local_symbol=local_symbol,
        strike=strike,
        right=right,
        multiplier=multiplier,
        sec_id=sec_id,
        sec_id_type=sec_id_type,
        description=description,
        include_expired=include_expired,
    )
    try:
        con = universe.CONTRACTS[(t, s)]
        con_d = con._asdict()
    except:
        raise ValueError(bar, t, s)
    kwargs = {
        k: (
            v if v is not None
            else con_d[k]
        ) for k, v in kwargs.items()
    }
    return Contract.new(
        t,
        s, 
        **{
            k: v for k, v 
            in kwargs.items() 
            if v is not None
        } # type: ignore
    )

#  ------------------

BARS_SCHEMA = {
    "date": polars.Date,
    "open": polars.Float64,
    "high": polars.Float64,
    "low": polars.Float64,
    "close": polars.Float64,
    "volume": polars.Float64,
    "wap": polars.Float64,
}

BAR_FIELD_INDICES = {
    k: i - 1
    for i, k in enumerate(BARS_SCHEMA)
    if k != "date"
}


#  ------------------

def get_daily_level(
    bar: Bar,
    start: datetime.date,
    end: datetime.date,
    field: str | int,
    df: bool = False,
    at: datetime.date | None = None,
    fp: str | None = None,
):
    if df:
        assert isinstance(field, str), field
        return get_daily_bars(
            bar,
            start,
            end,
            df=True,
            at=at,
            fp=fp,
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
        fp=fp,
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
get_daily_wap = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["wap"]
)

#  ------------------
    
CACHE: dict[
    Bar, 
    tuple[
        datetime.date, 
        datetime.date,
        core.Array.np_2D
    ]
] = {}

FRD_SCHEMA = {
    "date": polars.Date,
    "open": polars.Float64,
    "high": polars.Float64,
    "low": polars.Float64,
    "close": polars.Float64,
    "volume": polars.Float64,
    "open_interest": polars.Float64,
}

@overload
def get_daily_bars(
    bar: Bar,
    start: datetime.date,
    end: datetime.date,
    df: Literal[False] = False,
    at: Literal[None] = None,
    fp: str | None = None,
) -> core.Array.np_2D: ...

@overload
def get_daily_bars(
    bar: Bar,
    start: datetime.date,
    end: datetime.date,
    df: Literal[False] = False,
    at: datetime.date = None,
    fp: str | None = None,
) -> core.Array.np_1D: ...

@overload
def get_daily_bars(
    bar: Bar,
    start: datetime.date,
    end: datetime.date,
    df: Literal[True] = True,
    at: datetime.date | None = None,
    fp: str | None = None,
) -> polars.DataFrame: ...

def get_daily_bars(
    bar: Bar,
    start: datetime.date,
    end: datetime.date,
    df: bool = False,
    at: datetime.date | None = None,
    fp: str | None = None,
):
    key = bar
    if key not in CACHE:
        cache_end = end
        cache_start = end + ONE_DAY
        res = numpy.empty((0, 6))
    else:
        cache_start, cache_end, res = CACHE[key]

    if bar.source == "FRD" or bar.source is None:
        f_get = get_daily_bars_frd
        schema = FRD_SCHEMA
    else:
        f_get = get_daily_bars_ib
        schema = BARS_SCHEMA

    if start < cache_start:
        bars: polars.DataFrame = f_get(
            bar,
            start,
            cache_start - ONE_DAY,
        )
        res = numpy.vstack((
            bars.select(*list(schema.keys())[1:])
            .to_numpy(),
            res,
        ))
        cache_start = start
    if end > cache_end:
        bars: polars.DataFrame = f_get(
            bar,
            end + ONE_DAY,
            cache_end,
        )
        res = numpy.vstack((
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
        i_l = (at - cache_start).days
        i_r = i_l + 1
    else:
        i_l = (start - cache_start).days
        i_r = (cache_end - end).days

    if at is not None:
        res = res[i_l]
    elif i_r > 0:
        res = res[i_l:-i_r,:]
    else:
        res = res[i_l:]
        
    if df:
        return polars.DataFrame(
            res.T,
            schema = {
                k: v for k, v in 
                schema.items()
                if k != "date"
            }
        ).with_columns(
            polars.date_range(
                start, end
            ).alias("date")
        ).select(*BARS_SCHEMA.keys())
    return res

PRODUCT_FOLDERS = {
    PRODUCTS.ETF: raw.FOLDERS.ETF,
    PRODUCTS.GEN: raw.FOLDERS.GEN,
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
    start: datetime.date,
    end: datetime.date,
    fp: str | None = None,
) -> polars.DataFrame:
    return raw.read_file(
        parent=env(
            FRD_DIRECTORY, fp
        ),
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
    start: datetime.date,
    end: datetime.date,
    fp: str | None = None,
) -> polars.DataFrame:
    # TODO: other kwrags as on the bar?
    # have to ignore from hash?
    con: Contract = bar_to_ib_contract(
        bar
    )
    return api.req_daily_bars(
        env("IB_DATABASE", fp),
        con,
        start,
        end,
        bar_method=env(
            "IB_BAR_METHOD",
            "MIDPOINT"
            # or trades if index
        ),
        use_rth=True,
        df=True,
    )

#  ------------------


class daily_bar_kw(NamedTuple):
    type: str
    #
    d: core.Ref.Object
    bar: Bar

def f_daily_bar(
    self: daily_bar | daily_level,
    event: core.Event,
    graph: core.Graph
):
    dx = graph.select(self.d, event, t = False, i = -1)

    d = graph.nodes[self.d.i]

    start = d.start # type: ignore
    end = d.end # type: ignore

    return get_daily_bars(
        self.bar,
        start,
        end,
        df=False,
        at=dx,
    )

class daily_bar(daily_bar_kw, core.Node.Col1D):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col1D, daily_bar_kw
    )

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar
    ):
        return cls(
            cls.DEF.name, d=d, bar = bar
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        return f_daily_bar(self, event, graph)

class daily_bars_kw(NamedTuple):
    type: str
    #
    d: core.Ref.Object
    bar: Bar
    window: int | datetime.timedelta | None

def f_daily_bars(
    self: daily_bars | daily_levels,
    event: core.Event,
    graph: core.Graph
):
    # time series of 2d, each size window (window meaning depends on flags)
    # TODO: drop none (and thus allow count in not none if itn)
    
    ds = graph.select(self.d, event, t = False)
    dx = ds[-1]

    d = graph.nodes[self.d.i]

    start = d.start # type: ignore
    end = d.end # type: ignore

    if isinstance(self.window, int):
        window = datetime.timedelta(days=self.window)
    else:
        window = self.window

    d_ref = dx + window

    if d_ref > dx:
        d_start = dx
        d_end = d_ref
    elif d_ref < dx:
        d_start = dx
        d_end = d_ref
    else:
        d_start = d_end = dx
    
    vs = get_daily_bars(
        self.bar,
        start,
        end,
        df=False,
        at = (
            None if d_start != d_end else d_start
        )
    )

    if d_start == d_end:
        return vs
    elif window is None:
        return vs
    
    ex = (end - d_end).days
    if ex > 0:
        vs = vs[:, :-ex]

    sx = (d_start - start).days
    if sx > 0:
        vs = vs[:, sx:]

    return vs

class daily_bars(daily_bars_kw, core.Node.Col2D):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col2D, daily_bars_kw
    )

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | datetime.timedelta | None,
    ):
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            window=window,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        return f_daily_bars(self, event, graph)

#  ------------------

class daily_level_kw(NamedTuple):
    type: str
    #
    d: core.Ref.Object
    bar: Bar
    field: str | int

class daily_level(
    daily_level_kw, core.Node.Col
):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, daily_bar_kw
    )
    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        field: str | int,
    ):
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        vs = f_daily_bar(self, event, graph)
        
        if isinstance(self.field, str):
            fx = BAR_FIELD_INDICES[self.field]
        else:
            fx = self.field
        
        return vs[fx]

class daily_levels_kw(NamedTuple):
    type: str
    #
    d: core.Ref.Object
    bar: Bar
    field: str | int
    window: int | datetime.timedelta | None

class daily_levels(daily_levels_kw, core.Node.Col1D):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col1D, daily_bars_kw
    )
    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | datetime.timedelta | None,
        field: str | int,
        method: str | None = None,
        db: str | None = None,
    ):
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
            window=window,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        
        vs = f_daily_bars(self, event, graph)
        
        if isinstance(self.field, str):
            fx = BAR_FIELD_INDICES[self.field]
        else:
            fx = self.field
        
        return vs[:, fx]

#  ------------------

class daily_open(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        field: str | int = "open",
    ):
        assert field == "open", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
        )

class daily_opens(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | None,
        field: str | int = "open",
    ):
        assert field == "open", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
            window=window,
        )


class daily_high(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        field: str | int = "high",
    ):
        assert field == "high", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
        )

class daily_highs(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | None,
        field: str | int = "high",
    ):
        assert field == "high", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
            window=window,
        )

class daily_low(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        field: str | int = "low",
    ):
        assert field == "low", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
        )

class daily_lows(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | None,
        field: str | int = "low",
    ):
        assert field == "low", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
            window=window,
        )

class daily_close(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        field: str | int = "close",
    ):
        assert field == "close", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
        )

class daily_closes(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | None,
        field: str | int = "close",
    ):
        assert field == "close", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
            window=window,
        )

class daily_volume(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        field: str | int = "volume",
    ):
        assert field == "volume", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
        )

class daily_volumes(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | None,
        field: str | int = "volume",
    ):
        assert field == "volume", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
            window=window,
        )

class daily_wap(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        field: str | int = "wap",
    ):
        assert field == "wap", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
        )

class daily_waps(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | None,
        field: str | int = "wap",
    ):
        assert field == "wap", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
            window=window,
        )

class daily_open_interest(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        field: str | int = "open_interest",
    ):
        assert field == "open_interest", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
        )

class daily_open_interests(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        bar: Bar,
        window: int | None,
        field: str | int = "open_interest",
    ):
        assert field == "open_interest", (cls, field)
        return cls(
            cls.DEF.name,
            d=d,
            bar=bar,
            field=field,
            window=window,
        )

#  ------------------
