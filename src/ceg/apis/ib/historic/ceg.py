from __future__ import annotations

#  ------------------

from typing import ClassVar, NamedTuple, Literal, overload

import os
import datetime
from functools import partial

import polars
import numpy

from ..utils import datetime_to_str
from ..db import DB, Contract, Query, Bar
from ..api import Requests, connect
from ..contracts import contract

from . import api

from .... import core

#  ------------------

IB_BAR_METHOD = "IB_BAR_METHOD"
IB_DATABASE = "IB_DATABASE"

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
    raise ValueError(key)

#  ------------------


class daily_bar_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    d: core.Ref.Object
    contract: Contract
    method: str | None
    db: str | None

def f_daily_bar(
    self: daily_bar | daily_level,
    event: core.Event,
    graph: core.Graph
):
    ds = graph.select(self.d, event, t = False)
    dx = ds[-1]

    d = graph.nodes[self.d.i]

    start = d.start # type: ignore
    end = d.end # type: ignore

    return api.get_daily_bars(
        env(IB_DATABASE, self.db),
        self.contract,
        start,
        end,
        env(IB_BAR_METHOD, self.method),
        # use_rth,
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
        contract: Contract,
        method: str | None = None,
        db: str | None = None,
    ):
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            method=method,
            db=db,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        return f_daily_bar(self, event, graph)

class daily_bars_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    d: core.Ref.Object
    contract: Contract
    window: int | datetime.timedelta | None
    method: str | None
    db: str | None

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
    
    vs = api.get_daily_bars(
        env(IB_DATABASE, self.db),
        self.contract,
        start,
        end,
        env(IB_BAR_METHOD, self.method),
        # use_rth,
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
        contract: Contract,
        window: int | datetime.timedelta | None,
        method: str | None = None,
        db: str | None = None,
    ):
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            window=window,
            method=method,
            db=db,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        return f_daily_bars(self, event, graph)

#  ------------------

class daily_level_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    d: core.Ref.Object
    contract: Contract
    field: str | int
    method: str | None
    db: str | None

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
        contract: Contract,
        field: str | int,
        method: str | None = None,
        db: str | None = None,
    ):
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            method=method,
            db=db,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        vs = f_daily_bar(self, event, graph)
        
        if isinstance(self.field, str):
            fx = api.BAR_FIELD_INDICES[self.field]
        else:
            fx = self.field
        
        return vs[fx]

class daily_levels_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    d: core.Ref.Object
    contract: Contract
    field: str | int
    window: int | datetime.timedelta | None
    method: str | None
    db: str | None

class daily_levels(daily_levels_kw, core.Node.Col1D):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col1D, daily_bars_kw
    )
    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        window: int | datetime.timedelta | None,
        field: str | int,
        method: str | None = None,
        db: str | None = None,
    ):
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            window=window,
            method=method,
            db=db,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        
        vs = f_daily_bars(self, event, graph)
        
        if isinstance(self.field, str):
            fx = api.BAR_FIELD_INDICES[self.field]
        else:
            fx = self.field
        
        return vs[:, fx]

#  ------------------

class daily_open(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        field: str | int = "open",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "open", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            method=method,
            db=db,
        )

class daily_opens(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        window: int | None,
        field: str | int = "open",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "open", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            window=window,
            method=method,
            db=db,
        )


class daily_high(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        field: str | int = "high",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "high", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            method=method,
            db=db,
        )

class daily_highs(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        window: int | None,
        field: str | int = "high",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "high", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            window=window,
            method=method,
            db=db,
        )

class daily_low(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        field: str | int = "low",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "low", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            method=method,
            db=db,
        )

class daily_lows(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        window: int | None,
        field: str | int = "low",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "low", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            window=window,
            method=method,
            db=db,
        )

class daily_close(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        field: str | int = "close",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "close", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            method=method,
            db=db,
        )

class daily_closes(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        window: int | None,
        field: str | int = "close",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "close", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            window=window,
            method=method,
            db=db,
        )

class daily_volume(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        field: str | int = "volume",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "volume", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            method=method,
            db=db,
        )

class daily_volumes(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        window: int | None,
        field: str | int = "volume",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "volume", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            window=window,
            method=method,
            db=db,
        )

class daily_wap(daily_level):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        field: str | int = "wap",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "wap", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            method=method,
            db=db,
        )

class daily_waps(daily_levels):

    @classmethod
    def new(
        cls,
        d: core.Ref.Object,
        contract: Contract,
        window: int | None,
        field: str | int = "wap",
        method: str | None = None,
        db: str | None = None,
    ):
        assert field == "wap", (cls, field)
        return cls(
            *cls.args(),
            d=d,
            contract=contract,
            field=field,
            window=window,
            method=method,
            db=db,
        )

#  ------------------
