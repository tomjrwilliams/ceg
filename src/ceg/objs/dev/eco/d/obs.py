from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    overload,
    cast,
    final,
)
import datetime
import datetime as dt

import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

import numpy as np
import polars as pl

from ..... import apis
from .....apis import t
from .....apis import kv
from .....engine import *

#  ------------------


@dataclass(frozen=True)
class EcoDailyTags:
    symbol: str
    source: str
    units: str
    frequency: str
    seasonal: str

    asof: str | None = None
    lag: dt.timedelta | None = None

    transform: str | None = None


def apply_transform(
    df: pl.DataFrame,
    transform: str | None = None,
    units: str | None = None,
    v: str = VALUE,
) -> pl.DataFrame:
    if transform is None:
        return df
    else:
        assert units is not None, (transform, units)
    if transform == "index" and units.startswith(
        "Growth rate same period previous year"
    ):
        raise ValueError((transform, units))
    elif transform == "index" and units.startswith(
        "Growth rate previous period"
    ):
        return df.with_columns(
            pl.col(v)
            .fill_nan(None)
            .fill_null(0)
            .mul(1 / 100)
            .add(1)
            .cum_prod()
            .pipe(apis.pl.null_mask, pl.col(v))
            .alias(v)
        )
    else:
        raise ValueError((transform, units, v))


@dataclass(frozen=True)
class EcoDailyObservation(
    DependsGraphSync,
    DailyChunked,
):
    """
    >>> obj = EcoDailyObservation.new(
    ...     symbol="VIXCLS",
    ...     source="fred",
    ... )
    >>> domain = (
    ...    EcoDailyObservationDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-eco-d-VIXCLS-fred' [f64]
    [
        15.78
        15.41
        null
        null
        16.66
        16.93
        15.77
        14.85
    ]
    >>> obj = EcoDailyObservation.new(
    ...     symbol="US_CAP_UTIL",
    ...     source="fred",
    ...     asof="3W",
    ... )
    >>> domain = (
    ...    EcoDailyObservationDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-eco-d-US_CAP_UTIL-fred-3W' [f64]
    [
        null
        null
        null
        null
        null
        null
        null
        75.8847
    ]
    """

    parent: Project | str | None

    tags: EcoDailyTags

    version: int = 0
    type: Literal["EcoDailyObservation"] = (
        "EcoDailyObservation"
    )

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["EcoDailyObservationSnap"] = (
            "EcoDailyObservationSnap"
        )

    @classmethod
    def new(
        cls,
        symbol: str,
        units: str | None = None,
        frequency: str | None = None,
        seasonal: str | None = None,
        source: str | None = None,
        asof: str | None = None,
        lag: dt.timedelta | None = None,
        transform: str | None = None,
        parent: Project | str | None = None,
    ):
        s_lag: str | None = None

        if ":" in symbol:
            symbol, *asof_lag = symbol.split(":")
            for s in asof_lag:
                if s == "INDEX" or s == "IDX":
                    transform = "index"
                elif lag is None and s[0] == "L":
                    s_lag = s
                    lag = apis.cal.parse_lag(s_lag[1:])
                elif asof is None and s[0] != "L":
                    asof = s

        if units is None:
            if source == "fred":
                s = apis.data.fred.load_spec(
                    symbol,
                    os.environ.get(
                        "FRED_DATA", "./__local__/fred"
                    ),
                )
                units = s.units
                frequency = s.frequency
                seasonal = s.seasonal_adjustment
            else:
                raise ValueError((symbol, source))
        assert frequency is not None, symbol
        assert seasonal is not None, seasonal
        assert source is not None, source
        ident = f"{symbol}-{source}"
        if asof is not None:
            ident = f"{ident}-{asof}"
        if s_lag is not None:
            ident = f"{ident}-{s_lag}"
        if transform is not None:
            ident = f"{ident}-{transform}"
        return cls(
            id=cls.build_id(
                parent,
                "eco-d-obs",
                ident,
                strict=False,
            ),
            parent=parent,
            tags=EcoDailyTags(
                symbol=symbol,
                source=source,
                units=units,
                frequency=frequency,
                seasonal=seasonal,
                asof=asof,
                lag=lag,
                transform=transform,
            ),
        )

    # resampling etc.?

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ObjectChange):
            assert self.id == event.obj, (self, event)
            return HandleNewEcoDailyObservation()
        raise ValueError((self, event))


@dataclass(frozen=True)
class HandleUpdateEcoDailyObservation(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: EcoDailyObservation,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return


@dataclass(frozen=True)
class HandleNewEcoDailyObservation(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):
    def handle(
        self,
        event: ObjectChange,
        snap: EcoDailyObservation.Snap,
        obj: EcoDailyObservation,
    ):
        tags = obj.tags

        if tags.source == "fred":
            df = apis.data.fred.load_time_series(
                tags.symbol,
                os.environ.get(
                    "FRED_DATA", "./__local__/fred"
                ),
                dense=True,
                asof=tags.asof,
                lag=tags.lag,
            ).with_columns(
                pl.col(tags.symbol)
                # .forward_fill()
            )
        else:
            raise ValueError(obj)

        if tags.transform is not None:
            df = df.pipe(
                apply_transform,
                tags.transform,
                units=tags.units,
                v=tags.symbol,
            )

        v = self.set_value_from_df(
            obj,
            df,
            tags.symbol,
            t=self.key_value_type(obj),
        )

        new_snap = self.update_snap_from_df(snap, df)
        if (
            new_snap.first != snap.first
            and new_snap.last != snap.last
        ):
            return ValueUpdate.from_snap(
                self.update_snap_from_df(snap, df),
                into=frozendict(),
            )
        return frozendict().set(obj.id, (snap, []))


# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------


@dataclass(frozen=True)
class DependsEcoDailyObservationParent(DependsParentValue):
    parent: EcoDailyObservation | str

    def get_parent(self) -> EcoDailyObservation:
        return context.get().get(
            self.parent, EcoDailyObservation
        )


#  ------------------

__all__ = [
    "EcoDailyObservation",
    "HandleUpdateEcoDailyObservation",
    "HandleNewEcoDailyObservation",
    "DependsEcoDailyObservationParent",
]
# TODO: if __name__ == __main__:
# run domain by name
