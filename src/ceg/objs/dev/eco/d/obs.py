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
    Daily,
    DependsParent,
    HasDataFrameValue
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
    @dataclass(frozen=True)
    class Params(Daily.Params):
        symbol: str
        source: str
        units: str
        frequency: str
        seasonal: str

        asof: str | None = None
        lag: dt.timedelta | None = None

        transform: str | None = None

        @classmethod
        def new(
            cls,
            symbol: str,
            source: str,
            units: str | None = None,
            frequency: str | None = None,
            seasonal: str | None = None,
            asof: str | None = None,
            lag: dt.timedelta | None = None,
            transform: str | None = None,
        ):
            assert source == "fred", source
            if units is None:
                s = apis.data.fred.load_spec(
                    symbol,
                    os.environ.get(
                        "FRED_DATA", "./__local__/fred"
                    ),
                )
                units = s.units
                frequency = s.frequency
                seasonal = s.seasonal_adjustment
            units, frequency, seasonal = (
                assert_is(units, str),
                assert_is(frequency, str),
                assert_is(seasonal, str)
            )
            return cls(
                symbol=symbol,
                source=source,
                units=units,
                frequency=frequency,
                seasonal=seasonal,
                asof=asof,
                lag=lag,
                transform=transform,
            )

    parent: str | Date
    params: Params

    def self(self) -> EcoDailyObservation:
        return self
    
    def get_parent(self) -> Date:
        return get_parent(self)

    def encode_type(self, params: EcoDailyObservation.Params):
        return "fut-d-ord"

    def encode_params(self, params: EcoDailyObservation.Params):
        tags = []
        if params.asof is not None:
            tags.append(params.asof)
        if params.lag is not None:
            tags.append(f"{params.lag}-{encode_timedelta(params.lag)}")
        if params.transform is not None:
            tags.append(params.transform)
        return "-".join(map(str, [
            params.symbol,
            params.source,
            # params.units,
            # params.frequency,
            # params.seasonal,
            # TODO: and fx if given
        ] + tags))

    def handle(self, event: Event, snap: Snap):

        # TODO: possibly later on, deal with obj storage def changes?
        # TODO: and data appends versus full regen

        event = assert_is(event, ValueUpdate)
        snap = assert_is(snap, Daily.Snap)

        tags = self.params

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
            raise ValueError(self)

        if tags.transform is not None:
            df = df.pipe(
                apply_transform,
                tags.transform,
                units=tags.units,
                v=tags.symbol,
            )

        _ = self.set_value_from_df(
            df,
            tags.symbol,
        )
        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(df),
        )


eco_daily_observation = new(EcoDailyObservation, EcoDailyObservation.Params.new)


#  ------------------

__all__ = [
    "EcoDailyObservation",
    "eco_daily_observation",
]