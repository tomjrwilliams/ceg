from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    ParamSpec,
    Callable,
    Concatenate,
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
from .....apis import t, kv, assert_is
from .....apis.pl import *

from .....engine import *

from ...bars.d import *

P = ParamSpec("P")
O = TypeVar("O", bound=Object)

#  ------------------


@dataclass(frozen=True)
class DailyMParams(DailyParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    def build_id(
        self,
        cls: Type[DailyM],
    ):
        window = self.window
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return f"{cls.SUFFIX}-" + params


@dataclass(frozen=True)
class DailyM(DailyWithDailyParent, DailyChunked):
    parent: Daily | str
    params: DailyMParams

    project: Project | str | None = None

    version: int = 0
    type: Literal["DailyM"] = "DailyM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "M"
    # NOTE: to override

    @classmethod
    def new(
        cls,
        parent: Daily,
        window: int | None = None,
        project: Project | None = None,
    ):
        params = DailyMParams(
            window=window,
        )
        assert cls.SUFFIX != "M", cls
        id=parent.extend_id(params.build_id(cls))
        if project is not None:
            id=project.extend_id(path)
        return cls(
            id=id,
            parent=parent,
            params=params,
            project=project,
        )


#  ------------------


@dataclass(frozen=True)
class HandleNewDailyM(
    HandleNewDailyFromDailyParentPl,
):
    pass


#  ------------------


@dataclass(frozen=True)
class DailyMA(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMA.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ma-4' [f64]
    [
        5707.5
        null
        null
        5766.875
        5821.0
        5800.5
        5785.1875
        5737.0625
    ]
    """

    version: int = 0
    type: Literal["DailyMA"] = "DailyMA"

    SUFFIX: ClassVar[str] = "ma"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyMASnap"] = "DailyMASnap"

    def handler_object_new(self) -> HandleNewDailyMA:
        return HandleNewDailyMA()


@dataclass(frozen=True)
class HandleNewDailyMA(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return pl.col(VALUE).rolling_mean(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyKurt(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyKurt.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-kurt-4' [f64]
    [
        -1.116847
        null
        null
        -2.0
        -2.0
        -1.5
        -1.070286
        -1.379246
    ]
    """

    version: int = 0
    type: Literal["DailyKurt"] = "DailyKurt"

    SUFFIX: ClassVar[str] = "kurt"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyKurtSnap"] = "DailyKurtSnap"

    def handler_object_new(self) -> HandleNewDailyKurt:
        return HandleNewDailyKurt()


@dataclass(frozen=True)
class HandleNewDailyKurt(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return pl.col(VALUE).rolling_kurtosis(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyMax(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMax.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-max-4' [f64]
    [
        5729.75
        null
        null
        5815.5
        5826.5
        5826.5
        5826.5
        5826.5
    ]
    """

    version: int = 0
    type: Literal["DailyMax"] = "DailyMax"

    SUFFIX: ClassVar[str] = "max"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyMaxSnap"] = "DailyMaxSnap"

    def handler_object_new(self) -> HandleNewDailyMax:
        return HandleNewDailyMax()


@dataclass(frozen=True)
class HandleNewDailyMax(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return pl.col(VALUE).rolling_max(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyMed(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMed.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-med-4' [f64]
    [
        5715.5
        null
        null
        5766.875
        5821.0
        5815.5
        5787.5
        5749.375
    ]
    """

    version: int = 0
    type: Literal["DailyMed"] = "DailyMed"

    SUFFIX: ClassVar[str] = "med"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyMedSnap"] = "DailyMedSnap"

    def handler_object_new(self) -> HandleNewDailyMed:
        return HandleNewDailyMed()


@dataclass(frozen=True)
class HandleNewDailyMed(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return pl.col(VALUE).rolling_median(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyMid(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMid.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-mid-4' [f64]
    [
        5699.5
        null
        null
        5766.875
        5821.0
        5793.0
        5782.875
        5724.75
    ]
    """

    params: DailyMidParams

    version: int = 0
    type: Literal["DailyMid"] = "DailyMid"

    SUFFIX: ClassVar[str] = "mid"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyMidSnap"] = "DailyMidSnap"

    def handler_object_new(self) -> HandleNewDailyMid:
        return HandleNewDailyMid()

    @classmethod
    def new(
        cls,
        parent: Daily,
        window: int | None = None,
        width: float | None = None,
        project: Project | None = None,
    ):
        params = DailyMidParams(
            window=window,
            width=width,
        )
        assert cls.SUFFIX != "M", cls
        id=parent.extend_id(params.build_id(cls))
        if project is not None:
            id=project.extend_id(path)
        return cls(
            id=id,
            parent=parent,
            params=params,
            project=project,
        )


@dataclass(frozen=True)
class DailyMidParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    width: float | None = None

    def build_id(
        self,
        cls: Type[DailyM],
    ):
        window = self.window
        width = self.width
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return (
            f"{cls.SUFFIX}-"
            + params
            + ("" if width is None else f"-{width}")
        )


@dataclass(frozen=True)
class HandleNewDailyMid(HandleNewDailyM):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyMid,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
            obj,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyMid,
    ) -> pl.DataFrame:
        params = obj.params
        e = pl.col(VALUE)
        u = (
            e.rolling_max(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 + params.width,
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 - params.width,
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
        )
        e = (u + l) / 2

        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
        return df


@dataclass(frozen=True)
class DailyMin(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMin.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-min-4' [f64]
    [
        5669.25
        null
        null
        5718.25
        5815.5
        5759.5
        5739.25
        5623.0
    ]
    """

    version: int = 0
    type: Literal["DailyMin"] = "DailyMin"

    SUFFIX: ClassVar[str] = "min"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyMinSnap"] = "DailyMinSnap"

    def handler_object_new(self) -> HandleNewDailyMin:
        return HandleNewDailyMin()


@dataclass(frozen=True)
class HandleNewDailyMin(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params

        return pl.col(VALUE).rolling_min(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyMOscParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    slow: int | None = None
    vol: int | None = None
    activation: str | None = None

    # TODO: norm

    def build_id(
        self,
        cls: Type[DailyOsc],
    ):
        window = self.window
        slow = self.slow
        vol = self.vol
        activation = self.activation

        fast = 1 if window is None else window
        s_slow = "ex" if slow is None else slow
        s_vol = "ex" if vol is None else vol
        res = f"{cls.SUFFIX}-{fast}-{s_slow}-{s_vol}"
        if activation is not None:
            return f"{res}-{activation}"
        return res


@dataclass(frozen=True)
class DailyOsc(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyOsc.new(ord, fast=4, slow=12, vol=8)
    >>> unit = DailyOsc.new(ord, fast=4, slow=12, vol=8, activation="tanh")
    >>> Graph().with_objs(obj, unit).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-osc-4-12-8' [f64]
    [
        0.979022
        null
        null
        1.240631
        1.349691
        1.105384
        0.828809
        0.060248
    ]
    >>> unit.value_key_val().get(unit.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-osc-4-12-8-tanh' [f64]
    [
        0.752642
        null
        null
        0.845635
        0.87398
        0.802425
        0.679836
        0.060175
    ]
    """

    parent: (
        str
        | Daily
        | FutDailyOrdinal
        | FutDailyRx
        | EqDailyEtf
        | EqDailyStock
        | EqDailyRx
        | IndexDailyPx
        | IndexDailyRx
        | FxDailyPx
        | FxDailyRx
    )

    params: DailyMOscParams

    version: int = 0
    type: Literal["DailyOsc"] = "DailyOsc"

    SUFFIX: ClassVar[str] = "osc"

    @classmethod
    def new(
        cls,
        parent: (
            Daily
            | FutDailyOrdinal
            | FutDailyRx
            | EqDailyEtf
            | EqDailyStock
            | EqDailyRx
            | IndexDailyPx
            | IndexDailyRx
            | FxDailyPx
            | FxDailyRx
        ),
        # --
        fast: int | None = None,
        slow: int | None = None,
        vol: int | None = None,
        activation: str | None = None,
        #
        project: Project | None = None,
    ):
        params = DailyMOscParams(
            window=fast,
            slow=slow,
            vol=vol,
            activation=activation,
        )
        assert cls.SUFFIX != "M", cls
        id=parent.extend_id(params.build_id(cls))
        if project is not None:
            id=project.extend_id(path)
        return cls(
            id=id,
            parent=parent,
            params=params,
            project=project,
        )

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyOscSnap"] = "DailyOscSnap"

    def handler_object_new(self) -> HandleNewDailyOsc:
        return HandleNewDailyOsc()


@dataclass(frozen=True)
class HandleNewDailyOsc(
    HandleNewDailyFromDailyParentPl,
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: Event,
        obj: DailyOsc,
    ) -> pl.DataFrame:
        params = obj.params
        parent = obj.get_parent()
        if isinstance(parent, FutDailyRx):
            tags = parent.get_parent().tags
            rx = pl.col(VALUE)
            if tags.adjust == "pct":
                px = (1 + rx).cum_prod().pipe(null_mask, rx)
            elif tags.adjust == "abs":
                px = rx.cum_sum().pipe(null_mask, rx)
            else:
                raise ValueError(obj)
        elif isinstance(parent, FutDailyOrdinal):
            tags = parent.tags
            px = pl.col(VALUE)
            if tags.adjust == "pct":
                rx = (
                    px.forward_fill()
                    .pct_change()
                    .pipe(null_mask, px)
                )
            elif tags.adjust == "abs":
                rx = (
                    px.forward_fill()
                    .diff()
                    .pipe(null_mask, px)
                )
            else:
                raise ValueError(obj)
        else:
            raise ValueError((obj, parent))

        df = df.select(VALUE=pl.col(VALUE), px=px, rx=rx)
        px = pl.col("px")
        rx = pl.col("rx")

        px_fast = (
            px
            if params.window is None or params.window == 1
            else px.rolling_mean(
                params.window, min_samples=2
            )
        )
        px_slow = (
            px.rolling_mean(len(df), min_samples=2)
            if params.slow is None
            else px.rolling_mean(params.slow, min_samples=2)
        )
        # TODO: or default to rms?
        rx_vol = (
            rx.rolling_std(len(df), min_samples=2)
            if params.vol is None
            else px.rolling_std(params.vol, min_samples=2)
        )

        e = ((px_fast - px_slow) / rx_vol).pipe(
            null_mask, pl.col(VALUE)
        )
        if params.activation is None:
            pass
        elif params.activation == "tanh":
            e = e.tanh()
        else:
            raise ValueError(obj)

        df = df.select(e.alias(VALUE))
        return df


@dataclass(frozen=True)
class DailyPLast(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyPLast.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-plast-4' [f64]
    [
        0.333333
        null
        null
        0.0
        0.0
        0.666667
        1.0
        1.0
    ]
    """

    version: int = 0
    type: Literal["DailyPLast"] = "DailyPLast"

    SUFFIX: ClassVar[str] = "plast"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyPLastSnap"] = "DailyPLastSnap"

    def handler_object_new(self) -> HandleNewDailyPLast:
        return HandleNewDailyPLast()


@dataclass(frozen=True)
class HandleNewDailyPLast(HandleNewDailyM):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
            obj,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.DataFrame:
        params = obj.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_p_last(
            v,
            v,
            is_null,
            l=len(df)
            if params.window is None
            else params.window,
            norm=False,  # TODO: parametrise in path?
        )

        return pl.DataFrame({"mask": s, VALUE: res}).select(
            pl.when(pl.col("mask").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )


import numba as nb

# rolling window, indicator series of gt last
# then exp decay over binary series?
# so actually needs sliding window of weights


@nb.jit()
def calc_p_last(
    v: np.ndarray,
    loc: np.ndarray,
    is_null: np.ndarray,
    l: int,
    norm: bool,
):
    n: int = len(v)
    acc: float = 0
    res = np.zeros(n, dtype=v.dtype)

    l -= 1

    j = 0

    min_v = 1
    max_v = 0

    i: int = 0
    while i < n:
        if i == 0:
            acc = 0.5
            res[i] = acc
        elif not is_null[i]:
            j = 1
            k = 0

            acc = 0
            w_acc = 0

            while k < l and j <= i:
                ij = i - j
                if not is_null[ij]:
                    vv = (
                        0
                        if v[ij] < loc[i]
                        else 0.5
                        if v[ij] == loc[i]
                        else 1
                    )
                    w = 1 / (l + 1)
                    acc += vv * w
                    w_acc += w
                    k += 1
                j += 1

            if w_acc > 0:
                scale = 1 / w_acc
                acc *= scale

                res[i] = acc

                if acc < min_v:
                    min_v = acc
                if acc > max_v:
                    max_v = acc

        i += 1

    return res


@dataclass(frozen=True)
class DailyQile(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyQile.new(ord, window=4, quantile=.5)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-qile-4-0.5' [f64]
    [
        5718.25
        null
        null
        5815.5
        5826.5
        5815.5
        5815.5
        5759.5
    ]
    """

    params: DailyQileParams

    version: int = 0
    type: Literal["DailyQile"] = "DailyQile"

    SUFFIX: ClassVar[str] = "qile"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyQileSnap"] = "DailyQileSnap"

    def handler_object_new(self) -> HandleNewDailyQile:
        return HandleNewDailyQile()

    @classmethod
    def new(
        cls,
        parent: Daily,
        window: int | None = None,
        quantile: float | None = None,
        project: Project | None = None,
    ):
        params = DailyQileParams(
            window=window,
            quantile=quantile,
        )
        assert cls.SUFFIX != "M", cls
        id=parent.extend_id(params.build_id(cls))
        if project is not None:
            id=project.extend_id(path)
        return cls(
            id=id,
            parent=parent,
            params=params,
            project=project,
        )


@dataclass(frozen=True)
class DailyQileParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    quantile: float | None = None

    def build_id(
        self,
        cls: Type[DailyM],
    ):
        window = self.window
        quantile = self.quantile
        assert quantile is not None, (cls, quantile)
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return f"{cls.SUFFIX}-" + params + f"-{quantile}"


@dataclass(frozen=True)
class HandleNewDailyQile(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyQile,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyQile,
    ) -> pl.Expr:
        params = obj.params
        assert params.quantile is not None, obj

        return pl.col(VALUE).rolling_quantile(
            params.quantile,
            window_size=len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyRank(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRank.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rank-4' [f64]
    [
        3.0
        null
        null
        2.0
        2.0
        1.0
        1.0
        1.0
    ]
    """

    version: int = 0
    type: Literal["DailyRank"] = "DailyRank"

    SUFFIX: ClassVar[str] = "rank"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyRankSnap"] = "DailyRankSnap"

    def handler_object_new(self) -> HandleNewDailyRank:
        return HandleNewDailyRank()


@dataclass(frozen=True)
class HandleNewDailyRank(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return pl.col(VALUE).rolling_rank(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyRms(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRms.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-rms-4' [f64]
    [
        0.007817
        null
        null
        0.012045
        0.0121
        0.011903
        0.010457
        0.011816
    ]
    """

    version: int = 0
    type: Literal["DailyRms"] = "DailyRms"

    SUFFIX: ClassVar[str] = "rms"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyRmsSnap"] = "DailyRmsSnap"

    def handler_object_new(self) -> HandleNewDailyRms:
        return HandleNewDailyRms()


@dataclass(frozen=True)
class HandleNewDailyRms(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE)
            .pow(2)
            .rolling_mean(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            .sqrt()
        )


@dataclass(frozen=True)
class DailyRng(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRng.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rng-4' [f64]
    [
        60.5
        null
        null
        97.25
        11.0
        67.0
        87.25
        203.5
    ]
    """

    params: DailyRngParams

    version: int = 0
    type: Literal["DailyRng"] = "DailyRng"

    SUFFIX: ClassVar[str] = "rng"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyRngSnap"] = "DailyRngSnap"

    def handler_object_new(self) -> HandleNewDailyRng:
        return HandleNewDailyRng()

    @classmethod
    def new(
        cls,
        parent: Daily,
        window: int | None = None,
        width: float | None = None,
        project: Project | None = None,
    ):
        params = DailyRngParams(
            window=window,
            width=width,
        )
        assert cls.SUFFIX != "M", cls
        id=parent.extend_id(params.build_id(cls))
        if project is not None:
            id=project.extend_id(path)
        return cls(
            id=id,
            parent=parent,
            params=params,
            project=project,
        )


@dataclass(frozen=True)
class DailyRngParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    width: float | None = None

    def build_id(
        self,
        cls: Type[DailyRng],
    ):
        window = self.window
        width = self.width
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return (
            f"{cls.SUFFIX}-"
            + params
            + ("" if width is None else f"-{width}")
        )


@dataclass(frozen=True)
class HandleNewDailyRng(HandleNewDailyM):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyRng,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
            obj,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyRng,
    ) -> pl.DataFrame:
        params = obj.params
        e = pl.col(VALUE)
        u = (
            e.rolling_max(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 + params.width,
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 - params.width,
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
        )
        e = u - l

        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
        return df


@dataclass(frozen=True)
class DailySDev(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailySDev.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-sdev-4' [f64]
    [
        0.009001
        null
        null
        0.011345
        0.010688
        0.014262
        0.012023
        0.009659
    ]
    """

    version: int = 0
    type: Literal["DailySDev"] = "DailySDev"

    SUFFIX: ClassVar[str] = "sdev"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailySDevSnap"] = "DailySDevSnap"

    def handler_object_new(self) -> HandleNewDailySDev:
        return HandleNewDailySDev()


@dataclass(frozen=True)
class HandleNewDailySDev(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return pl.col(VALUE).rolling_std(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailySkew(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailySkew.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-skew-4' [f64]
    [
        0.15138
        null
        null
        2.6070e-14
        3.3046e-14
        0.073938
        0.464041
        -0.228863
    ]
    """

    version: int = 0
    type: Literal["DailySkew"] = "DailySkew"

    SUFFIX: ClassVar[str] = "skew"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailySkewSnap"] = "DailySkewSnap"

    def handler_object_new(self) -> HandleNewDailySkew:
        return HandleNewDailySkew()


@dataclass(frozen=True)
class HandleNewDailySkew(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return pl.col(VALUE).rolling_skew(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyVar(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyVar.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-var-4' [f64]
    [
        0.000081
        null
        null
        0.000129
        0.000114
        0.000203
        0.000145
        0.000093
    ]
    """

    version: int = 0
    type: Literal["DailyVar"] = "DailyVar"

    SUFFIX: ClassVar[str] = "var"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyVarSnap"] = "DailyVarSnap"

    def handler_object_new(self) -> HandleNewDailyVar:
        return HandleNewDailyVar()


@dataclass(frozen=True)
class HandleNewDailyVar(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        return pl.col(VALUE).rolling_var(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )


@dataclass(frozen=True)
class DailyZ(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyZ.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-z-4' [f64]
    [
        0.406191
        null
        null
        0.707107
        0.707107
        -1.141092
        -1.083191
        -1.346812
    ]
    """

    version: int = 0
    type: Literal["DailyZ"] = "DailyZ"

    SUFFIX: ClassVar[str] = "z"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyZSnap"] = "DailyZSnap"

    def handler_object_new(self) -> HandleNewDailyZ:
        return HandleNewDailyZ()


@dataclass(frozen=True)
class HandleNewDailyZ(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        mu = pl.col(VALUE).rolling_mean(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
        sig = pl.col(VALUE).rolling_std(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
        return ((pl.col(VALUE) - mu) / sig).fill_nan(None)


@dataclass(frozen=True)
class DailyZW(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyZW.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-zw-4' [f64]
    [
        0.406191
        null
        null
        0.707107
        0.707107
        -1.141092
        -1.083191
        -1.346812
    ]
    """

    version: int = 0
    type: Literal["DailyZW"] = "DailyZW"

    SUFFIX: ClassVar[str] = "zw"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyZWSnap"] = "DailyZWSnap"

    def handler_object_new(self) -> HandleNewDailyZ:
        return HandleNewDailyZ()


@dataclass(frozen=True)
class HandleNewDailyZW(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        params = obj.params
        mu = pl.col(VALUE).rolling_mean(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
        sig = (
            pl.col(VALUE)
            .forward_fill()
            .diff()
            .pipe(apis.pl.null_mask, pl.col(VALUE))
            .rolling_std(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
        )
        return ((pl.col(VALUE) - mu) / sig).fill_nan(None)


@dataclass(frozen=True)
class DailyRZParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    width: float = 0.25
    symmetric: bool = True
    rescale: bool = False

    def build_id(
        self,
        cls: Type[DailyRZ],
    ):
        window = self.window
        width = self.width
        symmetric = self.symmetric
        rescale = self.rescale
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return (
            f"{cls.SUFFIX}-"
            + params
            + ("" if width is None else f"-{width}")
            + ("-sym" if symmetric else "")
            + ("-norm" if rescale else "")
        )


@dataclass(frozen=True)
class DailyRZ(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRZ.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rz-4-0.25-sym' [f64]
    [
        0.285714
        null
        null
        2.0
        2.0
        -3.343284
        -1.512243
        -3.825922
    ]
    """

    params: DailyRZParams

    version: int = 0
    type: Literal["DailyRZ"] = "DailyRZ"

    SUFFIX: ClassVar[str] = "rz"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyRZSnap"] = "DailyRZSnap"

    def handler_object_new(self) -> HandleNewDailyRZ:
        return HandleNewDailyRZ()

    @classmethod
    def new(
        cls,
        parent: Daily,
        window: int | None = None,
        width: float = 0.25,
        symmetric: bool = True,
        rescale: bool = False,
        project: Project | None = None,
    ):
        params = DailyRZParams(
            window=window,
            width=width,
            symmetric=symmetric,
            rescale=rescale,
        )
        assert cls.SUFFIX != "M", cls
        id=parent.extend_id(params.build_id(cls))
        if project is not None:
            id=project.extend_id(path)
        return cls(
            id=id,
            parent=parent,
            params=params,
            project=project,
        )


@dataclass(frozen=True)
class HandleNewDailyRZ(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyRZ,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyRZ,
    ) -> pl.Expr:
        params = obj.params
        assert params.width is not None, params
        v = pl.col(VALUE)
        med = v.rolling_median(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
        if params.symmetric:
            q_up = v.rolling_quantile(
                0.5 + params.width,
                interpolation="linear",
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            q_dn = v.rolling_quantile(
                0.5 - params.width,
                interpolation="linear",
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            iqr = q_up - q_dn
            mad = iqr / 2
            # for symmetric, where median=average(up, dn):
            # iqr/2 == mad
        else:
            med = v.rolling_median(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            mad = ((v - med).abs()).rolling_median(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )

        if params.rescale:
            if not params.symmetric:
                mad = mad * 1.4826
            else:
                mad = mad * apis.stats.iqr_ratio(
                    params.width
                )

        return ((v - med) / mad).fill_nan(None)


@dataclass(frozen=True)
class DailyRZW(DailyRZ):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRZW.new(ord, window=4)
    >>> Graph().with_obj(obj)
    """

    # .sync()

    params: DailyRZParams

    version: int = 0
    type: Literal["DailyRZW"] = "DailyRZW"

    SUFFIX: ClassVar[str] = "rzw"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyRZWSnap"] = "DailyRZWSnap"

    def handler_object_new(self) -> HandleNewDailyRZW:
        return HandleNewDailyRZW()


@dataclass(frozen=True)
class HandleNewDailyRZW(HandleNewDailyM):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyRZ,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyRZ,
    ) -> pl.Expr:
        params = obj.params
        raise NotImplementedError()


#  ------------------

__all__ = [
    "P",
    "O",
    "DailyMParams",
    "DailyM",
    "HandleNewDailyM",
    "HandleNewDailyMA",
    "DailyMA",
    "HandleNewDailyKurt",
    "DailyKurt",
    "HandleNewDailyMax",
    "DailyMax",
    "HandleNewDailyMed",
    "DailyMed",
    "DailyMidParams",
    "DailyMid",
    "HandleNewDailyMid",
    "HandleNewDailyMin",
    "DailyMin",
    "HandleNewDailyPLast",
    "DailyPLast",
    "calc_p_last",
    "DailyQileParams",
    "DailyQile",
    "HandleNewDailyQile",
    "DailyQile",
    "HandleNewDailyRank",
    "DailyRank",
    "HandleNewDailyRms",
    "DailyRms",
    "DailyRngParams",
    "DailyRng",
    "HandleNewDailyRng",
    "HandleNewDailySDev",
    "DailySDev",
    "HandleNewDailySkew",
    "DailySkew",
    "HandleNewDailyVar",
    "DailyVar",
    "HandleNewDailyZ",
    "DailyZ",
    "HandleNewDailyRZ",
    "DailyRZ",
    "HandleNewDailyZW",
    "DailyZW",
    "HandleNewDailyRZW",
    "DailyRZW",
    "DailyOsc",
]
