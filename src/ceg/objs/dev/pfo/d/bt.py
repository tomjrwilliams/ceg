from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
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
from .....apis import assert_is, t, kv, logs
from .....engine import *

from ...agg import *

from ...calc import *
from .....apis.calcs.weights import pfo_weights_df

from .core import *

logger, log = logs.logger(
    __file__,
    streams=dict(formatter=True),
)

#  ------------------


@dataclass(frozen=True)
class Backtest(Combination):
    @classmethod
    def cast(cls, comb: Combination):
        # assert index > -1?
        return Backtest(
            comb.refs,
            comb.key,
            comb.params,
            comb.index,
        )

    def rx(self, coll: Container, t: Type[O] = Daily):
        return self.get_obj("rx", coll, t)


#  ------------------


def rx_price_types(
    univ: Container,
    comb: Combination,
):
    from ...bars import FutDailyRx, EqDailyRx, FxDailyRx
    from .pfo import DailyPfolioRx

    rx_ref = Ref.decode(assert_is(comb.refs.get("rx"), str))
    assert rx_ref.type == "obj", (comb, rx_ref)
    rx = context.get().get(univ.objects[rx_ref.i], Daily)
    if isinstance(
        rx,
        (
            FutDailyRx,
            EqDailyRx,
            FxDailyRx,
        ),
    ):
        return [type(rx)]
    elif isinstance(rx, DailyBtRx):
        return rx.get_parent().price_types()
    elif isinstance(rx, DailyPfolioRx):
        raise ValueError(rx)
    else:
        raise ValueError(rx)


def rx_price_symbols(
    univ: Container,
    comb: Combination,
):
    from ...bars import FutDailyRx, EqDailyRx, FxDailyRx
    from .pfo import DailyPfolioRx

    rx_ref = Ref.decode(assert_is(comb.refs.get("rx"), str))
    assert rx_ref.type == "obj", (comb, rx_ref)
    rx = context.get().get(univ.objects[rx_ref.i], Daily)
    if isinstance(
        rx,
        (
            FutDailyRx,
            EqDailyRx,
            FxDailyRx,
        ),
    ):
        px = rx.get_parent()
        return [px.tags.symbol]
    elif isinstance(rx, DailyBtRx):
        return rx.get_parent().price_symbols()
    elif isinstance(rx, DailyPfolioRx):
        raise ValueError(rx)
    else:
        raise ValueError(rx)


@dataclass(frozen=True)
class DailyBtXvar(DailyCombination):
    metric: str = "bt"  # TODO: or bt_cum?


@dataclass(frozen=True)
class DailyBtUvar(DailyBtXvar):
    metric: str = "bt-uv"  # TODO: or bt_cum?

    version: int = 0
    type: Literal["DailyBtUvar"] = "DailyBtUvar"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtUvarSnap"] = "DailyBtUvarSnap"

    @classmethod
    def new(
        cls,
        parent: Container,
        comb: int | Combination,
    ):
        return cls.new_(parent, "bt-uv", comb)

    def price_symbols(self):
        container= self.get_parent()
        try:
            comb = coll.comb[self.args[0]]
        except:
            raise ValueError((coll, self))
        return rx_price_symbols(coll, comb)

    def price_types(self):
        container= self.get_parent()
        comb = coll.comb[self.args[0]]
        return rx_price_types(coll, comb)

    def handler_object_new(self) -> HandleNewDailyBtUvar:
        return HandleNewDailyBtUvar()

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            return self.handler_object_new()
        raise ValueError((self, event))

    def params(self, coll: Container):
        assert len(self.args) == 1, self
        comb = coll.comb[self.args[0]]
        return assert_is(comb, Backtest)

    def rx(self):
        return DailyBtRx.new(self)

    def px(self):
        return DailyBtPx.new(self)

    def gross(self):
        return DailyBtGross.new(self)

    def pos(self):
        return DailyBtPos.new(self)

    def deriv(
        self,
        index: int | None = None,
        key: str | None = None,
    ):
        assert index is None, self
        if key == "pos":
            return DailyBtPos.new(self, key=key)
        elif key == "rx":
            return DailyBtPos.new(self, key=key)
        elif key == "px":
            return DailyBtPx.new(self, key=key)
        elif key == "gross":
            return DailyBtGross.new(self, key=key)
        return DailyCombinationDeriv.new(self, key=key, index=index)

    # def explorer_df(self):

    #     # TODO: return df with signal, scale, etc.

    #     # all stacked, relevant group keys

    #     return


@dataclass(frozen=True)
class DailyBtMvar(DailyBtXvar):
    metric: str = "bt-mv"  # TODO: or bt_cum?

    version: int = 0
    type: Literal["DailyBtMvar"] = "DailyBtMvar"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtMvarSnap"] = "DailyBtMvarSnap"

    @classmethod
    def new(
        cls,
        parent: Container,
        comb: tuple[int | Combination, ...],
    ):
        return cls.new_(parent, "bt-mv", comb)

    def price_symbols(self):
        container= self.get_parent()
        return list(
            set().union(
                *[
                    rx_price_symbols(coll, coll.comb[i])
                    for i in self.args[1:]
                ]
            )
        )

    def price_types(self):
        container= self.get_parent()
        return list(
            set().union(
                *[
                    rx_price_types(coll, coll.comb[i])
                    for i in self.args[1:]
                ]
            )
        )

    def handler_object_new(self) -> HandleNewDailyBtMvar:
        return HandleNewDailyBtMvar()

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            # assert self.get_parent().id == event.obj, (self, event)
            return self.handler_object_new()
        raise ValueError((self, event))

    def rx(self, index: int | None = None):
        return DailyBtRx.new(self, index)

    def px(self, index: int | None = None):
        return DailyBtPx.new(self, index)

    def gross(self, index: int | None = None):
        return DailyBtGross.new(self, index)

    def pos(self, index: int | None = None):
        return DailyBtPos.new(self, index)

    def params(self, coll: Container):
        comb, *combs = [
            assert_is(coll.comb[c], Backtest)
            for c in self.args
        ]
        return comb, combs

    def deriv(
        self,
        key: str | None = None,
        index: int | None = None,
    ):
        if key == "pos":
            return DailyBtPos.new(
                self, index=index, key=key
            )
        elif key == "rx":
            return DailyBtPos.new(
                self, index=index, key=key
            )
        elif key == "px":
            return DailyBtPx.new(self, index=index, key=key)
        elif key == "gross":
            return DailyBtGross.new(
                self, index=index, key=key
            )
        return DailyCombinationDeriv.new(self, key=key, index=index)

    # def explorer_df(self):

    #     # TODO: return df with signal, scale, etc.

    #     # all stacked, relevant group keys

    #     return


@dataclass(frozen=True)
class DailyBtWeights(DailyCombination):
    metric: str = "bt-w"

    version: int = 0
    type: Literal["DailyBtWeights"] = "DailyBtWeights"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtWeightsSnap"] = (
            "DailyBtWeightsSnap"
        )

    @classmethod
    def new(
        cls,
        parent: Container,
        comb: tuple[int | Combination, ...],
    ):
        return cls.new_(parent, "bt-w", comb)

    def price_symbols(self):
        container= self.get_parent()
        return list(
            set().union(
                *[
                    rx_price_symbols(coll, coll.comb[i])
                    for i in self.args[1:]
                ]
            )
        )

    def price_types(self):
        container= self.get_parent()
        return list(
            set().union(
                *[
                    rx_price_types(coll, coll.comb[i])
                    for i in self.args[1:]
                ]
            )
        )

    def handler_object_new(self) -> HandleNewDailyBtWeights:
        return HandleNewDailyBtWeights()

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            # assert self.get_parent().id == event.obj, (self, event)
            return self.handler_object_new()
        raise ValueError((self, event))

    def weight(self, key: str, index: int):
        return DailyBtWeight.new(self, index, key=key)

    def params(self, coll: Container):
        comb, *combs = [
            assert_is(coll.comb[c], Backtest)
            for c in self.args
        ]
        return comb, combs

    # TODO: method ot unpack param iteration
    # given index
    # given we pass tuples etc. so only run once per metric run

    def deriv(
        self,
        key: str | None = None,
        index: int | None = None,
    ):
        return DailyBtWeight.new(self, index, key)

    # def explorer_df(self):

    #     # TODO: return df with signal, scale, etc.

    #     # all stacked, relevant group keys

    #     return


#  ------------------


@dataclass(frozen=True)
class HandleNewDailyBtUvar(HandleNewDailyCombination):
    # TODO: shifts: ...
    unit: float = 1.0

    def comb_dfs(
        self,
        first: dt.date,
        last: dt.date,
        dfs_params: tuple[
            frozendict[str, pl.DataFrame],
            frozendict[str, TParamValue],
        ],
    ) -> (
        pl.DataFrame
        | tuple[
            pl.DataFrame, list[str | int | tuple[str, int]]
        ]
    ):
        dfs, params = dfs_params
        cap = params.get("cap", None)
        cap = cap if cap is None else float(cap)
        df = combine_backtest_dfs(
            assert_is(dfs.get("rx", None)),
            dfs.get("scale", None),
            dfs.get("signal", None),
            dfs.get("rebalance", None),
            dfs.get("calendar", None),
            dfs.get("weight", None),
            dfs.get("cap", None),
            dfs.get("fx", None),
            assert_is(dfs.get("px", None)),
            assert_is(dfs.get("gross", None)),
            # **shifts
            cap=cap,
        )

        df = df.pipe(
            run_backtest,
            signal=None
            if "signal" not in dfs
            else "signal",
            rebal=None
            if "rebalance" not in dfs
            else "rebalance",
            calendar=None
            if "calendar" not in dfs
            else "calendar",
            fx=None if "fx" not in dfs else "fx",
            cap=None if "cap" not in dfs else "cap",
            unit=self.unit,
        )

        return df, ["rx", "px", "gross", "pos"]


#  ------------------


def combine_mv_dfs(
    df_uv: pl.DataFrame | None,
    df_mv: pl.DataFrame | None,
    k: str,
):
    if df_uv is None:
        return df_mv
    elif df_mv is None:
        return df_uv
    return (
        df_uv.join(
            df_mv.rename({k: VALUE}),
            on="date",
            how="full",
            coalesce=True,
        )
        .sort(by="date")
        .select(
            "date", (pl.col(VALUE) * pl.col(k)).alias(k)
        )
    )


def combine_mv_scale(
    df_uv: pl.DataFrame | None,
    df_mv: pl.DataFrame | None,
):
    return combine_mv_dfs(df_uv, df_mv, "scale")


def combine_mv_signal(
    df_uv: pl.DataFrame | None,
    df_mv: pl.DataFrame | None,
):
    return combine_mv_dfs(df_uv, df_mv, "signal")


@dataclass(frozen=True)
class HandleNewDailyBtMvar(HandleNewDailyCombination):
    unit: float = 1.0

    def comb_dfs(
        self,
        first: dt.date,
        last: dt.date,
        *dfs_params: tuple[
            frozendict[str, pl.DataFrame],
            frozendict[str, TParamValue],
        ],
    ) -> (
        pl.DataFrame
        | tuple[
            pl.DataFrame, list[str | int | tuple[str, int]]
        ]
    ):

        dfs_ps_mv, *dfs_ps_uvs = dfs_params

        dfs_mv, ps_mv = dfs_ps_mv

        mv_df_scale = dfs_mv.get("scale")
        mv_df_signal = dfs_mv.get("signal")
        mv_df_rebalance = dfs_mv.get("rebalance")

        acc = []

        for i, dfs_p_uv in enumerate(dfs_ps_uvs):
            dfs_uv, p_uv = dfs_p_uv

            df_rx = assert_is(dfs_uv.get("rx", None))

            df_scale = dfs_uv.get("scale", None)
            df_scale = combine_mv_scale(
                df_scale, mv_df_scale
            )

            df_signal = dfs_uv.get("signal", None)
            df_signal = combine_mv_signal(
                df_signal, mv_df_signal
            )

            df_rebalance = dfs_uv.get("rebalance", None)
            df_rebalance = mv_df_rebalance or df_rebalance

            if df_scale is None:
                df_scale = df_rx.select(
                    "date", pl.lit(1.0).alias("scale")
                )

            cap = p_uv.get("cap", None)

            df = combine_backtest_dfs(
                df_rx,
                assert_is(df_scale),
                df_signal,
                df_rebalance,
                dfs_uv.get("calendar", None),
                dfs_uv.get("weight", None),
                dfs_uv.get("cap", None),
                dfs_uv.get("fx", None),
                dfs_uv.get("px", None),
                dfs_uv.get("gross", None),
                cap=(cap if cap is None else float(cap)),
                # **shifts
            )

            df = df.pipe(
                run_backtest,
                signal=None
                if "signal" not in dfs_uv
                else "signal",
                rebal=None
                if "rebalance" not in dfs_uv
                else "rebalance",
                calendar=None
                if "calendar" not in dfs_uv
                else "calendar",
                fx=None if "fx" not in dfs_uv else "fx",
                cap=None if "cap" not in dfs_uv else "cap",
                unit=self.unit,
            )

            acc.append(df)

        n = len(dfs_ps_uvs)

        res = (
            apis.pl.join_dfs(
                [
                    df.select(
                        "date", "pos", "gross", "px", "rx"
                    ).rename(
                        {
                            "gross": f"gross_{i}",
                            "px": f"px_{i}",
                            "pos": f"pos_{i}",
                            "rx": f"rx_{i}",
                        }
                    )
                    for i, df in enumerate(acc)
                ],
                align=True,
                on="date",
            )
            .with_columns(
                pl.all_horizontal(
                    [
                        pl.col(f"rx_{i}").is_null()
                        for i in range(n)
                    ]
                ).alias("mask"),
                pl.sum_horizontal(
                    *[pl.col(f"rx_{i}") for i in range(n)]
                ).alias("rx"),
                pl.all_horizontal(
                    [
                        pl.col(f"px_{i}").is_null()
                        for i in range(n)
                    ]
                ).alias("px_mask"),
                pl.sum_horizontal(
                    *[
                        pl.col(f"px_{i}").forward_fill()
                        for i in range(n)
                    ]
                ).alias("px"),
                pl.all_horizontal(
                    [
                        pl.col(f"gross_{i}").is_null()
                        for i in range(n)
                    ]
                ).alias("gross_mask"),
                pl.sum_horizontal(
                    *[
                        pl.col(f"gross_{i}").forward_fill()
                        for i in range(n)
                    ]
                ).alias("gross"),
            )
            .with_columns(
                pl.when(pl.col("mask"))
                .then(None)
                .otherwise("rx")
                .alias("rx"),
                pl.when(pl.col("px_mask"))
                .then(None)
                .otherwise("px")
                .alias("px"),
                pl.when(pl.col("gross_mask"))
                .then(None)
                .otherwise("gross")
                .alias("gross"),
            )
            .drop("mask", "px_mask", "gross_mask")
        )

        k_is: list[str | int | tuple[str, int]] = (
            ["rx", "px", "gross", "px"]
            + [
                (k, i)
                for i in list(range(n))
                for k in ["rx", "pos", "gross", "px"]
            ]
            + [0][1:]
        )

        return res, k_is


@dataclass(frozen=True)
class HandleNewDailyBtWeights(HandleNewDailyCombination):
    unit: float = 1.0

    def comb_dfs(
        self,
        first: dt.date,
        last: dt.date,
        *dfs_params: tuple[
            frozendict[str, pl.DataFrame],
            frozendict[str, TParamValue],
        ],
    ) -> (
        pl.DataFrame
        | tuple[
            pl.DataFrame, list[str | int | tuple[str, int]]
        ]
    ):

        dfs_ps_mv, *dfs_ps_uvs = dfs_params

        dfs_mv, ps_mv = dfs_ps_mv

        log("INFO")(ps=ps_mv)

        mv_df_scale = dfs_mv.get("scale")
        mv_df_signal = dfs_mv.get("signal")
        mv_df_rebalance = dfs_mv.get("rebalance")

        dfs = []
        params = []

        for i, dfs_p_uv in enumerate(dfs_ps_uvs):
            dfs_uv, p_uv = dfs_p_uv

            params.append(p_uv)

            df_rx = assert_is(dfs_uv.get("rx", None))

            df_scale = dfs_uv.get("scale", None)
            df_scale = combine_mv_scale(
                df_scale, mv_df_scale
            )

            df_signal = dfs_uv.get("signal", None)
            df_signal = combine_mv_signal(
                df_signal, mv_df_signal
            )

            df_rebalance = dfs_uv.get("rebalance", None)
            df_rebalance = mv_df_rebalance or df_rebalance

            if df_scale is None:
                df_scale = df_rx.select(
                    "date", pl.lit(1.0).alias("scale")
                )

            cap = p_uv.get("cap", None)

            # TODO: if only rx, skip backtest

            df = combine_backtest_dfs(
                df_rx,
                assert_is(df_scale),
                df_signal,
                df_rebalance,
                dfs_uv.get("calendar", None),
                dfs_uv.get("weight", None),
                dfs_uv.get("cap", None),
                dfs_uv.get("fx", None),
                dfs_uv.get("px", None),
                dfs_uv.get("gross", None),
                cap=(cap if cap is None else float(cap)),
                # **shifts
            )

            df = df.pipe(
                run_backtest,
                signal=None
                if "signal" not in dfs_uv
                else "signal",
                rebal=None
                if "rebalance" not in dfs_uv
                else "rebalance",
                calendar=None
                if "calendar" not in dfs_uv
                else "calendar",
                fx=None if "fx" not in dfs_uv else "fx",
                cap=None if "cap" not in dfs_uv else "cap",
                unit=self.unit,
            )

            dfs.append(df)

        n = len(dfs_ps_uvs)

        return pfo_weights_df(
            first,
            last,
            dfs,
            params,
            ps_mv,
        )  # type: ignore


#  ------------------


@dataclass(frozen=True)
class DailyBtPos(
    DailyCombinationDeriv,
):
    parent: DailyBtUvar | DailyBtMvar | str

    key: str = "pos"

    version: int = 0
    type: Literal["DailyBtPos"] = "DailyBtPos"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtPosSnap"] = "DailyBtPosSnap"

    def get_parent(self) -> DailyBtUvar | DailyBtMvar:
        return context.get().get(
            self.parent, DailyBtUvar | DailyBtMvar
        )


@dataclass(frozen=True)
class DailyBtRx(
    DailyCombinationDeriv,
):
    parent: DailyBtUvar | DailyBtMvar | str

    key: str = "rx"

    version: int = 0
    type: Literal["DailyBtRx"] = "DailyBtRx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtRxSnap"] = "DailyBtRxSnap"

    def get_parent(self) -> DailyBtUvar | DailyBtMvar:
        return context.get().get(
            self.parent, DailyBtUvar | DailyBtMvar
        )


@dataclass(frozen=True)
class DailyBtPx(
    DailyCombinationDeriv,
):
    parent: DailyBtUvar | DailyBtMvar | str

    key: str = "px"

    version: int = 0
    type: Literal["DailyBtPx"] = "DailyBtPx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtPxSnap"] = "DailyBtPxSnap"

    def get_parent(self) -> DailyBtUvar | DailyBtMvar:
        return context.get().get(
            self.parent, DailyBtUvar | DailyBtMvar
        )


@dataclass(frozen=True)
class DailyBtGross(
    DailyCombinationDeriv,
):
    parent: DailyBtUvar | DailyBtMvar | str

    key: str = "gross"

    version: int = 0
    type: Literal["DailyBtGross"] = "DailyBtGross"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtGrossSnap"] = (
            "DailyBtGrossSnap"
        )

    def get_parent(self) -> DailyBtUvar | DailyBtMvar:
        return context.get().get(
            self.parent, DailyBtUvar | DailyBtMvar
        )


@dataclass(frozen=True)
class DailyBtWeight(
    DailyCombinationDeriv,
):
    parent: DailyBtWeights | str

    # NOTE: key is the param comb
    # key: str = "weight"

    version: int = 0
    type: Literal["DailyBtWeight"] = "DailyBtWeight"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtWeightSnap"] = (
            "DailyBtWeightSnap"
        )

    def get_parent(self) -> DailyBtWeights:
        return context.get().get(
            self.parent, DailyBtWeights
        )


#  ------------------
# New typed bt types — parented directly on StrategyTyped / BasketTyped.
# No DailyCombination / Container.comb / args[idx] indirection.
# Handlers read refs directly off the parent's typed slot fields.
# Legacy DailyBtUvar / DailyBtMvar / DailyBtWeights stay above for now.
#  ------------------


@dataclass(frozen=True)
class DailyBtUvarTyped(DailyChunked, DependsParentValue):
    """Univariate backtest parented on a StrategyTyped namespace.

    Depends on the strategy's ValueUpdate (which fires once all its data
    slots have first-published, via TriageUpstream). The handler reads
    strategy.rx / px / gross / signal / scale / cap directly.
    """

    # parent type kept loose at the bt level; concrete typing happens at the
    # call site / get_parent.
    parent: "Object | str"

    version: int = 0
    type: Literal["DailyBtUvarTyped"] = "DailyBtUvarTyped"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtUvarTypedSnap"] = "DailyBtUvarTypedSnap"

    @classmethod
    def new(cls, parent):
        return cls(id=parent.extend_id("bt-uv"), parent=parent)

    def get_parent(self):
        # StrategyTyped resolved at call time to avoid circular imports
        from ...strat.params import StrategyTyped
        return context.get().get(self.parent, StrategyTyped)

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            return HandleNewDailyBtUvarTyped()
        raise ValueError((self, event))

    def rx(self):    return DailyBtRx.new(self, key="rx")
    def px(self):    return DailyBtPx.new(self, key="px")
    def gross(self): return DailyBtGross.new(self, key="gross")
    def pos(self):   return DailyBtPos.new(self, key="pos")


@dataclass(frozen=True)
class HandleNewDailyBtUvarTyped(Handler):
    """Univariate backtest handler.

    Reads the parent StrategyTyped's typed slot Refs, fetches each Daily
    series, runs combine_backtest_dfs and writes the bt's value plus
    fan-out children (DailyBtRx/Px/Gross/Pos).

    Body deferred — needs framework data pipeline; this stub keeps the
    construction/dispatch contract validated by tests.
    """

    def handle(
        self,
        event: ValueUpdate,
        snap: "DailyBtUvarTyped.Snap",
        obj: "DailyBtUvarTyped",
    ):
        raise NotImplementedError(
            "DailyBtUvarTyped handler body deferred to follow-up; the "
            "construction / depends() / dispatch shape is established."
        )


@dataclass(frozen=True)
class DailyBtMvarTyped(DailyChunked, DependsParentValue):
    """Multivariate backtest parented on a BasketTyped namespace.

    Depends on the basket's ValueUpdate (which fires once every member
    strategy has first-published). The handler iterates basket.members,
    fetches each Strategy's rx/scale/etc., combines them.
    """

    parent: "Object | str"

    version: int = 0
    type: Literal["DailyBtMvarTyped"] = "DailyBtMvarTyped"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtMvarTypedSnap"] = "DailyBtMvarTypedSnap"

    @classmethod
    def new(cls, parent):
        return cls(id=parent.extend_id("bt-mv"), parent=parent)

    def get_parent(self):
        from ...strat.params import BasketTyped
        return context.get().get(self.parent, BasketTyped)

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            return HandleNewDailyBtMvarTyped()
        raise ValueError((self, event))

    def rx(self, index: int | None = None):
        return DailyBtRx.new(self, key="rx", index=index)

    def px(self, index: int | None = None):
        return DailyBtPx.new(self, key="px", index=index)

    def gross(self, index: int | None = None):
        return DailyBtGross.new(self, key="gross", index=index)

    def pos(self, index: int | None = None):
        return DailyBtPos.new(self, key="pos", index=index)


@dataclass(frozen=True)
class HandleNewDailyBtMvarTyped(Handler):
    """Multivariate backtest handler. Body deferred (same as Uvar)."""

    def handle(
        self,
        event: ValueUpdate,
        snap: "DailyBtMvarTyped.Snap",
        obj: "DailyBtMvarTyped",
    ):
        raise NotImplementedError(
            "DailyBtMvarTyped handler body deferred to follow-up."
        )


__all__ = [
    "Backtest",
    "DailyBtUvar",
    "DailyBtMvar",
    "DailyBtWeights",
    "DailyBtPos",
    "DailyBtRx",
    "DailyBtPx",
    "DailyBtWeight",
    "DailyBtGross",
    "DailyBtWeight",
    "HandleNewDailyBtUvar",
    "HandleNewDailyBtMvar",
    # New typed forms (alongside legacy)
    "DailyBtUvarTyped",
    "DailyBtMvarTyped",
    "HandleNewDailyBtUvarTyped",
    "HandleNewDailyBtMvarTyped",
]
