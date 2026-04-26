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

from ...calc import *

from .core import *
from .bt import *

logger, log = logs.logger(
    __file__,
    streams=dict(formatter=True),
)

#  ------------------


@dataclass(frozen=True)
class DailyPfolio(UseRootStorage, DependsParent):
    parent: DailyBtUvar | DailyBtMvar | str

    version: int = 0
    type: Literal["DailyPfolio"] = "DailyPfolio"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyPfolioSnap"] = "DailyPfolioSnap"

    def get_parent(self) -> DailyBtUvar | DailyBtMvar:
        return context.get().get(
            self.parent, DailyBtUvar | DailyBtMvar
        )

    @classmethod
    def new(
        cls,
        parent: DailyBtUvar | DailyBtMvar,
    ):
        return cls(
            parent=parent,
            id=parent.extend_id("pf"),
        )

    def depends(self) -> list[Dependency]:
        parent = self.get_parent()
        rx = DailyBtRx.new(parent)
        return [
            Dependency(
                self.id,
                rx.id,
                events=("ValueUpdate",),
                # TODO: append, prepend?
            )
        ]

    def handler_object_new(self) -> HandleNewDailyPfolio:
        return HandleNewDailyPfolio()

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            # assert self.get_parent().id == event.obj, (self, event)
            return self.handler_object_new()
        raise ValueError((self, event))

    # def explorer_df(self):

    #     # TODO: return df with signal, scale, etc.

    #     # all stacked, relevant group keys

    #     return


def recurse_bt_pos(
    bt_pos: DailyBtPos,
    spec: Backtest,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):
    df_pos = daily_pl_values(
        "df",
        bt_pos,
        alias=VALUE,
    )
    if acc is not None:
        acc = acc.join(
            df_pos,
            on="date",
            how="full",
            coalesce=True,
        ).with_columns(
            pl.col(VALUE)
            .mul(
                1
                if parent is None
                else pl.col(parent)
                .fill_nan(None)
                .fill_null(1)
            )
            .alias(VALUE)
        )
    else:
        acc = df_pos

    univ = bt_pos.get_parent().get_parent()

    # spec has to be from the comb?
    # rx is then bt.parent().params().rx(coll).id

    if spec.key is not None:
        obj_rx = spec.rx(univ, Daily).id
        # assert spec.rx is not None, spec
        # obj_rx = univ.objects[spec.rx]
        assert isinstance(obj_rx, str), spec

        key = f"{spec.key}={obj_rx}"

        if acc is None:
            return df_pos.rename({VALUE: key})
        return acc.with_columns(
            pl.col(VALUE)
            .add(
                (
                    0
                    if key not in acc.columns
                    else pl.col(key)
                    .fill_nan(None)
                    .fill_null(0)
                )
            )
            .alias(key)
        ).drop(VALUE)

    parent = bt_pos.path
    acc = acc.rename({VALUE: bt_pos.path})

    univ = bt_pos.get_parent().get_parent()

    # assert spec.rx is not None, spec
    obj_rx = spec.rx(univ, DailyBtRx)
    # obj_rx = context.get().get(
    #     univ.objects[spec.rx],
    #     DailyBtRx,
    # )
    assert obj_rx.index is None, obj_rx
    # TODO: ie. no mvar individual bt, only total or uvar

    return acc_bt_pos(
        obj_rx.get_parent(),
        parent=parent,
        acc=acc,
    ).drop(parent)


def acc_bt_pos_uvar(
    bt: DailyBtUvar,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):

    bt_pos = DailyBtPos.new(bt)

    assert len(bt.args) == 1, bt

    univ = bt.get_parent()
    # spec = univ.bt[univ.uvar[bt.index].bt]
    spec = Backtest.cast(univ.comb[bt.args[0]])

    return recurse_bt_pos(
        bt_pos, spec, parent=parent, acc=acc
    )


def acc_bt_pos_mvar(
    bt: DailyBtMvar,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):
    univ = bt.get_parent()
    specs = [
        # univ.bt[i] for i in univ.mvar[bt.index].bts
        Backtest.cast(univ.comb[i])
        for i in bt.args[1:]
    ]
    for i, spec in enumerate(specs):
        bt_pos = DailyBtPos.new(bt, index=i)

        acc = recurse_bt_pos(
            bt_pos, spec, parent=parent, acc=acc
        )

    assert isinstance(acc, pl.DataFrame), bt
    return acc


def acc_bt_pos(
    bt: DailyBtUvar | DailyBtMvar,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if isinstance(bt, DailyBtUvar):
        return acc_bt_pos_uvar(bt, parent, acc)
    elif isinstance(bt, DailyBtMvar):
        return acc_bt_pos_mvar(bt, parent, acc)
    else:
        raise ValueError(bt)


@dataclass(frozen=True)
class DailyPfolioSeries:
    key: str
    rx_id: str
    asset: t.pl.DataFrame  # rx
    rx: t.pl.DataFrame  # scaled
    pos: t.pl.DataFrame


@dataclass(frozen=True)
class DailyPfolioData:
    bt: DailyBtUvar | DailyBtMvar
    start: dt.date
    end: dt.date
    series: list[DailyPfolioSeries]


def get_daily_pfolio_data(
    obj: DailyPfolio,
):
    bt = obj.get_parent()

    df_pos = acc_bt_pos(bt).sort(by="date")

    (first, last) = df_pos.pipe(
        apis.pl.select_min_max, "date", dt.date
    )
    df_pos = pl.select(
        pl.date_range(first, last).alias("date")
    ).join(df_pos, on="date", how="left", coalesce=True)

    res = []

    for k_rx in df_pos.columns:
        if k_rx == "date":
            continue

        df_pos = df_pos.with_columns(
            pl.col(k_rx).alias("pos")
        )

        # log(get_pfo_data=k_rx)

        k, rx_path = k_rx.split("=")
        obj_rx = context.get().get(rx_path, Daily)

        df_asset = daily_pl_df(obj_rx, alias="rx")

        df_rx = df_asset.join(
            df_pos.select("date", "pos"),
            on="date",
            how="full",
            coalesce=True,
        ).sort(by="date")

        v_pos, v_pos_null = apis.pl.to_numpy_with_null(
            df_rx.get_column("pos")
        )
        v_rx, v_rx_null = apis.pl.to_numpy_with_null(
            df_rx.get_column("rx")
        )

        df_rx = df_rx.with_columns(
            pl.Series(
                apis.perf.backtest_rx_pos(
                    v_rx,
                    v_pos,
                    v_rx_null,
                    v_pos_null,
                )
            )
            .alias("rx")
            .fill_nan(None)
        ).drop("pos")

        # TODO: check sum!

        res.append(
            DailyPfolioSeries(
                k,
                rx_path,
                # k_rx = {k}={rx_path}
                t.pl.DataFrame.cast(df_asset),
                t.pl.DataFrame.cast(df_rx),
                t.pl.DataFrame.cast(df_pos),
            )
        )

    return DailyPfolioData(bt, first, last, res)


@dataclass(frozen=True)
class HandleNewDailyPfolio(HandleNewValueFromPl):
    def handle(
        self,
        event: ObjectChange,
        snap: DailyPfolio.Snap,
        obj: DailyPfolio,
    ):

        data = assert_is(get_daily_pfolio_data(obj))

        dfs_rx = []

        res: frozendict[str, tuple[Snap, Event]] = (
            frozendict()
        )

        for ser in data.series:
            k = ser.key

            obj_pos = DailyPfolioPos.new(obj, k)
            obj_rx = DailyPfolioRx.new(obj, k)

            context.get().with_obj(obj_pos)
            context.get().with_obj(obj_rx)

            # log(handle_pfo=k)

            _ = self.set_value_from_df(
                obj_pos,
                ser.pos.select("date", "pos"),
                "pos",
                t=self.key_value_type(obj),
            )

            _ = self.set_value_from_df(
                obj_rx,
                ser.rx,
                "rx",
                t=self.key_value_type(obj),
            )

            res = ValueUpdate.from_snap(
                replace(
                    obj_rx.get_snap(DailyPfolioRx.Snap),
                    first=data.start,
                    last=data.end,
                ),
                into=res,
            )
            res = ValueUpdate.from_snap(
                replace(
                    obj_pos.get_snap(DailyPfolioPos.Snap),
                    first=data.start,
                    last=data.end,
                ),
                into=res,
            )

            dfs_rx.append(
                ser.rx.rename(
                    {"rx": f"{ser.key}={ser.rx_path}"}
                )
            )

        df_rx = apis.pl.join_dfs(
            dfs_rx,
            on="date",
            how="full",
            coalesce=True,
        ).sort(by="date")

        df_rx = df_rx.select(
            "date",
            pl.when(
                pl.all_horizontal(
                    [
                        pl.col(k).is_null()
                        for k in df_rx.columns
                        if k != "date"
                    ]
                )
            )
            .then(None)
            .otherwise(
                pl.sum_horizontal(pl.exclude("date"))
            )
            .alias("rx"),
        )

        res = ValueUpdate.from_snap(
            replace(
                snap,
                first=data.start,
                last=data.end,
            ),
            into=res,
        )

        pfo_rx = DailyPfolioRx.new(obj)
        context.get().with_obj(pfo_rx)

        _ = self.set_value_from_df(
            pfo_rx,
            df_rx,
            "rx",
            t=self.key_value_type(obj),
        )

        res = ValueUpdate.from_snap(
            replace(
                pfo_rx.get_snap(DailyPfolioRx.Snap),
                first=data.start,
                last=data.end,
            ),
            into=res,
        )
        return res


#  ------------------


@dataclass(frozen=True)
class DailyPfolioPos(
    DailyChunked,
):
    parent: DailyPfolio | str

    key: str

    version: int = 0
    type: Literal["DailyPfolioPos"] = "DailyPfolioPos"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyPfolioPosSnap"] = (
            "DailyPfolioPosSnap"
        )

    def get_parent(self) -> DailyPfolio:
        return context.get().get(self.parent, DailyPfolio)

    @classmethod
    def new(
        cls,
        parent: DailyPfolio,
        key: str,
    ):
        return cls(
            parent=parent,
            id=parent.extend_id(f"pos-{key}"),
            key=key,
        )

    def depends(self) -> list[Dependency]:
        return []


@dataclass(frozen=True)
class DailyPfolioRx(
    DailyChunked,
):
    parent: DailyPfolio | str

    key: str | None = None

    version: int = 0
    type: Literal["DailyPfolioRx"] = "DailyPfolioRx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyPfolioRxSnap"] = (
            "DailyPfolioRxSnap"
        )

    def get_parent(self) -> DailyPfolio:
        return context.get().get(self.parent, DailyPfolio)

    @classmethod
    def new(
        cls,
        parent: DailyPfolio,
        key: str | None = None,
    ):
        id="rx" if key is None else f"rx-{key}"
        return cls(
            parent=parent,
            id=parent.extend_id(path),
            key=key,
        )

    def depends(self) -> list[Dependency]:
        return []


#  ------------------

__all__ = [
    "DailyPfolio",
    "DailyPfolioPos",
    "DailyPfolioRx",
    "HandleNewDailyPfolio",
    "DailyPfolioData",
    "get_daily_pfolio_data",
]

# TODO:

# re run backtest given pos series per concrete, check matches sum rx

# and chek that sum matches rx sum
