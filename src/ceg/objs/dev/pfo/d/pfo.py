
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
from .....apis import t
from .....apis import kv
from .....engine import *

from ...calc import *

from ..bt import *
from .core import *
from .bt import *

#  ------------------


@dataclass(frozen=True)
class DailyPfolio(
    UseRootConfigs,
    DependsSelfNew,
    DependsParent
):
    parent: DailyBtUvar | DailyBtMvar | str

    version: int = 0
    type: Literal["DailyPfolio"] = "DailyPfolio"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyPfolioSnap"] = "DailyPfolioSnap"

    def get_parent(
        self, domain: Domain
    ) -> DailyBtUvar | DailyBtMvar:
        return domain.get(self.parent, DailyBtUvar | DailyBtMvar)
    
    def get_snap(
        self,
        domain: Domain,
        t: Type[Daily.Snap] | None = None,
    ) -> DailyPfolio.Snap | None:
        return self.snap_key_val(domain).get(self.path, DailyPfolio.Snap)
    
    @classmethod
    def new(
        cls,
        parent: DailyBtUvar | DailyBtMvar,
    ):
        return cls(
            parent=parent,
            path=parent.extend_path("pf"),
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_parent(domain)
        )
    
    def depends_parent(self, domain: Domain) -> list[Dependency]:
        parent = self.get_parent(domain)
        rx = DailyBtRx.new(parent)
        return [
            Dependency(
                self.path,
                rx.path,
                events=("ValueUpdate",)
                # TODO: append, prepend?
            )
        ]

    def handler_object_new(
        self
    ) -> HandleNewDailyPfolio:
        return HandleNewDailyPfolio()

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, ValueUpdate):
            # assert self.get_parent(domain).path == event.obj, (self, event)
            res.append(self.handler_object_new())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res
    
    # def explorer_df(self, domain: Domain):

    #     # TODO: return df with signal, scale, etc.

    #     # all stacked, relevant group keys

    #     return    

def recurse_bt_pos(
    bt_pos: DailyBtPos,
    domain: Domain,
    spec: Backtest,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):
    df_pos = daily_pl_values(
        domain,
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
            .mul(1 if parent is None else pl.col(parent))
            .alias(VALUE)
        )
    else:
        acc = df_pos

    univ = bt_pos.get_parent(domain).get_parent(domain)

    if spec.key is not None:
        assert spec.rx is not None, spec
        obj_rx = univ.objects[spec.rx]
        assert isinstance(obj_rx, str), spec

        key = f"{spec.key}={obj_rx}"

        if acc is None:
            return df_pos.rename({VALUE: key})
        return acc.with_columns(
            pl.col(VALUE)
            .add((
                0 
                if key not in acc.columns
                else pl.col(key)
            ))
            .alias(key)
        ).drop(VALUE)
    
    parent = bt_pos.path
    acc = acc.rename({VALUE: bt_pos.path})

    univ = bt_pos.get_parent(domain).get_parent(domain)

    assert spec.rx is not None, spec
    obj_rx = domain.get(
        univ.objects[spec.rx],
        DailyBtRx,
    )
    assert obj_rx.index is None, obj_rx
    # TODO: ie. no mvar individual bt, only total or uvar

    return acc_bt_pos(
        obj_rx.get_parent(domain),
        domain,
        parent=parent,
        acc=acc,
    ).drop(parent)

def acc_bt_pos_uvar(
    bt: DailyBtUvar,
    domain: Domain,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):

    bt_pos = DailyBtPos.new(bt)

    univ = bt.get_parent(domain)
    spec = univ.bt[univ.uvar[bt.index].bt]

    return recurse_bt_pos(
        bt_pos, domain, spec, parent=parent, acc=acc
    )

def acc_bt_pos_mvar(
    bt: DailyBtMvar,
    domain: Domain,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):
    univ = bt.get_parent(domain)
    specs = [
        univ.bt[i] for i in univ.mvar[bt.index].bts
    ]
    for i, spec in enumerate(specs):
        bt_pos = DailyBtPos.new(bt, index=i)
        
        acc = recurse_bt_pos(
            bt_pos,
            domain,
            spec,
            parent=parent,
            acc = acc
        )

    assert isinstance(acc, pl.DataFrame), bt
    return acc

def acc_bt_pos(
    bt: DailyBtUvar | DailyBtMvar,
    domain: Domain,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if isinstance(bt, DailyBtUvar):
        return acc_bt_pos_uvar(bt, domain, parent, acc)
    elif isinstance(bt, DailyBtMvar):
        return acc_bt_pos_mvar(bt, domain, parent, acc)
    else:
        raise ValueError(bt)

@dataclass(frozen=True)
class HandleNewDailyPfolio(
    HandleNewValueFromPl
):

    def handle(
        self,
        event: ObjectNew,
        obj: DailyPfolio,
        domain: Domain,
    ):

        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None
            
        bt = obj.get_parent(domain)

        df_pos = acc_bt_pos(bt, domain).sort(by="date")

        (first, last) = df_pos.pipe(apis.pl.select_min_max, "date", dt.date)

        dfs_rx = []

        for k_rx in df_pos.columns:
            if k_rx == "date":
                continue

            df_pos = df_pos.with_columns(
                pl.col(k_rx).alias("pos")
            )
                
            k, rx_path = k_rx.split("=")
            obj_rx = domain.get(rx_path, Daily)

            df_rx = daily_pl_df(
                obj_rx, domain, alias="rx"
            )
            
            obj_pos = DailyPfolioPos.new(obj, k)
            obj_rx = DailyPfolioRx.new(obj, k)

            e_pos, _ = self.set_value_from_df(
                obj_pos,
                domain,
                df_pos.select("date", "pos"),
                "pos",
                t = self.key_value_type(obj, domain),
                publish=True,
            )

            df_rx = df_rx.join(
                df_pos.select("date", "pos"),
                on="date",
                how="full",
                coalesce=True,
            ).sort(by="date")

            v_pos = df_rx.get_column("pos").to_numpy()
            v_rx = df_rx.get_column("rx").to_numpy()

            df_rx = df_rx.with_columns(
                pl.Series(
                    apis.perf.backtest_rx_pos(v_rx, v_pos)
                ).alias("rx").fill_nan(None)
            ).drop("pos")

            e_rx, _ = self.set_value_from_df(
                obj_rx,
                domain,
                df_rx,
                "rx",
                t = self.key_value_type(obj, domain),
                publish=True,
            )

            (
                obj_pos.snap_key_val(domain)
                .set(obj_pos.path, type(obj_pos).Snap(
                    event,
                    obj_pos.path,
                    dt.datetime.now(),
                    first,
                    last,
                ))
            )
            (
                obj_rx.snap_key_val(domain)
                .set(obj_rx.path, type(obj_rx).Snap(
                    event,
                    obj_rx.path,
                    dt.datetime.now(),
                    first,
                    last,
                ))
            )

            dfs_rx.append(df_rx.rename({"rx": k_rx}))

        df_rx = apis.pl.join_dfs(
            *dfs_rx,
            on="date",
            how="full",
            coalesce=True,
        ).sort(by="date")

        df_rx = df_rx.select(
            "date",
            pl.when(
                pl.all_horizontal([
                    pl.col(k).is_null()
                    for k in df_rx.columns
                    if k != "date"
                ])
            )
            .then(None)
            .otherwise(pl.sum_horizontal(
                pl.exclude("date")
            ))
            .alias("rx")
        )

        (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event,
                obj.path,
                dt.datetime.now(),
                first,
                last,
            ), DailyPfolio.Snap)
        )

        pfo_rx = DailyPfolioRx.new(obj)
        e_rx, _ = self.set_value_from_df(
            pfo_rx,
            domain,
            df_rx,
            "rx",
            t = self.key_value_type(obj, domain),
            publish=True,
        )
        (
            pfo_rx.snap_key_val(domain)
            .set(pfo_rx.path, type(pfo_rx).Snap(
                event,
                pfo_rx.path,
                dt.datetime.now(),
                first,
                last,
            ), DailyPfolioRx.Snap)
        )

        return e_pos, e_rx
    
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
        type: Literal["DailyPfolioPosSnap"] = "DailyPfolioPosSnap"

    def get_parent(self, domain: Domain) -> DailyPfolio:
        return domain.get(
            self.parent, DailyPfolio
        )

    @classmethod
    def new(
        cls,
        parent: DailyPfolio,
        key: str,
    ):
        return cls(
            parent=parent,
            path=parent.extend_path(f"pos-{key}"),
            key=key,
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
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
        type: Literal["DailyPfolioRxSnap"] = "DailyPfolioRxSnap"

    def get_parent(self, domain: Domain) -> DailyPfolio:
        return domain.get(
            self.parent, DailyPfolio
        )

    @classmethod
    def new(
        cls,
        parent: DailyPfolio,
        key: str | None = None,
    ):
        path = "rx" if key is None else f"rx-{key}"
        return cls(
            parent=parent,
            path=parent.extend_path(path),
            key=key,
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return []
    
#  ------------------

__all__ = [
    "DailyPfolio",
    "DailyPfolioPos",
    "DailyPfolioRx",
    "HandleNewDailyPfolio",
]

# TODO:

# re run backtest given pos series per concrete, check matches sum rx

# and chek that sum matches rx sum