
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

#  ------------------


@dataclass(frozen=True)
class DailyBtUvarValues(BtValues):
    # i: int
    pos: str
    rx: str

@dataclass(frozen=True)
class DailyBtUvarResult(BtResult):
    res: DailyBtUvarValues | None = None
    
@dataclass(frozen=True)
class DailyBtMvarValues(BtValues):
    agg: str
    pos: tuple[str, ...]
    rx: tuple[str, ...]

@dataclass(frozen=True)
class DailyBtMvarResult(BtResult):
    res: DailyBtMvarValues | None = None

#  ------------------

@dataclass(frozen=True)
class UniverseDaily(Universe):

    type: Literal["UniverseDaily"] = "UniverseDaily"

    def into_uvar(
        self,
        res: BtResult
    ):
        assert isinstance(res, DailyBtUvarResult), res
        assert isinstance(self, UniverseDaily), self
        assert res.i is not None, res
        bt = DailyBtUvar.new(self, res.i)
        bt_pos = DailyBtPos.new(bt)
        bt_rx = DailyBtRx.new(bt)
        res = replace(
            res,
            res = DailyBtUvarValues(
                # res.i,
                bt_pos.path,
                bt_rx.path,
            )
        ).with_obj((bt_pos, bt_rx))
        return self.with_uvar(BtUvar(bt.index)), res

@dataclass(frozen=True)
class DailyBtXvar(
    UseRootConfigs,
    DependsSelfNew,
    DependsParentValue
):
    parent: UniverseDaily | str

    index: int # of a uvar line in parent

    rescale: Daily | str | None = None

    def get_parent(self, domain: Domain) -> UniverseDaily:
        return domain.get(self.parent, UniverseDaily)
    
    @classmethod
    def new(
        cls,
        parent: Universe | UniverseDaily,
        index: int,
        rescale: Daily | str | None = None,
    ):
        assert isinstance(parent, UniverseDaily), (
            parent, cls
        )
        index_pad = str(index).zfill(3)
        path = f"bt-{index_pad}"
        if rescale is not None:
            assert isinstance(rescale, DailyERms), rescale
            rescale_path = obj_path(rescale.parent)
            assert rescale_path == cls.new(
                parent, index=index, rescale=None
            ).path + "-rx", (rescale, rescale_path)
            rescale_span = rescale.path.split("-erms")[-1]
            path = path + "-sc-erms-" + rescale_span
        return cls(
            parent=parent,
            path=parent.extend_path(path),
            index=index,
            rescale=rescale,
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        res = (
            self.depends_self(domain)
            + self.depends_parent(domain)
        )
        if self.rescale is None:
            return res
        return res + [
            Dependency(
                self.path, 
                obj_path(self.rescale), 
                events=("ValueUpdate",)
            )
        ]

@dataclass(frozen=True)
class DailyBtUvar(DailyBtXvar):

    version: int = 0
    type: Literal["DailyBtUvar"] = "DailyBtUvar"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyBtUvarSnap"] = "DailyBtUvarSnap"

    def get_snap(
        self,
        domain: Domain,
        t: Type[DailySnap] | None = None,
    ) -> DailyBtUvar.Snap | None:
        return self.snap_key_val(domain).get(self.path, DailyBtUvar.Snap)
    
    def handler_object_new(
        self
    ) -> HandleNewDailyBtUvar:
        return HandleNewDailyBtUvar()

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
    
    def explorer_df(self, domain: Domain):

        # TODO: return df with signal, scale, etc.

        # all stacked, relevant group keys

        return    

@dataclass(frozen=True)
class DailyBtMvar(DailyBtXvar):
    version: int = 0
    type: Literal["DailyBtMvar"] = "DailyBtMvar"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyBtMvarSnap"] = "DailyBtMvarSnap"

    def get_snap(
        self,
        domain: Domain,
        t: Type[DailySnap] | None = None,
    ) -> DailyBtUvar.Snap | None:
        return self.snap_key_val(domain).get(self.path, DailyBtUvar.Snap)
    
    def handler_object_new(
        self
    ) -> HandleNewDailyBtUvar:
        return HandleNewDailyBtUvar()

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
    
    def explorer_df(self, domain: Domain):

        # TODO: return df with signal, scale, etc.

        # all stacked, relevant group keys

        return    
    
#  ------------------

@dataclass(frozen=True)
class HandleNewDailyBtUvar(
    HandleNewValueFromPl
):

    def handle(
        self,
        event: ObjectNew,
        obj: DailyBtUvar,
        domain: Domain,
    ):

        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None

        if obj.rescale is not None:
            rescale = domain.get(obj.rescale, Object)
            if rescale is None:
                return None
            rescale_snap = rescale.get_snap(
                domain, Daily.Snap
            )
            if rescale_snap is None:
                return None
            
        univ = obj.get_parent(domain)

        uvar = univ.uvar[obj.index]
        bt = univ.bt[uvar.i]

        (
            df_rx,
            df_scale,
            df_signal,
            df_rebalance,
            df_calendar,
            df_weight,
            df_cap,
            df_fx,
        ) = get_backtest_dfs(
            bt, domain, univ
        )

        df = run_backtest(
            df_rx,
            df_scale,
            df_signal=df_signal,
            df_rebalance=df_rebalance,
            df_calendar=df_calendar,
            df_weight=df_weight,
            df_cap=df_cap,
            df_fx=df_fx,
            # TODO: lags 
            unit=bt.unit,
        )

        obj_rx = DailyBtRx.new(obj)
        obj_pos = DailyBtPos.new(obj)
        
        df_pos = df.select("pos")
        df_rx = df.select("rx")

        (
            first,
            last
        ) = df.select(
            pl.col("date").min().item().alias("min"),
            pl.col("date").max().item().alias("max"),
        ).row(0)

        e_pos, _ = self.set_value_from_df(
            obj_pos,
            domain,
            df_pos,
            "pos",
            t = self.key_value_type(obj, domain),
            publish=True,
        )
        e_rx, _ = self.set_value_from_df(
            obj_rx,
            domain,
            df_rx,
            "rx",
            t = self.key_value_type(obj, domain),
            publish=True,
        )

        snap_new: DailyBtUvar.Snap = (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event,
                obj.path,
                dt.datetime.now(),
                first,
                last,
            ), DailyBtUvar.Snap)
        )

        # TODO: if numpy we assume dense, use snap to infer dates

        snap_pos = (
            obj_pos.snap_key_val(domain)
            .set(obj_pos.path, type(obj_pos).Snap(
                event,
                obj_pos.path,
                dt.datetime.now(),
                snap_new.first,
                snap_new.last,
            ))
        )
        snap_rx = (
            obj_rx.snap_key_val(domain)
            .set(obj_rx.path, type(obj_rx).Snap(
                event,
                obj_rx.path,
                dt.datetime.now(),
                snap_new.first,
                snap_new.last,
            ))
        )

        return e_pos, e_rx
    
#  ------------------

@dataclass(frozen=True)
class HandleNewDailyBtMvar(
    HandleNewValueFromPl
):

    def handle(
        self,
        event: ObjectNew,
        obj: DailyBtUvar,
        domain: Domain,
    ):

        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None

        if obj.rescale is not None:
            rescale = domain.get(obj.rescale, Object)
            if rescale is None:
                return None
            rescale_snap = rescale.get_snap(
                domain, Daily.Snap
            )
            if rescale_snap is None:
                return None
            
        univ = obj.get_parent(domain)
        mv = univ.mvar[obj.index]
        
        bts = [
            univ.bt[i] for i in mv.i
        ]
        
        df_pos = None
        df_rx = None

        for i, bt in enumerate(bts):
            i_str = str(i)
            (
                df_rx,
                df_scale,
                df_signal,
                df_rebalance,
                df_calendar,
                df_weight,
                df_cap,
                df_fx,
            ) = get_backtest_dfs(
                bt, domain, univ
            )

            df = run_backtest(
                df_rx,
                df_scale,
                df_signal=df_signal,
                df_rebalance=df_rebalance,
                df_calendar=df_calendar,
                df_weight=df_weight,
                df_cap=df_cap,
                df_fx=df_fx,
                # TODO: lags 
                unit=bt.unit,
            )
            if df_pos is None:
                df_pos = df.select(
                    "date", pl.col("pos").alias(i_str)
                )
            else:
                df_pos = df_pos.join(
                    df.select(
                        "date", pl.col("pos").alias(i_str)
                    ),
                    on="date",
                    how="full",
                    coalesce=True,
                )
            if df_rx is None:
                df_rx = df.select(
                    "date", pl.col("rx").alias(i_str)
                )
            else:
                df_rx = df_rx.join(
                    df.select(
                        "date", pl.col("rx").alias(i_str)
                    ),
                    on="date",
                    how="full",
                    coalesce=True,
                )
        
        assert df_rx is not None, obj
        assert df_pos is not None, obj

        (
            first,
            last
        ) = df_rx.select(
            pl.col("date").min().item(),
            pl.col("date").max().item(),
        ).row(0)

        assert (first, last) == df_pos.select(
            pl.col("date").min().item(),
            pl.col("date").max().item(),
        ).row(0)

        for i, bt in enumerate(bts):

            obj_rx = DailyBtRx.new(obj, index=i)
            obj_pos = DailyBtPos.new(obj, index=i)

            e_pos, _ = self.set_value_from_df(
                obj_pos,
                domain,
                df_pos.select(pl.col(str(i)).alias("pos")),
                "pos",
                t = self.key_value_type(obj, domain),
                publish=True,
            )
            e_rx, _ = self.set_value_from_df(
                obj_rx,
                domain,
                df_rx.select(pl.col(str(i)).alias("rx")),
                "rx",
                t = self.key_value_type(obj, domain),
                publish=True,
            )

            snap_pos = (
                obj_pos.snap_key_val(domain)
                .set(obj_pos.path, type(obj_pos).Snap(
                    event,
                    obj_pos.path,
                    dt.datetime.now(),
                    first,
                    last,
                ))
            )
            snap_rx = (
                obj_rx.snap_key_val(domain)
                .set(obj_rx.path, type(obj_rx).Snap(
                    event,
                    obj_rx.path,
                    dt.datetime.now(),
                    first,
                    last,
                ))
            )

        df_agg = df_rx.select(
            pl.col("date"),
            pl.all_horizontal([
                pl.col(str(i)).is_null()
                for i, _ in enumerate(bts)
            ]).alias("mask"),
            pl.sum_horizontal(*[
                pl.col(str(i)) for i, _ in enumerate(bts)
            ]).alias("rx")
        ).select(
            "date",
            pl.when(pl.col("mask"))
            .then(None)
            .otherwise("rx")
            .alias("rx")
        )

        snap_mvar: DailyBtMvar.Snap = (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event,
                obj.path,
                dt.datetime.now(),
                first,
                last,
            ), DailyBtMvar.Snap)
        )

        obj_rx_agg = DailyBtRx.new(obj)

        e_agg, _ = self.set_value_from_df(
            obj_rx_agg,
            domain,
            df_agg,
            "rx",
            t = self.key_value_type(obj, domain),
            publish=True,
        )

        snap_rx = (
            obj_rx_agg.snap_key_val(domain)
            .set(obj_rx_agg.path, type(obj_rx_agg).Snap(
                event,
                obj_rx_agg.path,
                dt.datetime.now(),
                first,
                last,
            ))
        )

        snap_agg = (
            obj_rx_agg.snap_key_val(domain)
            .set(obj_rx_agg.path, type(obj_rx_agg).Snap(
                event,
                obj_rx_agg.path,
                dt.datetime.now(),
                first,
                last,
            ))
        )
        return e_pos, e_rx, e_agg
    
#  ------------------

    
@dataclass(frozen=True)
class DailyBtPos(
    DailyChunked,
):
    parent: DailyBtUvar | DailyBtMvar | str

    index: int | None = None

    version: int = 0
    type: Literal["DailyBtPos"] = "DailyBtPos"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtPosSnap"] = "DailyBtPosSnap"

    def get_parent(self, domain: Domain) -> DailyBtUvar | DailyBtMvar:
        return domain.get(
            self.parent, DailyBtUvar | DailyBtMvar
        )

    @classmethod
    def new(
        cls,
        parent: DailyBtUvar | DailyBtMvar,
        index: int | None = None,
    ):
        if isinstance(parent, DailyBtMvar):
            assert index is not None, (index, parent)
        # TODO: index in path
        return cls(
            parent=parent,
            path=parent.extend_path("pos"),
            index=index,
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return []
    
@dataclass(frozen=True)
class DailyBtRx(
    DailyChunked,
):
    parent: DailyBtUvar | DailyBtMvar | str

    index: int | None = None

    version: int = 0
    type: Literal["DailyBtRx"] = "DailyBtRx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtRxSnap"] = "DailyBtRxSnap"

    def get_parent(self, domain: Domain) -> DailyBtUvar | DailyBtMvar:
        return domain.get(
            self.parent, DailyBtUvar | DailyBtMvar
        )

    @classmethod
    def new(
        cls,
        parent: DailyBtUvar | DailyBtMvar,
        index: int | None = None,
    ):
        # TODO: index in path
        return cls(
            parent=parent,
            path=parent.extend_path("rx"),
            index=index,
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return []
    
#  ------------------


__all__ = [
    "UniverseDaily",
    "DailyBtUvar",
    "DailyBtMvar",
    "DailyBtPos",
    "DailyBtRx",
    "DailyBtUvarValues",
    "DailyBtUvarResult",
    "DailyBtMvarValues",
    "DailyBtMvarResult",
    "HandleNewDailyBtUvar",
    "HandleNewDailyBtMvar",
]