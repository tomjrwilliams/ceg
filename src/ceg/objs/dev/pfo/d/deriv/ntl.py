

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

from ...... import apis
from ......apis import assert_is, t, kv
from ......engine import *

from ....calc import *

from ..bt import *
from ..core import *
from ..bt import *
from ..pfo import *
from ..derivs import *

#  ------------------

@dataclass(frozen=True)
class DailyPfolioNotional(DailyPfolioDeriv):
    parent: DailyPfolio | str

    prices: tuple[tuple[str, int], ...] = () # (key, universe object index)

    cap: float = 1.
    # TODO: or daily series
    # eg. use to apply sub-portfolio weighting

    method: str = "ratio"

    # TODO: method: str
    # ie. so can cap diff per key eg.

    version: int = 0
    type: Literal["DailyPfolioNotional"] = "DailyPfolioNotional"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyPfolioNotionalSnap"] = "DailyPfolioNotionalSnap"
    
    def handler_object_new(
        self
    ):
        return HandleNewDailyPfolioNotional()
    
    @classmethod
    def new(
        cls,
        parent: DailyPfolio,
        cap: float = 1.,
        **px_indices: int,
    ):
        px = tuple(px_indices.items())
        path = "ntl-" + f"-".join([f"{k}-{i}" for k, i in px])
        return cls(
            parent=parent,
            path=parent.extend_path(path),
            prices=px,
            cap=cap,
        )
    
    
def calc_notional_df(
    px: pl.DataFrame,
    pos: pl.DataFrame,
):
    return px.join(
        pos, how="full", coalesce=True, on="date",
    ).sort(by="date").select(
        "date",
        pl.when(pl.col("px").is_null())
        .then(None)
        .otherwise(
            pl.col("px")
            .fill_nan(None)
            .forward_fill()
            .mul(
                pl.col("pos")
                .fill_nan(None)
                .forward_fill()
            )
        )
        .alias("notional")
    )
    

@dataclass(frozen=True)
class HandleNewDailyPfolioNotional(
    HandleNewValueFromPl
):
    def handle(
        self,
        event: ObjectChange,
        snap: DailyPfolioNotional.Snap,
        obj: DailyPfolioNotional,
    ):
        pf = obj.get_parent()
        data = assert_is(get_daily_pfolio_data(pf))

        univ = pf.get_parent().get_parent()

        pos = {
            ser.key: ser.pos
            for ser in data.series
        }
        prices = {
            k: daily_pl_df(
                context.get().get(univ.objects[i], Daily),
                alias="px",
            )
            for k, i in obj.prices
            if k in pos.keys()
        }

        try:
            notionals = {
                k: calc_notional_df(df_px, pos[k])
                for k, df_px in prices.items()
            }
        except:
            raise ValueError((
                list(pos.keys()),
                list(prices.keys()),
            ))
        
        if not len(notionals):
            raise ValueError(dict(
                pos=pos,
                prices={
                    k: context.get().get(univ.objects[i], Daily)
                    for k, i in obj.prices
                }
            ))
        
        df = apis.pl.join_dfs(
            [
                df.rename({"notional": k})
                for k, df in notionals.items()
            ],
            how="full",
            coalesce=True,
            on="date",
        ).sort(by="date").with_columns(
            pl.when(pl.all_horizontal([
                pl.col(k)
                .is_null() for k in notionals.keys()
            ]))
            .then(None)
            .otherwise(pl.sum_horizontal(
                *[
                    pl.col(k).abs()
                    for k in notionals.keys()
                ]
            ))
            .alias("gross")
        )

        # TODO: cap relative to rolling / total sharpe?

        res: frozendict[
            str, tuple[Snap, Event]
        ] = frozendict()

        for k in df.drop("date").schema.keys():
            obj_val = DailyPfolioNotionalVal.new(obj, k)
            context.get().with_obj(obj_val)

            _ = self.set_value_from_df(
                obj_val,
                df,
                k,
                t = self.key_value_type(obj_val),
            )
            res = ValueUpdate.from_snap(
                replace(
                    obj_val.get_snap(
                        DailyPfolioNotionalVal.Snap
                    ),
                    first=data.start,
                    last=data.end,
                ),
                into=res,
            )
        
        k = "gross"
        v = pl.col("gross")
        cap = obj.cap

        obj_cap = DailyPfolioNotionalCap.new(obj, k)
        context.get().with_obj(obj_cap)

        df_cap = df.select(
            "date",
            (
                pl.when(v.is_null())
                .then(None)
                .when(v.abs() < cap)
                .then(1)
                .when(v < 0)
                .then(v.abs() / cap)
                .when(v > 0)
                .then(v / cap)
                .otherwise(v)
                .alias(VALUE)
                .fill_nan(None)
            )
        )

        _ = self.set_value_from_df(
            obj_cap,
            df_cap,
            VALUE,
            t = self.key_value_type(obj_cap),
        )
        res = ValueUpdate.from_snap(
            replace(
                obj_cap.get_snap(
                    DailyPfolioNotionalCap.Snap
                ),
                first=data.start,
                last=data.end,
            ),
            into=res,
        )
        res = ValueUpdate.from_snap(
            replace(
                snap,
                first=data.start,
                last=data.end,
            ),
            into=res,
        )
        return res

    
#  ------------------

@dataclass(frozen=True)
class DailyPfolioNotionalVal(
    DailyChunked,
):
    parent: DailyPfolioNotional | str

    key: str | None = None

    version: int = 0
    type: Literal["DailyPfolioNotionalVal"] = "DailyPfolioNotionalVal"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyPfolioNotionalValSnap"] = "DailyPfolioNotionalValSnap"

    def get_parent(self) -> DailyPfolioNotional:
        return context.get().get(
            self.parent, DailyPfolioNotional
        )

    @classmethod
    def new(
        cls,
        parent: DailyPfolioNotional,
        key: str | None = None,
    ):
        key = key or "gross"
        return cls(
            parent=parent,
            path=parent.extend_path(key),
            key=key,
        )
    
    def depends(self) -> list[Dependency]:
        return []
    
@dataclass(frozen=True)
class DailyPfolioNotionalCap(
    DailyChunked,
):
    parent: DailyPfolioNotional | str

    # TODO: also subclass pfolio deriv?

    key: str | None = None

    version: int = 0
    type: Literal["DailyPfolioNotionalCap"] = "DailyPfolioNotionalCap"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyPfolioNotionalCapSnap"] = "DailyPfolioNotionalCapSnap"

    def get_parent(self) -> DailyPfolioNotional:
        return context.get().get(
            self.parent, DailyPfolioNotional
        )

    @classmethod
    def new(
        cls,
        parent: DailyPfolioNotional,
        key: str | None = None,
    ):
        key = key or "gross"
        path = f"cap-{key}"
        return cls(
            parent=parent,
            path=parent.extend_path(path),
            key=key,
        )
    
    def depends(self) -> list[Dependency]:
        return []
    
#  ------------------

# interesting: notional cap based on stress indicator
# between 2 sub portfolios:
# - bond and bond momentum
# - equity and equity momentum

__all__ = [
    "DailyPfolioNotional",
    "DailyPfolioNotionalCap",
    "DailyPfolioNotionalVal",
]