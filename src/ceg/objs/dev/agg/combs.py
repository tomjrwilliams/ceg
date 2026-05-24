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
from functools import partial

from frozendict import frozendict

import numpy as np
import polars as pl
import numba as nb

from .... import apis
from ....apis import assert_is, field_default, t, kv, logs
from ....engine import *

from ..calc import *

from .containers import *
from .aggs import *

from ....apis.calcs.regimes.dd import *

from ....apis.calcs import *

logger, log = logs.logger(
    __file__, streams=dict(formatter=True)
)

#  ------------------


@dataclass(frozen=True)
class DailyCombination(
    DailyChunked,
    DependsParentValue,
):
    parent: Container | str

    args: tuple[int, ...]
    metric: str

    # TODO: peers? can be either agg or comb?

    version: int = 0
    type: Literal["DailyCombination"] = "DailyCombination"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyCombinationSnap"] = (
            "DailyCombinationSnap"
        )

    def get_parent(self) -> Container:
        return context.get().get(self.parent, t=Container)

    @classmethod
    def new(
        cls,
        parent: Container,
        metric: str,
        comb: int
        | Combination
        | tuple[int | Combination, ...],
    ):
        return cls.new_(parent, metric, comb)

    @classmethod
    def new_(
        cls,
        parent: Container,
        metric: str,
        comb: int
        | Combination
        | tuple[int | Combination, ...],
    ):
        comb = (
            (comb,)
            if isinstance(comb, (int, Combination))
            else comb
        )
        i = tuple(
            (
                c if isinstance(c, int) else c.index
                for c in comb
            )
        )
        args_pad = f"comb-" + "-".join(map(str, i))
        return cls(
            id=parent.extend_id(args_pad),
            parent=parent,
            metric=metric,
            args=i,
        )

    def handler(self, event: Event, snap: Snap) -> Handler:
        parent = self.get_parent()

        obj = context.get().get(event.obj, Object)

        parent_update = (
            isinstance(event, ValueUpdate)
            and obj.id == parent.id
        )

        # TODO: pass params

        if parent_update and self.metric == "hsum":
            return HandleNewDailyHSum()
        elif parent_update:
            raise ValueError(self)

        # TODO: if upstream target change

        raise ValueError((self, event))

    def deriv(
        self,
        key: str | None = None,
        index: int | None = None,
    ):
        return DailyCombinationDeriv.new(
            self, index=index, key=key
        )


def get_comb_dfs(combs: tuple[int, ...], coll: Container):
    acc = []
    d0 = d1 = None

    for c in combs:
        dfs = frozendict()
        comb = coll.comb[c]

        if comb.params is None:
            params = {}
        else:
            params = coll.params[comb.params]

        agg_method = params.get("agg_method", None)

        for k, r in comb.refs.items():
            ref = Ref.decode(r)
            if ref.type == "obj":
                o = coll.get_object(ref.i, Daily)
                df = daily_pl_df(
                    o,
                    alias=k,
                    scale=assert_is(
                        params.get(f"{k}_scale", 1.0), float
                    ),
                )
                _d0, _d1 = apis.pl.select_items(
                    df,
                    (
                        pl.col("date").min(),
                        pl.col("date").max(),
                    ),
                    dt.date,
                )
            elif ref.type == "agg":
                method = assert_is(
                    params.get(
                        f"{k}_agg_method", agg_method
                    ),
                    str,
                )

                agg_params = frozendict(
                    {
                        kk[len(f"{k}_") :]: v
                        for kk, v in params.items()
                        if kk.startswith(f"{k}_")
                        and kk
                        not in [
                            f"{k}_agg_method",
                            f"{k}_scale",
                        ]
                        # TODO: or k in h.kwarg_names()?
                    }
                )

                try:
                    h = agg_handler(method, agg_params)
                except:
                    raise ValueError(
                        dict(
                            comb=comb,
                            params=params,
                            k=k,
                        )
                    )

                # TODO: assert singular df handler method

                # TODO: get_agg_dfs removed in DailyComposition migration; this
                # legacy path (DailyCombination over Combination.refs of "agg_*")
                # is not used by current framework callers. Re-implement against
                # DailyComposition refs if/when this path is reintroduced.
                raise NotImplementedError(
                    "Combination.refs with 'agg_*' ref encoding is not supported"
                    " after the DailyComposition migration."
                )
                df_ = h.agg_dfs(_d0, _d1, *_dfs)  # unreachable

                df = assert_is(df_, pl.DataFrame)
                if VALUE in df.schema:
                    df = df.rename({VALUE: k})

            else:
                raise ValueError(comb, combs)

            d0 = _d0 if d0 is None else min([d0, _d0])
            d1 = _d1 if d1 is None else min([d1, _d1])
            dfs = dfs.set(k, df)

        acc.append((dfs, params))

    d0, d1 = assert_is(d0), assert_is(d1)

    return acc, d0, d1


@dataclass(frozen=True)
class DailyCombinationDeriv(Output, DailyChunked):
    parent: DailyCombination | str

    version: int = 0
    type: Literal["DailyCombinationDeriv"] = (
        "DailyCombinationDeriv"
    )

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyCombinationDerivSnap"] = (
            "DailyCombinationDerivSnap"
        )

    def get_parent(self) -> DailyCombination:
        return context.get().get(
            self.parent, DailyCombination
        )

    # new() and depends() inherited from Output


@dataclass(frozen=True)
class HandleNewDailyCombination(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):
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
    ): ...

    def handle(
        self,
        event: ObjectChange,
        snap: DailyCombination.Snap,
        obj: DailyCombination,
    ):
        container= obj.get_parent()

        d0 = d1 = None

        # TODO: shared params? maybe on the DailyCombination instead

        acc, d0, d1 = get_comb_dfs(obj.args, container)

        if snap.first and snap.last:
            if d0 == snap.first and d1 == snap.last:
                return frozendict().set(
                    obj.id, (snap, ())
                )

        # log("INFO")( agg=obj.id)

        res = self.comb_dfs(d0, d1, *acc)

        df = None

        # log("INFO")( store=obj.id)

        snaps = frozendict()

        if isinstance(res, pl.DataFrame):
            df = res
            assert len(df) == ((d1 - d0).days + 1), (
                df,
                d1,
                d0,
            )

            v = self.set_value_from_df(
                obj,
                df,
                VALUE,
                t=self.key_value_type(obj),
            )
        elif isinstance(res, tuple):
            df, k_is = res

            df = df.filter(
                pl.col("date").is_between(d0, d1)
            )

            assert len(df) == ((d1 - d0).days + 1), (
                df,
                d1,
                d0,
            )

            _snaps = {}

            def store_child(
                k_i,
            ):
                if isinstance(k_i, str):
                    k = col = k_i
                    i = None
                elif isinstance(k_i, int):
                    i = k_i
                    col = f"v_{i}"
                    k = None
                else:
                    k, i = k_i
                    col = f"{k}_{i}"
                child = obj.deriv(key=k, index=i)
                context.get().with_obj(child)
                child_snap = child.get_snap(Daily.Snap)
                _ = self.set_value_from_df(
                    child,
                    df,
                    col,
                    t=self.key_value_type(child),
                )
                _snaps[child.id] = (
                    replace(
                        child_snap,
                        first=d0,
                        last=d1,
                    ),
                    ValueUpdate(child.id),
                )

            # TODO: fold / accumulate
            _ = list(map(store_child, k_is))
            for k, v in _snaps.items():
                snaps = snaps.set(k, v)
        else:
            raise ValueError((type(res), self))

        # log(agg=obj.id, mess="setting snap")

        return ValueUpdate.from_snap(
            self.update_snap_from_df(
                snap,
                assert_is(df),
                d_min=d0,
                d_max=d1,
            ),
            into=snaps,
        )


@dataclass(frozen=True)
class HandleNewDailyHSum(HandleNewDailyCombination):
    def comb_dfs(
        self,
        first: dt.date,
        last: dt.date,
        dfs_params: tuple[
            frozendict[str, pl.DataFrame],
            frozendict[str, TParamValue],
        ],
    ) -> pl.DataFrame:
        raise ValueError()


__all__ = [
    "get_comb_dfs",
    "DailyCombination",
    "DailyCombinationDeriv",
    "HandleNewDailyCombination",
]
