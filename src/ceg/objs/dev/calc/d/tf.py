
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
from .....apis import t
from .....apis import kv
from .....engine import *

P = ParamSpec("P")
O = TypeVar("O", bound=Object)

#  ------------------

@dataclass(frozen=True)
class DailyTFParams(DailyParams):
    chunk: int = 32
    # TODO: db?

    f: str | None = None

    a: float | None = None
    b: float | None = None

    m: float | None = None
    c: float | None = None

    params: tuple[float | None, ...] | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.


@dataclass(frozen=True)
class DailyTF(DailyWithDailyParent, DailyChunked):
    parent: Object | str
    # tags: DailyTags
    
    params: DailyTFParams

    version: int = 0
    type: Literal["DailyTF"] = "DailyTF"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "tf"
    # NOTE: to override
    
    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyTFSnap"] = "DailyTFSnap"
 
    @classmethod
    def params_suffix(
        cls,
        f: str | None = None,
        a: float | None = None,
        b: float | None = None,
        m: float | None = None,
        c: float | None = None,
        params: tuple[float | None, ...] | None = None,
    ):
        s = ""
        if a is not None:
            s += ("_" if len(s) else "") + f"a{a}"
        if b is not None:
            s += ("_" if len(s) else "") + f"b{b}"
        if f is not None:
            s += ("_" if len(s) else "") + f"f{f}"
        if params is not None:
            for p in params:
                if p is None:
                    s += ("_" if len(s) else "") + f"N"
                else:
                    s += ("_" if len(s) else "") + f"{p}"
        if m is not None:
            s += ("_" if len(s) else "") + f"m{m}"
        if c is not None:
            s += ("_" if len(s) else "") + f"c{c}"
        s = s.replace("-", "n").replace(".0_", "_")
        if s.endswith(".0"):
            s = s[:-2]
        assert len(s) > 0
        return s
    
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: None = None,
        f: str | None = None,
        a: float | None = None,
        b: float | None = None,
        m: float | None = None,
        c: float | None = None,
        params: tuple[float | None, ...] | None = None,
    ) -> DailyTF: ...

    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ],
        f: str | None = None,
        a: float | None = None,
        b: float | None = None,
        m: float | None = None,
        c: float | None = None,
        params: tuple[float | None, ...] | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DailyTF: ...

    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ] | None = None,
        f: str | None = None,
        a: float | None = None,
        b: float | None = None,
        m: float | None = None,
        c: float | None = None,
        params: tuple[float | None, ...] | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DailyTF:
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            parent = t(parent, *args, **kwargs)
        return cls(
            path=parent.extend_path(cls.params_suffix(
                f=f,
                a=a,
                b=b,
                m=m,
                c=c,
                params=params,
            )),
            parent=parent,
            params=DailyTFParams(
                f=f,
                a=a,
                b=b,
                m=m,
                c=c,
                params=params,
            ),
        )
    
    def handler_object_new(
        self
    ) -> HandleNewDailyTF:
        return HandleNewDailyTF()

@dataclass(frozen=True)
class HandleNewDailyTF(HandleNewDailyFromDailyParentPl):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyTF,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyTF,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        f = params.f
        v = pl.col(VALUE).fill_nan(None)
        if params.b is not None:
            v = params.b * v
        if params.a is not None:
            v = params.a + v
        
        if f is None:
            pass
        elif f.startswith("tanh"):
            v = v.tanh()

            if f == "tanhlsign":
                f = "lsignl"
            elif f == "tanhrsign":
                f = "rsign"
            elif f == "tanhlog":
                v = v.log()
            else:
                pass

        # elif f == "unit_tanh":
        #     return (1 + ser.tanh().fill_nan(None)) / 2
        # elif f == "neg":
        #     return (-1 * ser).fill_nan(None)
        elif f.startswith("log"):
            v = v.log()

            if f == "logtanh":
                v = v.tanh()
            else:
                pass

        # elif f == "neg_tanhlog":
        #     return (
        #         -1 * ser.log().tanh()
        #     ).fill_nan(None)
        # elif f == "neg_tanhlog_neg":
        #     return (
        #         -1 * (-1 * ser).log().tanh()
        #     ).fill_nan(None)

        # TODO: outright sign, needs three param for 0
        elif f == "lsign" or f == "rsign":
            pass
        else:
            raise ValueError(obj)

        if f == "lsign" or f == "rsign":
            assert params.params is not None, params
            assert len(params.params) == 2, params
            l, r = params.params
            if l is not None:
                v = (
                    (
                        pl.when(v <= 0)
                        if f == "lsign"
                        else pl.when(v < 0)
                    )
                    .then(l)
                    .otherwise(v)
                )
            if r is not None:
                v = (
                    (
                        pl.when(v >= 0)
                        if f == "rsign"
                        else pl.when(v > 0)
                    )
                    .then(r)
                    .otherwise(v)
                )

        if params.m is not None:
            v = params.m * v
        if params.c is not None:
            v = params.c + v

        return v

__all__ = [
    "DailyTFParams",
    "DailyTF",
    "HandleNewDailyTF",
]