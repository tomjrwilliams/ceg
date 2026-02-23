
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

    def build_path(
        self,
        cls: Type[DailyTF],
    ):
        f = self.f
        a = self.a
        b = self.b
        m = self.m
        c = self.c
        params = self.params
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

@dataclass(frozen=True)
class DailyTF(DailyWithDailyParent, DailyChunked):
    parent: Object | str
    # tags: DailyTags
    
    params: DailyTFParams

    project: Project | None = None

    version: int = 0
    type: Literal["DailyTF"] = "DailyTF"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "tf"
    # NOTE: to override
    
    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyTFSnap"] = "DailyTFSnap"
 
    @classmethod
    def new(
        cls,
        parent: Daily,
        f: str | None = None,
        a: float | None = None,
        b: float | None = None,
        m: float | None = None,
        c: float | None = None,
        params: tuple[float | None, ...] | None = None,
        project: Project | None = None,
    ) -> DailyTF:
        param=DailyTFParams(
            f=f,
            a=a,
            b=b,
            m=m,
            c=c,
            params=params,
        )
        assert cls.SUFFIX != "M", cls
        path = parent.extend_path(
            param.build_path(cls)
        )
        if project is not None:
            path = project.extend_path(path)
        return cls(
            path=path,
            parent=parent,
            params=param,
            project=project,
        )
    
    def handler_object_new(
        self
    ) -> HandleNewDailyTF:
        return HandleNewDailyTF()

def binary(v: pl.Expr):
    return pl.when(v.is_null()).then(None).when(v > 0).then(1).when(v < 0).then(-1).otherwise(0)

@dataclass(frozen=True)
class HandleNewDailyTF(HandleNewDailyFromDailyParentPl):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyTF,
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
        obj: DailyTF,
    ) -> pl.Expr:
        params = obj.params
        f = params.f
        v = pl.col(VALUE).fill_nan(None)
        if params.b is not None:
            v = params.b * v
        if params.a is not None:
            v = params.a + v
        
        e_sq = v.pow(2)

        # sd = e.rolling_std(len(df), min_samples=2)
        sd = e_sq.rolling_mean(len(df), min_samples=2).sqrt()
        mu = v.rolling_mean(len(df), min_samples=2)

        l_v = r_v = l_v_2 = r_v_2 = None

        if f is None:
            pass
        elif f == "bls":
            v = binary(v)
        elif f == "blo":
            v = binary(v).clip(lower_bound=0.)
        elif f == "bso":
            v = binary(v).clip(upper_bound=0.)
        elif f == "ls":
            pass
        elif f == "lo":
            v = v.clip(lower_bound=0.)
        elif f == "so":
            v = v.clip(upper_bound=0.)
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

        elif f == "capratio":
            assert params.params is not None, params
            if len(params.params) == 1:
                cap, = params.params
                
                v = (
                    pl.when(v.is_null())
                    .then(None)
                    .when(v.abs() < cap)
                    .then(1.)
                    .when(v < 0)
                    .then(v.abs() / cap)
                    .when(v > 0)
                    .then(v / cap)
                    .otherwise(v)
                    .alias(VALUE)
                    .fill_nan(None)
                )

            else:
                raise ValueError(self)

        # TODO: outright sign, needs three param for 0
        elif f == "lsign" or f == "rsign":
            pass

        elif f == "zclip" or f == "zflip":
            # TODO: rms variant?
            assert params.params is not None, params
            if len(params.params) == 2:
                z_span, z_width = params.params
                v_l = v.log()
                z_sd = v_l.ewm_std(
                    span=z_span,
                    ignore_nulls=True,
                    min_samples=2
                )
                z_mu = v_l.ewm_mean(
                    span=z_span,
                    ignore_nulls=True,
                    min_samples=2
                )
                mul = -1 if f == "zflip" else 1.
                v_d = (v_l - z_mu)
                v_w = (
                    z_sd * z_width
                    if z_width != 0
                    else v_d
                )
                lim_up = (
                    v_w if z_width != 0 else 0
                )
                lim_dn = - (
                    v_w if z_width != 0 else 0
                )
                v_up = (
                    z_mu + (v_w * mul)
                )
                v_dn = z_mu - (v_w * mul)
                v = (
                    pl.when(v_d >= lim_up)
                    .then(v_up)
                    .when(v_d <= lim_dn)
                    .then(v_dn)
                    .otherwise(
                        z_mu + (mul * v_d)
                    )
                ).exp().fill_nan(None)
            else:
                raise ValueError(params)

        elif f == "levl" or f == "levlt":
            assert params.params is not None, params
            if len(params.params) == 2:
                n, l = params.params
                assert l is not None, l
                r = l + 1
            elif len(params.params) == 3:
                n, l, r = params.params
            else:
                raise ValueError(params)
            
            # TODO: same approach but fixed levels
            # eg. _, -75, -25, 25, 75, _

        # TODO: allow specifying r > l
        # to indicate left of and right of
        # eg. for TN, does well strong pos or neg
        # but not middle

        elif f == "qile" or f == "qilet":
            assert params.params is not None, params
            if len(params.params) == 2:
                n, l = params.params
                assert l is not None, l
                r = l + 1
            elif len(params.params) == 3:
                n, l, r = params.params
            else:
                raise ValueError(params)
            
            assert n is not None, params
            n = int(n)

            assert l is not None, params
            assert r is not None, params
            
            w = 1 / n
            l = None if l == 0 else l * w
            r = None if r == n else r * w

            l_v = (
                None if l is None
                else v.rolling_quantile(
                    l, window_size=len(df), min_samples=10
                )
            )
            r_v = (
                None if r is None
                else v.rolling_quantile(
                    r, window_size=len(df), min_samples=10
                )
            )
        
        elif f == "zile" or f == "zilet":
            assert params.params is not None, params
            if len(params.params) == 3:
                n, w, i = params.params
                assert n is not None, params
                assert w is not None, params
                assert i is not None, params
                l, r = apis.stats.sdev_increments(
                    w, int(n), mu = 0
                )[int(i)]

            elif len(params.params) == 4:
                n, w, il, ir = params.params
                assert n is not None, params
                assert w is not None, params
                assert il is not None, params
                assert ir is not None, params
                if il < ir:
                    l, _ = apis.stats.sdev_increments(
                        w, int(n), mu = 0
                    )[int(il)]
                    _, r = apis.stats.sdev_increments(
                        w, int(n), mu = 0
                    )[int(ir) - 1]
                else:
                    l = None
                    _, r = apis.stats.sdev_increments(
                        w, int(n), mu = 0
                    )[int(ir) - 1]
                    r2 = None
                    l2, _ = apis.stats.sdev_increments(
                        w, int(n), mu = 0
                    )[int(il)]

                    l_v_2 = None if l2 is None else l2 * sd
                    # r_v_2 = None if r2 is None else r2 * sd
            else:
                raise ValueError(params)
            
            l_v = None if l is None else l * sd
            r_v = None if r is None else r * sd
            
        else:
            raise ValueError((obj, f))

        if l_v is not None or r_v is not None:
            assert params.a is None, params
            assert params.b is None, params
            v = apis.pl.mask_col(VALUE, lambda e: ((
                pl.lit(True) 
                if l_v is None
                else e > l_v
            ) & (
                pl.lit(True) 
                if r_v is None
                else e < r_v
            )).or_(
                pl.lit(False)
                if l_v_2 is None and r_v_2 is None
                else (
                    pl.lit(True) 
                    if l_v_2 is None
                    else e > l_v_2
                ) & (
                    pl.lit(True) 
                    if r_v_2 is None
                    else e < r_v_2
                )
            ))
            if f == "zilet":
                v = v.tanh()
            if f == "qilet":
                v = v.tanh()

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

        return v.fill_nan(None)

__all__ = [
    "DailyTFParams",
    "DailyTF",
    "HandleNewDailyTF",
]