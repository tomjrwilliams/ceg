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

import math
import numpy as np
import polars as pl

from .... import apis
from ....apis import t, kv, assert_is
from ....engine import *

from ..bars.d import *
from ..calc.d import *
from ..eco.d import *

from ..agg import *
from ..pfo import *

from .assets import *

#  ------------------


@dataclass(frozen=True)
class ArbitraryFactor:
    factors: str | tuple[str, ...]
    key: str
    val: float = 1.0

    weights: tuple[tuple[str, ...], ...] | None = None

    # TODO: agg method = "prod"

    def label(self):
        return self.key

    def unpack(
        self,
        bound: Bound,
        coll: Collection,
        force: bool = False,
    ):
        return replace(
            self,
            factors=unpack_bound(
                self.factors,
                "factors",
                bound,
                coll,
                force=force,
            )
        )

    def obj(
        self,
        params: HasAsset,
    ):
        if isinstance(self.factors, str):
            return (
                context.get().get(self.factors, Daily),
                dict(scale=self.val),
            )
        elif self.weights is None:
            # TODO: params scale_val=1.
            return tuple(
                [
                    context.get().get(
                        (
                            rx_s
                            if isinstance(rx_s, str)
                            else rx_s[0]
                        ),
                        Daily,
                    )
                    for rx_s in self.factors
                ]
            ), dict(scale=self.val, agg_method="hprod")
        else:
            assert all(
                [
                    len(w) == len(self.factors)
                    for w in self.weights
                ]
            ), self
            return tuple(
                [
                    context.get().get(
                        (
                            rx_s
                            if isinstance(rx_s, str)
                            else rx_s[0]
                        ),
                        Daily,
                    )
                    for rx_s in self.factors
                ]
                + sum(
                    [
                        [
                            context.get().get(
                                (
                                    rx_s
                                    if isinstance(rx_s, str)
                                    else rx_s[0]
                                ),
                                Daily,
                            )
                            for rx_s in w
                        ]
                        for w in self.weights
                    ],
                    [],
                )
            ), dict(
                scale=self.val,
                agg_method="hdot",
                dot_groups=1 + len(self.weights),
            )


@dataclass(frozen=True)
class ScaleFactor(ArbitraryFactor):
    pass


@dataclass(frozen=True)
class WeightFactor(ScaleFactor):
    pass


@dataclass(frozen=True)
class ScaleNotional:
    val: float = 1.0

    def unpack(
        self,
        bound: Bound,
        coll: Collection,
        force: bool = False,
    ):
        # TODO: assert no others
        return self

    def label(self):
        return "cn"

    def obj(
        self,
        params: HasAsset,
    ):
        if isinstance(params.asset, AssetConcrete):
            return params.asset.px(), dict(scale=self.val)
        elif isinstance(params.asset, AssetSynthetic):
            return params.asset.px(), dict(scale=self.val)
        else:
            raise ValueError(
                (
                    self,
                    params.asset,
                )
            )


@dataclass(frozen=True)
class ScaleVolBeta:
    bench: str
    alias: str | None = None

    rx: str | None = None
    # eg. if we already have a vol scaled rx

    vol: int | None = 48  # span

    span: int | None = 256
    alpha: float | None = None

    def apply(
        self,
        rx: Daily,
        ann: bool = False,
        # rms is ann scaled: not needed, just bench
    ):
        if self.vol is not None:
            # TODO: allow passing as string
            bench_rms = DailyERms.new(
                context.get().get(self.bench, Daily),
                span=self.vol,
                alpha=None,
                ann=False,
                cum=False,
            )
            context.get().with_obj(bench_rms)
            bench_rms = bench_rms.id
        else:
            bench_rms = 1.0
        if self.rx is not None:
            rx = context.get().get(self.rx, Daily)
        return DailyEBeta.new(
            rx,
            span=self.span,
            alpha=self.alpha,
            bench=self.bench,
            bench_rms=bench_rms,
            alias=self.alias,
            # bench_scale = (
            #     1 / math.sqrt(252)
            #     if self.ann
            #     else 1.
            # ),
            cum=False,  # type: ignore
            ann=False,
        )


@dataclass(frozen=True)
class ScaleVol:
    span: int
    # method = rms or sdev

    ann: bool = True

    rx: str | tuple[str | tuple[str, int], ...] | None = (
        None
    )

    z_clip: tuple[str, int, float] | None = None
    # TODO: pro-vol?

    val: float = 1.0

    weight: RxWeight | None = None

    beta: ScaleVolBeta | None = None

    def unpack(
        self,
        bound: Bound,
        coll: Collection,
        force: bool = False,
    ):
        # TODO: rx weight, beta
        if self.rx is None:
            return self
        return replace(
            self,
            rx=unpack_bound(
                self.rx,
                "rx",
                bound,
                coll,
                force=force,
            )
        )

    def prefix(self):
        return "cv" + (
            "" if self.val == 1.0 else f"{self.val}"
        )

    def z_clip_f(self):
        assert self.z_clip is not None, self
        z = self.z_clip[0]
        if z == "z":
            return "c"
        return z

    def label(self):
        s = f"{self.prefix()}-{self.span}"
        if self.z_clip is not None:
            z, z_span, z_width = self.z_clip
            s += (
                f"-z{self.z_clip_f()}lip-{z_span}-{z_width}"
            )
        if self.weight is not None:
            s += f"-{self.weight.label()}"
        return s

    def obj(
        self,
        params: HasAsset,
    ):

        if self.rx is None:
            assert not isinstance(params.asset, tuple), (
                params
            )
            rx = params.asset.rx()
            scale_rest = []
        elif isinstance(self.rx, str):
            rx = context.get().get(self.rx, Daily)
            scale_rest = []
        else:
            rx, *rx_rest = [
                context.get().get(
                    (
                        rx_s
                        if isinstance(rx_s, str)
                        else rx_s[0]
                    ),
                    Daily,
                )
                for rx_s in self.rx
            ]
            scale_rest = [
                DailyERms.new(
                    rx_o,
                    span=(
                        self.span
                        if isinstance(rx_s, str)
                        else rx_s[1]
                    ),
                    alpha=None,
                    ann=self.ann,
                )
                for rx_o, rx_s in zip(rx_rest, self.rx[1:])
            ]
        if self.weight is not None:
            rx = self.weight.apply(rx)

        scale = DailyERms.new(
            rx,
            span=self.span,
            alpha=None,
            ann=self.ann,
        )
        if self.z_clip is not None:
            scale = DailyTF.new(
                scale,
                f=f"z{self.z_clip_f()}lip",
                params=tuple(self.z_clip[1:]),
            )

        if self.beta is not None:
            scale_rest.append(
                self.beta.apply(
                    rx,
                    # scale,
                    ann=self.ann,
                )
            )

        if len(scale_rest):
            return (scale,) + tuple(scale_rest), dict(
                scale=self.val, agg_method="hprod"
            )
        return scale, dict(scale=self.val)


@dataclass(frozen=True)
class CapNotional(ScaleNotional):
    others: tuple[tuple[str, Daily], ...] = ()

    def label(self):
        s = "cap-n" + (
            "" if self.val == 1.0 else f"{self.val}"
        )
        if len(self.others):
            return s + "-".join([k for k, o in self.others])
        return s

    # TODO: the re-cap needs to be a scaling factor not a cap
    # as the cap might not have been reached, might still not

    # we effectively need to cap, then re-scale again...


@dataclass(frozen=True)
class CapVol(ScaleVol):
    def prefix(self):
        return "capv" + (
            "" if self.val == 1.0 else f"{self.val}"
        )


#  ------------------


@dataclass(frozen=True)
class SignalFactor(ArbitraryFactor):
    pass


@dataclass(frozen=True)
class SignalQile:
    n: int
    i: int
    r: int | None
    m: float | None = None
    c: float | None = None


@dataclass(frozen=True)
class SignalZile:
    n: int
    w: float
    i: int
    r: int | None
    m: float | None = None
    c: float | None = None

    def label(self):
        s = f"zile-{self.n}-{self.w}-{self.i}"
        if self.r is not None:
            s += f"-{self.r}"
        if self.m is not None:
            s += f"-{self.m}"
        if self.c is not None:
            s += f"-{self.c}"
        return s


@dataclass(frozen=True)
class SignalEco:
    symbol: str
    method: str
    speed: int  # span or window
    activation: str | None = "tanh"
    scale: float = 1.0

    qile: SignalQile | None = None
    zile: SignalZile | None = None

    def label(self):
        s = f"{self.symbol}-{self.method}-{self.speed}"
        if self.activation is not None:
            return f"{s}-{self.activation}"
        return s

    def obj(
        self,
        params: HasAsset,
    ):
        v_eco = EcoDailyObservation.new(
            symbol=self.symbol,
            source="fred",
        )
        sig: Daily
        if self.method == "z":
            sig = DailyZ.new(v_eco, window=self.speed)
        elif self.method == "ez":
            sig = DailyEZ.new(v_eco, span=self.speed)
        elif self.method == "zw":
            sig = DailyZW.new(v_eco, window=self.speed)
        elif self.method == "ezw":
            sig = DailyEZW.new(v_eco, span=self.speed)
        elif self.method == "rz":
            sig = DailyRZ.new(v_eco, window=self.speed)
        elif self.method == "erz":
            sig = DailyERZ.new(v_eco, span=self.speed)
        elif self.method == "dema":
            sig = DailyDEMA.new(v_eco, span=self.speed)
        elif self.method == "demax":
            sig = DailyDEMax.new(v_eco, span=self.speed)
        elif self.method == "zwdemin":
            demax = DailyDEMin.new(v_eco, span=self.speed)
            sig = DailyEZW.new(demax, span=self.speed)
        elif self.method == "zwdemax":
            demax = DailyDEMax.new(v_eco, span=self.speed)
            sig = DailyEZW.new(demax, span=self.speed)
        elif self.method == "raw":
            sig = v_eco
        else:
            raise ValueError((bt))
        if self.activation is not None:
            act = self.activation
            if act == "tanh":
                sig = DailyTF.new(sig, f="tanh")
            elif act == "unit_tanh":
                sig = DailyTF.new(
                    sig, f="tanh", a=1.0, m=0.5
                )
            elif act == "n_tanh_p_one":
                sig = DailyTF.new(
                    sig, f="tanhrsign", params=(None, 1)
                )
            elif act == "ng_tnlog_ng":
                sig = DailyTF.new(
                    sig,
                    b=-1,
                    f="logtanh",
                    m=-1,
                )
            else:
                raise ValueError(self)
        sig = apply_ile(sig, self)

        return sig, dict(scale=self.scale)


@dataclass(frozen=True)
class SignalWmThold:
    span: int
    field: str

    threshold: float = -2.0
    reset: float = 1.0
    reentry: bool | float | None = None  # 2.
    rethreshold: float = 1.0

    log: bool = True

    activation: str | None = None

    constraint: str | None = None

    qile: SignalQile | None = None
    zile: SignalZile | None = None

    rx: str | None = None

    weight: RxWeight | None = None

    val: float = 1.0
    # if none, uses the result.rx (ie. asset)

    def label(self):
        k = DailyEWmThold.Params.format_args(
            field=self.field,
            threshold=self.threshold,
            reset=self.reset,
            reentry=self.reentry,
            rethreshold=self.rethreshold,
            log=self.log,
        )
        return (
            f"wm-{k}"
            + (
                ""
                if self.activation is None
                or self.activation == ""
                else f"-{self.activation}"
            )
            + ("" if self.qile is None else f"-{self.qile}")
            + (
                ""
                if self.constraint is None
                else f"-{self.constraint}"
            )
            + (
                ""
                if self.zile is None
                else f"-{self.zile.label()}"
            )
            + (
                ""
                if self.weight is None
                else f"-{self.weight.label()}"
            )
        )

    def obj(
        self,
        params: HasAsset,
    ):

        if self.rx is None:
            assert not isinstance(params.asset, tuple), (
                params
            )
            rx = params.asset.rx()
        else:
            rx = context.get().get(self.rx, Daily)

        if self.weight is not None:
            rx = self.weight.apply(rx)

        sig = DailyEWmThold.new(
            rx,
            span=float(self.span),
            field=self.field,
            threshold=self.threshold,
            reset=self.reset,
            reentry=self.reentry,
            rethreshold=self.rethreshold,
            log=self.log,
        )
        if self.constraint is not None:
            sig = apply_constraint(
                sig,
                self.constraint,
            )

        return sig, dict(scale=self.val)


@dataclass(frozen=True)
class SignalOsc:
    fast: int
    slow: int | None
    vol: int
    activation: str = "tanh"

    constraint: str | None = None

    qile: SignalQile | None = None
    zile: SignalZile | None = None

    weight: PxWeight | None = None

    val: float = 1.0

    def label(self):
        return (
            f"osc-{self.fast}"
            + ("" if self.slow is None else f"-{self.slow}")
            + f"-{self.vol}"
            + (
                ""
                if self.activation is None
                or self.activation == ""
                else f"-{self.activation}"
            )
            + ("" if self.qile is None else f"-{self.qile}")
            + (
                ""
                if self.constraint is None
                else f"-{self.constraint}"
            )
            + (
                ""
                if self.zile is None
                else f"-{self.zile.label()}"
            )
            + (
                ""
                if self.weight is None
                else f"-{self.weight.label()}"
            )
        )

    def obj(
        self,
        params: HasAsset,
    ):
        asset = params.asset
        if isinstance(asset, AssetConcrete):
            px_adj = "pct" if asset.pct else "abs"
            px = (
                FutDailyOrdinal.new(
                    symbol=asset.symbol,
                    ident=1,
                    field="close",
                    adjust=px_adj,
                    source="frd",
                )
                if asset.product == "FUT"
                else EqDailyEtf.new(
                    symbol=asset.symbol,
                    field="close",
                    adjust="price",
                    source="frd",
                )
                if asset.product == "ETF"
                else None
            )
            if self.weight is not None:
                px = self.weight.apply(
                    assert_is(px, Daily),
                )
            assert px is not None, (params, self)
            context.get().with_obj(px)
            sig = DailyEOsc.new(
                px,
                fast=self.fast,
                slow=self.slow,
                vol=self.vol,
                activation=self.activation,
            )
        elif isinstance(asset, AssetSynthetic):
            rx = context.get().get(asset.id, t=Daily)
            if self.weight is not None:
                rx = self.weight.apply(rx, rx=True)
            assert rx is not None, (params, self)
            context.get().with_obj(rx)
            sig = DailyEOsc.new(
                rx,
                fast=self.fast,
                slow=self.slow,
                vol=self.vol,
                activation=self.activation,
                cum=True,
            )
        else:
            raise ValueError(self)
        sig = apply_ile(sig, self, suffix="t")
        if self.constraint is not None:
            sig = apply_constraint(
                sig,
                self.constraint,
            )

        return sig, dict(scale=self.val)


def apply_constraint(sig: Daily, constraint: str):
    if constraint == "bls":
        return DailyTF.new(sig, f=constraint)
    elif constraint == "blo":
        return DailyTF.new(sig, f=constraint)
    elif constraint == "bso":
        return DailyTF.new(sig, f=constraint)
    elif constraint == "ls":
        return sig
    elif constraint == "lo":
        return DailyTF.new(sig, f=constraint)
    elif constraint == "so":
        return DailyTF.new(sig, f=constraint)
    else:
        raise ValueError((sig, constraint))


def apply_ile(
    sig: Daily,
    self: SignalEco | SignalOsc,
    suffix: str = "",
):
    if self.qile is None and self.zile is None:
        pass
    elif self.qile is not None:
        sig = DailyTF.new(
            sig,
            f="qile" + suffix,
            params=(
                self.qile.n,
                self.qile.i,
            )
            + (
                ()
                if self.qile.r is None
                else (self.qile.r,)
            ),
            m=self.qile.m,
            c=self.qile.c,
        )
    elif self.zile is not None:
        sig = DailyTF.new(
            sig,
            f="zile" + suffix,
            params=(
                self.zile.n,
                self.zile.w,
                self.zile.i,
            )
            + (
                ()
                if self.zile.r is None
                else (self.zile.r,)
            ),
            m=self.zile.m,
            c=self.zile.c,
        )
    else:
        raise ValueError(self)
    return sig


__all__ = [
    "SignalEco",
    "SignalOsc",
    "SignalQile",
    "SignalZile",
    "SignalFactor",
    "SignalWmThold",
    "ScaleNotional",
    "ScaleVol",
    "ScaleFactor",
    "WeightFactor",
    "CapNotional",
    "CapVol",
]
