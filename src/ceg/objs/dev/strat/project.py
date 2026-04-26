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

import itertools
import functools

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

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
from .calcs import *
from .params import *
from .domain import *

from ....vis import explorer


# TODO: rename page
# as thats what these are


# as project is an object type for namespacing data / events

# TODO: in theory could use separate storage config per project root (the object) and hence just drop db for poject while using main constant

# then need to detect in grtaph that has been deleted? but poss quicker than traverse and delete one by one (much)


# if we know all are in the same db?

# so i guess iterate and just check, if not delete manually


# nice thing about separate namespace per object and no global counter is can genuineky have separate storage per object type if better / or by data type (sensitive or not, local if sim or remote if shared data)


@dataclass(frozen=True)
class StratExplorer(explorer.Explorer):
    PROJECT: ClassVar[str] = "p"
    SPAN_FAST: ClassVar[list[int]] = []
    SPAN_RATIO: ClassVar[list[float]] = []
    SPAN_VOL_SIG: ClassVar[list[int]] = []
    SPAN_VOL_SCALE: ClassVar[list[int]] = []

    def univ_name(
        self,
        name: str,
    ):
        return self.PROJECT + "-" + name

    def page_name(
        self,
        i: int,
        name: str,
        suffix: str | None = None,
        pad: int = 3,
    ):
        i_str = str(i).rjust(pad, "0") + "-"
        prefix, *name_segs = name.split("-")
        if (
            prefix[0] == "p"
            and len(prefix) == 4
            and prefix[1:].isnumeric()
        ):
            i_str = f"{prefix}-{i_str}"
            name = "-".join(name_segs)
        return (
            i_str
            + name
            + (
                ""
                if suffix is None or not len(suffix)
                else f"-{suffix}"
            )
        )

    def rebal_variants(self):
        return [
            ("", None),
            ("-w", "3D"),
        ]

    @classmethod
    def qile(
        cls,
        n: int,
        i: int,
        r: int | None = None,
        m: float | None = None,
        c: float | None = None,
        f: str = "q",
        w: float = 0.5,
    ):
        if f == "q":
            return SignalQile(n, i, r, m=m, c=c)
        elif f == "z":
            return SignalZile(n, w, i, r, m=m, c=c)
        else:
            raise ValueError((cls, f))

    def with_perf_table(
        self,
        res: list[Strategy | Basket],
        t: Type[Proj],
        name: str = "perf",
        template: str = "perf_table",
    ):
        self = self.with_objects_from_template(
            name,
            template,
            [bt.vis_spec_perf() for bt in res],
            t=StratExplorer,
        )
        assert isinstance(self, t), (self, t)
        return self

    def with_rx_graph(
        self,
        res: list[Strategy | Basket],
        t: Type[Proj],
        name: str = "rx",
        template: str = "line_cum",
    ):
        self = self.with_objects_from_template(
            name,
            template,
            [bt.vis_spec_rx() for bt in res],
            t=StratExplorer,
        )
        assert isinstance(self, t), (self, t)
        return self

    def with_signal_graph(
        self,
        res: list[Strategy | Basket],
        t: Type[Proj],
        name: str = "signal",
        template: str = "line",
    ):
        # TODO: put quantiles on the y axis
        self = self.with_objects_from_template(
            name,
            template,
            [
                bt.vis_spec_signal()
                for bt in res
                if bt.params.signal is not None
            ],
            t=StratExplorer,
        )
        assert isinstance(self, t), (self, t)
        return self

    def with_partn_heatmap(
        self,
        res: list[Strategy | Basket],
        t: Type[Proj],
        by: str = "period",
        ref: (
            str | Literal["y"] | Literal["q"] | Literal["m"]
        ) = "y",
        name: str = "sharpe per year",
        template: str = "heatmap",
        value: str = "shrp",
        include: Callable[[Strategy | Basket], bool]
        | None = None,
        threshold: tuple[float, int] | None = None,
        # (val, span)
    ):
        # obj, label, params:dict
        if include is not None:
            i_include = [
                i for i, v in enumerate(res) if include(v)
            ]
            i_ignore = list(
                sorted(
                    set(range(len(res))).difference(
                        i_include
                    )
                )
            )
        else:
            i_include = list(range(len(res)))
            i_ignore = []
        self, res_include = self.with_perf(
            [res[i] for i in i_include],
            partition=DailyBtPartition(
                by,  # type: ignore
                ref,
                threshold=threshold,
            ),
        )
        # TODO: option to not bind, if already done (and just plotting a new field)
        self = self.with_objects_from_template(
            name,
            template,
            [
                (
                    assert_is(bt.perf_variants)[-1],
                    bt.label(),
                    dict(
                        # TODO: ...
                    ),
                )
                for bt in res_include
            ],
            t=StratExplorer,
            params=f"x=key:mu,y=label,v={value}",
        )
        assert isinstance(self, t), (self, t)
        if not len(i_ignore):
            res = res_include
        else:
            res_all = res_include + [
                res[i] for i in i_ignore
            ]
            i_all = i_include + i_ignore
            res = [
                r
                for i, r in sorted(
                    zip(i_all, res_all),
                    key=lambda iv: iv[0],
                )
            ]
        return self, res

    def with_mutn_heatmap(
        self,
        res: list[Strategy | Basket],
        t: Type[Proj],
        target: str = "sig",
        change: str = "levl",
        params: tuple[str | float, ...] = (),
        name: str | None = None,
        template: str = "heatmap",
        value: str = "shrp_g",
        include: Callable[[Strategy | Basket], bool]
        | None = None,
    ):
        name = name or (
            f"{value} varying {target} {change} {params}"
        )
        if include is not None:
            i_include = [
                i for i, v in enumerate(res) if include(v)
            ]
            i_ignore = list(
                sorted(
                    set(range(len(res))).difference(
                        i_include
                    )
                )
            )
        else:
            i_include = list(range(len(res)))
            i_ignore = []
        self, res_include = self.with_perf(
            [res[i] for i in i_include],
            mutation=DailyBtMutation(
                target,  # type: ignore
                change,  # type: ignore
                params,  # type: ignore
            ),
        )
        # TODO: option to not bind, if already done (and just plotting a new field)
        self = self.with_objects_from_template(
            name,
            template,
            [
                (
                    assert_is(bt.perf_variants)[-1],
                    bt.label(),
                    dict(
                        # TODO: ...
                    ),
                )
                for bt in res_include
            ],
            t=StratExplorer,
            params=f"x=key:mu,y=label,v={value}",
            # TODO: if not given x, or given x_left
            # then use all to the left of x_left (?)
        )
        assert isinstance(self, t), (self, t)
        if not len(i_ignore):
            res = res_include
        else:
            res_all = res_include + [
                res[i] for i in i_ignore
            ]
            i_all = i_include + i_ignore
            res = [
                r
                for i, r in sorted(
                    zip(i_all, res_all),
                    key=lambda iv: iv[0],
                )
            ]
        return self, res

    def with_perf(
        self,
        results: list[Strategy | Basket],
        partition: DailyBtPartition
        | list[DailyBtPartition | None]
        | None = None,
        mutation: DailyBtMutation
        | list[DailyBtMutation | None]
        | None = None,
    ):
        acc = []
        if not isinstance(partition, list):
            partition = [partition]
        if not isinstance(mutation, list):
            mutation = [mutation]
        for res in results:
            rx = res.rx()
            assert isinstance(rx, DailyBtRx), rx
            variants = []
            for p, m in itertools.product(
                partition, mutation
            ):
                perf = DailyBtPerf.new(
                    rx,
                    partition=p,
                    mutation=m,
                )
                context.get().with_obj(perf)
                variants.append(perf)

            res = res.with_perf_variants(*variants)
            acc.append(res)
        return self, acc


@dataclass(frozen=True)
class MomExplorer(StratExplorer):
    SPAN_FAST: ClassVar[list[int]] = []
    SPAN_RATIO: ClassVar[list[float]] = []
    SPAN_VOL_SIG: ClassVar[list[int]] = []
    SPAN_VOL_SCALE: ClassVar[list[int]] = []

    def mom_params(
        self,
        product_symbols: list[tuple[str, str]],
        span_fast: list[int] = [],
        span_ratio: list[float] = [],
        span_vol_sig: list[int] = [],
        span_vol_scale: list[int] = [],
        rebal: str | None = None,
        with_no_signal: bool = True,
        qiles: dict[
            tuple[str, str, float, int, float],
            list[SignalQile | SignalZile | None],
        ]
        | None = None,
        qiles_default: list[
            SignalQile | SignalZile | None
        ] = [None],
        # TODO: rebal rules
        notional_caps: dict[str, float] | None = None,
        notional_caps_default: float | None = None,
        z_clip: list[tuple[str, int, float] | None]
        | None = None,
    ):
        cls = type(self)

        z_clip = z_clip or [None]
        qiles = qiles or {}

        if not len(span_fast):
            span_fast = cls.SPAN_FAST
        if not len(span_ratio):
            span_ratio = cls.SPAN_RATIO
        if not len(span_vol_sig):
            span_vol_sig = cls.SPAN_VOL_SIG
        if not len(span_vol_scale):
            span_vol_scale = cls.SPAN_VOL_SCALE

        signals = lambda p, sym: (
            ([None] if with_no_signal else [])
            + [
                SignalOsc(
                    fast,
                    int(fast * ratio),
                    vol,
                    qile=qile
                    if isinstance(qile, SignalQile)
                    else None,
                    zile=qile
                    if isinstance(qile, SignalZile)
                    else None,
                    activation=(
                        "tanh" if qile is None else ""
                    ),
                )
                for fast in span_fast
                for vol in span_vol_sig
                for ratio in span_ratio
                for qile in qiles.get(
                    (p, sym, fast, vol, ratio),
                    qiles_default,
                )
            ]
        )
        scales = [
            ScaleVol(span=span, z_clip=z_c)
            for span in span_vol_scale
            for z_c in z_clip
        ]

        params = [
            Strategy(
                asset=AssetConcrete(
                    product,
                    symbol,
                ),
                params=StrategyParams(
                    scale=scale,
                    signal=signal,
                    rebal=rebal,
                    cap=(
                        None
                        if cap is None
                        else CapNotional(cap)
                    ),
                ),
                # TODO: fx?
            )
            for product, symbol in product_symbols
            for scale in scales
            for signal in signals(product, symbol)
            for cap in (
                [None]
                if notional_caps is None
                else ([None] if signal is None else [])
                + [
                    notional_caps.get(
                        symbol, notional_caps_default
                    )
                ]
            )
        ]

        return params


Proj = TypeVar("Proj", bound=StratExplorer)

__all__ = ["StratExplorer", "MomExplorer"]
