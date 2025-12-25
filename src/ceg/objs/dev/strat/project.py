
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

import numpy as np
import polars as pl

from .... import apis
from ....apis import t
from ....apis import kv
from ....engine import *

from ..bars.d import *
from ..calc.d import *
from ..eco.d import *

from ..agg import *
from ..pfo import *

from .assets import *
from .scales import *
from .signals import *
from .uvar import *
from .mvar import *
from .domain import *

from ...vis import explorer

@dataclass(frozen=True)
class StratProject(explorer.Project):

    SPAN_FAST: ClassVar[list[int]] = []
    SPAN_RATIO: ClassVar[list[float]] = []
    SPAN_VOL_SIG: ClassVar[list[int]] = []
    SPAN_VOL_SCALE: ClassVar[list[int]] = []

    def mom_params(
        self,
        product_symbols: list[tuple[str, str]],
        span_fast: list[int]=[],
        span_ratio: list[float]=[],
        span_vol_sig: list[int]=[],
        span_vol_scale: list[int]=[],
        span_vol_rescale: list[int] | None = None,
        rebal: str | None = None,
        with_no_signal: bool = True,
        # TODO: rebal rules
    ):
        cls = type(self)

        if not len(span_fast):
            span_fast = cls.SPAN_FAST
        if not len(span_ratio):
            span_ratio = cls.SPAN_RATIO
        if not len(span_vol_sig):
            span_vol_sig = cls.SPAN_VOL_SIG
        if not len(span_vol_scale):
            span_vol_scale = cls.SPAN_VOL_SCALE

        signals = (
            [None]
            if with_no_signal
            else []
        ) + [
            SignalOsc(
                fast, int(fast * ratio), vol
            )
            for fast in span_fast
            for vol in span_vol_sig
            for ratio in span_ratio
        ]
        scales = [
            ScaleVol(span=span)
            for span in span_vol_scale
        ]
        rescales = [
            None
        ] if span_vol_rescale is None else [
            ScaleVol(span=span)
            for span in span_vol_rescale
        ]

        params = [
            StratParams(
                asset=AssetConcrete(
                    product,
                    symbol,
                ),
                scale=scale,
                signal=signal,
                rescale=rescale,
                rebal=rebal,
            )
            for rescale in rescales
            for scale in scales
            for signal in signals
            for product, symbol in product_symbols
        ]

        return params

    def with_bt_page(
        self,
        name: str,
        d: Domain,
        bts: list[StratResult | BasketResult],
        # TODO: or BasketResult or basketresult and index
        # TODO: and pfolio?
    ):
        self = self.with_domains(**{
            name: lambda sync: explorer.AppDomain(
                [d], sync
            )
        }).with_page(
            name,
            type="daily",
            domain=name,
        )

        # TODO: indicate in params which to show in results
        # including sub-items within basket

        # TODO: ideally, across multiple levels of depth

        # --

        # one table where individual perf rows are indicated, stacked, displayed

        # another where spread to benchmark perf rows are indicated (with extra key to indicate benchmark)
        # if any 

        # --

        # then, per partn / mutation:
        # relevant distribution visualisation across strategies for tha partn/mutation comb

        # so pass the partn/mutn in here or unpack dynamically?
        
        # --

        # and, again per partn / mutation across all strategy benchmark spreads

        for bt in bts:
            assert bt.res is not None, bt
            self = self.with_object(
                bt.res.perf,
                graph=0,
                label=bt.params.label(),
                group="perf",
            ).with_object(
                bt.res.rx,
                graph=1,
                label=bt.params.label(),
                group="rx",
            )
            # .with_object(
            #     bt.res.pos,
            #     graph=0,
            #     label=bt.params.label(),
            #     group="pos"
            # )
            if bt.signal is not None:
                assert isinstance(bt.signal, str), bt
                # TODO: graph many
                self = self.with_object(
                    bt.signal,
                    graph=2,
                    label=bt.params.label(),
                    group="sig"
                )
        self = self.with_graph(
            0,
            "bench"
        ) 
        self = self.with_graph(
            1,
            "line_cum"
        )
        self = self.with_graph(
            2,
            "line"
        )
        return cast(StratProject, self)
    
__all__ = ["StratProject"]