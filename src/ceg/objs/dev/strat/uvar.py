
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

#  ------------------

# TODO: drop eng from sig and just getparent as required

@dataclass(frozen=True)
class ScaleNotional:
    
    def label(self):
        return "cn"
    
    def with_scale(
        self,
        params: UvarBtParams,
        result: UvarBtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        if (
            isinstance(params.asset, AssetConcrete)
            and params.asset.product == "FUT"
        ):
            scale = FutDailyOrdinal.new(
                eng,
                symbol=params.asset.symbol,
                ident = 1,
                field="close",
                adjust="raw",
                source="frd"
            )
            univ, res = universe.with_scale(
                scale,
                into=result,
                scale=1.,
            )
            return univ, res
        raise ValueError(self)

@dataclass(frozen=True)
class ScaleVol:
    span: int
    # method = rms or sdev

    ann: bool = True

    def label(self):
        return f"cv-{self.span}"
    
    def with_scale(
        self,
        params: UvarBtParams,
        result: UvarBtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        assert result.rx is not None, result
        assert result.domain is not None, result
        rx = result.domain.get(result.rx, Daily)
        scale = DailyERms.new(
            rx,
            span=self.span,
            alpha=None,
            ann=self.ann,
        )
        univ, res = universe.with_scale(
            scale,
            into=result,
            scale=1.,
        )
        return univ, res
    
    def with_rescale(
        self,
        params: UvarBtParams,
        result: UvarBtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        i_sc = result.i_scale
        assert isinstance(result.scale, str), result
        assert result.res is not None, result
        assert result.domain is not None, result
        rescale = DailyERms.new(
            result.domain.get(
                result.res.rx, Daily
            ),
            span=self.span,
            ann=True,
        )
        res = result.with_obj(rescale)
        assert result.i is not None, result
        bt = DailyBtUvar.new(
            universe,
            result.i,
            rescale=rescale
        )
        bt_pos = DailyBtPos.new(bt)
        bt_rx = DailyBtRx.new(bt)
        res = replace(
            res,
            res = DailyBtUvarValues(
                bt_pos.path,
                bt_rx.path,
            )
        ).with_obj((bt, bt_pos, bt_rx))
        return universe, res

@dataclass(frozen=True)
class SignalEco:
    symbol: str
    method: str
    speed: int # span or window
    activation: str | None = "tanh"
    scale: float = 1.

    def label(self):
        s = f"{self.symbol}-{self.method}-{self.speed}"
        if self.activation is not None:
            return f"{s}-{self.activation}"
        return s
    
    def with_signal(
        self,
        params: UvarBtParams,
        result: UvarBtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        v_eco = EcoDailyObservation.new(
            eng,
            symbol=self.symbol,
            source="fred",
        )
        sig: Daily
        if self.method == "z":
            sig = DailyZ.new(
                v_eco, window=self.speed
            )
        elif self.method == "ez":
            sig = DailyEZ.new(
                v_eco, span=self.speed
            )
        elif self.method == "zw":
            sig = DailyZW.new(
                v_eco, window=self.speed
            )
        elif self.method == "ezw":
            sig = DailyEZW.new(
                v_eco, span=self.speed
            )
        elif self.method == "rz":
            sig = DailyRZ.new(
                v_eco, window=self.speed
            )
        elif self.method == "erz":
            sig = DailyERZ.new(
                v_eco, span=self.speed
            )
        elif self.method == "dema":
            sig = DailyDEMA.new(
                v_eco, span=self.speed
            )
        elif self.method == "demax":
            sig = DailyDEMax.new(
                v_eco, span=self.speed
            )
        elif self.method == "zwdemin":
            demax = DailyDEMin.new(
                v_eco, span=self.speed
            )
            sig = DailyEZW.new(
                demax, span=self.speed
            )
        elif self.method == "zwdemax":
            demax = DailyDEMax.new(
                v_eco, span=self.speed
            )
            sig = DailyEZW.new(
                demax, span=self.speed
            )
        elif self.method == "raw":
            sig = v_eco
        else:
            raise ValueError((bt))
        if self.activation is not None:
            act = self.activation
            if act == "tanh":
                sig = DailyTF.new(
                    sig, f="tanh"
                )
            elif act == "n_tanh_p_one":
                sig = DailyTF.new(
                    sig, f="tanhrsign",params=(None,1)
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
        univ, res = universe.with_signal(
            sig,
            into=result,
            scale=self.scale,
            # activation=self.activation,
        )
        return univ, res
        
@dataclass(frozen=True)
class SignalOsc:
    fast: int
    slow: int
    vol: int
    activation: str = "tanh"

    def label(self):
        return f"osc-{self.fast}-{self.slow}-{self.vol}-{self.activation}"

    def with_signal(
        self,
        params: UvarBtParams,
        result: UvarBtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        asset = params.asset
        if isinstance(asset, AssetConcrete):
            px_adj = "pct" if asset.pct else "abs"
            px = (
                FutDailyOrdinal.new(
                    eng,
                    symbol=asset.symbol,
                    ident = 1,
                    field="close",
                    adjust=px_adj,
                    source="frd"
                )
                if asset.product == "FUT"
                else EqDailyEtf.new(
                    eng,
                    symbol=asset.symbol,
                    field="close",
                    adjust="price",
                    source="frd"
                )
                if asset.product == "ETF"
                else None
            )
            assert px is not None, (params, self)
            res = result.with_obj(px)
            sig = BarDailyEOsc.new(
                px,
                fast=self.fast,
                slow=self.slow,
                vol=self.vol,
                activation=self.activation
            )
            univ, res = universe.with_signal(
                sig,
                into=res,
            )
            return univ, res
        raise ValueError(self)

@dataclass(frozen=True)
class AssetConcrete:
    product: str
    symbol: str
    pct: bool = False

    def label(self):
        pct = "-pct" if self.pct else ""
        return f"{self.symbol}{pct}"
    
    def with_rx(
        self,
        params: UvarBtParams,
        result: UvarBtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        rx = (
            FutDailyRx.new(
                eng,
                symbol=self.symbol,
                ident = 1,
                field="close",
                adjust="abs",
                source="frd"
            )
            if self.product == "FUT"
            else EqDailyRx.new(
                eng,
                symbol=self.symbol,
                field="close",
                adjust="full",
                source="frd"
            )
            if self.product == "ETF"
            else None
        )
        assert rx is not None, self
        return universe.with_rx(
            rx, into=result
        )

@dataclass(frozen=True)
class AssetSynthetic:
    name: str
    alias: str | None = None
    
    def label(self):
        return self.alias or self.name
    
    def with_rx(self, 
        params: UvarBtParams,
        result: UvarBtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        assert result.domain is not None, result
        rx = result.domain.get(self.name, t=Daily)
        assert rx is not None, self
        return universe.with_rx(
            rx, into=result
        )


@dataclass(frozen=True)
class UvarBtParams(BtParams):
    asset: AssetConcrete | AssetSynthetic
    scale: ScaleNotional | ScaleVol
    signal: SignalOsc | SignalEco | None = None
    rescale: ScaleVol | None = None
    rebal: str | None = None

    def label(self):
        return (
            f"{self.asset.label()}-{self.scale.label()}"
            + (
                "" 
                if self.signal is None 
                else f"-{self.signal.label()}"
            )
            + (
                "" 
                if self.rescale is None 
                else f"-{self.rescale.label()}"
            )
            + (
                "" 
                if self.rebal is None 
                else f"-{self.rebal}"
            )
        )

@dataclass(frozen=True)
class UvarBtResult(DailyBtUvarResult):
    params: UvarBtParams

    # symbol: str
    # fut_px: FutDailyOrdinal
    # fut_rx: FutDailyRx

    def assume_res(self):
        assert self.res is not None, self
        return self.res

    def cast_self(
        self, univ_result: tuple[Universe, BtResult]
    ) -> tuple[UniverseDaily, UvarBtResult]:
        univ, res = univ_result
        assert isinstance(univ, UniverseDaily), univ
        assert isinstance(res, UvarBtResult), res
        return univ, res

    def load(self, univ: UniverseDaily, eng: EngineRoot):
        params = self.params
        cast_self = self.cast_self
        
        univ, self = cast_self(params.asset.with_rx(
            params,
            self,
            univ,
            eng,
        ))
        univ, self = cast_self(params.scale.with_scale(
            params,
            self,
            univ,
            eng,
        ))
        if params.signal is not None:
            univ, self = cast_self(params.signal.with_signal(
                params,
                self,
                univ,
                eng,
            ))

        univ, self = cast_self(univ.with_bt(
            self, 
            rebalance=self.rebalance,
            unit=1.,
        ))

        univ, self = cast_self(univ.into_uvar(self))

        if params.rescale is None:
            return univ, self.done()
            
        univ, self = cast_self(params.rescale.with_rescale(
            params,
            self,
            univ,
            eng,
        ))
        
        return univ, self.done()
    
@dataclass(frozen=True)
class UvarBtDomain(Domain):
    """
    >>> spec = UvarBtParams(
    ...     asset=AssetConcrete("FUT", "ES"),
    ...     signal=SignalOsc(32, 96, 48),
    ...     scale=ScaleVol(span=48, ann=False)
    ... )
    >>> dom, univ = UvarBtDomain.new("example")
    >>> dom, univ, bts = dom.with_bts(univ, spec)
    >>> rx = Collection.new(
    ...     univ.get_parent(dom),
    ...     univ.path + "-rx"
    ... )
    >>> dom, _, agg = dom.with_agg(
    ...     rx, bts
    ... )
    >>> dom = dom.sync()
    >>> pos = dom.get(bts[0].res.pos, Daily)
    >>> rx = dom.get(bts[0].res.rx, Daily)
    >>> v_pos = pos.value_key_val(dom).get(pos.path).ser
    >>> v_rx = rx.value_key_val(dom).get(rx.path).ser
    >>> v_agg = agg.value_key_val(dom).get(agg.path).ser
    >>> v_pos.tail(8)
    shape: (8,)
    Series: 'example-bt-000-pos' [f64]
    [
        -0.006311
        null
        null
        -0.006753
        -0.006675
        -0.006903
        -0.007089
        -0.007451
    ]
    >>> v_rx.tail(8)
    shape: (8,)
    Series: 'example-bt-000-rx' [f64]
    [
        -0.032185
        null
        null
        -0.613726
        -0.074286
        0.447206
        0.139791
        0.824148
    ]
    >>> v_agg.tail(8)
    shape: (8,)
    Series: 'example-rx-000' [f64]
    [
        -0.032185
        0.0
        0.0
        -0.613726
        -0.074286
        0.447206
        0.139791
        0.824148
    ]
    >>> pl.DataFrame(dict(pos=v_pos, rx=v_rx)).pipe(
    ...     apis.perf.perf_row, round=4, repr=True
    ... )
    mu 4.2188
    sd 11.2066
    rms 11.2085
    skew -1.3384
    shrp 0.3794
    sort 0.4375
    dd 26.6185
    calm 0.1585
    p_mu 0.0293
    p_sd 0.0341
    p_skew 0.1727
    """
    
    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

    @dataclass(frozen=True)
    class Root(UseRootConfigs):

        version: int = 0
        type: Literal["UvarBtDomain"] = "UvarBtDomain"

        @dataclass(frozen=True)
        class Snap(Snap):
            event: (
                DomainSync
                | ObjectNew
                | ObjectChange
                | ValueUpdate
            )
            type: Literal["UvarBtDomainSnap"] = "UvarBtDomainSnap"

        @classmethod
        def new(
            cls,
            parent: EngineRoot,
            name: str = "bt-univ",
        ):
            return cls(
                path = parent.extend_path(name),
                parent=parent
            )

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | FutDailyOrdinal
            | FutDailyRx
            | EqDailyEtf
            | EqDailyRx
            | BarDailyEOsc
            | DailyERms
            | DailyBtUvar
            | DailyBtPos
            | DailyBtRx
            | DailyAggregate
            | DailyZ | DailyEZ
            | DailyRZ | DailyERZ
            | DailyZW | DailyEZW
            | DailyRZW | DailyERZW
            | DailyDEMax
            | DailyDEMin
            | DailyDEMA
            | EcoDailyObservation
            | Universe
            | DailyAggregate
            | Collection
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | EqDailyEtf.Snap
            | EqDailyRx.Snap
            | BarDailyEOsc.Snap
            | DailyERms.Snap
            | DailyBtUvar.Snap
            | DailyBtPos.Snap
            | DailyBtRx.Snap
            | DailyAggregate.Snap
            | DailyZ.Snap | DailyEZ.Snap
            | DailyRZ.Snap | DailyERZ.Snap
            | DailyZW.Snap | DailyEZW.Snap
            | DailyRZW.Snap | DailyERZW.Snap
            | DailyDEMin.Snap
            | DailyDEMax.Snap
            | DailyDEMA.Snap
            | EcoDailyObservation.Snap
            | Universe.Snap
            | DailyAggregate.Snap
            | UvarBtDomain.Root.Snap
            | Collection.Snap
        )

    @classmethod
    def new(
        cls,
        name: str,
        eng: EngineRoot | None = None,
    ):
        if eng is None:
            eng = EngineRoot.new(ROOT)
        
        self = cls.empty(eng)

        univ = UniverseDaily.new(
            eng, name,
        )
        return self, univ
    
    def with_univ(
        self,
        name: str,
        eng: EngineRoot | None = None,
    ):
        if eng is None:
            eng = EngineRoot.new(ROOT)

        univ = UniverseDaily.new(
            eng, name,
        )
        return self, univ


    def with_bts(
        self,
        univ: UniverseDaily,
        *bt_params: UvarBtParams,
    ):
        eng = univ.get_parent(self)

        res: list[UvarBtResult] = []

        for params in bt_params:
            univ, (result, self) = UvarBtResult(
                params=params,
                domain=self,
            ).load(univ, eng)
            res.append(
                cast(UvarBtResult, result)
            )
        
        self = self.with_objs(
            univ,
            force=True,
            t=UvarBtDomain
        )

        # TODO: return on notional is relevant when vol scaling

        return self, univ, res

    def with_agg(
        self,
        coll: Collection,
        res: list[UvarBtResult]
    ):
        
        # TODO: replace with a basket backtest (mvar)

        # which we can eg. vol rescale as well
        
        agg = []
        for r in res:
            assert r.res is not None, (r)
            coll, i = coll.with_object(r.res.rx)
            agg.append(i)
        
        coll, i_agg = coll.with_agg(tuple(agg))

        rx_sum = DailyAggregate.new(coll, "hsum", i_agg)
        
        return self.with_objs(
            coll,
            rx_sum,
            force=True,
            t=UvarBtDomain
        ), coll, rx_sum

from ...vis import explorer

@dataclass(frozen=True)
class UvarBtProject(explorer.Project):

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
            UvarBtParams(
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
        bts: list[UvarBtResult],
        rx_agg: DailyAggregate | None = None,
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

        # --

        if rx_agg is not None:
            self = self.with_object(
                rx_agg.path,
                graph=0,
                label=name,
                group="rx",
            ).with_object(
                rx_agg.path,
                graph=0,
                label=name,
                group="pos"
            )
        for bt in bts:
            assert bt.res is not None, bt
            self = self.with_object(
                bt.res.rx,
                graph=0,
                label=bt.params.label(),
                group="rx",
            ).with_object(
                bt.res.pos,
                graph=0,
                label=bt.params.label(),
                group="pos"
            )
            if bt.signal is not None:
                assert isinstance(bt.signal, str), bt
                # TODO: graph many
                self = self.with_object(
                    bt.signal,
                    graph=1,
                    label=bt.params.label(),
                    group="sig"
                )
        self = self.with_graph(
            0,
            "perf"
        )
        self = self.with_graph(
            1,
            "line"
        )
        return cast(UvarBtProject, self)