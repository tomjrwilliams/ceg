
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

#  ------------------

@dataclass(frozen=True)
class StratDomain(Domain):

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

    @dataclass(frozen=True)
    class Root(UseRootConfigs):

        version: int = 0
        type: Literal["StratDomain"] = "StratDomain"

        @dataclass(frozen=True)
        class Snap(Snap):
            event: (
                DomainSync
                | ObjectNew
                | ObjectChange
                | ValueUpdate
            )
            type: Literal["StratDomainSnap"] = "StratDomainSnap"

        @classmethod
        def new(
            cls,
            parent: EngineRoot,
            name: str = "strat",
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
            | DailyBtBench
            | DailyTF
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
            | Collection.Snap
            | StratDomain.Root.Snap
            | DailyBtBench.Snap
            | DailyTF.Snap
        )

    @classmethod
    def new(
        cls,
        eng: EngineRoot | None = None,
    ):
        if eng is None:
            eng = EngineRoot.new(ROOT)
        
        self = cls.empty(eng)

        return self
    
    def with_univ(
        self,
        name: str,
        depth: int = 0,
        eng: EngineRoot | None = None,
    ):
        if eng is None:
            eng = EngineRoot.new(ROOT)

        univ = UniverseDaily.new(
            eng, f"{name}-{depth}",
        )
        return self, univ

    def with_univ_fork(
        self, univ: UniverseDaily, depth: int = 0
    ):
        univ = univ.fork(self, depth=depth)
        return self, univ

    def with_strat(
        self,
        params: StratParams,
        univ: UniverseDaily,
        eng: EngineRoot
    ):
        result = StratResult(
            params=params,
            domain=self,
        )
        cast_self = result.cast_self

        univ, result = result.with_bt(univ, eng)
        
        univ, result = cast_self(result.into_uvar(univ))

        if params.rescale is None:
            return univ, result.done()
            
        univ, result = cast_self(params.rescale.with_rescale(
            params,
            result,
            univ,
            eng,
        ))
        
        return univ, result.done()

    def with_basket(
        self,
        params: BasketParams,
        univ: UniverseDaily,
        eng: EngineRoot,
        with_i_perf: bool = False,
        with_i_signal: bool = False,
        with_i_scale: bool = False,
    ):
        result = BasketResult(
            params=params,
            domain=self,
        )
        cast_self = result.cast_self

        bts = []
        
        for strat in params.asset:
                
            strat_result = StratResult(
                params=strat,
                domain=self,
            )
            univ, strat_result = strat_result.with_bt(
                univ, eng, assert_scale=False
            )

            strat_result, self = strat_result.done()

            assert strat_result.i is not None, strat_result
            bts.append(strat_result.i)
            
        result = BasketResult(
            params=params,
            domain=self,
        )
        univ, result = result.with_bt(
            univ, eng
        )
        
        univ, result = cast_self(result.into_mvar(
            univ, bts,
            with_i_perf=with_i_perf,
            with_i_signal=with_i_signal,
            with_i_scale=with_i_scale,
        ))

        if params.rescale is None:
            return univ, result.done()
            
        univ, result = cast_self(params.rescale.with_rescale(
            params,
            result,
            univ,
            eng,
        ))
        
        return univ, result.done()

    def with_backtest(
        self,
        univ: UniverseDaily,
        params: StratParams | BasketParams,
        into: list[StratResult | BasketResult],
        with_i_perf: bool = False,
        with_i_signal: bool = False,
        with_i_scale: bool = False,
        # TODO: into as overload
        # if given append, else return result directly
    ):
        eng = univ.get_parent(self)

        if isinstance(params, StratParams):
            univ, (result, self) = self.with_strat(
                params, univ, eng
            )
            result = cast(StratResult, result)
        elif isinstance(params, BasketParams):
            univ, (result, self) = self.with_basket(
                params, univ, eng,
                with_i_perf=with_i_perf,
                with_i_signal=with_i_signal,
                with_i_scale=with_i_scale,
            )
            result = cast(BasketResult, result)
        else:
            raise ValueError(params)

        into.append(result)
        assert isinstance(self, StratDomain), self

        self = self.with_objs(
            univ,
            force=True,
            t=StratDomain
        )

        # TODO: return on notional is relevant when vol scaling

        return self, univ, into

    # TODO: deprecated?
    def with_rx_agg(
        self,
        coll: Collection,
        res: list[StratResult]
    ):
                
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
            t=StratDomain
        ), coll, rx_sum
    
__all__ = ["StratDomain"]