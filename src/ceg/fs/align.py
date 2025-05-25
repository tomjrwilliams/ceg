from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from ..core import Graph, Node, Ref, Event, Loop, Defn, define, steps

#  ------------------

class align_d0_date_kw(NamedTuple):
    v: Ref.Scalar_Date
    to: Ref.Any
    tx: float

class align_d0_date(align_d0_date_kw, Node.D0_Date):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_Date, align_d0_date_kw
    )

    @classmethod
    def new(cls, v: Ref.Scalar_Date, to: Ref.Any, tx = 10e-6):
        return cls(v=v, to=to, tx=tx)

    def __call__(
        self, event: Event, graph: Graph
    ):
        assert event.ref == self.to, dict(
            self=self, event=event
        )
        if event.prev is None:
            return self.v.history(graph).last_before(event.t)
        return self.v.history(graph).last_between(
            event.prev.t + self.tx, event.t
        )
        
class align_d0_f64_kw(NamedTuple):
    v: Ref.Scalar_F64
    to: Ref.Any
    tx: float

class align_d0_f64(align_d0_f64_kw, Node.D0_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, align_d0_f64_kw
    )

    @classmethod
    def new(cls, v: Ref.Scalar_F64, to: Ref.Any, tx = 10e-6):
        return cls(v=v, to=to, tx=tx)

    def __call__(
        self, event: Event, graph: Graph
    ):
        assert event.ref == self.to, dict(
            self=self, event=event
        )
        if event.prev is None:
            return self.v.history(graph).last_before(event.t)
        return self.v.history(graph).last_between(
            event.prev.t + self.tx, event.t
        )
#  ------------------

class lag_d0_f64_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    w: int


# NOTE: probably used after align
class lag_d0_f64(lag_d0_f64_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, lag_d0_f64_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64, w: int
    ):
        return cls(cls.DEF.name, v=v, w=w)

    def __call__(
        self, event: Event, graph: Graph
    ):
        return self.v.history(graph).last_n_before(self.w, event.t)[0]
        
class lag_d0_date_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_Date
    w: int


# NOTE: probably used after align
class lag_d0_date(lag_d0_date_kw, Node.Scalar_Date):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_Date, lag_d0_date_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_Date, w: int
    ):
        return cls(cls.DEF.name, v=v, w=w)

    def __call__(
        self, event: Event, graph: Graph
    ):
        return self.v.history(graph).last_n_before(self.w, event.t)[0]
        
#  ------------------

class align:

    d0_date = align_d0_date
    scalar_date = align_d0_date

    d0_f64 = align_d0_f64
    scalar_f64 = align_d0_f64

class lag:

    d0_date = lag_d0_date
    scalar_date = lag_d0_date

    d0_f64 = lag_d0_f64
    scalar_f64 = lag_d0_f64

#  ------------------

# TODO: nan_mask, cumsum zero fills, so if you want to put back nans