from typing import NamedTuple, ClassVar

import datetime as dt
import numpy as np

from ..core import (
    Graph,
    Node,
    Ref,
    Event,
    Ready,
    Defn,
    define,
    steps,
    batches,batch_until
)

#  ------------------


class align_d0_date_kw(NamedTuple):
    v: Ref.Scalar_Date
    to: Ref.Any
    tx: float


class align_d0_date(align_d0_date_kw, Node.D0_Date):
    """
    >>> g = Graph.new()
    >>> from . import dates
    >>> d0 = dt.date(2025, 1, 1)
    >>> d1 = dt.date(2025, 2, 1)
    >>> g, r0 = dates.daily.fs().loop(g, d0, d1, n = 1.)
    >>> g, r1 = dates.daily.fs().loop(g, d0, d1, n = 2, step=2., keep = 3)
    >>> g, r2 = dates.daily.fs().loop(g, d0, d1, n = 3, step=3., keep = 3)
    >>> with g.implicit() as (bind, done):
    ...     r10 = bind(align.scalar_date.new(r1, r0), when=Ready.ref(r0), keep=2)
    ...     r20 = bind(align.scalar_date.new(r2, r0), when=Ready.ref(r0), keep=2)
    ...     g = done()
    ...
    >>> es = map(Event.zero, (r0, r1, r2))
    >>> day = lambda d: "N" if d is None else d.day
    >>> for g, es, _ in batch_until(
    ...     g, lambda _, e: e.ref.i == 0, *es, n=5, next=True, iter=True
    ... )():
    ...     t = es[0].t
    ...     v0 = day(r0.last_before(g, t))
    ...     v1 = day(r1.last_before(g, t))
    ...     v10 = day(r10.last_before(g, t))
    ...     v2 = day(r2.last_before(g, t))
    ...     v20 = day(r20.last_before(g, t))
    ...     print(t, v0, v1, v10, v2, v20)
    0.0 1 1 1 1 1
    1.0 2 1 N 1 N
    2.0 3 3 3 1 N
    3.0 4 3 N 4 4
    4.0 5 5 5 4 N
    """


    DEF: ClassVar[Defn] = define.node(
        Node.Scalar_Date, align_d0_date_kw
    )

    @classmethod
    def new(cls, v: Ref.Scalar_Date, to: Ref.Any, tx=10e-6):
        return cls(v=v, to=to, tx=tx)

    def __call__(self, event: Event, graph: Graph):
        if event.prev is None:
            return self.v.history(graph).last_before(
                event.t
            )
        return self.v.history(graph).last_between(
            event.prev.t + self.tx, event.t
        )


class align_d0_f64_kw(NamedTuple):
    v: Ref.Scalar_F64
    to: Ref.Any
    tx: float


class align_d0_f64(align_d0_f64_kw, Node.D0_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.fs().walk(g, step = 1.)
    >>> g, r1 = rand.gaussian.fs().walk(g, step = 1.5, keep = 3)
    >>> g, r2 = rand.gaussian.fs().walk(g, step = 2., keep = 3)
    >>> with g.implicit() as (bind, done):
    ...     r10 = bind(align.scalar_f64.new(r1, r0), when=Ready.ref(r0), keep=2)
    ...     r20 = bind(align.scalar_f64.new(r2, r0), when=Ready.ref(r0), keep=2)
    ...     g = done()
    ...
    >>> es = map(Event.zero, (r0, r1, r2))
    >>> for g, es, _ in batch_until(
    ...     g, lambda _, e: e.ref.i == 0, *es, n=5, next=True, iter=True
    ... )():
    ...     t = es[0].t
    ...     v0 = round(r0.last_before(g, t), 2)
    ...     v1 = round(r1.last_before(g, t), 2)
    ...     v10 = round(r10.last_before(g, t), 2)
    ...     v2 = round(r2.last_before(g, t), 2)
    ...     v20 = round(r20.last_before(g, t), 2)
    ...     print(t, v0, v1, v10, v2, v20)
    0.0 0.13 -0.13 -0.13 0.64 0.64
    1.0 0.23 -0.13 nan 0.64 nan
    2.0 0.59 -0.67 -0.67 1.94 1.94
    3.0 1.54 -1.37 -1.37 1.94 nan
    4.0 0.27 -1.37 nan 1.32 1.32
    """

    DEF: ClassVar[Defn] = define.node(
        Node.Scalar_F64, align_d0_f64_kw
    )

    @classmethod
    def new(cls, v: Ref.Scalar_F64, to: Ref.Any, tx=10e-6):
        return cls(v=v, to=to, tx=tx)

    def __call__(self, event: Event, graph: Graph):
        if event.prev is None:
            return self.v.history(graph).last_before(
                event.t
            )
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

    DEF: ClassVar[Defn] = define.node(
        Node.Scalar_F64, lag_d0_f64_kw
    )

    @classmethod
    def new(cls, v: Ref.Scalar_F64, w: int):
        return cls(cls.DEF.name, v=v, w=w)

    def __call__(self, event: Event, graph: Graph):
        return self.v.history(graph).last_n_before(
            self.w, event.t
        )[0]


class lag_d0_date_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_Date
    w: int


# NOTE: probably used after align
class lag_d0_date(lag_d0_date_kw, Node.Scalar_Date):

    DEF: ClassVar[Defn] = define.node(
        Node.Scalar_Date, lag_d0_date_kw
    )

    @classmethod
    def new(cls, v: Ref.Scalar_Date, w: int):
        return cls(cls.DEF.name, v=v, w=w)

    def __call__(self, event: Event, graph: Graph):
        return self.v.history(graph).last_n_before(
            self.w, event.t
        )[0]


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
