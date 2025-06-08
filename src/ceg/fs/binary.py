# add
# div
# sub
# mul
# pow, log

from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from ..core import (
    Graph,
    Node,
    Ref,
    Event,
    Loop,
    Defn,
    define,
    steps,
    batches,
)

#  ------------------


class ratio_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64


class ratio(ratio_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> g, r1 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(ratio.new(r0, r1))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v2 = round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    0.13 -0.13 -0.95
    0.64 0.1 6.11
    -0.54 0.36 -1.48
    1.3 0.95 1.38
    -0.7 -1.27 0.56
    """

    DEF: ClassVar[Defn] = define.node(Node.Scalar_F64, ratio_kw)

    @classmethod
    def new(cls, l: Ref.Scalar_F64, r: Ref.Scalar_F64):
        return cls(cls.DEF.name, l=l, r=r)

    def __call__(self, event: Event, graph: Graph):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        if l is None or np.isnan(l) or r is None or np.isnan(r):
            return np.NAN
        return l / r


#  ------------------


class pct_diff_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64
    shift: tuple[int | None, ...] | None
    m: float | None = None


class pct_diff(pct_diff_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> g, r1 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(pct_diff.new(r0, r1))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v2 = round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    0.13 -0.13 -1.95
    0.64 0.1 5.11
    -0.54 0.36 -2.48
    1.3 0.95 0.38
    -0.7 -1.27 -0.44
    """

    DEF: ClassVar[Defn] = define.node(
        Node.Scalar_F64, pct_diff_kw
    )

    @classmethod
    def new(
        cls,
        l: Ref.Scalar_F64,
        r: Ref.Scalar_F64,
        shift: tuple[int | None, ...] | None = None,
        m: float | None = None,
    ):
        return cls(cls.DEF.name, l=l, r=r, shift=shift, m=m)

    def __call__(self, event: Event, graph: Graph):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        if l is None or np.isnan(l) or r is None or np.isnan(r):
            return np.NAN

        return ((l / r) - 1) * (
            1 if self.m is None else self.m
        )


#  ------------------


class subtract_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64


class subtract(subtract_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> g, r1 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(subtract.new(r0, r1))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v2 = round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    0.13 -0.13 0.26
    0.64 0.1 0.54
    -0.54 0.36 -0.9
    1.3 0.95 0.36
    -0.7 -1.27 0.56
    """

    DEF: ClassVar[Defn] = define.node(Node.Scalar_F64, ratio_kw)

    @classmethod
    def new(cls, l: Ref.Scalar_F64, r: Ref.Scalar_F64):
        return cls(cls.DEF.name, l=l, r=r)

    def __call__(self, event: Event, graph: Graph):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        if l is None or np.isnan(l) or r is None or np.isnan(r):
            return np.NAN
        return l - r


diff = subtract
