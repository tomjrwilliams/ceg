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


class norm_range_pct_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64
    v: Ref.Scalar_F64
    a: float
    b: float

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(
        cls, 
        l: Ref.Scalar_F64, 
        r: Ref.Scalar_F64, 
        v: Ref.Scalar_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return norm_range_pct("norm_pct", l=l, r=r, v=v, a=a, b=b)


class norm_range_pct(norm_range_pct_kw, Node.Scalar_F64):
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
    >>> g, r2 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r3 = bind(norm_pct.new(r0, r1, r2))
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
    ...     v3 = round(
    ...         r3.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2, v3)
    0.13 -0.13 -0.95
    0.64 0.1 6.11
    -0.54 0.36 -1.48
    1.3 0.95 1.38
    -0.7 -1.27 0.56
    """

    DEF: ClassVar[Defn] = define.node(Node.Scalar_F64, norm_range_pct_kw)
    bind = define.bind_from_new(norm_range_pct_kw.new, norm_range_pct_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        v = self.v.history(graph).last_before(event.t)
        if l is None or np.isnan(l) or r is None or np.isnan(r) or v is None or np.isnan(v):
            return np.nan
        norm = (r - l)
        if norm == 0:
            return np.nan
        res = (v - l) / norm
        return (res + self.a) * self.b

class norm_range_pct_vec_kw(NamedTuple):
    type: str
    #
    vec: Ref.Vector_F64
    a: float
    b: float

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Vector_F64:
        return Ref.d1_f64(i, slot=slot)

    @classmethod
    def new(
        cls, 
        vec: Ref.Vector_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return norm_range_pct_vec("norm_range_pct_vec", vec=vec, a=a, b=b)


class norm_range_pct_vec(norm_range_pct_vec_kw, Node.Vector_F64):
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
    >>> g, r2 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r3 = bind(norm_pct.new(r0, r1, r2))
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
    ...     v3 = round(
    ...         r3.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2, v3)
    0.13 -0.13 -0.95
    0.64 0.1 6.11
    -0.54 0.36 -1.48
    1.3 0.95 1.38
    -0.7 -1.27 0.56
    """

    DEF: ClassVar[Defn] = define.node(Node.Vector_F64, norm_range_pct_vec_kw)
    bind = define.bind_from_new(norm_range_pct_vec_kw.new, norm_range_pct_vec_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(graph).last_before(event.t)
        l = int(len(v)/2)
        mid = (v[l-1] + v[l]) / 2
        range = v[0] - v[-1]
        # res = (v - mid) / range
        res = (v - mid) / mid
        # print(res)
        return res
