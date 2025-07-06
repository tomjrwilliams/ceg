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
    dataclass,
    define,
    steps,
    batches,
)

#  ------------------




@dataclass(frozen=True)
class norm_range_pct(Node.Scalar_F64):
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
    ...     r3 = bind(norm_range_pct.new(r0, r1, r2))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1), Event.zero(r2)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=4, iter=True
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
    0.13 -0.13 0.64 -2.0
    0.1 -0.54 0.36 -0.4
    1.3 0.95 -0.7 5.63
    -1.27 -0.62 0.04 2.03
    -2.33 -0.22 -1.25 0.51
    """
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64
    v: Ref.Scalar_F64
    a: float
    b: float

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
    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

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


@dataclass(frozen=True)
class norm_mid_pct_vec(Node.Vector_F64):
    """
    note: assumes sorted ascending, and goes a bit funky as mid -> 0 (particularly if inputs span both signs)
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian_vec.new((3,)),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(norm_mid_pct_vec.new(r0))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = np.round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = np.round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    [ 0.1  -0.54  0.36] [ -40.45   40.45 -201.93]
    [ 1.3   0.95 -0.7 ] [ 0.16 -0.16 -1.63]
    [-2.33 -0.22 -1.25] [ 0.34 -0.34 -1.04]
    [-0.73 -0.54 -0.32] [ 0.15 -0.15 -0.5 ]
    [ 1.37 -0.67  0.35] [-0.43  0.43 -1.18]
    """

    type: str
    #
    vec: Ref.Vector_F64
    a: float
    b: float

    @classmethod
    def new(
        cls, 
        vec: Ref.Vector_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return norm_mid_pct_vec("norm_mid_pct_vec", vec=vec, a=a, b=b)


    bind = define.bind_from_new(new, Node.Vector_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(graph).last_before(event.t)
        l = int(len(v)/2)
        mid = (v[l-1] + v[l]) / 2
        res = (v - mid) / mid
        return res


@dataclass(frozen=True)
class norm_mid_vec(Node.Vector_F64):
    """
    note: assumes sorted ascending
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian_vec.new((3,)),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(norm_mid_vec.new(r0))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = np.round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = np.round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    [ 0.1  -0.54  0.36] [ 0.13 -0.13  0.64]
    [ 1.3   0.95 -0.7 ] [ 0.18 -0.18 -1.83]
    [-2.33 -0.22 -1.25] [-0.32  0.32  0.99]
    [-0.73 -0.54 -0.32] [-0.09  0.09  0.32]
    [ 1.37 -0.67  0.35] [-0.32  0.32 -0.86]
    """
    type: str
    #
    vec: Ref.Vector_F64
    a: float
    b: float

    @classmethod
    def new(
        cls, 
        vec: Ref.Vector_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return norm_mid_vec("norm_mid_vec", vec=vec, a=a, b=b)
    bind = define.bind_from_new(new, Node.Vector_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(graph).last_before(event.t)
        l = int(len(v)/2)
        mid = (v[l-1] + v[l]) / 2
        res = v - mid
        return res



@dataclass(frozen=True)
class norm_mid_inner_vec( Node.Vector_F64):
    """
    note: assumes sorted ascending
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian_vec.new((3,)),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(norm_mid_inner_vec.new(r0))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = np.round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = np.round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    [ 0.13 -0.13  0.64] [ 0.5 -0.5  2.5]
    [ 1.3   0.95 -0.7 ] [ 0.5 -0.5  0.9]
    [-1.27 -0.62  0.04] [ 0.5  -0.5  -1.53]
    [-0.73 -0.54 -0.32] [ 0.5  -0.5  -0.01]
    [ 0.41  1.04 -0.13] [ 0.5  -0.5   1.36]
    """

    type: str
    #
    vec: Ref.Vector_F64
    a: float
    b: float

    @classmethod
    def new(
        cls, 
        vec: Ref.Vector_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return norm_mid_inner_vec("norm_mid_inner_vec", vec=vec, a=a, b=b)

    bind = define.bind_from_new(new, Node.Vector_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(graph).last_before(event.t)
        l = int(len(v)/2)
        mid = (v[l-1] + v[l]) / 2
        inner_range = v[l-1] - v[l]
        res = (v - mid) / inner_range
        return res
