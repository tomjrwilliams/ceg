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


class truncate_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    bound: Ref.Scalar_F64

    b: float

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(cls, v: Ref.Scalar_F64, bound: Ref.Scalar_F64, b: float = 1.):
        return truncate("truncate", v=v, bound=bound, b=b)

class truncate(truncate_kw, Node.Scalar_F64):
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
    ...     r2 = bind(truncate.new(r0, r1))
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

    DEF: ClassVar[Defn] = define.node(Node.Scalar_F64, truncate_kw)
    bind = define.bind_from_new(truncate_kw.new, truncate_kw.ref)


    def __call__(self, event: Event, graph: Graph):
        l = self.v.history(graph).last_before(event.t)
        r = self.bound.history(graph).last_before(event.t)
        if l is None or np.isnan(l) or r is None or np.isnan(r):
            return np.nan
        r = r * self.b
        if l < r and l > -r:
            return 0
        return l

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
            return np.nan
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
            return np.nan

        return ((l / r) - 1) * (
            1 if self.m is None else self.m
        )


#  ------------------


class subtract_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64

    a: float
    b: float

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(cls, l: Ref.Scalar_F64, r: Ref.Scalar_F64, a: float = 0., b: float = 1.):
        return subtract("subtract", l=l, r=r, a=a,b=b)



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
    bind = define.bind_from_new(subtract_kw.new, subtract_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        if l is None or np.isnan(l) or r is None or np.isnan(r):
            return np.nan
        return self.a + ((l - r) * self.b)


diff = subtract

class add_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64

    a: float
    b: float

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(cls, l: Ref.Scalar_F64, r: Ref.Scalar_F64, a: float = 0., b: float = 1.):
        return add("add", l=l, r=r, a=a,b=b)



class add(add_kw, Node.Scalar_F64):
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
    ...     r2 = bind(add.new(r0, r1))
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
    bind = define.bind_from_new(add_kw.new, add_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        if l is None or np.isnan(l) or r is None or np.isnan(r):
            return np.nan
        return self.a + ((l + r) * self.b)



class subtract_vec_i_kw(NamedTuple):
    type: str
    #
    l: Ref.Vector_F64
    r: Ref.Vector_F64

    il: int
    ir: int

    a: float
    b: float

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(cls, l: Ref.Vector_F64, r: Ref.Vector_F64, il: int, ir: int, a: float = 0., b: float = 1.):
        return subtract_vec_i("subtract_vec_i", l=l, r=r, il=il, ir=ir, a=a,b=b)



class subtract_vec_i(subtract_vec_i_kw, Node.Scalar_F64):
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
    bind = define.bind_from_new(subtract_vec_i_kw.new, subtract_vec_i_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        # if l is None or np.isnan(l) or r is None or np.isnan(r):
        #     return np.nan
        return self.a + ((l[self.il] - r[self.ir]) * self.b)


diff = subtract