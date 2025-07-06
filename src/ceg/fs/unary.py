import logging
from typing import NamedTuple, ClassVar, cast
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
    Ready,
)

logger = logging.Logger(__file__)
#  ------------------

# abs, log, pow, shift, scale, clip, tanh

# sq, sqrt, neg

# various activations (sigmoid, rev_sigmoid, relu, etc.)



@dataclass(frozen=True)
class abs(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(g, keep=2)
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(abs.new(r0))
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    0.13 0.13
    -0.01 0.01
    0.63 0.63
    0.74 0.74
    0.2 0.2
    """

    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(cls, v: Ref.Scalar_F64):
        return abs("abs", v=v)

    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    # @classmethod
    # def bind(cls, g: Graph, v: Ref.Scalar_F64, keep: int = 4):
    #     n = cls.new(v.select(keep))
    #     return g.bind(n, when=Ready.ref_not_nan(v))

    def __call__(self, event: Event, graph: Graph):

        hist = self.v.history(graph)

        v0 = hist.last_before(event.t)
        if v0 is None:
            return v0

        return v0 * np.sign(v0)

@dataclass(frozen=True)
class abs_change(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(g)
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(abs_change.new(r0))
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    0.13 nan
    -0.01 -0.13
    0.63 0.64
    0.74 0.1
    0.2 -0.54
    """
    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(cls, v: Ref.Scalar_F64):
        return abs_change("abs_change", v=v)
    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    # @classmethod
    # def bind(cls, g: Graph, v: Ref.Scalar_F64, keep: int = 4):
    #     n = cls.new(v.select(keep))
    #     return g.bind(n, when=Ready.ref_not_nan(v))

    def __call__(self, event: Event, graph: Graph):
        if event.prev is None:
            return np.nan  # or 0?

        hist = self.v.history(graph)

        v0 = hist.last_before(event.t)
        v1 = hist.last_before(event.prev.t, allow_nan=False)
        
        if v0 is None or v1 is None or np.isnan(v0) or np.isnan(v1):
            return np.nan
            
        return v0 - v1


@dataclass(frozen=True)
class pct_change(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=2
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(pct_change.new(r0))
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    1.13 nan
    1.99 0.77
    3.63 0.82
    4.74 0.3
    5.2 0.1
    """
    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(cls, v: Ref.Scalar_F64):
        return pct_change("pct_change", v=v)
    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    # @classmethod
    # def bind(cls, g: Graph, v: Ref.Scalar_F64, keep: int = 4):
    #     n = cls.new(v.select(keep))
    #     return g.bind(n, when=Ready.ref_not_nan(v))

    def __call__(self, event: Event, graph: Graph):
        if event.prev is None:
            return np.nan  # or 0?

        hist = self.v.history(graph)

        v0 = hist.last_before(event.t)
        v1 = hist.last_before(event.prev.t, allow_nan=False)
        
        if v0 is None or v1 is None or np.isnan(v0) or np.isnan(v1):
            return np.nan
        return (v0 / v1) - 1




@dataclass(frozen=True)
class sqrt(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(g)
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(sqrt.new(r0))
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    0.13 0.35
    -0.01 -0.08
    0.63 0.8
    0.74 0.86
    0.2 0.45
    """

    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(cls, v: Ref.Scalar_F64):
        return cls("sqrt", v=v)

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        if v is None or np.isnan(v):
            return v
        if v < 0:
            return -1 * (np.sqrt(-1 * v))
        return np.sqrt(v)


@dataclass(frozen=True)
class sq(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(g)
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(sq.new(r0))
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    0.13 0.02
    -0.01 -0.0
    0.63 0.4
    0.74 0.55
    0.2 0.04
    """
    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(cls, v: Ref.Scalar_F64):
        return cls("sq", v=v)

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        if v is None or np.isnan(v):
            return v
        if v < 0:
            return -1 * (np.square(-1 * v))
        return np.square(v)



@dataclass(frozen=True)
class cum_sum(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ...     keep=2,
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(cum_sum.new(r0), keep=2)
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    0.13 0.13
    -0.13 -0.01
    0.64 0.63
    0.1 0.74
    -0.54 0.2
    """
    type: str
    #
    v: Ref.Scalar_F64
    @classmethod
    def new(cls, v: Ref.Scalar_F64):
        return cum_sum("cum_sum", v=v.select(4))

    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        hist = self.v.history(graph)
        v = hist.last_before(event.t)
        if event.prev is None:
            return v
        acc = cast(Ref.Scalar_F64, event.ref)
        prev = acc.history(graph).last_before(event.prev.t, allow_nan=False)
        if v is None or np.isnan(v):
            return np.nan
        if prev is None or np.isnan(prev):
            return v
        return prev + v



@dataclass(frozen=True)
class cum_prod(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian.new(),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(
    ...         cum_prod.new(r0, a=1.0), keep=2
    ...     )
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    0.13 1.13
    -0.13 0.98
    0.64 1.6
    0.1 1.77
    -0.54 0.82
    """
    type: str
    #
    v: Ref.Scalar_F64
    a: float

    @classmethod
    def new(cls, v: Ref.Scalar_F64, a: float = 0.0):
        # NOTE: eg. a = 1 to compound pct rx
        return cls("cum_prod", v=v, a=a)

    def __call__(self, event: Event, graph: Graph):
        hist = self.v.history(graph)
        v = hist.last_before(event.t)
        if v is None:
            return v
        if event.prev is None:
            return self.a + v
        acc = cast(Ref.Scalar_F64, event.ref)
        prev = acc.history(graph).last_before(event.prev.t)
        if prev is None or np.isnan(prev):
            return self.a + v
        elif v is None or np.isnan(v):
            return prev
        return prev * (self.a + v)
