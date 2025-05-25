from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from ..core import Graph, Node, Ref, Event, Loop, Defn, define, steps

#  ------------------

# abs, log, pow, shift, scale, clip, tanh

# sq, sqrt, neg

# various activations (sigmoid, rev_sigmoid, relu, etc.)

class pct_change_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class pct_change(pct_change_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(pct_change.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, pct_change_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64
    ):
        return cls(cls.DEF.name, v=v)

    def __call__(
        self, event: Event, graph: Graph
    ):
        if event.prev is None:
            return np.NAN # or 0?
        hist = self.v.history(graph)
        v0 = hist.last_before(event.t)
        v1 = hist.last_before(event.prev.t)
        if np.isnan(v0) or np.isnan(v1):
            return np.NAN
        return (v0 / v1)-1
        
class sqrt_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class sqrt(sqrt_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(sqrt.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, sqrt_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64
    ):
        return cls(cls.DEF.name, v=v)

    def __call__(
        self, event: Event, graph: Graph
    ):
        v = self.v.history(graph).last_before(event.t)
        if np.isnan(v):
            return v
        if v < 0:
            return -1 * (np.sqrt(-1 * v))
        return np.sqrt(v)
        
class sq_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class sq(sq_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(sq.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, sq_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64
    ):
        return cls(cls.DEF.name, v=v)

    def __call__(
        self, event: Event, graph: Graph
    ):
        v = self.v.history(graph).last_before(event.t)
        if np.isnan(v):
            return v
        if v < 0:
            return -1 * (np.square(-1 * v))
        return np.square(v)
        
class cum_sum_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class cum_sum(cum_sum_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(cum_sum.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, cum_sum_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64
    ):
        return cls(cls.DEF.name, v=v)

    def __call__(
        self, event: Event, graph: Graph
    ):
        hist = self.v.history(graph)
        v = hist.last_before(event.t)
        if event.prev is None:
            return v
        prev = hist.last_before(event.prev.t)
        if np.isnan(prev):
            return v
        elif np.isnan(v):
            return prev
        return prev + v
        
class cum_prod_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    a: float


class cum_prod(cum_prod_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(cum_prod.new(r, a = 1))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, cum_prod_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64, a: float = 0.
    ):
        # NOTE: eg. a = 1 to compound pct rx
        return cls(cls.DEF.name, v=v, a=a)

    def __call__(
        self, event: Event, graph: Graph
    ):
        hist = self.v.history(graph)
        v = hist.last_before(event.t)
        if event.prev is None:
            return self.a + v
        prev = hist.last_before(event.prev.t)
        if np.isnan(prev):
            return self.a + v
        elif np.isnan(v):
            return prev
        return prev * (self.a + v)