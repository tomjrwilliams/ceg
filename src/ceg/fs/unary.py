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
        v = self.v.history(graph).last_before(event.t)

        # TODO: hmmm. still have to select(window) and iter back?
        # unless can do a last_n (not none) quickly and efficiently?
        
        # unless we never actually bind the nan values? we skip them
        # and then plots are *always* resampled, effectively discretised?
        
        # so need eg. an index series of booleans on times
        # that we agg into, rolling

        # and then the plot can just be over that agg'd series?

        if np.isnan(v):
            return v
        r = 0
        vv = None
        for vv in v[::-1][1:]:
            if np.isnan(vv):
                continue
            r = (vlast / vv)-1
            break
        return r

class lag_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    w: int


class lag(lag_kw, Node.Scalar_F64):
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
        Node.Scalar_F64, lag_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64, w: int
    ):
        return cls(cls.DEF.name, v=v, w=w)

    def __call__(
        self, event: Event, graph: Graph
    ):
        v = graph.select(self.v, event, t=False)
        if not len(v):
            return numpy.NAN
        for vv in v[:-self.w][::-1]:
            if not np.isnan(vv):
                return vv
        return np.NAN
        
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
        if vlast < 0:
            return -1 * (np.square(-1 * vlast))
        return np.square(vlast)
        
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
        v = graph.select(self.v, event, t=False)
        if not len(v):
            return numpy.NAN
        if np.isnan(v[-1]):
            return np.NAN
        return np.nansum(v)
        
class compound_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class compound(compound_kw, Node.Scalar_F64):
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
        Node.Scalar_F64, compound_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64
    ):
        return cls(cls.DEF.name, v=v)

    def __call__(
        self, event: Event, graph: Graph
    ):
        v = graph.select(self.v, event, t=False)
        if not len(v):
            return numpy.NAN
        return np.nanprod(1 + v)