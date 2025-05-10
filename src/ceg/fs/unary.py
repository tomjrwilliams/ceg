from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from .. import core

#  ------------------

# abs, log, pow, shift, scale, clip, tanh

# sq, sqrt, neg

# various activations (sigmoid, rev_sigmoid, relu, etc.)

class pct_change_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class pct_change(pct_change_kw, core.Node.Col):
    """
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(pct_change.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, pct_change_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col
    ):
        return cls(*cls.args(), v=v)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        v = graph.select(self.v, event, t=False)
        if not len(v):
            return numpy.NAN
        vlast = v[-1]
        if np.isnan(vlast):
            return vlast
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
    schedule: core.Schedule
    #
    v: core.Ref.Col
    w: int


class lag(lag_kw, core.Node.Col):
    """
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(sqrt.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, lag_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col, w: int
    ):
        return cls(*cls.args(), v=v, w=w)

    def __call__(
        self, event: core.Event, graph: core.Graph
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
    schedule: core.Schedule
    #
    v: core.Ref.Col


class sqrt(sqrt_kw, core.Node.Col):
    """
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(sqrt.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, sqrt_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col
    ):
        return cls(*cls.args(), v=v)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        v = graph.select(self.v, event, t=False)
        if not len(v):
            return numpy.NAN
        vlast = v[-1]
        if np.isnan(vlast):
            return vlast
        if vlast < 0:
            return -1 * (np.sqrt(-1 * vlast))
        return np.sqrt(vlast)
        
class sq_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class sq(sq_kw, core.Node.Col):
    """
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(sqrt.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, sq_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col
    ):
        return cls(*cls.args(), v=v)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        v = graph.select(self.v, event, t=False)
        if not len(v):
            return numpy.NAN
        vlast = v[-1]
        if np.isnan(vlast):
            return vlast
        if vlast < 0:
            return -1 * (np.square(-1 * vlast))
        return np.square(vlast)
        
class cum_sum_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class cum_sum(cum_sum_kw, core.Node.Col):
    """
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(sqrt.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, cum_sum_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col
    ):
        return cls(*cls.args(), v=v)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        v = graph.select(self.v, event, t=False)
        if not len(v):
            return numpy.NAN
        if np.isnan(v[-1]):
            return np.NAN
        return np.nansum(v)
        
class compound_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class compound(compound_kw, core.Node.Col):
    """
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     ch = bind(sqrt.new(r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(ch, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, compound_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col
    ):
        return cls(*cls.args(), v=v)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        v = graph.select(self.v, event, t=False)
        if not len(v):
            return numpy.NAN
        return np.nanprod(1 + v)