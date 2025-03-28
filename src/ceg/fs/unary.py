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
        if vlast == numpy.NAN:
            return vlast
        for vv in v[:-1][::-1]:
            if vv == numpy.NAN:
                continue
            r = (vlast / vv)-1
            return r
        return 0
        
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
        if vlast == numpy.NAN:
            return vlast
        if vlast < 0:
            return -1 * (np.sqrt(-1 * vlast))
        return np.sqrt(vlast)
        