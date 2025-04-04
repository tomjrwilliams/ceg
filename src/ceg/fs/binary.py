# add
# div
# sub
# mul
# pow, log

from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from .. import core

#  ------------------

class ratio_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    l: core.Ref.Col
    r: core.Ref.Col


class ratio(ratio_kw, core.Node.Col):
    """
    scalar mean (optional rolling window)
    v: core.Ref.Col
    window: float | None
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     rat = bind(ratio.new(r, r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, ratio_kw
    )

    @classmethod
    def new(
        cls, l: core.Ref.Col, r: core.Ref.Col
    ):
        return cls(*cls.args(), l=l, r=r)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        l = graph.select(self.l, event, t=False)[-1]
        r = graph.select(self.r, event, t= False)[-1]
        if l == np.NAN or r == np.NAN:
            return np.NAN
        return l / r
