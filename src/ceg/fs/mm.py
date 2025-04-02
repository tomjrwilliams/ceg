from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from .. import core
from . import stats

from .stats import window_mask, window_null_mask

#  ------------------

class vs_x_vec_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    vs: tuple[core.Ref.Col, ...]
    vec: core.Ref.Col1D


class vs_x_vec(vs_x_vec_kw, core.Node.Col):
    """
    scalar mean (optional rolling window)
    v: core.Ref.Col
    window: float | None
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(mean.new(r))
    ...     mu_3 = bind(mean.new(r, window=3))
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
        core.Node.Col, vs_x_vec_kw
    )

    @classmethod
    def new(
        cls,
        vs: tuple[core.Ref.Col, ...],
        vec: core.Ref.Col1D,
    ):
        return cls(
            *cls.args(), vs=vs, vec=vec
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        vs = list(map(
            lambda v: graph.select(v, event, t=False)[-1],
            self.vs,
        ))
        vec = graph.select(self.vec, event, t=False)[-1]
        vs = np.array(vs)
        return np.dot(vs, vec)