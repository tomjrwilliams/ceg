from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from .. import core
from . import stats

from .stats import window_mask, window_null_mask

#  ------------------

# TODO: vs_x_mat to do more than one factor decomp (below is strictly oen)

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
        if not core.series(graph, self.vec).t.n:
            return np.NAN
        vec = graph.select(self.vec, event, t=False, i = -1, null=False)
        vs = np.array(vs)
        all_null = np.all(np.isnan(vs))
        if all_null:
            return np.NAN
        return np.dot(vs, vec)

#  ------------------

class v_x_vec_i_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col
    vec: core.Ref.Col1D
    i: int


class v_x_vec_i(v_x_vec_i_kw, core.Node.Col):
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
        core.Node.Col, v_x_vec_i_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
        vec: core.Ref.Col1D,
        i: int,
    ):
        return cls(
            *cls.args(), v=v, vec=vec, i=i
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        v = graph.select(self.v, event, t= False, i = -1)
        if not core.series(graph, self.vec).t.n:
            return np.NAN
        vec = graph.select(self.vec, event, t=False, i = -1)
        all_null = np.all(np.isnan(vec))
        if all_null:
            return np.NAN
        if np.isnan(v):
            return np.NAN
        return vec[self.i] * v