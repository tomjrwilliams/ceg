from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from ..core import Graph, Node, Ref, Event, Loop, Defn, define, steps

from . import stats
from .stats import window_mask, window_null_mask

#  ------------------

# TODO: vs_x_mat to do more than one factor decomp (below is strictly oen)

class vs_x_vec_kw(NamedTuple):
    type: str
    #
    vs: tuple[Ref.D0_F64, ...]
    vec: Ref.D1_F64


class vs_x_vec(vs_x_vec_kw, Node.D0_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.D0_F64
    window: float | None
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(mean.new(r))
    ...     mu_3 = bind(mean.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.D0_F64, vs_x_vec_kw
    )

    @classmethod
    def new(
        cls,
        vs: tuple[Ref.D0_F64, ...],
        vec: Ref.D1_F64,
    ):
        return cls(
            cls.DEF.name, vs=vs, vec=vec
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        vs = list(map(
            lambda v: v.history(graph).last_before(event.t),
            self.vs,
        ))
        if not series(graph, self.vec).t.n:
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
    #
    v: Ref.D0_F64
    vec: Ref.D1_F64
    i: int


class v_x_vec_i(v_x_vec_i_kw, Node.D0_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.D0_F64
    window: float | None
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(mean.new(r))
    ...     mu_3 = bind(mean.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.D0_F64, v_x_vec_i_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.D0_F64,
        vec: Ref.D1_F64,
        i: int,
    ):
        return cls(
            cls.DEF.name, v=v, vec=vec, i=i
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        v = self.v.history(graph).last_before(event.t)
        if not series(graph, self.vec).t.n:
            return np.NAN
        vec = graph.select(self.vec, event, t=False, i = -1)
        all_null = np.all(np.isnan(vec))
        if all_null:
            return np.NAN
        if np.isnan(v):
            return np.NAN
        return vec[self.i] * v