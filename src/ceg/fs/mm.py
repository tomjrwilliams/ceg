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


class vs_x_vec_kw(NamedTuple):
    type: str
    #
    vs: tuple[Ref.D0_F64, ...]
    vec: Ref.D1_F64


class vs_x_vec(vs_x_vec_kw, Node.D0_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> from .arrays import vs_to_vec
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2,
    ... )
    >>> g, r1 = rand.gaussian.walk(
    ...     g, mean=-0.2,
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(vs_to_vec.new((r0, r1)), keep=1)
    ...     r3 = bind(vs_x_vec.new((r0, r1), r2), keep=1)
    ...     g = done()
    ...
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), Event.zero(r1), n=5, g=4, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v3 = round(
    ...         r3.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v3)
    -0.07 -0.33 0.12
    0.37 -0.43 0.32
    -0.37 -0.27 0.21
    0.73 0.48 0.77
    -0.17 -0.98 1.0
    """

    DEF: ClassVar[Defn] = define.node(Node.D0_F64, vs_x_vec_kw)

    @classmethod
    def new(
        cls,
        vs: tuple[Ref.D0_F64, ...],
        vec: Ref.D1_F64,
    ):
        return cls(cls.DEF.name, vs=vs, vec=vec)

    def __call__(self, event: Event, graph: Graph):
        vs = list(
            map(
                lambda v: v.history(graph).last_before(
                    event.t
                ),
                self.vs,
            )
        )
        vec = self.vec.history(graph).last_before(event.t)
        vs = np.array(vs)
        all_null = np.all(np.isnan(vs))
        if all_null:
            return np.NAN
        # TODO: have to be careful of zero fill misisng entries?
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
    >>> g = Graph.new()
    >>> from . import rand
    >>> from .arrays import vs_to_vec
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2,
    ... )
    >>> g, r1 = rand.gaussian.walk(
    ...     g, mean=-0.2,
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(vs_to_vec.new((r0, r1)), keep=1)
    ...     r3 = bind(v_x_vec_i.new(r0, r2, 0), keep=1)
    ...     g = done()
    ...
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), Event.zero(r1), n=5, g=4, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v3 = round(
    ...         r3.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v3)
    -0.07 0.01
    0.37 0.13
    -0.37 0.14
    0.73 0.54
    -0.17 0.03
    """

    DEF: ClassVar[Defn] = define.node(Node.D0_F64, v_x_vec_i_kw)

    @classmethod
    def new(
        cls,
        v: Ref.D0_F64,
        vec: Ref.D1_F64,
        i: int,
    ):
        return cls(cls.DEF.name, v=v, vec=vec, i=i)

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        vec = self.vec.history(graph).last_before(event.t)
        all_null = np.all(np.isnan(vec))
        if all_null:
            return np.NAN
        if v is None:
            return None
        elif np.isnan(v):
            return v
        return vec[self.i] * v
