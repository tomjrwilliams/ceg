from typing import NamedTuple, ClassVar
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
)


#  ------------------



@dataclass(frozen=True)
class vs_x_vec(Node.D0_F64):
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

    type: str
    #
    vs: tuple[Ref.D0_F64, ...]
    vec: Ref.D1_F64

    @staticmethod
    def new(
        # cls,
        vs: tuple[Ref.D0_F64, ...],
        vec: Ref.D1_F64,
    ):
        return vs_x_vec("vs_x_vec", vs=vs, vec=vec)

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
            return np.nan
        # TODO: have to be careful of zero fill misisng entries?
        return np.dot(vs, vec)


#  ------------------



@dataclass(frozen=True)
class v_x_vec_i( Node.D0_F64):
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

    type: str
    #
    v: Ref.D0_F64
    vec: Ref.D1_F64
    i: int

    @staticmethod
    def new(
        # cls,
        v: Ref.D0_F64,
        vec: Ref.D1_F64,
        i: int,
    ):
        return v_x_vec_i("v_x_vec_i", v=v, vec=vec, i=i)

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        vec = self.vec.history(graph).last_before(event.t)
        all_null = np.all(np.isnan(vec))
        if all_null:
            return np.nan
        if v is None:
            return None
        elif np.isnan(v):
            return v
        return vec[self.i] * v

#  ------------------



@dataclass(frozen=True)
class vec_x_mat_i( Node.D0_F64):
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

    # TODO: tup in the name

    type: str
    #
    v: Ref.D1_F64
    vec: Ref.D1_F64_D2_F64
    f: int
    slot: int

    @staticmethod
    def new(
        # cls,
        v: Ref.D1_F64,
        vec: Ref.D1_F64_D2_F64,
        f: int,
        slot: int,
    ):
        return vec_x_mat_i("vec_x_mat_i", v=v, vec=vec, f=f,slot=slot)

    bind = define.bind_from_new(new, Node.D0_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        mat = self.vec.history(graph, slot=1).last_before(event.t)
        vec = mat[:, self.f]
        all_null = np.all(np.isnan(vec))
        if all_null:
            return np.nan
        if v is None:
            return None
        elif vec is None:
            return None
        res = np.dot(v, vec)
        return res
