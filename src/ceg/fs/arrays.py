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


class vs_to_vec_kw(NamedTuple):
    type: str
    #
    vs: tuple[Ref.D0_F64, ...]

class vs_to_vec(vs_to_vec_kw, Node.D1_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2,
    ... )
    >>> g, r1 = rand.gaussian.walk(
    ...     g, mean=-0.2,
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(vs_to_vec.new((r0, r1)), keep=1)
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0),Event.zero(r1), n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v2 = np.round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    -0.07 -0.33 [-0.07 -0.33]
    0.37 -0.43 [ 0.37 -0.43]
    -0.37 -0.27 [-0.37 -0.27]
    0.73 0.48 [0.73 0.48]
    -0.17 -0.98 [-0.17 -0.98]
    """

    DEF: ClassVar[Defn] = define.node(Node.D1_F64, vs_to_vec_kw)

    @classmethod
    def new(
        cls,
        vs: tuple[Ref.D0_F64, ...],
    ):
        return cls(cls.DEF.name, vs=vs)

    def __call__(self, event: Event, graph: Graph):
        return np.array(list(
            map(
                lambda v: v.history(graph).last_before(
                    event.t
                ),
                self.vs,
            )
        ))

class v_args_to_vec_kw(NamedTuple):
    type: str
    #
    v0: Ref.Scalar_F64
    v1: Ref.Scalar_F64 | None = None
    v2: Ref.Scalar_F64 | None = None
    v3: Ref.Scalar_F64 | None = None
    v4: Ref.Scalar_F64 | None = None
    v5: Ref.Scalar_F64 | None = None
    v6: Ref.Scalar_F64 | None = None
    v7: Ref.Scalar_F64 | None = None
    v8: Ref.Scalar_F64 | None = None
    v9: Ref.Scalar_F64 | None = None

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Vector_F64:
        return Ref.d1_f64(i, slot=slot)

    @classmethod
    def new(
        cls,
        v0: Ref.Scalar_F64,
        v1: Ref.Scalar_F64 | None = None,
        v2: Ref.Scalar_F64 | None = None,
        v3: Ref.Scalar_F64 | None = None,
        v4: Ref.Scalar_F64 | None = None,
        v5: Ref.Scalar_F64 | None = None,
        v6: Ref.Scalar_F64 | None = None,
        v7: Ref.Scalar_F64 | None = None,
        v8: Ref.Scalar_F64 | None = None,
        v9: Ref.Scalar_F64 | None = None,
    ):
        return v_args_to_vec("v_args_to_vec", v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)

class v_args_to_vec(v_args_to_vec_kw, Node.D1_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2,
    ... )
    >>> g, r1 = rand.gaussian.walk(
    ...     g, mean=-0.2,
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(vs_to_vec.new((r0, r1)), keep=1)
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0),Event.zero(r1), n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v2 = np.round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    -0.07 -0.33 [-0.07 -0.33]
    0.37 -0.43 [ 0.37 -0.43]
    -0.37 -0.27 [-0.37 -0.27]
    0.73 0.48 [0.73 0.48]
    -0.17 -0.98 [-0.17 -0.98]
    """

    DEF: ClassVar[Defn] = define.node(Node.D1_F64, v_args_to_vec_kw)
    bind = define.bind_from_new(v_args_to_vec_kw.new, v_args_to_vec_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        vs = [
            self.v0,
            self.v1,
            self.v2,
            self.v3,
            self.v4,
            self.v5,
            self.v6,
            self.v7,
            self.v8,
            self.v9,
        ]
        vs = [v for v in vs if v is not None]
        return np.array(list(
            map(
                lambda v: v.history(graph).last_before(
                    event.t
                ),
                vs,
            )
        ))

#  ------------------

# etc. stack vecs, index out entries

#  ------------------