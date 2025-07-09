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
class vs_to_vec( Node.D1_F64):
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

    type: str
    #
    vs: tuple[Ref.D0_F64, ...]

    @staticmethod
    def new(
        # cls,
        vs: tuple[Ref.D0_F64, ...],
    ):
        return vs_to_vec("vs_to_vec", vs=vs)

    def __call__(self, event: Event, graph: Graph):
        return np.array(list(
            map(
                lambda v: v.history(graph).last_before(
                    event.t
                ),
                self.vs,
            )
        ))



@dataclass(frozen=True)
class v_args_to_vec(Node.D1_F64):
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
    v10: Ref.Scalar_F64 | None = None
    v11: Ref.Scalar_F64 | None = None
    v12: Ref.Scalar_F64 | None = None
    v13: Ref.Scalar_F64 | None = None
    v14: Ref.Scalar_F64 | None = None
    v15: Ref.Scalar_F64 | None = None

    @staticmethod
    def new(
        # cls,
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
        v10: Ref.Scalar_F64 | None = None,
        v11: Ref.Scalar_F64 | None = None,
        v12: Ref.Scalar_F64 | None = None,
        v13: Ref.Scalar_F64 | None = None,
        v14: Ref.Scalar_F64 | None = None,
        v15: Ref.Scalar_F64 | None = None,
    ):
        return v_args_to_vec("v_args_to_vec", v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
    bind = define.bind_from_new(new, Node.D1_F64.ref)

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
            self.v10, self.v11, self.v12, self.v13, self.v14, self.v15
        ]
        vs = [v for v in vs if v is not None]
        vals = list(
            map(
                lambda v: v.history(graph).last_before(
                    event.t, strict=False
                ),
                vs,
            )
        )
        if None in vals:
            return None
        return np.array(vals)

#  ------------------

# etc. stack vecs, index out entries

#  ------------------



@dataclass(frozen=True)
class mat_tup_to_v(Node.D1_F64):

    type: str
    vec: Ref.D1_F64_D2_F64
    i0:int
    i1: int
    slot: int

    @staticmethod
    def new(
        # cls,
        vec: Ref.D1_F64_D2_F64,
        i0: int,
        i1: int,
        slot: int=1,
    ):
        return mat_tup_to_v("mat_tup_to_v", vec, i0, i1, slot)
    bind = define.bind_from_new(new, Node.D1_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(
            graph,
            slot=1
        ).last_before(event.t)
        if v is None:
            return None
        return v[self.i0][self.i1]