# sum
# prod

# weighted thereof

# all, any

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

class mean_vec_kw(NamedTuple):
    type: str
    #
    vec: Ref.Vector_F64
    a: float
    b: float

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(
        cls, 
        vec: Ref.Vector_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return sum_vec("sum_vec", vec=vec, a=a, b=b)

class mean_vec(mean_vec_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian_vec.new(),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(sum_vec.new(r0))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = np.round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    0.13 -0.13 -0.95
    0.64 0.1 6.11
    -0.54 0.36 -1.48
    1.3 0.95 1.38
    -0.7 -1.27 0.56
    """

    DEF: ClassVar[Defn] = define.node(Node.Scalar_F64, mean_vec_kw)
    bind = define.bind_from_new(mean_vec_kw.new, mean_vec_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(graph).last_before(event.t)
        return np.mean(self.b * v) + self.a

#  ------------------

class sum_vec_kw(NamedTuple):
    type: str
    #
    vec: Ref.Vector_F64
    a: float
    b: float

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(
        cls, 
        vec: Ref.Vector_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return sum_vec("sum_vec", vec=vec, a=a, b=b)

class sum_vec(sum_vec_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian_vec.new(),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(sum_vec.new(r0))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = np.round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    0.13 -0.13 -0.95
    0.64 0.1 6.11
    -0.54 0.36 -1.48
    1.3 0.95 1.38
    -0.7 -1.27 0.56
    """

    DEF: ClassVar[Defn] = define.node(Node.Scalar_F64, sum_vec_kw)
    bind = define.bind_from_new(sum_vec_kw.new, sum_vec_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(graph).last_before(event.t)
        return np.sum(self.b * v) + self.a

#  ------------------

class sum_mat_i_kw(NamedTuple):
    type: str
    #
    mat: Ref.D1_F64_D2_F64
    slot: int
    i: int
    a: float
    b: float
    t: bool

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(
        cls, 
        mat: Ref.D1_F64_D2_F64,
        slot: int,
        i: int,
        t: bool=False,
        a: float = 0.,
        b: float = 1.,
    ):
        return sum_mat_i("sum_mat_i", mat=mat,slot=slot,i=i,t=t, a=a, b=b)

class sum_mat_i(sum_mat_i_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define.node(Node.Scalar_F64, sum_mat_i_kw)
    bind = define.bind_from_new(sum_mat_i_kw.new, sum_mat_i_kw.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.mat.history(
            graph, slot=1 # TODO fix
        ).last_before(event.t)
        if self.t:
            v = v[:, self.i]
        else:
            v = v[self.i]
        return np.sum(self.b * v) + self.a