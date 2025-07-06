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
    dataclass,
    define,
    steps,
    batches,
)
#  ------------------

@dataclass(frozen=True)
class mean_vec(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian_vec.new((3,)),
    ...     when=Loop.every(1),
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(mean_vec.new(r0))
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
    [ 0.13 -0.13  0.64] 0.21
    [ 1.3   0.95 -0.7 ] -0.02
    [-1.27 -0.62  0.04] -0.62
    [-0.73 -0.54 -0.32] -1.26
    [ 0.41  1.04 -0.13] 0.44
    """

    type: str
    #
    vec: Ref.Vector_F64
    a: float
    b: float

    @classmethod
    def new(
        cls, 
        vec: Ref.Vector_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return mean_vec("mean_vec", vec=vec, a=a, b=b)

    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(graph).last_before(event.t)
        return np.mean(self.b * v) + self.a

#  ------------------


@dataclass(frozen=True)
class sum_vec(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = g.bind(
    ...     rand.gaussian_vec.new((3,)),
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
    [ 0.13 -0.13  0.64] 0.63
    [ 1.3   0.95 -0.7 ] -0.07
    [-1.27 -0.62  0.04] -1.85
    [-0.73 -0.54 -0.32] -3.79
    [ 0.41  1.04 -0.13] 1.33
    """

    type: str
    #
    vec: Ref.Vector_F64
    a: float
    b: float

    @classmethod
    def new(
        cls, 
        vec: Ref.Vector_F64,
        a: float = 0.,
        b: float = 1.,
    ):
        return sum_vec("sum_vec", vec=vec, a=a, b=b)
    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.vec.history(graph).last_before(event.t)
        return np.sum(self.b * v) + self.a

#  ------------------

class sum_mat_i_kw(NamedTuple):

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)


@dataclass(frozen=True)
class sum_mat_i(Node.Scalar_F64):

    type: str
    #
    mat: Ref.D1_F64_D2_F64
    slot: int
    i: int
    a: float
    b: float
    t: bool

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
    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.mat.history(
            graph, slot=1 # TODO fix
        ).last_before(event.t)
        if self.t:
            v = v[:, self.i]
        else:
            v = v[self.i]
        return np.sum(self.b * v) + self.a