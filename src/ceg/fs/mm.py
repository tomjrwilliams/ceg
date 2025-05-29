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
)

#  ------------------

# TODO: vs_x_mat to do more than one factor decomp (below is strictly oen)


class vs_x_vec_kw(NamedTuple):
    type: str
    #
    vs: tuple[Ref.D0_F64, ...]
    vec: Ref.D1_F64


class vs_x_vec(vs_x_vec_kw, Node.D0_F64):

    DEF: ClassVar[Defn] = define(Node.D0_F64, vs_x_vec_kw)

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
        return np.dot(vs, vec)


#  ------------------


class v_x_vec_i_kw(NamedTuple):
    type: str
    #
    v: Ref.D0_F64
    vec: Ref.D1_F64
    i: int


class v_x_vec_i(v_x_vec_i_kw, Node.D0_F64):

    DEF: ClassVar[Defn] = define(Node.D0_F64, v_x_vec_i_kw)

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
        if np.isnan(v):
            return np.NAN
        return vec[self.i] * v
