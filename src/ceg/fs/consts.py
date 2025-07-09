
from typing import NamedTuple, ClassVar, cast

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
class const_float(Node.D0_F64):
    """
    >>> g = Graph.new()
    >>> g, r0 = g.pipe(const_float.zero_every, 1.)
    >>> e = Event.zero(r0)
    >>> for g, e, t in steps(g, e, n=3, iter=True)():
    ...     print(r0.history(g).last_before(t))
    0.0
    0.0
    0.0
    """

    type: str
    #
    v: float
    rf: Ref.Scalar_F64 | None

    @staticmethod
    def new(v: float, rf: Ref.Scalar_F64 | None = None):
        return const_float("const_float", v=v, rf=rf)

    @classmethod
    def one(cls, sign = 1):
        return cls.new(1. * sign)

    @classmethod
    def zero(cls):
        return cls.new(0.)

    @classmethod
    def one_every(
        cls,
        g: Graph,
        step=1.0,
        keep: int = 1,
        sign: float = 1.,
    ):
        return cls.bind(g, sign, step=step, keep=keep)

    @classmethod
    def zero_every(
        cls,
        g: Graph,
        step=1.0,
        keep: int = 1,
    ):
        return cls.bind(g, 0., step=step, keep=keep)

    @classmethod
    def bind(
        cls,
        g: Graph,
        v: float,
        step=1.0,
        keep: int = 1,
    ):
        g, r = g.bind(None, Ref.Scalar_F64)
        g, r = cls.new(
            v=v, rf=r.select(last=keep)
        ).pipe(g.bind, r, Loop.every(step), keep=keep)
        return g, cast(Ref.Scalar_F64, r)

    def __call__(self, event: Event, graph: Graph):
        return self.v

# const float, date etc.



# the easiest way to streamllit, is probably just to run the graph

# indicage which nodes to store the full history (aligned on dates)


# and then just pass the graph to the front end to pull the relevant series out, and do what it wants with?


# not try to wrap the plots into the graph run



# later can look at that, but work backwards from workig plot apps rather than trying to design up front