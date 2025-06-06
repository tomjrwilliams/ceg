
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


class const_float_kw(NamedTuple):
    type: str
    #
    v: float


class const_float(const_float_kw, Node.D0_F64):
    """
    >>> g = Graph.new()
    >>> g, r0 = const_float(1.)
    >>> for g, e, t in steps(g, Event.zero(r0), n=5):
    ...     print(r0.history(g).last_before(t))
    -0.07 0.01
    0.37 0.13
    -0.37 0.14
    0.73 0.54
    -0.17 0.03
    """

    DEF: ClassVar[Defn] = define(Node.D0_F64, const_float_kw)

    @classmethod
    def new(cls, v: float):
        return cls(cls.DEF.name, v=v)

    def __call__(self, event: Event, graph: Graph):
        return self.v

# const float, date etc.



# the easiest way to streamllit, is probably just to run the graph

# indicage which nodes to store the full history (aligned on dates)


# and then just pass the graph to the front end to pull the relevant series out, and do what it wants with?


# not try to wrap the plots into the graph run



# later can look at that, but work backwards from workig plot apps rather than trying to design up front