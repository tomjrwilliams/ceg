from typing import NamedTuple, ClassVar, Type

from frozendict import frozendict

from .. import Ref
from ..types import *
from ..types import Sync

# TODO: separate guard types
import datetime as dt
import numpy

rng = numpy.random.default_rng(42069)


class Loop(Sync):
    """ """

    pass


#  ------------------


class FixedKw(NamedTuple):
    step: float


class Fixed(FixedKw, Loop):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        assert event.ref == ref, (self, node, ref, event)
        return event._replace(t=event.t + self.step)

class FixedUntilDateKw(NamedTuple):
    step: float
    until: dt.date


class FixedUntilDate(FixedUntilDateKw, Loop):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        assert event.ref == ref, (self, node, ref, event)
        vs = graph.select(ref, event, t = False)
        if len(vs) and vs[-1] == self.until:
                return None
        return event._replace(t=event.t + self.step)


class RandKw(NamedTuple):
    dist: str
    params: tuple[float, ...]


class Rand(RandKw, Loop):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        assert event.ref == ref, (self, node, ref, event)
        if self.dist == "normal":
            step = rng.normal(*self.params, size=None)
        else:
            raise ValueError(self)
        return event._replace(t=event.t + step)


#  ------------------
