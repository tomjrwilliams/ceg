from typing import NamedTuple, ClassVar, Type

from frozendict import frozendict

from .. import Ref

from ..types import *
from ..types import Sync

# TODO: separate guard types

import numpy

rng = numpy.random.default_rng(42069)

# TODO: put the rng behind a cache, with different variations for if you want a fresh one (with that seed) or the current acc state


class Wait(Sync):
    """ """

    pass


#  ------------------


class FixedKw(NamedTuple):
    step: float


class Fixed(FixedKw, Wait):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        # assert event.ref == ref, (self, node, ref, event)
        return event._replace(
            ref=ref, t=event.t + self.step
        )


class RandKw(NamedTuple):
    dist: str
    params: tuple[float, ...]


class Rand(RandKw, Wait):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        # assert event.ref == ref, (self, node, ref, event)
        if self.dist == "normal":
            step = rng.normal(*self.params, size=None)
        else:
            raise ValueError(self)
        return event._replace(ref=ref, t=event.t + step)


#  ------------------
