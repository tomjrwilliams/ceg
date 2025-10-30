from __future__ import annotations
from typing import Callable
import itertools

from dataclasses import dataclass

from .objects import (
    Object,
    Tags,
    # Index,
)

# just object state, doesnt track which global events have been handled yet?

# so no guarantee of object state relative to global event ordering?



class Snap:
    # used to decide when work needs to be done, summary of value state (eg. start / end)

    # minimum: event id (ordered) and queue identifier (for that event?)

    # ie. this is a subscription: it stores the latest processed (and other relevant data about the value series for deciding if work needs doing), given a queue

    # but then we can only target one queue?

    # so otherwise we need a mapping to each of the queues we subscribe to in the values? that's in theory fine as well
    pass


class Value:
    # this is the col value, for time series often presumably chunked
    pass


class Signature:
    # describes what kind of values it holds: eg. time series, daily
    # 1d float
    # no null etc.

    # each node implementation should be able to calc its own signature given those of input nodes, and its args
    # every stored needs a signature, every dynamic can then calc (and hence has as well)
    pass


@dataclass(frozen=True)
class Store:
    # index: Index

    def snap(self, obj: Object) -> Snap | None:
        return

    def value(self, obj: Object) -> Value | None:
        return


# for now, orthogonal to existing set up
# eg. just load / save / update value series


# id generation given string is straight forward: unique index on string, to auto incr id

# then use id instead

# then upsert id is get -> (insert if needed) -> return
