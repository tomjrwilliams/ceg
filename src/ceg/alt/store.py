from __future__ import annotations
from typing import Callable
import itertools

from dataclasses import dataclass

from .indices import (
    cartesian_enumeration,
    Key, 
    Tags, 
    Index,
)

# generic col store


# contains both a snap for the latest value
# so you can more easily work out if need to act


# headers mapping to ids in the store
# for equality check, if not match header with given, recalc

# and values
# including possibly some kind of indexing indicator


# storage col headers are contextual to app, but you can obviously share schemas across apps for global series

class Snap:
    # used to decide when work needs to be done, summary of value state (eg. start / end)
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
    index: Index

    def snap(self, key: Key) -> Snap | None:
        return

    def value(self, key: Key) -> Value | None:
        return


# for now, orthogonal to existing set up
# eg. just load / save / update value series


# id generation given string is straight forward: unique index on string, to auto incr id

# then use id instead

# then upsert id is get -> (insert if needed) -> return