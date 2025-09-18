from __future__ import annotations
from typing import Callable
import itertools

from dataclasses import dataclass

from .keyval import KeyVal

def cartesian_enumeration(
    filters: list[Callable[..., bool]] = [],
    **iterables: list,
):
    ks = list(iterables.keys())
    prods = itertools.product(*iterables.values())
    for filt in filters:
        prods = filter(filt, prods)
    for i, prod in enumerate(prods):
        yield i, dict(zip(ks, prod))

@dataclass(frozen=True)
class Tags:
    data: dict

@dataclass(frozen=True)
class Key:
    s: str
    
    def segments(self):
        segs = self.s.split(".")
        s = segs[0]
        yield Key(s)
        for ss in segs[1:]:
            s += f".{ss}"
            yield Key(s)

class KeyTags(KeyVal[Key, str, Tags, str]):
    pass


# users then mix in keytags with their backend (eg. sqlite) key val class at call time


@dataclass(frozen=True)
class Index:
    tags: KeyTags # user mixin with backend (sqlite) at call time

    # TODO: some kind of value mask?
    # ie. so that you can force a certain path (and descendants)
    # to always eval to no-match (ie. not be cached)
    # eg. if in dev

    def has(self, key: Key, index: Index) -> bool:
        for segment in key.segments():
            if not self.get(segment) == index.get(segment):
                return False
        return True

    def get(self, key: Key) -> Tags | None:
        return self.tags.get(key, None)

    def set(self, key: Key, tags: Tags) -> Index:
        # TODO: timed expiry, etc.
        return self


# TODO: load index for existing store
# build index for current app

# compare keys in app with store
# if match store, can pull
# if not, have to calculate, and update value


# TODO: the question is then how you back store the index element that changed?

# i think you probably just re-calc, re-store (overwrite), even if that means temporary downstream inconsistency?

# until all the downstream trigger events have flushed through?


# and then the assumption is that prod is constant, until you decide to merge in a dev branch with new definitions (that trigger downstream rewrites)


# q is if you publish an event for the delta, or if you just recalc?
# or if the recalc is what fires the event?


# if pull, can still check snap for if any update required
# if required (include if not exist), update and store


# in calculate / update case, up to you to fire an event to indicate this has been done
# for relevant listeners


# TODO: the storage can be orthogonal to the index
# ie. can have many stores, thats up to you to map based on key-value

# ie. above need some kind of partition handler (on key?), and thus many tags, dict on partition?

# TODO: and likewise on stores?


# to temporarily 'overwrite'
# you just have to assign a new app local key, around the relevant code
# ie. you force a cache miss, but don't change the underlying data

# then every so often, when you're ready to deploy to prod
# you can bump the version instead of aliasing in the path

# and then you just need to clear the old version (s)




# arbitrary key value mapping, eg. for universe meta data
def load_index():
    return


# eg. col headers
def save_index():
    return


def with_header():
    return

# replace certain prefixes with others, eg. for app local column storage (include an app in the col series)
def with_prefix():
    return