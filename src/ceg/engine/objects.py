from __future__ import annotations
from typing import Callable
import itertools

from dataclasses import dataclass

from ..apis import kv



class Event:
    # type: define, change, create, update
    object: Object



class Queue:

    # one per object, and then others by string

    # object paths for their own define/change/update/create

    # and then any others as per broadcast

    pass



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
    # sublcass for the relevant object tags as fields
    # easy to go to and from the dict repr with pydnatic 


    # TODO: have some way of indicating which should be collapsed (overriding downward, eg. dev env variables lower down)

    # and which should be kept mapped to their [i, s] pair
    # when consolidated

    # with a lookup method on name, optionally pass i (path index) or first / last etc.

    pass


@dataclass(frozen=True)
class Object:
    parent: Object | None
    s: str

    path: str

    # TODO: tags as fields

    # TODO: snap, values, etc. config references
    # are probably tags of parent objects, so they can be shared

    # ie. the root object is essentially an environment variable registry?

    def broadcasts(self):
        # boradcasts *to*
        # return a queue (s?)
        # eg. by traversing upwards until we find
        # probably local to the domain?

        # TODO: ideally the broadcasts and specific field values are filled in by the app logic, config (eg. from static data etc.), possibly env dependent

        # TODO: not necessarily just a string, ie. can return subs that truncate the upward path, reinstate, etc.
        # as we inherit (merging) from parents
        # so cache this presumably after that merge op

        # the upward path above being for definition change, not for create / update (as thats a value question not an object question)
        return

    @classmethod
    def new(cls, parent: Object | None, s: str):
        # fill in path str from parent.path

        # parent eg. could be module, root registry, etc.
        return

    def decode(self):
        # used when loading from kv store

        # include cast parent, call decode (recursively)
        return

    # convenience method to return
    def tags(self):
        return

    def on_define(self, e: Event, obj: Object):
        # if we have the depencies, run calc?
        return

    def on_change(self, e: Event, obj: Object):
        # of any node in the path - though we ideally dont need to actually bind all the intermediate
        # but probably do have some caching so we dont repeat checks for those with shared parents

        # either run calc or err, obj dependent
        return

    def on_create(self, e: Event, obj: Object):
        # this is presumably triggered by another object create, not this
        return

    def on_update(self, e: Event, obj: Object):
        # this is presumably triggered by another object update, not this
        # presume update calc
        return

    def on_event(self, e: Event):
        return

    def sync(self):
        # include snap guard that exits if not required?

        # do the calc, store snap, etc.

        # using the given index, snap, values
        # ie objects contain their own back-end references?

        # TODO: those then presumably are part of the tags?
        # to force re-calc on change

        return


class ObjectTags(kv.KeyValIO[kv.ID, str, str, Object, Tags]):
    pass


