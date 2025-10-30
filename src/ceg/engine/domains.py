
from typing import Callable

from .objects import Event, Object
from .subs import Subscription

# --------------------------------------

TBC = None

class Domain:

    objects: dict[str, Object]
    
    # lhs listen for events on rhs
    subscribes: dict[Object, list[Object]]

    # TODO: load the last (fully) handled event (so we dont have to re-iterate beyond that)

    def with_object(self, obj: Object):
        return

    # TODO: with_objects 

    def with_sub(self, sub: Subscription):

        # object here might use star in the path?
        # eg. to say all under this prefix sub to this?

        # star on both?
        # ie. sub lhs / rhs are generalisations of actual objects to include wildcards

        return

    def on_sync(self, f: Callable):
        # eg. used to check external systems for events to fire
        # eg. if upstream dependent data has updated
        return

    def sync(self):

        # check for updates
        # fire events


        # whether an event needs handling by a sub is determined by the snap of the boject: it should have an event id 

        # so new objects automatically traverse the full history

        
        # but the snap can also check more eg. last_date of the series to see if existing objects do need updates dpeending on what upstream changed

        return


# ------------------------------------------------------








# TODO: need populate hooks for the histories
# and write callbacks

# make as generic functions presumably

# then juts need to seed the event at the right point
# or to no-op through the first already-loaded data

# where this includes both raw data obviously but also synthetics weve saved down


# the object contains the calc, any (env and calc dependent) storage

# and so the data be relatively thin apis for just getting the data, that the obj can then persist in the relevant kv store?


# maybe have all the dailies inherit from a day counter
# but the calcs dont need them: they point to either, implicitly daily or not 



# NOTE:

# to temporarily 'overwrite'
# you just have to assign a new app local key, around the relevant code
# ie. you force a cache miss, but don't change the underlying data

# then every so often, when you're ready to deploy to prod
# you can bump the version instead of aliasing in the path

# and then you just need to clear the old version (s)