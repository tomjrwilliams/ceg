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


# and listeners


# these are completely orthogonal to the storage element

# on recalc, we can fire events, but the logic to do so sits at teh event layer, not the storage layer

class Event:
    pass

class Log:
    index: Index

# persist somewhere (possibly namespaced? with the same name spacing as the indices?)
def publish(self, event: Event):
    return

# arguably if you persist event log (but prune) can then just loop through looking for stuff to do
# use the snap to decide if need to update or not
def listen(self):
    return


# TODO: take target (dot sep index string), datetime, json blob of context

# registr listeners on a broker

# scan through on each new event the listeners to see if match

# target based on prefix (?) with optional json map

# and then what they do depends on the actual event json (eg. prefix match on the data source, and then what they do depends on the ticker in the json)






# TODO: need a write time, json blob

# channels? or a specific channel?

# how do we subscribe - just on the json?

# or i guess it's related to the index nodes and the store surely?


# TODO: then last questions are:

# - what to do when an index definition changes, eg. an enumerated universe, the item for that number has changed?
# have to invalidate all downstream?

# ie. we overwrite, and trigger downstrem overwrites?




# the listen relationship then sits at app level
# as well as any local aliasing





# should the even thave the snap as well?
# or i guess just a json blob and up to you to parse, so yes it can

# and then each separate consumer, identified by some kind ofname

# /that itselfcould even be namedspaced? index style
# as a wya of keeping track of the handlers as well


# so eg. register a new handler, trawls history for stuff to do

# then every now and then, if you can rebuild from scratch not incrementally
# clean the event log, even the store? arguably store should be fine

# the event log you want for adding new downstream dependencies in generation
# but then can be cleaned and one shot (?) - but then it needs to be able to understand the one shot (that it must repeat?)

# eg. so no-op still implies checking all downstream ?
# and then its ultimately up to the snap, the values code, to actualyl decide to calc or not



# TODO: so just need to be able to design an ordered persistent event log data structure?

# with the same approach as key-val