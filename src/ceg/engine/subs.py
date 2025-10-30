
from typing import Callable
from dataclasses import dataclass

from .objects import Queue, Event, Object


class Subscription:
    obj: Object
    listens: Object
    queue: Queue 
    
    # TODO: queue should default to listens.broadcasts()
    # but we can override if we want? eg. for old queue data stores? or for subscribing to several?

    type: str | None
    anti: bool = False

    # if none: assume create or update

# NOTE: we assume objects subscribe to their own definition / change
# anti can be used to turn this off

# NOTE: and also those of all their parents? so i guess anti is stronger, int hat it allows you maybe to truncate that list?
# and then even reinstate it further down


# TODO: but create / update are of the value not the object def, so by default they're nt subscribed to by children (only object itself, brodcasts its own value)



# TODO:

# you can load one domain into another

# therefore you can iterate it

# mapover it

# itertools product, filter, etc.


# use that rather than wildcards to create programmatic subscription / implicitly defined universes


# TODO:

# then the snap just needs an event id (and queue config?) for us to know what needs update or not

# snap can have more, because just because is a new event still doesnt mean work automatically needs doing