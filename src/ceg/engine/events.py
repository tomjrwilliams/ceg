from __future__ import annotations
from typing import Callable
import itertools

from dataclasses import dataclass

from .objects import (
    Object,
    Event,
    Tags,
    # Index,
)

def publish(self, event: Event):
    return

def listen(self):
    return

# TODO: events are id on the log / queue id

# ideally, under the same kind of if_new / update etc. logic as the calc nodes?


# TODO: then subscriptions can instead be a key-val store on the index (not a log: just writing their own last processed), per pair of consumer and queue


# the only remaining question is what to do on the change of the log itself? is that such a thing - or they have no props and hence new name is just new queue?

# thats by far the simplest


# TODO: need a way of registeringt hese to fire / run on startup (eg. to check for new data)