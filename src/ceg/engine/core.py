from __future__ import annotations

# TODO: pydantic?
from dataclasses import dataclass

from frozendict import frozendict

# TODO: possibly annotate with separate co and contr variant bound type variables rather than the base class themselves?

@dataclass(frozen=True)
class Object:
    parent: Object | None
    path: str

    # TODO: save / load on own tags, on diff publish event
    # different events for diff tag changes?
    
    # TODO: save, load, diff, event publish on diff
    def depends(self) -> list[Dependency]: ...

    # rather than storing all of them up front, return the handler (s) to run for a given event)
    def handlers(self, event: Event) -> list[Handler]: ...

    # likewise sub-type with the snap for this object and it's value (s??)
    def snap(self) -> Snap: ...

    # TODO: override with more specific output per object?
    def value(self) -> Value: ...

    def latest(self) -> Event: ...
    # cache this? for easier checking if we need to do anything for a given event?
    # we need to know the events are always globally ordered? (or at least handled ordered?)

    # TODO: value / values?

@dataclass(frozen=True)
class Domain:
    current: Event | None
    objects: frozendict[str, Object]
    dependencies: frozendict[
        str, frozendict[str, Dependency]
    ]

    # for those to run on startup, others we want to add on objects directly? but then the event is none as well if on startup? or the event is startup. can then probably simplify the sig, eg. just list of handlers?
    handlers: frozendict[str | None, list[Handler]]

    def with_obj(self, obj: Object) -> Domain: ...

    def with_handler(self, handler: Handler) -> Domain: ...

    def sync(self) -> Domain: ...

    def latest(self) -> Event: ...

    # TODO: need to presumably take a generic interface onto the backend storage (kv etc.?)
    # partially filled in with the o-types, then up to the app implementatin to specify which concrete implementation

@dataclass(frozen=True)
class Dependency:
    obj: Object
    depends: Object
    parent: bool = False
    active: bool = True
    # TODO: active is just len(handlers) > 0?

    # TODO: save / load

@dataclass(frozen=True)
class Handler:
    obj: Object
    # event: Event # ? so no need in handle() sig?
    
    def handle(
        self,
        # possibly take the object here not in fields
        event: Event,
        queue: Queue,
    ) -> Queue: ...

@dataclass(frozen=True)
class Event:
    obj: Object
    # type?

@dataclass(frozen=True)
class Queue:

    buffer: list[Event] | None = None
    # acc buffer into the queue, flush as required
    # so might have several copies of the queue we pass around

    # buffer also means we can iter directly from the queue into the domain before we flush? or we want to ensure publish before acting (probably that, for other domains)
    # then i guess just want to ensure publish before we mark done on the handler?

    def publish(self, event: Event): ...

    # iter?

    def latest(self): ...

    # TODO: need to presumably take a generic interface onto the backend storage (kv etc.?)
    # partially filled in with the o-types, then up to the app implementatin to specify which concrete implementation?

    # one way to do that is to have a switch function that takes the root object, and uses that to identify which back end type to instantiate for the actual back end object?


@dataclass(frozen=True)
class Snap:
    # obj: Object # convenience reference? needed?
    event: Event # latest handled

    # TODO: need to presumably take a generic interface onto the backend storage (kv etc.?)
    # partially filled in with the o-types, then up to the app implementatin to specify which concrete implementation?


@dataclass(frozen=True)
class Value:
    # TODO: different sub types
    # with different corresponding event sub types to publish
    

    # TODO: need to presumably take a generic interface onto the backend storage (kv etc.?)
    # partially filled in with the o-types, then up to the app implementatin to specify which concrete implementation?

    # value type specific (eg. time series vs blob etc.)

    pass