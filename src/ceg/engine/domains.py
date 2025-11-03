from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    overload,
    cast,
    final,
)
import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from pydantic import (
    TypeAdapter,
    BaseModel,
    Field,
)
from frozendict import frozendict

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .objects import *

@dataclass(frozen=True)
class Domain:
    root: Object # snap is used for domain on_run trigger
    objects: frozendict[str, Object]
    dependencies: frozendict[
        str, frozendict[str, Dependency]
    ]

    def with_obj(self, obj: Object) -> Domain:
        assert obj.path not in self.objects, obj.path
        # TODO: dependencies
        return replace(
            self, objects=self.objects.set(obj.path, obj)
        )
    
    # NOTE: we dont need to diff the domain itself
    # if we add objects then their dependencies will be picked up on sync, and we know to iter the whole queue again, regardless of the domains own latest
    def sync(self) -> Domain:
        
        # TODO: or object.value not snap?
        snap = self.root.get_snap()
        curr = snap.event

        q = self.root.event_queue()

        events: list[Event]
        history: list[Event] | None = None

        for i_path, triggers in self.dependencies.items():
            for o_path, dep in triggers.items():
                obj = self.objects.get(o_path)
                assert obj is not None, (o_path)

                if not dep.active:
                    dep = dep.set(
                        obj.dependency_key_val()
                    )
                    continue

                dep_prev = dep.get(
                    dep.key(),
                    obj.dependency_key_val()
                )

                events = []
                if dep_prev is None:
                    history = history or q.head(
                        n=-1,
                        where_i = [("le", curr.i)]
                        # TODO: head = from first?
                    )
                    events = [
                        e for e in history
                        if e.obj == i_path
                    ]
                elif dep.events is not None:
                    blob_events = dep_prev.events or []
                    e_types = [
                        e_type for e_type in dep.events
                        if e_type not in blob_events
                    ]
                    if len(e_types):
                        history = history or q.head(
                            n=-1,
                            where_i = [("le", curr.i)]
                            # TODO: head = from first?
                        )
                        events = [
                            e for e in history
                            if e.obj == i_path
                            and e.type in e_types    
                        ]

                for e in events:
                    handlers = obj.handlers(e)
                    for handler in handlers:
                        handler.handle(obj, e)

                dep = dep.set(
                    obj.dependency_key_val()
                )

        # TODO: publish self.domain_sync()
        # objects might also filter on their own snap for domain start, eg. if only has to run daily after a certain time, etc.

        # TODO: detect object change, whether it's a def change is entirely by version fields set on object as fields with default values
        # handled client side: no separate event type (?), same logic as for other field changes
        # maybe fire separate event? delegate to object? obj.difference() -> events

        events = q.head(
            n=-1,
            where_i = [("gt", curr.i)],
            # TODO: tail = from first, asecnding?
        )

        while len(events):

            for e in events:
                i_path = e.obj
                triggers = self.dependencies.get(i_path)
                if triggers is None:
                    continue
                for o_path, dep in triggers.items():
                    if not dep.active:
                        continue
                    elif dep.events is None:
                        pass
                    elif e.type not in dep.events:
                        continue
                    
                    obj = self.objects.get(o_path)
                    assert obj is not None, o_path

                    handlers = obj.handlers(e)
                    for handler in handlers:
                        handler.handle(obj, e)

                curr = e
            
            events = q.head(
                n=-1,
                where_i = [("gt", curr.i)],
                # TODO: tail = from first, asecnding?
            ) # type: ignore

        snap = replace(snap, event=curr)
        snap.set(self.root.snap_key_val())
        
        return self

__all__ = [
    "Domain"
]