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
from .objects import *

# TODO: on domain run / start we publish an event on the root object, that the objects can subscribe to (for eg. on-run style triggers)
# though perhaps filter on the object own state, eg. if only has to run daily after a certain time, etc.

@dataclass(frozen=True)
class Domain:
    root: Object # snap is used for domain on_run trigger
    objects: frozendict[str, Object]
    dependencies: frozendict[
        str, frozendict[str, Dependency]
    ]

    def with_obj(self, obj: Object) -> Domain: ...
    
    # NOTE: we dont need to diff the domain itself
    # if we add objects then their dependencies will be picked up on sync, and we know to iter the whole queue again, regardless of the domains own latest
    def sync(self) -> Domain: ...

__all__ = [
    "Domain"
]