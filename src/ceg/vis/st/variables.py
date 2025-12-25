from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Type,
    Any,
    Callable,
    Literal,
    overload,
    cast,

)
from pathlib import Path
import datetime as dt
import numpy as np
import polars as pl

from ... import apis
st = apis.st.library()

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

T = TypeVar("T")

@dataclass(frozen=True)
class StateVariable:
    k: str

    @property
    def key(self) -> str:
        return self.k
    
    def exists(self) -> bool:
        return self.key in st.session_state

    @overload
    def get(self, *args, t: Type[T]) -> T: ...

    @overload
    def get(self, *args, t: None = None) -> Any: ...

    def get(self, *args, t: Type[T] | None = None):
        v = st.session_state.get(self.key, *args)
        if t is not None:
            assert isinstance(v, t), (v, t)
        return v
    
    def set(self, val):
        st.session_state[self.key] = val


@dataclass(frozen=True)
class VersionVariable(StateVariable):
    def incr(self):
        v = self.get(0, t = int) + 1
        self.set(v)
        return v
    
@dataclass(frozen=True)
class VersionedVariable(StateVariable):
    version: VersionVariable
    
    @property
    def key(self) -> str:
        return f"{self.k}_{self.version.get(0, t = int)}"

__all__ = [
    "T",
    "StateVariable",
    "VersionVariable",
    "VersionedVariable",
]