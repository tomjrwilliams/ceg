from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Type,
    Generic,
    Callable,
    cast,
    final,
)
from dataclasses import dataclass, field
from frozendict import frozendict

from ...t import *

# TODO: rename keyValstoreConfig
# TODO: convience registry on keyValStore to the class object
@dataclass(frozen=True)
class KeyValConfig:
    pass


@dataclass(frozen=True)
class KeyValStore(ABC, Generic[ID, KeyI, ValI]):
    """KeyValStore[ID, KeyI, ValI]"""
    config: KeyValConfig

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    # TODO: store the datetime last updated as well

    @abstractmethod
    def prepare(self) -> KeyValStore: ...

    @abstractmethod
    def lookup(self, id: ID) -> KeyI: ...

    @abstractmethod
    def identify(self, key: KeyI) -> None | ID: ...

    @abstractmethod
    def select(
        self,
        key: KeyI,
    ) -> None | ValI: ...

    @abstractmethod
    def iselect(
        self,
        key: KeyI,
    ) -> tuple[ID | None, ValI | None]: ...

    @abstractmethod
    def insert(
        self,
        key: KeyI,
        val: ValI,
    ) -> ID: ...

    @abstractmethod
    def update(
        self,
        key: KeyI,
        val: ValI,
        chunk: int | None = None,
        i: ID | None = None,
    ) -> ID: ...


@dataclass(frozen=True)
class KeyValIO(Generic[ID, KeyI, ValI, KeyO, ValO]):
    """KeyValIO[ID, KeyI, ValI, KeyO, ValO]"""
    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]
    type_key_o: Type[KeyO]
    type_val_o: Type[ValO]

    @abstractmethod
    def encode_key_type(self, t: Type[KeyO]) -> Type[KeyI]: ...

    @abstractmethod
    def encode_val_type(self, t: Type[ValO]) -> Type[ValI]: ...

    @abstractmethod
    def encode_key(self, key: KeyO) -> KeyI: ...

    @abstractmethod
    def encode_val(self, key: KeyO, val: ValO) -> ValI: ...

    @abstractmethod
    def decode_key(
        self,
        key: KeyI,
    ) -> KeyO: ...

    @abstractmethod
    def decode_val(
        self,
        key: KeyO,
        val: ValI,
    ) -> ValO: ...

    def check_type_id(self, id: ID | None) -> ID:
        assert isinstance(id, self.type_id)
        return id

    def check_type_key_i(self, key_i: KeyI) -> KeyI:
        assert isinstance(key_i, self.type_key_i)
        return key_i

    def check_type_key_o(self, key_o: KeyO) -> KeyO:
        assert isinstance(key_o, self.type_key_o)
        return key_o
    
    def check_type_val_i(self, val_i: ValI) -> ValI:
        assert isinstance(val_i, self.type_val_i)
        return val_i
        
    def check_type_val_o(self, val_o: ValO) -> ValO:
        assert isinstance(val_o, self.type_val_o)
        return val_o


@dataclass(frozen=True)
class KeyVal(
    KeyValIO[ID, KeyI, ValI, KeyO, ValO],
    KeyValStore[ID, KeyI, ValI],
):
    """KeyVal[ID, KeyI, ValI, KeyO, ValO]"""
    on_create: Callable[[KeyO, ValO], None] | None = None
    on_update: Callable[[KeyO, ValO], None] | None = None

    # NOTE: if on_update is none, cannot update

    @overload
    def get(
        self,
        key: KeyO,
        t: None=None,
        # TODO: default value if not found
    ) -> ValO | None: ...
    @overload
    def get(
        self,
        key: KeyO,
        t: Type[T],
        # TODO: default value if not found
    ) -> T | None: ...
    def get(
        self,
        key: KeyO,
        t: Type[T] | None = None,
        # TODO: default value if not found
    ) -> ValO | T | None:
        key_i = self.check_type_key_i(
            self.encode_key(self.check_type_key_o(key))
        )
        val_i = self.select(key_i)
        if val_i is None:
            return val_i
        val_o = self.check_type_val_o(
            self.decode_val(
                key, self.check_type_val_i(val_i)
            )
        )
        if t is not None:
            assert isinstance(val_o, t), (val_o, t)
        return val_o

    def iget(
        self,
        key: KeyO,
    ) -> tuple[ID | None, ValO | None]:
        key_i = self.check_type_key_i(
            self.encode_key(self.check_type_key_o(key))
        )
        i, val_i = self.iselect(key_i)
        if val_i is None:
            return i, val_i
        return (
            self.check_type_id(i), 
            self.check_type_val_o(self.decode_val(
                key, val_i
            ))
        )

    @overload
    def set(
        self,
        key: KeyO,
        val: ValO,
        t: None = None,
    ) -> ValO: ...

    @overload
    def set(
        self,
        key: KeyO,
        val: ValO,
        t: Type[T],
    ) -> T: ...

    def set(
        self,
        key: KeyO,
        val: ValO,
        t: Type[T] | None = None,
    ) -> ValO | T:
        _, v = self.iset(key, val)
        if t is not None:
            assert isinstance(v, t), (v, t)
            return v
        return v

    def iset(
        self,
        key: KeyO,
        val: ValO,
    ) -> tuple[ID, ValO]:

        i, val_o = self.iget(key)

        if i is not None and (
            val_o is None and val is None or val_o == val
        ):
            return i, val

        key_i = self.check_type_key_i(self.encode_key(key))
        val_i = self.check_type_val_i(self.encode_val(key, val))

        if i is None:
            if self.on_create is not None:
                self.on_create(key, val)
            i = self.insert(key_i, val_i)
        elif self.on_update is not None:
            self.on_update(key, val)
            i = self.update(key_i, val_i, i=i)
        else:
            raise ValueError((key, val, self.on_update))

        return i, val

    # TODO: get_many (return dict of the above by key)
    # optionally a list instead
    # or a dataframe with key in col vs val, etc.?

    # TODO: set_many

@dataclass(frozen=True)
class MemoryKeyVal(
    KeyVal[ID, KeyI, ValI, KeyO, ValO]
):
    """KeyVal[ID, KeyI, ValI, KeyO, ValO]"""
    data: dict[KeyI, ValI] = field(default_factory=dict)

    def prepare(self) -> MemoryKeyVal:
        return self

    def lookup(self, id: ID) -> KeyI:
        assert isinstance(id, int), id
        return list(self.data.keys())[id]

    def identify(self, key: KeyI) -> None | ID:
        if key not in self.data:
            return None
        for i, k in enumerate(self.data):
            if k == key:
                return cast(ID, i)
        return None

    def select(
        self,
        key: KeyI,
    ) -> None | ValI:
        return self.data.get(key)

    def iselect(
        self,
        key: KeyI,
    ) -> tuple[ID | None, ValI | None]:
        return self.identify(key), self.select(key)

    def insert(
        self,
        key: KeyI,
        val: ValI,
    ) -> ID:
        i = cast(ID, len(self.data))
        self.data[key] = val
        return i

    def update(
        self,
        key: KeyI,
        val: ValI,
        i: ID | None = None,
    ) -> ID:
        i = self.identify(key)
        if i is None:
            return self.insert(key, val)
        self.data[key] = val
        return i

@dataclass(frozen=True)
class ExampleKeyVal(KeyVal[int, str, int, str, float]):

    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[int] = int
    type_key_o: Type[str] = str
    type_val_o: Type[float] = float

    def encode_key_type(self, t: Type[str]) -> Type[str]:
        return t

    def encode_val_type(self, t: Type[float]) -> Type[int]:
        return int

    def encode_key(self, k: str) -> str:
        return k.upper()

    def encode_val(self, k: str, v: float) -> int:
        return int(v)

    def decode_key(self, k: str) -> str:
        return k.lower()

    def decode_val(self, k: str, v: int) -> float:
        return float(v)

@dataclass(frozen=True)
class KeyValStrKeyIO(
    KeyVal[ID, str, ValI, str, ValO]
):
    """KeyValStrKeyIO[ID, ValI, ValO]"""

    type_key_i: Type[str]
    type_key_o: Type[str]

    def encode_key_type(self, t: Type[str]) -> Type[str]:
        return t

    def encode_key(self, k: str) -> str:
        return k

    def decode_key(self, k: str) -> str:
        return k

@dataclass(frozen=True)
class KeyValBytesValI(
    KeyVal[ID, KeyI, bytes, KeyO, ValO]
):
    """KeyValBytesValI[ID, KeyI, KeyO, ValO]"""

    type_key_i: Type[str]
    type_val_i: Type[bytes]

    def encode_key_type(self, t: Type[str]) -> Type[str]:
        return t

    def encode_key(self, k: str) -> str:
        return k

    def decode_key(self, k: str) -> str:
        return k

@dataclass(frozen=True)
class KeyValStrKeyIOBytesValI(
    KeyValStrKeyIO[ID, bytes, ValO],
    KeyValBytesValI[ID, str, str, ValO]
):
    """KeyValStrKeyIOBytesValI"""

    type_key_i: Type[str]
    type_val_i: Type[bytes]
    type_key_o: Type[str]

    def encode_key_type(self, t: Type[str]) -> Type[str]:
        return t

    def encode_key(self, k: str) -> str:
        return k

    def decode_key(self, k: str) -> str:
        return k

@final
@dataclass(frozen=True)
class ExampleMemoryKeyVal(
    ExampleKeyVal,
    MemoryKeyVal[int, str, int, str, float]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> kv = ExampleMemoryKeyVal(
    ...     on_update = lambda k, v: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k)
    >>> v = kv.set(k, 0.0)
    >>> assert v == 0.0 == kv.get(k)
    >>> assert (0, 0.0) == kv.iget(k)
    >>> v = kv.set(j, 1.0)
    >>> v = kv.set(k, 2.0)
    >>> assert (1, 1.0) == kv.iget(j)
    >>> assert (0, 2.0) == kv.iget(k)
    >>> assert updated == [(k, 2.0)]
    >>> assert kv.data == {"K": 2, "J": 1}
    """
    config: KeyValConfig = KeyValConfig()

# NOTE: typing example
i, v = ExampleMemoryKeyVal().iselect("")

__all__ = [
    "KeyValConfig",
    "KeyValStore",
    "KeyValIO",
    "KeyVal",
    "MemoryKeyVal",
    "ExampleKeyVal",
    "KeyValStrKeyIO",
    "KeyValBytesValI",
    "KeyValStrKeyIOBytesValI",
]