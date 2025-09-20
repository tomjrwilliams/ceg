from typing import Type, Generic, Literal, overload, Callable, TypeGuard
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from .t import (
    _, ID, KeyI, KeyO, ValI, ValO
)
from .store import KeyValStore
from .io import KeyValIO

@dataclass(frozen=True)
class KeyValStoreConfig:
    pass

@dataclass(frozen=True)
class KeyValIO(ABC, Generic[KeyI, ValI, KeyO, ValO]):
    """
    Generic[KeyI, ValI, KeyO, ValO]
    """
    fields: dict[str, ValO] | None = None
    
    @abstractmethod
    def encode_key_type(
        self: KeyValIO[KeyI, ValI, KeyO, ValO],
        t: Type[KeyO]
    ) -> Type[KeyI]: ...

    @abstractmethod
    def encode_val_type(
        self: KeyValIO[KeyI, ValI, KeyO, ValO],
        t: Type[ValO]
    ) -> Type[ValI]: ...

    @abstractmethod
    def encode_key(
        self, key: KeyO
    ) -> KeyI: ...

    @abstractmethod
    def encode_val(
        self, val: ValO
    ) -> ValI: ...

    @abstractmethod
    def decode_key(
        self, key: KeyI, t: Type[KeyO]
    ) -> KeyO: ...

    @abstractmethod
    def decode_val(
        self, val: ValI, t: Type[ValO]
    ) -> ValO: ...