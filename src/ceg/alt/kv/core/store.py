from typing import Type, Generic, Literal, TypeGuard
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod

from .t import (
    _, ID, KeyI, KeyO, ValI, ValO
)
from .store import KeyValStore
from .io import KeyValIO

@dataclass(frozen=True)
class KeyValStoreConfig:
    pass

def is_singular_val(
    val: ValI | tuple[ValI, ...],
    val_type: Type[ValI] | dict[str, Type[ValI]]
) -> TypeGuard[ValI]:
    return not isinstance(val_type, dict)


# TODO: check we can define a type that just implements the signature below, and that the protocol is happy

# and thereofre that the free func select() etc. are happy


# and in the keyval layer? its okay as the sub fields should obey the protocols?



class KeyValStore(ABC, Generic[ID, KeyI, ValI]):
    """
    Generic[ID, KeyI, ValI]
    """
    key_type: Type[KeyI]
    val_type: Type[ValI] | dict[str, Type[ValI]]
    
    config: KeyValStoreConfig

    chunked: int | None = None # indicates chunk size

    def singular(self):
        return not isinstance(self.val_type, dict)

    def val_keys(self, field: tuple[str, ...]) -> tuple[str, ...]:
        if len(field):
            return field
        assert isinstance(self.val_type, dict), self
        return tuple(self.val_type.keys())

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def identify(self, key: KeyI) -> None | ID: ...

    @abstractmethod
    def lookup(self, id: ID) -> KeyI: ...

    @abstractmethod
    def select_val(
        self,
        key: KeyI, 
        field: str | None = None,
        chunk: int | None = None,
    ) -> None | ValI: ...

    @abstractmethod
    def select_vals(
        self,
        key: KeyI, 
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> None | tuple[ValI, ...]: ...

    @abstractmethod
    def iselect_val(
        self, 
        key: KeyI, 
        field: str | None = None,
        chunk: int | None = None,
    ) -> tuple[
        ID | None, ValI | None
    ]: ...

    @abstractmethod
    def iselect_vals(
        self, 
        key: KeyI, 
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> tuple[
        ID | None, tuple[ValI, ...] | None
    ]: ...

    # TODO: field is either tuple or none
    # if not singular, must pass dict (even if len =1)

    @abstractmethod
    def insert_val(
        self, 
        key: KeyI, 
        val: ValI,
        field: str | None = None,
        chunk: int | None = None,
    ) -> ID: ...

    @abstractmethod
    def insert_vals(
        self, 
        key: KeyI, 
        val: tuple[ValI, ...],
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> ID: ...

    @abstractmethod
    def update_val(
        self, 
        key: KeyI, 
        val: ValI, 
        field: str | None = None,
        chunk: int | None = None,
        i: ID | None = None
    ) -> ID: ...

    @abstractmethod
    def update_vals(
        self, 
        key: KeyI, 
        val: tuple[ValI, ...],
        field: tuple[str, ...] = (),
        chunk: int | None = None,
        i: ID | None = None
    ) -> ID: ...

