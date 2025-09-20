from typing import Type, Generic, Literal, TypeGuard, Callable
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod

from typing import TypeVar

ID = TypeVar("ID", covariant=False, contravariant=False)
KeyI = TypeVar("KeyI", covariant=False, contravariant=False)
ValI = TypeVar("ValI", covariant=False, contravariant=False)
KeyO = TypeVar("KeyO", covariant=False)
ValO = TypeVar("ValO", covariant=False)

_ = TypeVar("_", bound=object)


@dataclass(frozen=True)
class KeyValConfig:
    pass

@dataclass(frozen=True)
class KeyVal(ABC, Generic[ID, KeyI, ValI, KeyO, ValO]):
    """Generic[ID, KeyI, ValI, KeyO, ValO]
    
    config: KeyValConfig

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]
    type_key_o: Type[KeyO]
    type_val_o: Type[ValO]

    chunked: int | None = None
    """
    config: KeyValConfig

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]
    type_key_o: Type[KeyO]
    type_val_o: Type[ValO]

    chunked: int | None = None # indicates chunk size

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def identify(self, key: KeyI) -> None | ID: ...

    @abstractmethod
    def lookup(self, id: ID) -> KeyI: ...

    @abstractmethod
    def select(
        self,
        key: KeyI, 
        chunk: int | None = None,
    ) -> None | ValI: ...

    @abstractmethod
    def iselect(
        self, 
        key: KeyI, 
        field: str | None = None,
        chunk: int | None = None,
    ) -> tuple[
        ID | None, ValI | None
    ]: ...

    @abstractmethod
    def insert(
        self, 
        key: KeyI, 
        val: ValI,
        chunk: int | None = None,
    ) -> ID: ...

    @abstractmethod
    def update(
        self, 
        key: KeyI, 
        val: ValI, 
        chunk: int | None = None,
        i: ID | None = None
    ) -> ID: ...

    # TODO: encode id type?

    @abstractmethod
    def encode_key_type(
        self,
        t: Type[KeyO]
    ) -> Type[KeyI]: ...

    @abstractmethod
    def encode_val_type(
        self,
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
        self, key: KeyI, 
    ) -> KeyO: ...

    @abstractmethod
    def decode_val(
        self, val: ValI, 
    ) -> ValO: ...

    def get(
        self,
        key: KeyO,
        chunk: int | None = None,
        # TODO: default value if not found
    ) -> ValO | None:
        key_i = self.encode_key(key)
        val_i = self.select(
            key_i, 
            chunk=chunk,
        )
        if val_i is None:
            return val_i
        return self.decode_val(val_i)

    def iget(
        self,
        key: KeyO,
        chunk: int | None = None,
    ) -> tuple[ID | None, ValO | None]:
        key_i = self.encode_key(key)
        i, val_i = self.iselect(
            key_i, 
            chunk=chunk,
        )
        if val_i is None:
            return i, val_i
        return i, self.decode_val(val_i)

    def set(
        self,
        key: KeyO,
        val: ValO,
        chunk: int | None = None,
        on_create: Callable[
            [KeyO, ValO, int | None], None
        ] | None=None,
        on_update: Callable[
            [KeyO, ValO, int | None], None
        ] | None=None,
    ) -> ValO:
        i, v = self.iset(
            key,
            val,
            chunk=chunk,
            on_create=on_create,
            on_update=on_update,
        )
        return v

    def iset(
        self,
        key: KeyO,
        val: ValO,
        chunk: int | None = None,
        on_create: Callable[
            [KeyO, ValO, int | None], None
        ] | None=None,
        on_update: Callable[
            [KeyO, ValO, int | None], None
        ] | None=None,
    ) -> tuple[ID, ValO]:
    
        # TODO: if chunk, assert same length, or full (no partial, if multi field)

        # TODO: later on, can take a custom is eq funciton for eg. large blobs to have quicker / custom implementation

        # and even possibly store some kind of hash / quicksum to check, needs to guarantee match is the same

        # and then possibly if diff, can still query? i guess the question there is how often your upates are no-op

        i, val_o = self.iget(key)

        if i is not None and (
            val_o is None and val is None
            or val_o == val
        ):
            return i, val

        key_i = self.encode_key(key)
        val_i = self.encode_val(val)

        if i is None:
            if on_create is not None:
                on_create(key, val, chunk)
            i = self.insert(key_i, val_i, chunk=chunk)
        elif on_update is not None:
            on_update(key, val, chunk)
            i = self.update(key_i, val_i, chunk=chunk, i=i)
        else:
            raise ValueError((key, val, on_update))
        
        return i, val

    # TODO: get_many (return dict of the above by key)
    # optionally a list instead
    # or a dataframe with key in col vs val, etc.?

    # TODO: set_many

    def append(self):
        # set with chunk
        # re chunk if needed (?)
        # append values is a io concern, we just reset on size
        # so size and append should be an io concern as well

        # up to the io to encode  decode whatever is used for the list type
        # and to append / prepend values to it

        # given a fixed chunk size, we can presumably not append numpy arrays, but pre-alloc the chunk, and assign, leave the rest as null or use some marker (ie. store as a tuple of array and occupied int)

        # benchmark the two implementations: expanding aray with append, and pre-alloc the chunk and assign
        return

    def prepend(self):
        # set with chunk
        return