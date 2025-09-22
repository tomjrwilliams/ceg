from typing import (
    Type,
    Generic,
    Literal,
    TypeGuard,
    Callable,
)
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod

from typing import TypeVar

T = TypeVar("T")
ID = TypeVar("ID", covariant=False, contravariant=False)
KeyI = TypeVar("KeyI", covariant=False, contravariant=False)
ValI = TypeVar("ValI", covariant=False, contravariant=False)
KeyO = TypeVar("KeyO", covariant=False)
ValO = TypeVar("ValO", covariant=False)
ValT = TypeVar("ValT", covariant=False)

START = FIRST = HEAD = 0
MIDDLE = BODY = 1
END = LAST = FOOT = TAIL = 2

_ = TypeVar("_", bound=object)


def not_none(v: T | None) -> T:
    assert v is not None, v
    return v


@dataclass(frozen=True)
class KeyValConfig:
    pass


@dataclass(frozen=True)
class KeyValStore(ABC, Generic[ID, KeyI, ValI]):
    config: KeyValConfig

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    @abstractmethod
    def prepare(self) -> None: ...

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
class KeyValChunkedConfig(KeyValConfig):
    size: int


@dataclass(frozen=True)
class KeyValChunkedStore(ABC, Generic[ID, KeyI, ValI]):
    config: KeyValChunkedConfig

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def lookup(self, id: ID) -> KeyI: ...

    @abstractmethod
    def identify(self, key: KeyI) -> None | tuple[ID, int]: ...

    @abstractmethod
    def select(
        self,
        key: KeyI,
        chunk: int,
    ) -> None | ValI: ...

    @abstractmethod
    def iselect(
        self,
        key: KeyI,
        chunk: int,
    ) -> tuple[ID | None, ValI | None]: ...

    @abstractmethod
    def insert(
        self,
        key: KeyI,
        val: ValI,
        chunk: int,
    ) -> ID: ...

    @abstractmethod
    def update(
        self,
        key: KeyI,
        val: ValI,
        chunk: int,
        i: ID | None = None,
    ) -> ID: ...


@dataclass(frozen=True)
class KeyValIO(Generic[ID, KeyI, ValI, KeyO, ValO]):
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
    def encode_val(self, val: ValO) -> ValI: ...

    @abstractmethod
    def decode_key(
        self,
        key: KeyI,
    ) -> KeyO: ...

    @abstractmethod
    def decode_val(
        self,
        val: ValI,
    ) -> ValO: ...


@dataclass(frozen=True)
class KeyVal(
    KeyValIO[ID, KeyI, ValI, KeyO, ValO],
    KeyValStore[ID, KeyI, ValI],
):
    on_create: Callable[[KeyO, ValO], None] | None = None
    on_update: Callable[[KeyO, ValO], None] | None = None

    # NOTE: if on_update is none, cannot update

    def get(
        self,
        key: KeyO,
        # TODO: default value if not found
    ) -> ValO | None:
        key_i = self.encode_key(key)
        val_i = self.select(key_i)
        if val_i is None:
            return val_i
        return self.decode_val(val_i)

    def iget(
        self,
        key: KeyO,
    ) -> tuple[ID | None, ValO | None]:
        key_i = self.encode_key(key)
        i, val_i = self.iselect(key_i)
        if val_i is None:
            return i, val_i
        return i, self.decode_val(val_i)

    def set(
        self,
        key: KeyO,
        val: ValO,
    ) -> ValO:
        _, v = self.iset(key, val)
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

        key_i = self.encode_key(key)
        val_i = self.encode_val(val)

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
class KeyChunkIO(Generic[ID, KeyI, ValI, KeyO, ValO, ValT]):
    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]
    type_key_o: Type[KeyO]
    type_val_o: Type[ValO]
    type_val_t: Type[ValT]

    @abstractmethod
    def init_val(self) -> ValO: ...

    # eg. assign nan array, assign rather than append
    # when full, split and return two new arrays
    # size given by chunked param

    # up to you to implement whatever chunking mechanism you want, given the chunked param?
    # in that case, just pass the t?

    @abstractmethod
    def append_val(
        self, val: ValT, acc: ValO, tail: bool = True
    ) -> tuple[ValO, ValO | None]: ...

    @abstractmethod
    def extend_val(
        self, val: ValO, acc: ValO, tail: bool = True
    ) -> tuple[ValO, ValO | None]: ...

    @abstractmethod
    def concat_val(
        self, val: ValO, acc: ValO, tail: bool = True
    ) -> ValO: ...


@dataclass(frozen=True)
class KeyValChunked(
    KeyChunkIO[ID, KeyI, ValI, KeyO, ValO, ValT],
    KeyValIO[ID, KeyI, ValI, KeyO, ValO],
    KeyValChunkedStore[ID, KeyI, ValI],
):
    on_create: Callable[[KeyO, ValO, int], None] | None = None
    on_update: Callable[[KeyO, ValO, int], None] | None = None

    def get(
        self,
        key: KeyO,
        chunk: int,
        # TODO: default value if not found
    ) -> ValO | None:
        key_i = self.encode_key(key)
        val_i = self.select(key_i, chunk)
        if val_i is None:
            return val_i
        return self.decode_val(val_i)

    def get_head(self, key: KeyO) -> ValO | None:
        return self.get(key, HEAD)

    def get_body(self, key: KeyO) -> ValO | None:
        return self.get(key, BODY)

    def get_tail(self, key: KeyO) -> ValO | None:
        return self.get(key, TAIL)

    def iget(
        self,
        key: KeyO,
        chunk: int,
    ) -> tuple[ID | None, ValO | None]:
        key_i = self.encode_key(key)
        i, val_i = self.iselect(
            key_i,
            chunk,
        )
        if val_i is None:
            return i, val_i
        return i, self.decode_val(val_i)

    def iget_head(
        self, key: KeyO
    ) -> tuple[ID | None, ValO | None]:
        return self.iget(key, HEAD)

    def iget_body(
        self, key: KeyO
    ) -> tuple[ID | None, ValO | None]:
        return self.iget(key, BODY)

    def iget_tail(
        self, key: KeyO
    ) -> tuple[ID | None, ValO | None]:
        return self.iget(key, TAIL)

    def set(
        self,
        key: KeyO,
        val: ValO,
        chunk: int,
    ) -> ValO:
        _, v = self.iset(key, val, chunk)
        return v

    def set_head(
        self,
        key: KeyO,
        val: ValO,
    ) -> ValO:
        return self.set(key, val, HEAD)

    def set_body(
        self,
        key: KeyO,
        val: ValO,
    ) -> ValO:
        return self.set(key, val, BODY)

    def set_tail(
        self,
        key: KeyO,
        val: ValO,
    ) -> ValO:
        return self.set(key, val, TAIL)

    def iset(
        self,
        key: KeyO,
        val: ValO,
        chunk: int,
    ) -> tuple[ID, ValO]:

        i, val_o = self.iget(key, chunk)

        if i is not None and (
            val_o is None and val is None or val_o == val
        ):
            return i, val

        key_i = self.encode_key(key)
        val_i = self.encode_val(val)

        if i is None:
            if self.on_create is not None:
                self.on_create(key, val, chunk)
            i = self.insert(key_i, val_i, chunk)
        elif self.on_update is not None:
            self.on_update(key, val, chunk)
            i = self.update(key_i, val_i, chunk, i=i)
        else:
            raise ValueError((key, val, self.on_update))

        return i, val

    def iset_head(
        self,
        key: KeyO,
        val: ValO,
    ) -> tuple[ID, ValO]:
        return self.iset(key, val, HEAD)

    def iset_body(
        self,
        key: KeyO,
        val: ValO,
    ) -> tuple[ID, ValO]:
        return self.iset(key, val, BODY)

    def iset_tail(
        self,
        key: KeyO,
        val: ValO,
    ) -> tuple[ID, ValO]:
        return self.iset(key, val, TAIL)

    def append(
        self,
        key: KeyO,
        val: ValT,
        tail: bool = True,
    ) -> tuple[ID, ValO]:

        i, val_o = self.iget(key, chunk=TAIL if tail else HEAD)

        if i is None:
            assert val_o is None, val_o
            val_o = self.init_val()

            _, _ = self.iset(
                key, val_o, chunk=HEAD if tail else TAIL
            )
            _, _ = self.iset(
                key,
                val_o,
                chunk=BODY,
            )

            val_o, acc_o = self.append_val(val, val_o, tail)

        else:
            assert val_o is not None, (self, val_o)
            val_o, acc_o = self.append_val(val, val_o, tail)

        if acc_o is None:
            i, val_o = self.iset(
                key,
                val_o,
                chunk=TAIL if tail else HEAD,
            )
        else:
            acc_o = self.concat_val(
                not_none(self.get(key, BODY)),
                acc_o,
                tail=tail,
            )
            _, acc_o = self.iset(
                key,
                acc_o,
                chunk=BODY,
            )
            i, val_o = self.iset(
                key,
                val_o,
                chunk=TAIL if tail else HEAD,
            )

        return i, val_o

    def append_head(
        self,
        key: KeyO,
        val: ValT,
    ) -> tuple[ID, ValO]:
        return self.append(key, val, False)

    def append_tail(
        self,
        key: KeyO,
        val: ValT,
    ) -> tuple[ID, ValO]:
        return self.append(key, val, True)

    def extend(
        self,
        key: KeyO,
        val: ValO,
        tail: bool = True,
    ) -> tuple[ID, ValO]:

        i, val_o = self.iget(key, chunk=TAIL if tail else HEAD)

        if i is None:
            assert val_o is None, val_o
            val_o = self.init_val()

            _, _ = self.iset(
                key,
                val_o,
                chunk=HEAD if tail else TAIL,
            )
            _, _ = self.iset(
                key,
                val_o,
                chunk=BODY,
            )

            val_o, acc_o = self.extend_val(val, val_o, tail)
        else:
            assert val_o is not None, (self, val_o)
            val_o, acc_o = self.extend_val(val, val_o, tail)

        if acc_o is None:
            i, val_o = self.iset(
                key,
                val_o,
                chunk=TAIL if tail else HEAD,
            )
        else:
            acc_o = self.concat_val(
                not_none(self.get(key, BODY)),
                acc_o,
                tail=tail,
            )
            _, acc_o = self.iset(
                key,
                acc_o,
                chunk=BODY,
            )
            i, val_o = self.iset(
                key,
                val_o,
                chunk=TAIL if tail else HEAD,
            )

        return i, val_o

    def extend_head(
        self,
        key: KeyO,
        val: ValO,
    ):
        return self.extend(key, val, False)

    def extend_tail(
        self,
        key: KeyO,
        val: ValO,
    ):
        return self.extend(key, val, True)

    # TODO: append / extend many
