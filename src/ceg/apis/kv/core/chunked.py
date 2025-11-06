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
import numpy as np

from ...type_vars import *
from .key_val import *


START = FIRST = HEAD = 0
MIDDLE = BODY = 1
END = LAST = FOOT = TAIL = 2


@dataclass(frozen=True)
class KeyChunkConfig(KeyValConfig):
    size: int


@dataclass(frozen=True)
class KeyChunkStore(ABC, Generic[ID, KeyI, ValI]):
    """KeyChunkStore[ID, KeyI, ValI]"""
    config: KeyChunkConfig

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    @abstractmethod
    def prepare(self) -> KeyChunkStore: ...

    @abstractmethod
    def lookup(self, id: ID) -> tuple[KeyI, int] | tuple[None, None]: ...

    @abstractmethod
    def identify(self, key: KeyI, chunk: int) -> None | ID: ...

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
class KeyChunkIO(Generic[ID, KeyI, ValI, KeyO, ValO, ValT]):
    """KeyChunkIO[ID, KeyI, ValI, KeyO, ValO, ValT]"""
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

    def check_type_val_t(self, val_t: ValT) -> ValT:
        assert isinstance(val_t, self.type_val_t)
        return val_t


@dataclass(frozen=True)
class KeyChunk(
    KeyChunkIO[ID, KeyI, ValI, KeyO, ValO, ValT],
    KeyValIO[ID, KeyI, ValI, KeyO, ValO],
    KeyChunkStore[ID, KeyI, ValI],
):
    """KeyChunk[ID, KeyI, ValI, KeyO, ValO, ValT]"""
    config: KeyChunkConfig

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
        return self.decode_val(key, val_i)

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
        return i, self.decode_val(key, val_i)

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
            val_o is None and val is None
            # or val_o == val: TODO: np.all?
        ):
            return i, val

        key_i = self.encode_key(key)
        val_i = self.encode_val(key, val)

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

    # TODO: these only make sense for three chunk?
    # need a new name for that?

    def iappend(
        self,
        key: KeyO,
        val: ValT,
        tail: bool = True,
    ) -> tuple[ID, ValO]:

        i, val_o = self.iget(key, chunk=TAIL if tail else HEAD)

        if i is None:
            assert val_o is None, val_o
            val_o = self.init_val()

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
            acc_curr = self.get(key, BODY)
            if acc_curr is not None:
                acc_o = self.concat_val(
                    acc_o,
                    acc_curr,
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

    def iappend_head(
        self,
        key: KeyO,
        val: ValT,
    ) -> tuple[ID, ValO]:
        return self.iappend(key, val, False)

    def iappend_tail(
        self,
        key: KeyO,
        val: ValT,
    ) -> tuple[ID, ValO]:
        return self.iappend(key, val, True)

    def append(
        self,
        key: KeyO,
        val: ValT,
        tail: bool = True,
    ) -> ValO:
        i, val_o = self.iappend(key, val, tail=tail)
        return val_o

    def append_head(
        self,
        key: KeyO,
        val: ValT,
    ) -> ValO:
        i, val_o = self.iappend(key, val, False)
        return val_o

    def append_tail(
        self,
        key: KeyO,
        val: ValT,
    ) -> ValO:
        i, val_o = self.iappend(key, val, True)
        return val_o

    def iextend(
        self,
        key: KeyO,
        val: ValO,
        tail: bool = True,
    ) -> tuple[ID, ValO]:

        i, val_o = self.iget(key, chunk=TAIL if tail else HEAD)

        if i is None:
            assert val_o is None, val_o
            val_o = self.init_val()

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
            acc_curr = self.get(key, BODY)
            if acc_curr is not None:
                acc_o = self.concat_val(
                    acc_o,
                    acc_curr,
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

    def iextend_head(
        self,
        key: KeyO,
        val: ValO,
    ) -> tuple[ID, ValO]:
        return self.iextend(key, val, False)

    def iextend_tail(
        self,
        key: KeyO,
        val: ValO,
    ) -> tuple[ID, ValO]:
        return self.iextend(key, val, True)

    def extend(
        self,
        key: KeyO,
        val: ValO,
        tail: bool = True,
    ) -> ValO:
        i, val_o = self.iextend(key, val, tail=tail)
        return val_o
    
    def extend_head(
        self,
        key: KeyO,
        val: ValO,
    ) -> ValO:
        i, val_o = self.iextend(key, val, False)
        return val_o

    def extend_tail(
        self,
        key: KeyO,
        val: ValO,
    ) -> ValO:
        i, val_o = self.iextend(key, val, True)
        return val_o


    # TODO: append / extend many

@dataclass(frozen=True)
class MemoryKeyChunk(
    KeyChunk[ID, KeyI, ValI, KeyO, ValO, ValT]
):
    data: dict[tuple[KeyI, int], ValI] = field(default_factory=dict)

    def prepare(self) -> MemoryKeyChunk:
        return self

    def lookup(self, id: ID) -> tuple[KeyI, int] | tuple[None, None]:
        assert isinstance(id, int)
        return list(self.data.keys())[id]

    def identify(self, key: KeyI, chunk: int) -> ID | None:
        if (key, chunk) not in self.data:
            return None
        for id, (k, i) in enumerate(self.data.keys()):
            if k == key and chunk == i:
                return cast(ID, id)
        return None

    def select(
        self,
        key: KeyI,
        chunk: int,
    ) -> None | ValI:
        return self.data.get((key, chunk))

    def iselect(
        self,
        key: KeyI,
        chunk: int,
    ) -> tuple[ID | None, ValI | None]:
        id = self.identify(key, chunk)
        if id is None:
            return None, None
        return id, self.select(key, chunk)

    def insert(
        self,
        key: KeyI,
        val: ValI,
        chunk: int,
    ) -> ID:
        id = cast(ID, len(self.data))
        self.data[(key, chunk)] = val
        return id

    def update(
        self,
        key: KeyI,
        val: ValI,
        chunk: int,
        i: ID | None = None,
    ) -> ID:
        id = self.identify(key, chunk)
        if i is not None:
            assert i == id, (i, id)
        self.data[(key, chunk)] = val
        return i

@dataclass(frozen=True)
class ExampleKeyChunk(
    KeyChunk[int, str, bytes, str, np.ndarray, float]
):
    """MemoryKeyChunk[int, str, bytes, str, np.ndarray, float]"""
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes] = bytes
    type_key_o: Type[str] = str
    type_val_o: Type[np.ndarray] = np.ndarray
    type_val_t: Type[float] = float

    def init_val(self) -> np.ndarray:
        return np.zeros((1,))[1:]

    def chunk_val(self, acc: np.ndarray, tail: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        size = self.config.size
        if acc.size >= size * 2:
            if tail:
                # tail, body
                return acc[-size:], acc[:-size]
            else:
                # head, body
                return acc[:size], acc[size:]
        return acc, None

    def append_val(
        self, val: float, acc: np.ndarray, tail: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        acc = np.concat(
            (acc, np.array([val]))
            if tail
            else (np.array([val]), acc)
        )
        return self.chunk_val(acc, tail=tail)

    def extend_val(
        self, val: np.ndarray, acc: np.ndarray, tail: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        acc = np.concat(
            (acc, val) if tail else (val, acc)
        )
        return self.chunk_val(acc, tail=tail)

    def concat_val(
        self, val: np.ndarray, acc: np.ndarray, tail: bool = True
    ) -> np.ndarray:
        return np.concat(
            (acc, val) if tail else (val, acc)
        )

    def encode_key_type(self, t: Type[str]) -> Type[str]:
        return t

    def encode_val_type(
        self, t: Type[np.ndarray]
    ) -> Type[bytes]:
        return bytes

    def encode_key(self, k: str) -> str:
        return k.upper()

    def encode_val(self, k: str, v: np.ndarray) -> bytes:
        return v.tobytes()

    def decode_key(self, k: str) -> str:
        return k.lower()

    def decode_val(self, k: str, v: bytes) -> np.ndarray:
        # NOTE: assumes 1d only
        return np.frombuffer(v)

@final
@dataclass(frozen=True)
class ExampleMemoryKeyChunk(
    ExampleKeyChunk,
    MemoryKeyChunk[int, str, bytes, str, np.ndarray, float]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> v_arr = lambda v: np.array([v])
    >>> kv = ExampleMemoryKeyChunk(
    ...     on_update = lambda k, v, c: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k, HEAD)
    >>> v = kv.set(k, v_arr(0.), HEAD)
    >>> assert v == v_arr(0.) == kv.get(k, HEAD)
    >>> assert (0, v_arr(0.)) == kv.iget(k, HEAD)
    >>> _ = kv.set(j, v_arr(1.), HEAD)
    >>> _ = kv.set(k, v_arr(2.), HEAD)
    >>> assert (1, v_arr(1.)) == kv.iget(j, HEAD)
    >>> assert (0, v_arr(2.)) == kv.iget(k, HEAD)
    >>> assert updated == [(k, v_arr(2.))]
    >>> assert kv.data == {
    ...     ("K", HEAD): v_arr(2.).tobytes(),
    ...     ("J", HEAD): v_arr(1.).tobytes(),
    ... }
    >>> _ = kv.append_head(k, 1.)
    >>> _ = kv.append_head(k, 0.)
    >>> _ = kv.extend_head(k, np.array([2., 1.]))
    >>> _ = kv.append_tail(j, 2.)
    >>> _ = kv.append_tail(j, 3.)
    >>> _ = kv.extend_tail(j, np.array([2., 1., 0.]))
    >>> _ = kv.append_head(j, 0.)
    >>> kv.get(k, HEAD), kv.get(k, BODY)
    (array([2.]), array([1., 0., 1., 2.]))
    >>> kv.get(j, HEAD), kv.get(j, BODY), kv.get(j, TAIL)
    (array([0.]), array([1., 2., 3., 2., 1.]), array([0.]))
    """
    config: KeyChunkConfig = KeyChunkConfig(size=1)

# NOTE: typing examples
i, v = ExampleMemoryKeyChunk().iselect("", HEAD)
a = ExampleMemoryKeyChunk().get("", HEAD)

__all__ = [
    "START",
    "FIRST",
    "HEAD",
    "MIDDLE",
    "BODY",
    "END",
    "LAST",
    "FOOT",
    "TAIL",
    "KeyChunkConfig",
    "KeyChunkStore",
    "KeyChunkIO",
    "KeyChunk",
    "MemoryKeyChunk",
    "ExampleKeyChunk",
]