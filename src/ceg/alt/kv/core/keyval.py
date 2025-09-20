
from typing import Type, Generic, Literal, overload, Callable, TypeGuard, Protocol
from dataclasses import dataclass, replace

from .t import (
    _, ID, KeyI, KeyO, ValI, ValO
)
from .store import KeyValStore
from .io import KeyValIO

from . import store
from . import io

# --------------------------------------


@dataclass(frozen=True)
class KeyVal(Generic[ID, KeyI, ValI, KeyO, ValO]):
    store: KeyValStore[ID, KeyI, ValI]
    io: KeyValIO[KeyI, ValI, KeyO, ValO]


@dataclass(frozen=True)
class ExKV(KeyVal[int, float, str, bool, str]):
    pass



def is_singular_val(
    self, val: ValO | tuple[ValO, ...]
) -> TypeGuard[ValO]:
    return not isinstance(val, tuple)

def is_singular_val_type(
    self, val_type: Type[ValO] | dict[str, Type[ValO]]
) -> TypeGuard[Type[ValO]]:
    return not isinstance(val_type, dict)

def is_singular_val_func(
    self,
    func: (
        Callable[[KeyO, ValO, int | None], None]
        | Callable[[KeyO, dict[str, ValO], int | None], None]
    ),
    val: ValO | tuple[ValO, ...],
) -> TypeGuard[Callable[[KeyO, ValO, int | None], None]]:
    return not isinstance(val, tuple)

def is_not_singular_val(
    self, val: ValO | tuple[ValO, ...]
) -> TypeGuard[tuple[ValO, ...]]:
    return not self.is_singular_val(val)

def is_not_singular_val_type(
    self, val_type: Type[ValO] | dict[str, Type[ValO]]
) -> TypeGuard[dict[str, Type[ValO]]]:
    return not self.is_singular_val_type(val_type)
    
def is_not_singular_val_func(
    self,
    func: (
        Callable[[KeyO, ValO, int | None], None]
        | Callable[[KeyO, dict[str, ValO], int | None], None]
    ),
    val: ValO | tuple[ValO, ...],
) -> TypeGuard[Callable[[KeyO, dict[str, ValO], int | None], None]]:
    return not self.is_singular_val_func(func, val)


@overload
def get(
    self,
    key: KeyO,
    val: Type[ValO],
    chunk: int | None = None,
    identify: Literal[False] = False,
    # TODO: default value if not found
) -> ValO | None: ...

@overload
def get(
    self,
    key: KeyO,
    val: dict[str, Type[ValO]],
    chunk: int | None = None,
    identify: Literal[False] = False,
    # TODO: default value if not found
) -> dict[str, ValO] | None: ...

@overload
def get(
    self,
    key: KeyO,
    val: Type[ValO],
    chunk: int | None = None,
    identify: bool = True,
    # TODO: default value if not found
) -> tuple[ID | None, ValO | None]: ...

@overload
def get(
    self,
    key: KeyO,
    val: dict[str, Type[ValO]],
    chunk: int | None = None,
    identify: bool = True,
    # TODO: default value if not found
) -> tuple[ID | None, dict[str, ValO] | None]: ...

def get(
    self,
    key: KeyO,
    val: dict[str, Type[ValO]] | Type[ValO],
    chunk: int | None = None,
    identify: bool = False,
    # TODO: default value if not found
# ) -> (
#     ValO
#     | dict[str, ValO]
#     | tuple[int | None, ValO | None]
#     | tuple[int | None, dict[str, ValO] | None]
#     | None
):
    if chunk is None:
        assert self.store.chunked is None, self.store
    else:
        assert self.store.chunked
    if isinstance(val, dict):
        assert not self.store.singular()
    else:
        assert self.store.singular()
    key_i = self.io.encode_key(key)
    if identify and isinstance(val, dict):
        i, val_i = self.store.select(
            key_i, 
            identify=True,
            field=tuple(val.keys()),
            chunk=chunk,
        )
        if val_i is None:
            return i, val_i
        # assert isinstance(val_i, tuple), val_i
        # not needed for typing but useful sanity check?
        return i, {
            k: self.io.decode_val(v, t)
            for (k, t), v in zip(val.items(), val_i)
        }
    elif identify:
        assert self.is_singular_val_type(val), val
        i, val_i = self.store.select(
            key_i, 
            identify=True,
            chunk=chunk,
        )
        if val_i is None:
            return i, val_i
        return i, self.io.decode_val(val_i, val)
    elif isinstance(val, dict):
        assert not identify, self
        val_i = self.store.select(
            key_i, 
            field=tuple(val.keys()),
            chunk=chunk,
        )
        if val_i is None:
            return val_i
        return {
            k: self.io.decode_val(v, t)
            for (k, t), v in zip(val.items(), val_i)
        }
    else:
        assert self.is_singular_val_type(val), val
        val_i = self.store.select(
            key_i, 
            chunk=chunk,
        )
        if val_i is None:
            return val_i
        return self.io.decode_val(val_i, val)

@overload
def set(
    self,
    key: KeyO,
    val: ValO,
    val_type: Type[ValO],
    chunk: int | None = None,
    on_create: Callable[
        [KeyO, ValO, int | None], None
    ] | None=None,
    on_update: Callable[
        [KeyO, ValO, int | None], None
    ] | None=None,
    identify: Literal[False] = False,
) -> ValO: ...

@overload
def set(
    self,
    key: KeyO,
    val: tuple[ValO, ...],
    val_type: dict[str, Type[ValO]],
    chunk: int | None = None,
    on_create: Callable[
        [KeyO, dict[str, ValO], int | None], None
    ] | None=None,
    on_update: Callable[
        [KeyO, dict[str, ValO], int | None], None
    ] | None=None,
    identify: Literal[False] = False,
) -> dict[str, ValO]: ...

@overload
def set(
    self,
    key: KeyO,
    val: ValO,
    val_type: Type[ValO],
    chunk: int | None = None,
    on_create: Callable[
        [KeyO, ValO, int | None], None
    ] | None=None,
    on_update: Callable[
        [KeyO, ValO, int | None], None
    ] | None=None,
    identify: bool = True,
) -> tuple[ID, ValO]: ...

@overload
def set(
    self,
    key: KeyO,
    val: tuple[ValO, ...],
    val_type: dict[str, Type[ValO]],
    chunk: int | None = None,
    on_create: Callable[
        [KeyO, dict[str, ValO], int | None], None
    ] | None=None,
    on_update: Callable[
        [KeyO, dict[str, ValO], int | None], None
    ] | None=None,
    identify: bool = True,
) -> tuple[ID, dict[str, ValO]]: ...

def set(
    self,
    key: KeyO,
    val: tuple[ValO, ...] | ValO,
    val_type: dict[str, Type[ValO]] | Type[ValO],
    chunk: int | None = None,
    on_create: Callable[
        [KeyO, ValO, int | None], None
    ] | Callable[
        [KeyO, dict[str, ValO], int | None], None
    ] | None=None,
    on_update: Callable[
        [KeyO, ValO, int | None], None
    ] | Callable[
        [KeyO, dict[str, ValO], int | None], None
    ] | None=None,
    # TODO: on create / update need prev value maybe?
    identify: bool = False,
# ) -> (
# ...
):
    if chunk is None:
        assert self.store.chunked is None, self.store
    else:
        # TODO: assert chunk in [0, 1, 2]
        # first, body, last
        assert self.store.chunked

    # TODO: if chunk, assert same length, or full (no partial, if multi field)

    # TODO: later on, can take a custom is eq funciton for eg. large blobs to have quicker / custom implementation

    # and even possibly store some kind of hash / quicksum to check, needs to guarantee match is the same

    # and then possibly if diff, can still query? i guess the question there is how often your upates are no-op

    # NOTE: ie. multi field the chunsk hve to be consistent lengths, so can do partial update, but it cant change len

    # where len, append, extend etc. are all defined on the IO (the business logic)

    if isinstance(val_type, dict):
        assert not self.store.singular(), self.store
        assert self.is_not_singular_val(val), self

        i, val_o = self.get(key, val_type, identify=True)
    
        if i is not None and (
            val_o is None and val is None
            or (
                val_o is not None
                and tuple(val_o.values()) == val
            )
        ):
            return i, val
        
        key_i = self.io.encode_key(key)

        val_i = tuple(map(self.io.encode_val, val))
        val_d = dict(zip(val_type.keys(), val))
        
        if i is None:
            if on_create is not None:
                assert self.is_not_singular_val_func(on_create, val)
                on_create(key, val_d, chunk)
            i = self.store.insert(
                key_i, 
                val_i, 
                field=tuple(val_type.keys()),
                chunk=chunk
            )
        elif on_update is not None:
            assert self.is_not_singular_val_func(on_update, val)
            on_update(key, val_d, chunk)
            i = self.store.update(
                key_i,
                val_i,
                field=tuple(val_type.keys()),
                chunk=chunk,
                i=i
            )
        else:
            raise ValueError((key, val, on_update))
        
        if identify:
            return i, val_d
        return val_d
    else:
        assert self.store.singular(), self.store
        assert self.is_singular_val_type(val_type), self
        assert self.is_singular_val(val), self
        
        i, val_o = self.get(key, val_type, identify=True)

        if i is not None and (
            val_o is None and val is None
            or val_o == val
        ):
            return i, val

        key_i = self.io.encode_key(key)
        val_i = self.io.encode_val(val)

        if i is None:
            if on_create is not None:
                assert self.is_singular_val_func(on_create, val)
                on_create(key, val, chunk)
            i = self.store.insert(key_i, val_i, chunk=chunk)
        elif on_update is not None:
            assert self.is_singular_val_func(on_update, val)
            on_update(key, val, chunk)
            i = self.store.update(
                key_i, val_i, field=None, chunk=chunk, i=i
            )
        else:
            raise ValueError((key, val, on_update))
        
        if identify:
            return i, val
        return val

# TODO: get_many (return dict of the above by key)

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