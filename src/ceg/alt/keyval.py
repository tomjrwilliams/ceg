
from typing import Type, Generic, TypeVar, Literal, cast, overload, Callable, TypeGuard
from dataclasses import dataclass, replace

from ..apis import sqlite

# --------------------------------------

# backends determine permissive i types, what goes in
# I is fixed per backend (store)

# front end determines o types, map to i (both key and value)
# o ideally is broad as possible: np, polars, etc.

# o is business logic, needs to implement codign for whatever output (business) types you have, given the input stores you intend to use

ID = TypeVar("ID")
KeyI = TypeVar("KeyI")
ValI = TypeVar("ValI")
KeyO = TypeVar("KeyO")
ValO = TypeVar("ValO")

# --------------------------------------

@dataclass(frozen=True)
class KeyValStoreConfig:
    pass


@dataclass(frozen=True)
class KeyValStore(Generic[ID, KeyI, ValI]):
    key_type: Type[KeyI]
    val_type: Type[ValI] | dict[str, Type[ValI]]
    
    config: KeyValStoreConfig

    chunked: int | None = None # indicates chunk size

    def singular(self):
        return not isinstance(self.val_type, dict)

    def keys(self, field: tuple[str, ...]) -> tuple[str, ...]:
        if len(field):
            return field
        assert isinstance(self.val_type, dict), self
        return tuple(self.val_type.keys())

    def prepare(self) -> None: ...

    def identify(self, key: KeyI) -> None | ID: ...

    @overload
    def select(
        self,
        key: KeyI, 
        identify: Literal[False] = False,
        field: str | None = None,
        chunk: int | None = None,
    ) -> None | ValI: ...

    @overload
    def select(
        self,
        key: KeyI, 
        identify: Literal[False] = False,
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> None | tuple[ValI, ...]: ...

    @overload
    def select(
        self, 
        key: KeyI, 
        identify: bool = True,
        field: str | None = None,
        chunk: int | None = None,
    ) -> tuple[
        ID | None, ValI | None
    ]: ...

    @overload
    def select(
        self, 
        key: KeyI, 
        identify: bool = True,
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> tuple[
        ID | None, tuple[ValI, ...] | None
    ]: ...


    def select(
        self, 
        key: KeyI, 
        identify: bool = False,
        field: tuple[str, ...] | str | None = None,
        chunk: int | None = None,
    ) -> None | ValI: ...

    # TODO: field is either tuple or none
    # if not singular, must pass dict (even if len =1)

    @overload
    def insert(
        self, 
        key: KeyI, 
        val: ValI,
        field: str | None = None,
        chunk: int | None = None,
    ) -> ID: ...

    @overload
    def insert(
        self, 
        key: KeyI, 
        val: tuple[ValI, ...],
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> ID: ...

    def insert(
        self, 
        key: KeyI, 
        val: tuple[ValI, ...] | ValI,
        field: tuple[str, ...] | str | None = None,
        chunk: int | None = None,
    ) -> ID: ...

    @overload
    def update(
        self, 
        key: KeyI, 
        val: ValI, 
        field: str | None = None,
        chunk: int | None = None,
        i: ID | None = None
    ) -> ID: ...
    @overload
    def update(
        self, 
        key: KeyI, 
        val: tuple[ValI, ...],
        field: tuple[str, ...] = (),
        chunk: int | None = None,
        i: ID | None = None
    ) -> ID: ...
    def update(
        self, 
        key: KeyI, 
        val: tuple[ValI, ...] | ValI,
        field: tuple[str, ...] | str | None = None,
        chunk: int | None = None,
        i: ID | None = None
    ) -> ID: ...

# --------------------------------------

@dataclass(frozen=True)
class KeyValIO(Generic[KeyI, ValI, KeyO, ValO]):
    fields: dict[str, ValO] | None = None
    
    def encode_key_type(self, t: Type[KeyO]) -> Type[KeyI]: ...

    def encode_val_type(self, t: Type[ValO]) -> Type[ValI]: ...

    def encode_key(
        self, key: KeyO
    ) -> KeyI: ...

    def encode_val(
        self, val: ValO
    ) -> ValI: ...

    def decode_key(
        self, key: KeyI, t: Type[KeyO]
    ) -> KeyO: ...

    def decode_val(
        self, val: ValI, t: Type[ValO]
    ) -> ValO: ...

# --------------------------------------

@dataclass(frozen=True)
class KeyVal(Generic[ID, KeyI, ValI, KeyO, ValO]):
    store: KeyValStore[ID, KeyI, ValI] # chunk size here
    io: KeyValIO[KeyI, ValI, KeyO, ValO]

    # TODO: innately multi valued
    # if you have optional string for select, insert, update?

    # TODO: ideally chunked also then just takes an extra arg ot the select / insert / update for the chunk (as well as the col for multi vals)
    # and then set is unchanged, it just takes the extra arg optionally
    # and we can implement append in a subclass

    # encode / decode val then optionally take the field name?

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

# TODO: key vals, multiple val cols, named in spec
# can query individually, set jointly (?) - or even per?

# and key chunked vals (so i guess chunked is rather the col data structure)?

# where chunked means you can ask for first or last chunk, and set just that

# so it's up to you to ask for first / last (or snap), and use that to decide if a given update is incremental append, or requires re-set all the chunks

# i guess ideally you re-merge all the non first last after each new chunk
# so you only have first, body, last in terms of chunks

# TODO: how do chunks work with fields?

# ie. are the chunks across field values, or can different fields have different chunk ranges?

# --------------------------------------

# TODO: specific sqlite valid types, so not a generic anymore?


@dataclass(frozen=True)
class SQLiteConfig(KeyValStoreConfig):
    
    db: str
    name: str
    # if plural, dict, then overload on string literals when subclass (?)
    table: sqlite.Table | None = None
    spec: dict | None = None


@dataclass(frozen=True)
class SQLite(KeyValStore[ID, KeyI, ValI]):
    config: SQLiteConfig

    # TODO: write / update time needed if going to do timed expiry?
    # and anyway i guess useful context

    def prepare(self, io: KeyValIO):
        # if type is array -> bytes
        # TODO STRICT
        t = sqlite.Table.new(
            self.config.name,
            key = str, # TODO encode
            value = io.encode_val_type(self.val_type),
            # TODO: dict, col names
            unique=("value",)
        ).with_id()
        # TODO: if exists, assert types match!
        # TODO: add chunk to spec if given, assert there if exists


        return replace(
            self, 
            config = replace(
                self.config,
                table=t,
                spec=t.create(self.config.db)
            )
        )

    def where_id_eq(self, id: int):
        assert self.config.table is not None, self
        return {
            cast(sqlite.TTable, self.config.table): dict(
                id=f"={id}"
            )
            # str wrap in quotes?
        }

    def where_key_eq(self, key: KeyI):
        assert self.config.table is not None, self
        return {
            cast(sqlite.TTable, self.config.table): dict(
                key=f"={key}"
            )
            # str wrap in quotes?
        }

    # TODO: default value if not found
    def select(
        self, key # TODO: take col, chunk
    ) -> tuple[int | None, ValI | None]:

        assert self.config.table is not None, self
        res = sqlite.select(
            self.config.db,
            sqlite.query.select(self.config.table, "id", "value").acc,
            where=self.where_key_eq(key)
        )

        if not len(res):
            return None, None
        assert len(res) == 1, res
        r = res[0]
        id: int = r[f"{self.config.name}.id"]
        val = r[f"{self.config.name}.value"]

        # TODO; decode from string as well?
        return id, val

    def insert(
        self,
        key: KeyI,
        val: ValI,
    ) -> ID:
        assert self.config.table is not None, self

        res = sqlite.insert(
            self.config.db,
            self.config.table,
            [dict(key=key, val=val)] # type: ignore
        )

        return res[0]["id"]
    
    def update(self, key: KeyI, val: ValI, i: ID | None = None):
        if i is None:
            res = sqlite.update(
                self.config.db,
                self.config.name,
                dict(val=val),
                where=self.where_key_eq(key)
            )
        else:
            res = sqlite.update(
                self.config.db,
                self.config.name,
                dict(val=val),
                where=self.where_id_eq(i)
            )
        return res

# --------------------------------------

# TODO: key vals, multiple val cols, named in spec
# can query individually, set jointly (?) - or even per?

# and key chunked vals (so i guess chunked is rather the col data structure)?

# where chunked means you can ask for first or last chunk, and set just that

# so it's up to you to ask for first / last (or snap), and use that to decide if a given update is incremental append, or requires re-set all the chunks

# i guess ideally you re-merge all the non first last after each new chunk
# so you only have first, body, last in terms of chunks

# --------------------------------------

class Postgres(KeyValStore):
    pass

class S3(KeyValStore):
    pass

class CSV(KeyValStore):
    pass

class JSON(KeyValStore):
    pass

class Parquet(KeyValStore):
    pass

# clickhouse, redis, etc.

# --------------------------------------
