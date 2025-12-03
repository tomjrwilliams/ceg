from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
    Any,
    overload,
    cast,
    final,
)
import functools
import io
from pydantic import (
    TypeAdapter,
    BaseModel,
    Field,
)
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

from ..apis import t
from ..apis.t import ndarray, pl as pl_types
from ..apis import kv

import os
TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")

@dataclass(frozen=True)
class ValueParser:
    value: Value

    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)

@functools.lru_cache(maxsize=128, typed=True)
def value_parser(cls: Type[Value]) -> Type[ValueParser]:
    return dataclass(frozen=True)(type(
        cls.__name__ + "Parser",
        (ValueParser,),
        {
            "value": Field(discriminator="type"),
            "__annotations__": {"value": cls}
        }
    ))


@dataclass(frozen=True)
class Value:
    """
    >>> @dataclass(frozen=True)
    ... class ValueA(Value):
    ...     a: int
    ...     type: Literal['A'] = Value.type_field("A")
    >>> @dataclass(frozen=True)
    ... class ValueB(Value):
    ...     b: float
    ...     type: Literal['B'] = Value.type_field("B")
    >>> @dataclass(frozen=True)
    ... class ValueParserAB(ValueParser):
    ...     value: ValueA | ValueB = Value.union_field()
    >>> a, b = ValueA("a", 1), ValueB("b", 1.)
    >>> _ = ValueParserAB(a) # init parser type
    >>> Value.parse_json(a.dump_bytes(), ValueParserAB)
    ValueA(obj='a', a=1)
    >>> Value.parse_json(b.dump_bytes(), ValueParserAB) 
    ValueB(obj='b', b=1.0)
    """
    obj: str # NOTE: object path

    # NOTE: can have other fields for json-type value objects
    # NOTE: meta (calendar etc. for time series) should appear in snap instead

    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parser(cls) -> Type[ValueParser]:
        return value_parser(cls)

    def dump_bytes(self) -> bytes:
        return self.parser()(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[ValueParser]
    ) -> Value:
        parser: ValueParser = into.ta().validate_json(data)
        return parser.value
    
    @classmethod
    def get(
        cls,
        key: str,
        kv: ValueKeyVal,
    ) -> Value | None:
        # TODO: iget
        return kv.get(key)

    def set(
        self,
        kv: ValueKeyVal,
    ) -> Value:
        # TODO: iset
        return kv.set(self.obj, self)

@dataclass(frozen=True)
class ValueKeyVal(
    kv.KeyValStrKeyIO[int, t.ValI, Value]
):
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes] = bytes
    type_key_o: Type[str] = str
    type_val_o: Type[Value] = Value

    parser: Type[ValueParser] | None = None

    # TODO: get and set are generic, they just need the relevant encode and decode methods set in the types passed to the relevant key val for the value type

V_KV = TypeVar("V_KV", bound=ValueKeyVal)

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class ValueKeyValJSON(ValueKeyVal[bytes]):

    type_val_i: Type[bytes] = bytes
    
    def encode_val_type(self, t: Type[Value]) -> Type[bytes]:
        return bytes

    def encode_val(self, k: Any, v: Value) -> bytes:
        # TODO: extract any non json vals?
        # ie. if store numpy array directly?
        return v.dump_bytes()

    def decode_val(self, k: Any, v: bytes) -> Value:
        # TODO: extract any non json vals?
        # ie. if store numpy array directly?
        assert self.parser is not None, self
        return Value.parse_json(v, into=self.parser)

@dataclass(frozen=True)
class ValueKeyValJSONSQLite(
    ValueKeyValJSON,
    kv.SQLiteKeyVal[int, str, bytes, str, Value]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> @dataclass(frozen=True)
    ... class ValueEx(Value):
    ...     a: int
    ...     type: Literal['Ex'] = Value.type_field("Ex")
    >>> kv = ValueKeyValJSONSQLite.new(
    ...     TEST_DB, 
    ...     parser = ValueEx.parser(),
    ...     on_update = lambda k, v: updated.append((k, v))
    ... )
    >>> kv.get(k)
    >>> v = kv.set(k, ValueEx(k, 0.0))
    >>> assert v == ValueEx(k, 0.0) == kv.get(k), kv.get(k)
    >>> assert (1, ValueEx(k, 0.0)) == kv.iget(k), kv.iget(k)
    >>> v = kv.set(j, ValueEx(j, 1.0))
    >>> v = kv.set(k, ValueEx(k, 2.0))
    >>> assert (2, ValueEx(j, 1.0)) == kv.iget(j)
    >>> assert (1, ValueEx(k, 2.0)) == kv.iget(k)
    >>> assert updated == [(k, ValueEx(k, 2.0))]
    """

    @classmethod
    def new(
        cls,
        db: str, 
        parser: Type[ValueParser],
        table: str = "ValuesJSON",
        on_update: Callable[[str, Value], None] | None = None
    ) -> ValueKeyValJSONSQLite:
        self = cls(config=kv.SQLiteKeyValConfig(
            db=db, name=table
        ), on_update=on_update, parser=parser)
        return cast(ValueKeyValJSONSQLite, self.prepare())


@dataclass(frozen=True)
class ValueNumPy(Value):
    arr: ndarray
    # TODO: n dims?

    @classmethod
    def new(cls, k: str, v: np.ndarray) -> ValueNumPy:
        return cls(k, ndarray.cast(v))

    @property
    def size(self) -> int:
        return self.arr.size

    def __getitem__(self, sl: slice) -> ValueNumPy:
        return ValueNumPy(self.obj, ndarray.cast(
            self.arr[sl]
        ))

@dataclass(frozen=True)
class ValueKeyChunkNumPyF64(
    kv.KeyChunk[
        int, str, bytes, str, ValueNumPy, float
    ]
):
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes]= bytes
    type_key_o: Type[str] = str
    type_val_o: Type[ValueNumPy] = ValueNumPy
    type_val_t: Type[float] = float

    def encode_key_type(self, t: Type[str]) -> Type[str]:
        return t

    def encode_val_type(self, t: Type[ValueNumPy]) -> Type[bytes]:
        return bytes

    def encode_key(self, k: str) -> str:
        return k

    def decode_key(self, k: str) -> str:
        return k

    def encode_val(self, k: str, v: ValueNumPy) -> bytes:
        return v.arr.tobytes()

    def decode_val(self, k: str, v: bytes) -> ValueNumPy:
        return ValueNumPy(k, ndarray.cast(np.frombuffer(v)))
    
    def init_val(self) -> ValueNumPy:
        return ValueNumPy("", ndarray.cast(np.zeros((1,))[1:]))

    # TODO: generalise these out into numpy IO?
    # implement in api core, not engine?

    def chunk_val(self, acc: ValueNumPy, tail: bool = True) -> tuple[ValueNumPy, ValueNumPy | None]:
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
        self, val: float, acc: ValueNumPy, tail: bool = True
    ) -> tuple[ValueNumPy, ValueNumPy | None]:
        acc = ValueNumPy.new(
            acc.obj, 
            np.concat(
                (acc.arr, np.array([val]))
                if tail
                else (np.array([val]), acc.arr)
            )
        )
        return self.chunk_val(acc, tail=tail)

    def extend_val(
        self, val: ValueNumPy, acc: ValueNumPy, tail: bool = True
    ) -> tuple[ValueNumPy, ValueNumPy | None]:
        acc = ValueNumPy.new(acc.obj, np.concat(
            (acc.arr, val.arr) if tail else (val.arr, acc.arr)
        ))
        return self.chunk_val(acc, tail=tail)

    def concat_val(
        self, val: ValueNumPy, acc: ValueNumPy, tail: bool = True
    ) -> ValueNumPy:
        return ValueNumPy.new(acc.obj, np.concat(
            (acc.arr, val.arr) if tail else (val.arr, acc.arr)
        ))

@final
@dataclass(frozen=True)
class ValueKeyChunkNumPySQLite( # TODO: f64?
    ValueKeyChunkNumPyF64,
    kv.SQLiteKeyChunk[
        int, str, bytes, str, ValueNumPy, float
    ]
):
    def set_from_df(
        self,
        df: pl.DataFrame,
        k: str,
        obj: str,
        chunk: int,
    ) -> ValueNumPy:
        # : F64_1D
        v_np: np.ndarray = df.select(k).to_numpy().T[0]
        v = ValueNumPy(obj, t.ndarray.cast(v_np))
        return self.set(obj, v, chunk)

    @classmethod
    def new(
        cls,
        db: str,
        size: int = 32,
        table: str = "ChunksNumPyF64",
    ) -> ValueKeyChunkNumPySQLite:
        self = cls(config=kv.SQLiteKeyChunkConfig(
             size=size, db=db,name=table
        ))
        return cast(ValueKeyChunkNumPySQLite, self.prepare())


@dataclass(frozen=True)
class ValuePlSeries(Value):
    ser: pl_types.Series
    
    def head(self, n: int) -> pl.Series:
        return self.ser.head(n)

    def tail(self, n: int) -> pl.Series:
        return self.ser.tail(n)

    @classmethod
    def new(cls, k: str, ser: pl.Series) -> ValuePlSeries:
        return cls(k, t.pl.Series.cast(ser))
    
    @classmethod
    def synthetic(cls, ser: pl.Series) -> ValuePlSeries:
        return cls.new("SYNTHETIC", ser)

    @property
    def size(self):
        return len(self.ser)

    def __getitem__(self, sl: slice) -> ValuePlSeries:
        return self.new(self.obj, self.ser[sl])
    

@dataclass(frozen=True)
class ValueKeyChunkPlSeriesF64(
    kv.KeyChunk[
        int, str, bytes, str, ValuePlSeries, float
    ]
):
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes]= bytes
    type_key_o: Type[str] = str
    type_val_o: Type[ValuePlSeries] = ValuePlSeries
    type_val_t: Type[float] = float

    def encode_key_type(self, t: Type[str]) -> Type[str]:
        return t

    def encode_val_type(self, t: Type[ValuePlSeries]) -> Type[bytes]:
        return bytes

    def encode_key(self, k: str) -> str:
        return k

    def decode_key(self, k: str) -> str:
        return k

    def encode_val(self, k: str, v: ValuePlSeries) -> bytes:
        return pl.DataFrame({k: v.ser}).serialize()

    def decode_val(
        self, k: str, v: bytes
    ) -> ValuePlSeries:
        df = pl.DataFrame.deserialize(io.BytesIO(v))
        return ValuePlSeries.new(k, df.get_column(k))
    
    def init_val(self) -> ValuePlSeries:
        return ValuePlSeries.new("", pl.Series("", [], dtype=pl.Float64))

    # TODO: generalise these out into numpy IO?
    # implement in api core, not engine?

    def chunk_val(self, acc: ValuePlSeries, tail: bool = True) -> tuple[ValuePlSeries, ValuePlSeries | None]:
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
        self, val: float, acc: ValuePlSeries, tail: bool = True
    ) -> tuple[ValuePlSeries, ValuePlSeries | None]:
        acc = ValuePlSeries.new(
            acc.obj, 
            acc.ser.extend_constant(val, 1)
            if tail
            else pl.Series(acc.obj, [val]).extend(acc.ser)
        )
        return self.chunk_val(acc, tail=tail)

    def extend_val(
        self, val: ValuePlSeries, acc: ValuePlSeries, tail: bool = True
    ) -> tuple[ValuePlSeries, ValuePlSeries | None]:
        acc = ValuePlSeries.new(
            acc.obj, 
            acc.ser.extend(val.ser)
            if tail
            else val.ser.extend(acc.ser)
        )
        return self.chunk_val(acc, tail=tail)

    def concat_val(
        self, val: ValuePlSeries, acc: ValuePlSeries, tail: bool = True
    ) -> ValuePlSeries:
        return ValuePlSeries(
            acc.obj, 
            acc.ser.extend(val.ser)
            if tail
            else val.ser.extend(acc.ser)
        )

@final
@dataclass(frozen=True)
class ValueKeyChunkPlSeriesSQLite( # TODO: f64?
    ValueKeyChunkPlSeriesF64,
    kv.SQLiteKeyChunk[
        int, str, bytes, str, ValuePlSeries, float
    ]
):
    
    def set_from_df(
        self,
        df: pl.DataFrame,
        k: str,
        obj: str,
        chunk: int = kv.BODY
    ) -> ValuePlSeries:
        v_s = df.get_column(k)
        v = ValuePlSeries.new(obj, v_s)
        return self.set(obj, v, chunk)
    
    @classmethod
    def new(
        cls,
        db: str,
        size: int = 32,
        table: str = "ChunksPlSeriesF64",
    ) -> ValueKeyChunkPlSeriesSQLite:
        self = cls(config=kv.SQLiteKeyChunkConfig(
            size=size, db=db,name=table
        ))
        return cast(ValueKeyChunkPlSeriesSQLite, self.prepare())


__all__ = [
    "ValueParser",
    "Value",
    "ValueKeyVal",
    "V_KV",
    "ValueKeyValJSON",
    "ValueKeyValJSONSQLite",
    "ValueKeyChunkNumPySQLite",
    "ValueNumPy",
    "ValueKeyChunkPlSeriesSQLite",
    "ValuePlSeries",
]