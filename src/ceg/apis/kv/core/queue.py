from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Type,
    Generic,
    Literal,
    Callable,
    Protocol,
    overload,
    cast,
    final,
)
import operator
from functools import partial
import datetime
import datetime as dt
from dataclasses import dataclass, field

from ...t import *

gt = Literal["gt"]
ge = Literal["ge"]
lt = Literal["lt"]
le = Literal["le"]
eq = Literal["eq"]
ne = Literal["ne"]

V = TypeVar("V")

def comparator(comp: Comp, v: V) -> Callable[[V], bool]:
    op = getattr(operator, comp)
    return lambda val: op(val, v)

Comp = (
    gt
    | ge
    | lt
    | le
    | eq
    | ne
)


@dataclass(frozen=True)
class QueueConfig:
    pass


@dataclass(frozen=True)
class QueueStore(ABC, Generic[ID, DtI, ValI]):
    """QueueStore[ID, DtI, ValI]"""
    config: QueueConfig

    type_id: Type[ID]
    type_dt_i: Type[DtI]
    type_val_i: Type[ValI]

    @abstractmethod
    def prepare(self) -> QueueStore: ...
    # TODO: return self

    @abstractmethod
    def slice(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtI]] | None = None,
        head: bool = True,
    ) -> list[ValI]: ...

    @abstractmethod
    def islice(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtI]] | None = None,
        head: bool = True,
    ) -> list[tuple[ID, ValI, DtI]]: ...

    @abstractmethod
    def insert(
        self,
        val: ValI,
    ) -> tuple[ID, DtI]: ...


@dataclass(frozen=True)
class QueueIO(Generic[ID, DtI, ValI, DtO, ValO]):
    """QueueIO[ID, DtI, ValI, DtO, ValO]"""
    type_id: Type[ID]
    type_dt_i: Type[DtI]
    type_val_i: Type[ValI]
    type_dt_o: Type[DtO]
    type_val_o: Type[ValO]

    @abstractmethod
    def init_dt(self) -> DtO: ...

    @abstractmethod
    def encode_dt_type(self, t: Type[DtO]) -> Type[DtI]: ...

    @abstractmethod
    def encode_val_type(self, t: Type[ValO]) -> Type[ValI]: ...

    @abstractmethod
    def encode_dt(self, val: DtO) -> DtI: ...

    @abstractmethod
    def encode_val(self, val: ValO) -> ValI: ...

    @abstractmethod
    def decode_dt(
        self,
        dt: DtI,
    ) -> DtO: ...

    @abstractmethod
    def decode_val(
        self,
        val: ValI,
    ) -> ValO: ...

    def check_type_id(self, id: ID | None) -> ID:
        assert isinstance(id, self.type_id)
        return id

    def check_type_dt_i(self, dt_i: DtI) -> DtI:
        assert isinstance(dt_i, self.type_dt_i)
        return dt_i

    def check_type_dt_o(self, dt_o: DtO) -> DtO:
        assert isinstance(dt_o, self.type_dt_o)
        return dt_o
    
    def check_type_val_i(self, val_i: ValI) -> ValI:
        assert isinstance(val_i, self.type_val_i)
        return val_i
        
    def check_type_val_o(self, val_o: ValO) -> ValO:
        assert isinstance(val_o, self.type_val_o)
        return val_o

@dataclass(frozen=True)
class Queue(
    QueueIO[ID, DtI, ValI, DtO, ValO],
    QueueStore[ID, DtI, ValI],
):
    """Queue[ID, DtI, ValI, DtO, ValO]"""

    def encode_where_dt(
        self,
        where_dt: list[tuple[Comp, DtO]] | None = None,
    ) -> list[tuple[Comp, DtI]] | None:
        return (
            where_dt 
            if where_dt is None
            else [
                (comp, self.encode_dt(t))
                for comp, t in where_dt
            ]
        )

    def unzip_decode_slice(
        self,
        sl: list[tuple[ID, ValI, DtI]]
    ) -> tuple[list[ID], list[ValO], list[DtO]]:
        if not len(sl):
            return [], [], []
        i, v, t = cast(
            tuple[list[ID], list[ValI], list[DtI]],
            zip(*sl), 
        )
        return (
            list(i),
            list(map(self.decode_val, v)),
            list(map(self.decode_dt, t)), 
        )

    def head(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtO]] | None = None,
    ) -> list[ValO]:
        res = self.slice(
            n,
            where_i=where_i,
            where_dt = self.encode_where_dt(where_dt),
            head=True
        )
        return list(map(self.decode_val, res))

    def tail(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtO]] | None = None,
    ) -> list[ValO]:
        res = self.slice(
            n,
            where_i=where_i,
            where_dt = self.encode_where_dt(where_dt),
            head=False
        )
        return list(map(self.decode_val, res))

    def ihead(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtO]] | None = None,
    ) -> tuple[list[ID], list[ValO], list[DtO]]:
        res = self.islice(
            n,
            where_i=where_i,
            where_dt = self.encode_where_dt(where_dt),
            head=True
        )
        return self.unzip_decode_slice(res)

    def itail(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtO]] | None = None,
    ) -> tuple[list[ID], list[ValO], list[DtO]]:
        res = self.islice(
            n,
            where_i=where_i,
            where_dt = self.encode_where_dt(where_dt),
            head=False
        )
        return self.unzip_decode_slice(res)

    def append(
        self,
        val: ValO,
    ) -> ValO:
        _, v, _ = self.iappend(val)
        return v

    def iappend(
        self,
        val: ValO,
    ) -> tuple[ID, ValO, DtO]:
        val_i = self.encode_val(val)
        i, t = self.insert(val_i)
        return i, val, self.decode_dt(t)

    # TODO: append_many
    # TODO: == extend?


@dataclass(frozen=True)
class MemoryQueue(Queue[ID, DtI, ValI, DtO, ValO]):
    data: list[tuple[ValI, DtI]] = field(
        default_factory=list
    )

    def prepare(self) -> QueueStore:
        return self

    def slice(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtI]] | None = None,
        head: bool = True,
    ) -> list[ValI]:
        i_dt_vs = self.islice(
            n=n,
            where_i=where_i,
            where_dt=where_dt,
            head=head
        )
        if not len(i_dt_vs):
            return []
        i, vs, dt = zip(*i_dt_vs)
        return cast(list[ValI], list(vs))

    def islice(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtI]] | None = None,
        head: bool = True,
    ) -> list[tuple[ID, ValI, DtI]]:
        acc_n: int = 0
        acc: list[tuple[ID, ValI, DtI]] = []
        comp_i = [] if where_i is None else [
            comparator(comp, i) for comp, i in where_i
        ]
        comp_dt = [] if where_dt is None else [
            comparator(comp, dt) for comp, dt in where_dt
        ]
        for i, (v, dt) in (
            enumerate(self.data) 
            if head 
            else reversed(list(enumerate(self.data)))
        ):
            i = cast(ID, i)
            if not (all((
                wh(i) for wh in comp_i
            )) and all((
                wh(dt) for wh in comp_dt
            ))):
                continue
            acc.append((i, v, dt))
            acc_n += 1
            if acc_n == n:
                return acc
        return acc

    def insert(
        self,
        val: ValI,
    ) -> tuple[ID, DtI]:
        dt = self.encode_dt(self.init_dt())
        i = cast(ID, len(self.data))
        self.data.append((val, dt,))
        return i, dt


@dataclass(frozen=True)
class QueueTimestampAsFloat(
    Queue[ID, float, ValI, dt.datetime, ValO],
    # Generic[ID, ValI, ValO]
):
    """QueueTimestampAsFloat[ID, ValI, ValO]"""

    type_dt_i: Type[float]
    type_dt_o: Type[dt.datetime]

    def init_dt(self) -> dt.datetime:
        return dt.datetime.now()

    def encode_dt_type(self, t: Type[dt.datetime]) -> Type[float]:
        return float

    def encode_dt(self, dt: dt.datetime) -> float:
        return dt.timestamp()

    def decode_dt(self, dt: float) -> dt.datetime:
        return datetime.datetime.fromtimestamp(dt)

@dataclass(frozen=True)
class ExampleQueue(QueueTimestampAsFloat[int, int, float]):

    type_id: Type[int] = int
    type_dt_i: Type[float] = float
    type_val_i: Type[int] = int
    type_dt_o: Type[dt.datetime] = dt.datetime
    type_val_o: Type[float] = float

    def encode_dt_type(self, t: Type[dt.datetime]) -> Type[float]:
        return float

    def encode_val_type(self, t: Type[float]) -> Type[int]:
        return int

    def encode_dt(self, dt: dt.datetime) -> float:
        return dt.timestamp()

    def encode_val(self, v: float) -> int:
        return int(v)

    def decode_dt(self, dt: float) -> dt.datetime:
        return datetime.datetime.fromtimestamp(dt)

    def decode_val(self, v: int) -> float:
        return float(v)

@final
@dataclass(frozen=True)
class ExampleMemoryQueue(
    ExampleQueue,
    MemoryQueue[int, float, int, dt.datetime, float]
):
    """
    >>> import time
    >>> j, k = ("j", "k")
    >>> kv = ExampleMemoryQueue().prepare()
    >>> kv.head(1)
    []
    >>> kv.tail(1)
    []
    >>> v = kv.append(0.)
    >>> assert v == 0.0 == kv.head(1)[0]
    >>> assert ([0], [0.0]) == kv.ihead(1)[:2]
    >>> time.sleep(0.1)
    >>> dt = datetime.datetime.now()
    >>> time.sleep(0.1)
    >>> _ = kv.append(1.)
    >>> _ = kv.append(2.)
    >>> kv.tail(n=3, where_dt=[("gt", dt)])
    [2.0, 1.0]
    >>> kv.head(n=3, where_dt=[("gt", dt)])
    [1.0, 2.0]
    >>> kv.tail(n=3, where_dt=[("lt", dt)])
    [0.0]
    >>> kv.head(n=3, where_dt=[("lt", dt)])
    [0.0]
    >>> kv.tail(n=3, where_i=[("ge", 1)])
    [2.0, 1.0]
    >>> kv.head(n=3, where_i=[("ge", 1)])
    [1.0, 2.0]
    >>> kv.tail(n=3, where_i=[("le", 1)])
    [1.0, 0.0]
    >>> kv.head(n=3, where_i=[("le", 1)])
    [0.0, 1.0]
    """
    config: QueueConfig = QueueConfig()

__all__ = [
    "gt",
    "ge",
    "lt",
    "le",
    "eq",
    "ne",
    "Comp",
    "QueueConfig",
    "QueueStore",
    "QueueIO",
    "Queue",
    "QueueTimestampAsFloat",
    "MemoryQueue",
    "ExampleQueue",
]