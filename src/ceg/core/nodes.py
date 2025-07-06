from __future__ import annotations

import abc
from typing import (
    NamedTuple,
    Generic,
    Type,
    TypeVar,
    ParamSpec,
    Callable,
    Concatenate,
)
from typing import (
    Any,
    Iterable,
    Iterator,
    get_type_hints,
    get_origin,
    get_args,
    cast,
)

import datetime as dt

import numpy as np

from .types import dataclass, frozendict
from .refs import Ref, R, GraphInterface, Scope

#  ------------------

V = np.ndarray | bool | int | float | dt.date | dt.datetime | None

F = ParamSpec("F")
FRes = TypeVar("FRes")

#  ------------------

class Event(NamedTuple):
    """
    t: float
    ref: Ref.Any
    prev: Event | None
    """
    t: float
    ref: Ref.Any
    prev: Event | None

    @classmethod
    def new(
        cls,
        t: float,
        ref: Ref.Any,
        prev: Event | None = None,
    ):
        return cls(t, ref, prev=prev)

    @classmethod
    def zero(cls, ref: Ref.Any, prev: Event | None = None):
        return cls(0., ref, prev=prev)

#  ------------------


@dataclass(frozen=True)
class NodeInterface(abc.ABC, Generic[R]):
    type: str

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> R: ...

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> V: ...


N = TypeVar("N", bound=NodeInterface)

#  ------------------

@dataclass(frozen=True)
class Node_Null(NodeInterface[Ref.Any]):

    def pipe(
        self,
        f: Callable[Concatenate[Node_Null, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any | Ref.Any, slot: int | None = None
    ) -> Ref.Any:
        raise ValueError(cls)

    def __call__(self, event: Event, graph: GraphInterface):
        raise ValueError(self)


#  ------------------

@dataclass(frozen=True)
class Node_D0_Date(NodeInterface[Ref.D0_Date]):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D0_Date, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any, slot: int | None = None
    ) -> Ref.D0_Date:
        return Ref.d0_date(i, slot=slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> dt.date: ...


@dataclass(frozen=True)
class Node_D0_F64(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D0_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any, slot: int | None = None
    ) -> Ref.D0_F64:
        return Ref.d0_f64(i, slot=slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> float: ...


#  ------------------


@dataclass(frozen=True)
class Node_D1_Date(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D1_Date, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any, slot: int | None = None
    ) -> Ref.D1_Date:
        return Ref.d1_date(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class Node_D1_F64(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D1_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any, slot: int | None = None
    ) -> Ref.D1_F64:
        return Ref.d1_f64(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> np.ndarray: ...


#  ------------------


@dataclass(frozen=True)
class Node_D2_F64(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D2_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any, slot: int | None = None
    ) -> Ref.D2_F64:
        return Ref.d2_f64(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> np.ndarray: ...


#  ------------------

@dataclass(frozen=True)
class Node_D0_F64_3(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D0_F64_3, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any, slot: int | None = None
    ) -> Ref.D0_F64_3:
        return Ref.d0_f64_3(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> tuple[float, float, float]: ...

@dataclass(frozen=True)
class Node_D0_F64_4(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D0_F64_4, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any, slot: int | None = None
    ) -> Ref.D0_F64_4:
        return Ref.d0_f64_4(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> tuple[float, float, float, float]: ...


@dataclass(frozen=True)
class Node_D1_F64_D2_F64(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D1_F64_D2_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def ref(
        cls, i: int | Ref.Any, slot: int | None = None
    ) -> Ref.D1_F64_D2_F64:
        return Ref.d1_f64_d2_f64(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> tuple[np.ndarray, np.ndarray]: ...


#  ------------------


class Node:
    Any = NodeInterface

    D0_Date = Node_D0_Date
    Scalar_Date = Node_D0_Date

    D0_F64 = Node_D0_F64
    Scalar_F64 = Node_D0_F64

    D1_Date = Node_D1_Date
    Vector_Date = Node_D1_Date

    D1_F64 = Node_D1_F64
    Vector_F64 = Node_D1_F64

    D2_F64 = Node_D2_F64
    Matrix_F64 = Node_D2_F64

    D1_F64_D2_F64 = Node_D1_F64_D2_F64

    D0_F64_4 = Node_D0_F64_4
    D0_F64_3 = Node_D0_F64_3

    Null = Node_Null
    null = Node_Null("null")


#  ------------------


def rec_yield_param(k, v: Ref.Any | Iterable | Any):
    if isinstance(v, Ref.Any):
        yield (k, v.i, v.scope)
    elif isinstance(v, (tuple, Iterable)):
        yield from rec_yield_params(k, v)


def rec_yield_params(k: str, v: Iterable):
    if isinstance(v, dict):
        yield from rec_yield_params(k, v.keys())
        yield from rec_yield_params(k, v.values())
    elif isinstance(v, (tuple, Iterable)):
        for vv in v:
            yield from rec_yield_param(k, vv)


def yield_params(
    node: Node.Any,
) -> Iterator[tuple[str, int, Scope | None]]:
    for k in yield_param_keys(type(node)):
        v = getattr(node, k)
        yield from rec_yield_param(k, v)


def rec_yield_hint_types(hint):
    try:
        o = get_origin(hint)
        yield o
    except:
        pass
    try:
        args = get_args(hint)
        for a in args:
            if a == Ellipsis:
                continue
            yield from rec_yield_hint_types(a)
    except:
        pass
    yield hint


def yield_param_keys(t_kw: Type):
    seen: set[str] = set()
    for k, h in get_type_hints(t_kw).items():
        for h in rec_yield_hint_types(h):
            if k in seen:
                continue
            if not isinstance(h, type):
                continue
            if issubclass(h, Ref.Any):
                seen.add(k)
                yield k


#  ------------------
