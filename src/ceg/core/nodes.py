from __future__ import annotations

import abc
from typing import (
    Generic,
    ClassVar,
    Type,
    NamedTuple,
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

from dataclasses import dataclass
from heapq import heapify, heappush, heappop

import datetime as dt

from frozendict import frozendict
import numpy as np

from .refs import Ref, R, GraphInterface, Scope


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
    for k in node.DEF.params:
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


def yield_param_keys(t_kw: Type[NamedTuple]):
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


class Defn(NamedTuple):
    name: str
    params: tuple[str, ...]

#  ------------------

N = TypeVar("N", bound=NamedTuple)

V = np.ndarray | float

F = ParamSpec("F")
FRes = TypeVar("FRes")


class NodeInterface(abc.ABC, Generic[R, N]):

    DEF: ClassVar[Defn] = Defn("NULL", ())

    @abc.abstractmethod
    def pipe(
        self,
        f: Callable[
            Concatenate[NodeInterface[R, N], F], FRes
        ],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes: ...

    @abc.abstractmethod
    def ref(self, i: int, slot: int | None = None) -> R: ...

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> V: ...


#  ------------------


class Node_NullKW(NamedTuple):
    pass


class Node_Null(
    Node_NullKW, NodeInterface[Ref.Any, Node_NullKW]
):

    def pipe(
        self,
        f: Callable[Concatenate[Node_Null, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(
        self, i: int, slot: int | None = None
    ) -> Ref.Any:
        raise ValueError(self)

    def __call__(self, event: Event, graph: GraphInterface):
        raise ValueError(self)


#  ------------------


class Node_D0_Date(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D0_Date, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(
        self, i: int, slot: int | None = None
    ) -> Ref.D0_Date:
        return Ref.D0_Date.new(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> dt.date: ...


class Node_D0_F64(NodeInterface):

    
    @classmethod
    @abc.abstractclassmethod
    def new(cls, **kwargs) -> Node_D0_F64: ...

    def pipe(
        self,
        f: Callable[Concatenate[Node_D0_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(
        self, i: int, slot: int | None = None
    ) -> Ref.D0_F64:
        return Ref.D0_F64.new(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> float: ...


#  ------------------


class Node_D1_Date(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D1_Date, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(
        self, i: int, slot: int | None = None
    ) -> Ref.D1_Date:
        return Ref.D1_Date.new(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> np.ndarray: ...


class Node_D1_F64(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D1_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(
        self, i: int, slot: int | None = None
    ) -> Ref.D1_F64:
        return Ref.D1_F64.new(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> np.ndarray: ...


#  ------------------


class Node_D2_F64(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D2_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(
        self, i: int, slot: int | None = None
    ) -> Ref.D2_F64:
        return Ref.D2_F64.new(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> np.ndarray: ...



#  ------------------


class Node_D1_F64_D2_F64(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D1_F64_D2_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(
        self, i: int, slot: int | None = None
    ) -> Ref.D1_F64_D2_F64:
        return Ref.D1_F64_D2_F64.new(i, slot)

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
    Matrox_F64 = Node_D2_F64

    D1_F64_D2_F64 = Node_D1_F64_D2_F64

    Null = Node_Null
    null = Node_Null()


#  ------------------
