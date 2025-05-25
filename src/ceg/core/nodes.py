
from __future__ import annotations

import abc
from typing import Generic, ClassVar, Type, NamedTuple, TypeVar, ParamSpec, Callable, Concatenate

from dataclasses import dataclass
from heapq import heapify, heappush, heappop

import datetime as dt

from frozendict import frozendict
import numpy as np

from .refs import Ref, R, GraphInterface


#  ------------------

class Event(NamedTuple):
    """
    t: int
    ref: Ref.Any
    """
    t: int
    ref: Ref.Any

class Defn(NamedTuple):
    name: str
    params: tuple[str, ...]

#  ------------------

N = TypeVar("N", bound = NamedTuple)

V = np.ndarray | float

F = ParamSpec("F")
FRes = TypeVar("FRes")

class NodeInterface(abc.ABC, Generic[R, N]):
    
    DEF: ClassVar[Defn] = Defn("NULL", ())

    @abc.abstractmethod
    def pipe(
        self,
        f: Callable[Concatenate[NodeInterface[R, N], F], FRes],
        *args: F.args,
        **kwargs: F.kwargs
    ) -> FRes: ...
    
    @abc.abstractmethod
    def ref(self, i: int, slot: int | None=None) -> R: ...

    @abc.abstractmethod
    def __call__(self, event: Event, graph: GraphInterface) -> V: ...

#  ------------------

class Node_NullKW(NamedTuple):
    pass

class Node_Null(Node_NullKW, NodeInterface[Ref.Any, Node_NullKW]):

    def pipe(
        self,
        f: Callable[Concatenate[Node_Null, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(self, i: int, slot: int | None=None) -> Ref.Any:
        raise ValueError(self)

    def __call__(self, event: Event, graph: GraphInterface):
        raise ValueError(self)

#  ------------------

class Node_D0_Date(NodeInterface):

    def pipe(
        self,
        f: Callable[Concatenate[Node_D0_Date, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(self, i: int, slot: int | None=None) -> Ref.D0_Date:
        return Ref.D0_Date.new(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> dt.date: ...

class Node_D0_F64(NodeInterface):
    
    def pipe(
        self,
        f: Callable[Concatenate[Node_D0_F64, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs
    ) -> FRes:
        return f(self, *args, **kwargs)

    def ref(self, i: int, slot: int | None=None) -> Ref.D0_F64:
        return Ref.D0_F64.new(i, slot)

    @abc.abstractmethod
    def __call__(
        self, event: Event, graph: GraphInterface
    ) -> float: ...

#  ------------------

class Node:
    Any = NodeInterface

    D0_Date = Node_D0_Date
    Scalar_Date = Node_D0_Date

    D0_F64 = Node_D0_F64
    Scalar_F64 = Node_D0_F64

    Null = Node_Null
    null = Node_Null()    

#  ------------------
