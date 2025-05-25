from __future__ import annotations

import abc
from typing import NamedTuple, Protocol, Type, TypeVar, overload, Literal

import numpy as np

from .histories import History, Last, V

#  ------------------

Data = tuple[
    History.Any | tuple[History.Any, ...], ...
]

class GraphInterface(Protocol):
    
    @property
    def data(self) -> Data: ...

H = TypeVar("H", bound=History.Any)

@overload
def history(
    g: GraphInterface,
    i: int,
    t: Type[H],
    strict: bool | Literal[True] = True,
    slot: int | None = None,
) -> H: ...

@overload
def history(
    g: GraphInterface,
    i: int,
    t: Type[H],
    strict: Literal[False],
    slot: int | None = None,
) -> H | None: ...

def history(
    g: GraphInterface,
    i: int,
    t: Type[H],
    strict: bool = True,
    slot: int | None = None,
):
    h = g.data[i]
    if slot is None:
        if not strict and isinstance(h, History.Null):
            return None
        assert isinstance(h, t), dict(
            h=h, t=t, strict=strict # type: ignore
        )
        return h
    assert isinstance(h, tuple), (h, t, slot)
    h_slot = h[slot]
    if not strict and isinstance(h_slot, History.Null):
        return None
    assert isinstance(h_slot, t), dict(
        h=h_slot, t=t, strict=strict # type: ignore
    )
    return h_slot

#  ------------------

class Scope(NamedTuple):
    required: int

class RefInterface(abc.ABC):
    i: int
    slot: int | None
    scope: Scope | None

    def eq(self, other: RefInterface):
        if not isinstance(other, RefInterface):
            return False
        return self.i == other.i

    @abc.abstractmethod
    @overload
    def history(
        self, 
        g: GraphInterface, 
        strict: bool | Literal[True] = True
    ) -> History.Any: ...

    @abc.abstractmethod
    @overload
    def history(
        self, 
        g: GraphInterface, 
        strict: Literal[False]
    ) -> History.Any | None: ...

    @abc.abstractmethod
    def history(self, g: GraphInterface, strict: bool = True) -> History.Any | None: ...

    @classmethod
    @abc.abstractclassmethod
    def new(
        cls: Type[R],
        i: int,
        slot: int | None = None,
        scope: Scope | None = None,
    ) -> R: ...

    @abc.abstractmethod
    def select(self, last: bool | int) -> RefInterface: ...


class RefKwargs(NamedTuple):
    """
    NamedTuple direct subclasses can only have a single parent (NamedTuple)\n
    however, we *can* subclass a *subclass* of NamedTuple (but the fields are fixed by the parent)\n
    hence, here we define a NamedTuple interface that we can inherit from\n
    """

    i: int
    slot: int | None
    scope: Scope | None

    @classmethod
    def new(
        cls,
        i: int,
        slot: int | None = None,
        scope: Scope | None = None,
    ):
        return cls(i, slot, scope)
    
    def _select(self, last: bool | int):
        return self._replace(scope=Scope(required=int(last)))

R = TypeVar("R", bound=RefInterface)

#  ------------------

class Ref_D0_Date(RefKwargs, RefInterface):

    def select(self, last: bool | int) -> Ref_D0_Date:
        return self._select(last)
    
    @overload
    def history(
        self, g: GraphInterface, strict: Literal[True] = True
    ) -> History.D0_Date: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False]
    ) -> History.D0_Date | None: ...

    def history(
        self, g: GraphInterface, strict: bool = True
    ) -> History.D0_Date | None:
        return history(g, self.i, History.D0_Date, strict, self.slot)

class Ref_D0_F64(RefKwargs, RefInterface):

    def select(self, last: bool | int) -> Ref_D0_F64:
        return self._select(last)
    
    @overload
    def history(
        self, g: GraphInterface, strict: Literal[True] = True
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False]
    ) -> History.D0_F64 | None: ...

    def history(
        self, g: GraphInterface, strict: bool = True
    ) -> History.D0_F64 | None:
        return history(g, self.i, History.D0_F64, strict, self.slot)

#  ------------------

class Ref:
    Any = RefInterface

    D0_Date = Ref_D0_Date
    Scalar_Date = Ref_D0_Date

    D0_F64 = Ref_D0_F64
    Scalar_F64 = Ref_D0_F64

    @staticmethod
    def history(
        ref: Ref.Any,
        v: V,
        required: int | None,
        limit: int = 4,
    ) -> History.Any | tuple[History.Any, ...]:
        if not required:
            return History.null
        if isinstance(ref, Ref_D0_F64) and required > 1:
            return History.D0_F64.new(v, required, limit)
        elif isinstance(ref, Ref_D0_F64) and required == 1:
            return Last.D0_F64.new(v, required, 1)
        elif isinstance(ref, Ref_D0_Date) and required > 1:
            return History.D0_Date.new(v, required, limit)
        elif isinstance(ref, Ref_D0_Date) and required == 1:
            return Last.D0_Date.new(v, required, 1)
        raise ValueError(ref)

#  ------------------
