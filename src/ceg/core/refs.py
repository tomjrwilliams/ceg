from __future__ import annotations

import datetime as dt

import abc
from typing import (
    Any,
    NamedTuple,
    Protocol,
    Type,
    TypeVar,
    overload,
    cast,
    Literal,
)

import numpy as np

from .types import dataclass, replace
from .histories import History, Last, Unbounded, V

#  ------------------

Data = tuple[History.Any | tuple[History.Any, ...], ...]


class GraphInterface(Protocol):

    @property
    def data(self) -> Data: ...


H = TypeVar("H", bound=History.Any)


# @overload
# def history(
#     g: GraphInterface,
#     i: int,
#     t: Type[H],
#     strict: bool | Literal[True] = True,
#     slot: int | None = None,
# ) -> H: ...


# @overload
# def history(
#     g: GraphInterface,
#     i: int,
#     t: Type[H],
#     strict: Literal[False],
#     slot: int | None = None,
# ) -> H | None: ...


def history(
    g: GraphInterface,
    i: int,
    t: Type[H],
    strict: bool = True,
    slot: int | None = None,
):
    h = g.data[i]
    if slot is None:
        # if not strict and isinstance(h, History.Null):
        #     return None
        # assert isinstance(h, t), dict(
        #     h=h, t=t, strict=strict, node=g.nodes[i]  # type: ignore
        # )
        return cast(t, h)
    slot = int(slot)
    if isinstance(h, History.Null):
        return cast(t, h)
    assert isinstance(h, tuple), (h, t, slot)
    h_slot = h[slot]
    # if not strict and isinstance(h_slot, History.Null):
    #     return None
    # assert isinstance(h_slot, t), dict(
    #     h=h_slot, t=t, strict=strict, node=g.nodes[i]  # type: ignore
    # )
    return cast(t, h_slot)


#  ------------------

@dataclass(frozen=True)
class Scope:
    required: bool | int

@dataclass(frozen=True)
class RefInterface(abc.ABC):
    i: int
    slot: int | None
    scope: Scope | None

    def eq(self, other: RefInterface):
        if not isinstance(other, RefInterface):
            return False
        return self.i == other.i

    def __lt__(self, other: Any):
        if isinstance(other, RefInterface):
            return self.i < other.i
        raise ValueError((self, other))

    @abc.abstractmethod
    @overload
    def history(
        self,
        g: GraphInterface,
        strict: bool | Literal[True] = True,
    ) -> History.Any: ...

    @abc.abstractmethod
    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False]
    ) -> History.Any | None: ...

    @abc.abstractmethod
    def history(
        self, g: GraphInterface, strict: bool = True
    ) -> History.Any | None: ...

    @classmethod
    def new(
        cls,
        i: int,
        slot: int | None = None,
        scope: Scope | None = None,
    ):
        return cls(i, slot, scope)

    def _select(self, last: bool | int):
        return replace(
            self, scope=Scope(required=0 if not last else last)
        )

    def select_slot(self, slot: int) -> RefInterface:
        raise ValueError()

    @abc.abstractmethod
    def select(self, last: bool | int) -> RefInterface: ...

R = TypeVar("R", bound=RefInterface)

#  ------------------


@dataclass(frozen=True)
class Ref_D0_Date(RefInterface):

    def select(self, last: bool | int) -> Ref_D0_Date:
        return self._select(last)

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
    ) -> History.D0_Date: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False]
    ) -> History.D0_Date | None: ...

    def history(
        self, g: GraphInterface, strict: bool = True
    ) -> History.D0_Date | None:
        return history(
            g, self.i, History.D0_Date, strict, self.slot
        )

    def last_before(self, g: GraphInterface, t: float):
        return self.history(g).last_before(t)


@dataclass(frozen=True)
class Ref_D0_F64(RefInterface):

    def select(self, last: bool | int) -> Ref_D0_F64:
        return self._select(last)

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False]
    ) -> History.D0_F64 | None: ...

    def history(
        self, g: GraphInterface, strict: bool = True
    ) -> History.D0_F64 | None:
        return history(
            g, self.i, History.D0_F64, strict, self.slot
        )

    def last_before(self, g: GraphInterface, t: float) -> float | None:
        return self.history(g).last_before(t)


#  ------------------


@dataclass(frozen=True)
class Ref_D1_Date(RefInterface):

    def select(self, last: bool | int) -> Ref_D1_Date:
        return self._select(last)

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
    ) -> History.D1_Date: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False]
    ) -> History.D1_Date | None: ...

    def history(
        self, g: GraphInterface, strict: bool = True
    ) -> History.D1_Date | None:
        return history(
            g, self.i, History.D1_Date, strict, self.slot
        )


@dataclass(frozen=True)
class Ref_D1_F64(RefInterface):

    def select(self, last: bool | int) -> Ref_D1_F64:
        return self._select(last)

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
    ) -> History.D1_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False]
    ) -> History.D1_F64 | None: ...

    def history(
        self, g: GraphInterface, strict: bool = True
    ) -> History.D1_F64 | None:
        return history(
            g, self.i, History.D1_F64, strict, self.slot
        )


#  ------------------


@dataclass(frozen=True)
class Ref_D2_F64(RefInterface):

    def select(self, last: bool | int) -> Ref_D2_F64:
        return self._select(last)

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
    ) -> History.D2_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False]
    ) -> History.D2_F64 | None: ...

    def history(
        self, g: GraphInterface, strict: bool = True
    ) -> History.D2_F64 | None:
        return history(
            g, self.i, History.D2_F64, strict, self.slot
        )
#  ------------------


@dataclass(frozen=True)
class Ref_D1_F64_D2_F64(RefInterface):

    def select(self, last: bool | int) -> Ref_D1_F64_D2_F64:
        return self._select(last)

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[0] = 0,
    ) -> History.D1_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False],
        slot: Literal[0] = 0,
    ) -> History.D1_F64 | None: ...

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[1] = 1,
    ) -> History.D2_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False],
        slot: Literal[1] = 1,
    ) -> History.D2_F64 | None: ...

    def history(
        self, 
        g: GraphInterface, strict: bool = True, slot: int | None = None,
    ) -> History.D1_F64 | History.D2_F64 | None:
        assert slot is not None, self
        if slot == 0:
            return history(
                g, self.i, History.D1_F64, strict, slot
            )
        elif slot == 1:
            return history(
                g, self.i, History.D2_F64, strict, slot
            )
        else:
            raise ValueError(self)

@dataclass(frozen=True)
class Ref_D0_F64_3(RefInterface):

    def select(self, last: bool | int) -> Ref_D0_F64_3:
        return self._select(last)

    def select_slot(self, slot: int) -> Ref_D0_F64:
        return Ref_D0_F64(self.i, slot, scope=self.scope)

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[0] = 0,
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, 
        g: GraphInterface, strict: Literal[False],
        slot: Literal[0] = 0,
    ) -> History.D0_F64 | None: ...

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[1] = 1,
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False],
        slot: Literal[1] = 1,
    ) -> History.D0_F64 | None: ...

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[2] = 2,
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False],
        slot: Literal[2] = 2,
    ) -> History.D0_F64 | None: ...

    def history(
        self, 
        g: GraphInterface, strict: bool = True, slot: int | None = None,
    ) -> History.D0_F64 | None:
        slot = self.slot if slot is None else slot
        assert slot is not None, self
        if slot == 0:
            return history(
                g, self.i, History.D0_F64, strict, slot
            )
        elif slot == 1:
            return history(
                g, self.i, History.D0_F64, strict, slot
            )
        elif slot == 2:
            return history(
                g, self.i, History.D0_F64, strict, slot
            )
        else:
            raise ValueError(self)

@dataclass(frozen=True)
class Ref_D0_F64_4(RefInterface):

    def select(self, last: bool | int) -> Ref_D0_F64_4:
        return self._select(last)

    def select_slot(self, slot: int) -> Ref_D0_F64:
        return Ref_D0_F64(self.i, slot, scope=self.scope)

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[0] = 0,
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, 
        g: GraphInterface, strict: Literal[False],
        slot: Literal[0] = 0,
    ) -> History.D0_F64 | None: ...

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[1] = 1,
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False],
        slot: Literal[1] = 1,
    ) -> History.D0_F64 | None: ...

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[2] = 2,
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False],
        slot: Literal[2] = 2,
    ) -> History.D0_F64 | None: ...

    @overload
    def history(
        self,
        g: GraphInterface,
        strict: Literal[True] = True,
        slot: Literal[3] = 3,
    ) -> History.D0_F64: ...

    @overload
    def history(
        self, g: GraphInterface, strict: Literal[False],
        slot: Literal[3] = 3,
    ) -> History.D0_F64 | None: ...

    def history(
        self, 
        g: GraphInterface, strict: bool = True, slot: int | None = None,
    ) -> History.D0_F64 | None:
        slot = self.slot if slot is None else slot
        assert slot is not None, self
        if slot == 0:
            return history(
                g, self.i, History.D0_F64, strict, slot
            )
        elif slot == 1:
            return history(
                g, self.i, History.D0_F64, strict, slot
            )
        elif slot == 2:
            return history(
                g, self.i, History.D0_F64, strict, slot
            )
        elif slot == 3:
            return history(
                g, self.i, History.D0_F64, strict, slot
            )
        else:
            raise ValueError(self)

#  ------------------


class Ref:
    Any = RefInterface

    D0_Date = Ref_D0_Date
    Scalar_Date = Ref_D0_Date

    D0_F64 = Ref_D0_F64
    Scalar_F64 = Ref_D0_F64

    D1_Date = Ref_D1_Date
    Vector_Date = Ref_D1_Date

    D1_F64 = Ref_D1_F64
    Vector_F64 = Ref_D1_F64

    D2_F64 = Ref_D2_F64
    Matrix_F64 = Ref_D2_F64

    D1_F64_D2_F64 = Ref_D1_F64_D2_F64

    D0_F64_3 = Ref_D0_F64_3
    D0_F64_4 = Ref_D0_F64_4

    @staticmethod
    def history(
        ref: Ref.Any,
        v: V,
        required: bool | int | None,
        limit: int = 4,
    ) -> History.Any | tuple[History.Any, ...]:
        if not required:
            return History.null
        
        #
        if isinstance(ref, Ref_D0_F64) and isinstance(required, bool):
            return Unbounded.D0_F64.new(v, 32, 1)
        elif isinstance(ref, Ref_D0_F64) and required > 1:
            return History.D0_F64.new(v, required, limit)
        elif isinstance(ref, Ref_D0_F64) and required == 1:
            return Last.D0_F64.new(v, required, 1)

        elif isinstance(ref, Ref_D0_Date) and isinstance(required, bool):
            return Unbounded.D0_Date.new(v, 32, 1)
        elif isinstance(ref, Ref_D0_Date) and required > 1:
            return History.D0_Date.new(v, required, limit)
        elif isinstance(ref, Ref_D0_Date) and required == 1:
            return Last.D0_Date.new(v, required, 1)
        #
        elif isinstance(ref, Ref_D1_F64) and required > 1:
            return History.D1_F64.new(v, required, limit)
        elif isinstance(ref, Ref_D1_F64) and required == 1:
            return Last.D1_F64.new(v, required, 1)
        elif isinstance(ref, Ref_D1_Date) and required > 1:
            return History.D1_Date.new(v, required, limit)
        elif isinstance(ref, Ref_D1_Date) and required == 1:
            return Last.D1_Date.new(v, required, 1)
        #
        elif isinstance(ref, Ref_D2_F64) and required > 1:
            return History.D2_F64.new(v, required, limit)
        elif isinstance(ref, Ref_D2_F64) and required == 1:
            return Last.D2_F64.new(v, required, 1)
        #
        elif isinstance(ref, Ref_D1_F64_D2_F64) and required > 1:
            return (
                History.D1_F64.new(v, required, limit),
                History.D2_F64.new(v, required, limit),
            )
        elif isinstance(ref, Ref_D1_F64_D2_F64) and required == 1:
            return (
                Last.D1_F64.new(v, required, 1),
                Last.D2_F64.new(v, required, 1)
            )
        #
        elif isinstance(ref, Ref_D0_F64_3) and required > 1:
            assert isinstance(v, tuple), v
            return (
                History.D0_F64.new(v[0], required, limit),
                History.D0_F64.new(v[1], required, limit),
                History.D0_F64.new(v[2], required, limit),
            )
        elif isinstance(ref, Ref_D0_F64_3) and required == 1:
            assert isinstance(v, tuple), v
            return (
                Last.D0_F64.new(v[0], required, 1),
                Last.D0_F64.new(v[1], required, 1),
                Last.D0_F64.new(v[2], required, 1),
            )
        elif isinstance(ref, Ref_D0_F64_4) and required > 1:
            assert isinstance(v, tuple), v
            return (
                History.D0_F64.new(v[0], required, limit),
                History.D0_F64.new(v[1], required, limit),
                History.D0_F64.new(v[2], required, limit),
                History.D0_F64.new(v[3], required, limit),
            )
        elif isinstance(ref, Ref_D0_F64_4) and required == 1:
            assert isinstance(v, tuple), v
            return (
                Last.D0_F64.new(v[0], required, 1),
                Last.D0_F64.new(v[1], required, 1),
                Last.D0_F64.new(v[2], required, 1),
                Last.D0_F64.new(v[3], required, 1),
            )
        #
        raise ValueError(ref)
    
    @classmethod
    def d0_f64(
        cls,
        i: int | Ref.Any,
        slot: int | None = None
    ) -> Ref_D0_F64:
        if isinstance(i, Ref.Any):
            return cast(Ref.D0_F64, i)
        return Ref.D0_F64.new(i, slot)

    @classmethod
    def d1_f64(
        cls,
        i: int | Ref.Any,
        slot: int | None = None
    ) -> Ref_D1_F64:
        if isinstance(i, Ref.Any):
            return cast(Ref.D1_F64, i)
        return Ref.D1_F64.new(i, slot)

    @classmethod
    def d2_f64(
        cls,
        i: int | Ref.Any,
        slot: int | None = None
    ) -> Ref_D2_F64:
        if isinstance(i, Ref.Any):
            return cast(Ref.D2_F64, i)
        return Ref.D2_F64.new(i, slot)

    @classmethod
    def d0_date(
        cls,
        i: int | Ref.Any,
        slot: int | None = None
    ) -> Ref_D0_Date:
        if isinstance(i, Ref.Any):
            return cast(Ref.D0_Date, i)
        return Ref.D0_Date.new(i, slot)

    @classmethod
    def d1_date(
        cls,
        i: int | Ref.Any,
        slot: int | None = None
    ) -> Ref_D1_Date:
        if isinstance(i, Ref.Any):
            return cast(Ref.D1_Date, i)
        return Ref.D1_Date.new(i, slot)

    @classmethod
    def d0_f64_4(
        cls,
        i: int | Ref.Any,
        slot: int | None = None
    ) -> Ref_D0_F64_4:
        if isinstance(i, Ref.Any):
            return cast(Ref.D0_F64_4, i)
        return Ref.D0_F64_4.new(i, slot)

    @classmethod
    def d0_f64_3(
        cls,
        i: int | Ref.Any,
        slot: int | None = None
    ) -> Ref_D0_F64_3:
        if isinstance(i, Ref.Any):
            return cast(Ref.D0_F64_3, i)
        return Ref.D0_F64_3.new(i, slot)

    @classmethod
    def d1_f64_d2_f64(
        cls,
        i: int | Ref.Any,
        slot: int | None = None
    ) -> Ref_D1_F64_D2_F64:
        if isinstance(i, Ref.Any):
            return cast(Ref.D1_F64_D2_F64, i)
        return Ref.D1_F64_D2_F64.new(i, slot)

#  ------------------
