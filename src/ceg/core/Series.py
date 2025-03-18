from __future__ import annotations

from typing import NamedTuple, overload
from . import Array

ValueAny = float | Array.np | dict | list


class SeriesND(NamedTuple):
    t: Array.Scalar
    v: Array.Any
    # possibly c array (of structs if row)

    @classmethod
    def new(cls) -> SeriesND:
        return cls(t=Array.Scalar.new(), v=Array.Null.new())

    # rows as list? can be decided later
    def append(self, t: float, v: ValueAny) -> SeriesND:
        raise ValueError(self)

    def mask(self, at: float) -> Array.np:
        ts = self.t.data
        return ts <= at

    def select(
        self, at: float, t: bool = False
    ) -> ValueAny | tuple[Array.np_1D, ValueAny]:
        raise ValueError()


class SeriesNull(SeriesND):
    pass


Any = SeriesND
Null = SeriesNull
null = SeriesNull.new()


class SeriesCol(SeriesND):
    t: Array.Scalar
    v: Array.Scalar
    # possibly c array

    @classmethod
    def new(cls) -> SeriesND:
        return cls(
            t=Array.Scalar.new(), v=Array.Scalar.new()
        )

    def append(self, t: float, v: float):
        return self._replace(
            t=self.t.add(t),
            v=self.v.add(v),
        )

    @overload
    def select(
        self, at: float, t: bool = False
    ) -> Array.np_1D: ...

    @overload
    def select(
        self, at: float, t: bool = True
    ) -> tuple[Array.np_1D, Array.np_1D]: ...

    def select(self, at: float, t: bool = False):
        mask = self.mask(at)
        if t:
            return self.t.data[mask], self.v.data[mask]
        return self.v.data[mask]


Col = SeriesCol


class SeriesCol1D(SeriesND):
    t: Array.Scalar
    v: Array.Vec

    @classmethod
    def new(cls, v: Array.np_1D) -> SeriesND:
        return cls(t=Array.Scalar.new(), v=Array.Vec.new())

    def append(self, t: float, v: Array.np_1D):
        return self._replace(
            t=self.t.add(t),
            v=self.v.add(v),
        )

    def select(self, at: float):
        return


Col1D = SeriesCol1D


class SeriesCol2D(SeriesND):
    t: Array.Scalar
    v: Array.Vec

    @classmethod
    def new(cls, v: Array.np_2D) -> SeriesND:
        return cls(t=Array.Scalar.new(), v=Array.D2.new())

    def append(self, t: float, v: Array.np_2D):
        return self._replace(
            t=self.t.add(t),
            v=self.v.add(v),
        )

    def select(self, at: float):
        return


Col2D = SeriesCol2D

# TODO: const just returns the value on mask (and the given t presumably?)
# but you then have to check for if const - as will be different return type to a normal col?
