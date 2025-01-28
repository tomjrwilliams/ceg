
from __future__ import annotations

from typing import NamedTuple
from . import Array

ValueAny = float | Array.np | dict | list

class SeriesND(NamedTuple):
    t: Array.Scalar
    v: Array.Any
    # possibly c array (of structs if row)
    
    @classmethod
    def new(cls) -> SeriesND:
        return cls(t=Array.Scalar.new(), v = Array.Null.new())

    # rows as list? can be decided later
    def append(
        self,
        t: float,
        v: ValueAny
    ) -> SeriesND:
        raise ValueError(self)

    def mask(self, t: float):
        return

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
        return cls(t=Array.Scalar.new(), v = Array.Scalar.new())

    def append(self, t: float, v: float):
        return self._replace(
            t=self.t.add(t),
            v = self.v.add(v),
        )

    def mask(self, t: float):
        return

Col = SeriesCol

class SeriesCol1D(SeriesND):
    t: Array.Scalar
    v: Array.Vec

    @classmethod
    def new(cls, v: Array.np) -> SeriesND:
        return cls(t=Array.Scalar.new(), v = Array.Vec.new())

    def append(self, t: float, v: Array.np):
        return self._replace(
            t=self.t.add(t),
            v = self.v.add(v),
        )

    def mask(self, t: float):
        return

Col1D = SeriesCol1D