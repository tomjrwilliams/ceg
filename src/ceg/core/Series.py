from __future__ import annotations

from typing import NamedTuple, Callable, overload

import numpy as np

from . import Array

ValueAny = int | float | Array.np | dict | list

def ffill_0D(v: Array.np_1D):
    res = np.zeros_like(v) * np.NAN
    prev = None
    is_nan = np.isnan(v)
    for i, (vv, b) in enumerate(zip(v, is_nan)):
        if prev is None:
            res[i] = vv
        if not b:
            prev = vv
        res[i] = prev
    return res

def ffill_1D(v: Array.np_2D):
    res = np.zeros_like(v) * np.NAN
    prev = None
    is_nan = np.all(np.isnan(v), axis=1)
    for i, (vv, b) in enumerate(zip(v, is_nan)):
        if prev is None:
            res[i] = vv
        if not b:
            prev = vv
        res[i] = prev
    return res

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

    def mask(
        self, 
        at: float, 
        where: dict[str, Callable] | None=None,
        null: bool | str = True,
    ) -> Array.np:
        ts = self.t.data
        ts_at = ts <= at
        if where is None:
            return ts_at
        where_t: Callable | None = where.get("t")
        where_v: Callable | None=where.get("v")
        if where_t is not None and where_v is not None:
            vs = self.v.data
            return ts_at & where_t(ts) & where_v(vs)
        elif where_v is not None:
            vs = self.v.data
            return ts_at & where_v(vs)
        elif where_t is not None:
            return ts_at & where_t(ts)
        return ts_at

    def select(
        self,
        at: float,
        t: bool = False,
        i: int | slice | None = None,
        where: dict[str, Callable] | None = None,
        null: bool | str = True,
    ) -> ValueAny | tuple[
        Array.np_1D, ValueAny
    ] | tuple[
            list[float], list
        ]:
        raise ValueError()


class SeriesNull(SeriesND):
    pass


Any = SeriesND
Null = SeriesNull
null = SeriesNull.new()

class SeriesObject(SeriesND):
    t: Array.Scalar
    v: Array.Object
    # possibly c array

    @classmethod
    def new(cls) -> SeriesND:
        return cls(
            t=Array.Scalar.new(), v=Array.Object.new()
        )

    def append(self, t: float, v: float):
        return self._replace(
            t=self.t.add(t),
            v=self.v.add(v),
        )

    # @overload
    # def select(
    #     self, 
    #     at: float, 
    #     t: bool = False,
    #     i: int | slice | None = None,
    #     where: dict[str, Callable] | None = None,
    # ) -> list: ...

    # @overload
    # def select(
    #     self, 
    #     at: float, 
    #     t: bool = True,
    #     i: int | slice | None = None,
    #     where: dict[str, Callable] | None = None,
    # ) -> tuple[list[float], list]: ...

    # f_take = lambda v: [vv for vv, b in zip(v, mask) if b > 0]
    # if t:
    #     return f_take(self.t.data), f_take(self.v.data)
    # return f_take(self.v.data)
    def select(
        self,
        at: float,
        t: bool = False,
        i: int | slice | None = None,
        where: dict[str, Callable] | None = None,
        null: bool | str = True,
    ):
        mask = self.mask(at, where)
        v = np.array(self.v.data)
        if not null:
            mask = mask & np.logical_not(np.isnan(v))
        if null == "forward":
            v = ffill_0D(v[mask])
        else:
            v = v[mask]
        if i is None and t:
            return self.t.data[mask], v
        elif t:
            return self.t.data[mask][i], v[i]
        elif i is None:
            return v
        return v[i]

Object = SeriesObject

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
        self,
        at: float,
        t: bool = False,
        i: int | slice | None = None,
        where: dict[str, Callable] | None = None,
    ) -> Array.np_1D: ...

    @overload
    def select(
        self,
        at: float,
        t: bool = True,
        i: int | slice | None = None,
        where: dict[str, Callable] | None = None,
    ) -> tuple[Array.np_1D, Array.np_1D]: ...

    def select(
        self,
        at: float,
        t: bool = False,
        i: int | slice | None = None,
        where: dict[str, Callable] | None = None,
        null: str | None = None,
    ):
        mask = self.mask(at, where)
        v = self.v.data
        if not null:
            mask = mask & np.logical_not(np.isnan(v))
        if null == "forward":
            v = ffill_0D(v[mask])
        else:
            v = v[mask]
        if i is None and t:
            return self.t.data[mask], v
        elif t:
            return self.t.data[mask][i], v[i]
        elif i is None:
            return v
        return v[i]


Col = SeriesCol


class SeriesCol1D(SeriesND):
    t: Array.Scalar
    v: Array.Vec

    @classmethod
    def new(cls) -> SeriesND:
        return cls(t=Array.Scalar.new(), v=Array.Vec.new())

    def append(self, t: float, v: Array.np_1D):
        return self._replace(
            t=self.t.add(t),
            v=self.v.add(v),
        )

    def select(
        self,
        at: float,
        t: bool = False,
        i: int | slice | None = None,
        where: dict[str, Callable] | None = None,
        null: str | None = None,
    ):
        mask = self.mask(at, where)
        v = self.v.data
        if not null:
            mask = mask & np.logical_not(np.all(np.isnan(v), axis=1))
        if null == "forward":
            v = ffill_1D(v[mask])
        else:
            v = v[mask]
        if i is None and t:
            return self.t.data[mask], v
        elif t:
            return self.t.data[mask][i], v[i]
        elif i is None:
            return v
        return v[i]


Col1D = SeriesCol1D


class SeriesCol2D(SeriesND):
    t: Array.Scalar
    v: Array.Mat

    @classmethod
    def new(cls) -> SeriesND:
        return cls(t=Array.Scalar.new(), v=Array.D2.new())

    def append(self, t: float, v: Array.np_2D):
        return self._replace(
            t=self.t.add(t),
            v=self.v.add(v), # type: ignore
        )

    def select(self, at: float, t: bool = False):
        raise ValueError(self)


Col2D = SeriesCol2D

# TODO: const just returns the value on mask (and the given t presumably?)
# but you then have to check for if const - as will be different return type to a normal col?
