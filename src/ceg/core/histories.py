import logging
import abc
from typing import NamedTuple, cast, Type, Literal, overload
from dataclasses import dataclass

import datetime as dt

import numpy as np
import numba as nb

from . import algos

logger = logging.Logger(__file__)

#  ------------------

# TODO: abstract out nd implementatios, each class just needs:
# specific type casting
# return array annotatios

#  ------------------

V = np.ndarray | np.int64 | np.datetime64 | float | dt.date


@dataclass
class HistoryMutable:
    occupied: int  # NOTE: current index occupied up to


class HistoryKW(NamedTuple):
    shape: tuple[int, ...]
    size: int
    exponent: int  # NOTE: len = 2 ** exponent
    required: int  # NOTE: max history required to be held
    limit: int  # NOTE: multiple of required to hold
    times: np.ndarray
    values: np.ndarray  # NOTE: flattened 1D array
    mut: HistoryMutable

    @classmethod
    def new_mutable(cls):
        return HistoryMutable(0)

    @classmethod
    def new(
        cls,
        v: V,
        required: int,
        limit: int = 4,
    ):
        if not isinstance(v, np.ndarray):
            shape = ()
            if type(v) in {dt.date, dt.datetime}:
                # dtype = "datetime64[s]"
                dtype = np.int64
            else:
                dtype = np.dtype(type(v))
        else:
            shape = v.shape
            dtype = v.dtype
        if np.issubdtype(dtype, np.datetime64):
            dtype = np.int64
        exponent = (
            0
            if required == 0
            else np.log2(limit) + np.ceil(np.log2(required))
        )
        size = 0 if not len(shape) else np.prod(shape)
        values=np.empty(
            int(
                np.prod(np.array(shape)) * (2**exponent)
            ),
            dtype=dtype,
        )
        return cls(
            shape,
            int(size),
            exponent,
            required,
            limit,
            np.empty(
                int(2**exponent),
                dtype=dtype,
            ),
            values=values,
            mut=cls.new_mutable()
        )

    @property
    def length(self) -> int:
        return 2**self.exponent


class LastKW(NamedTuple):
    shape: tuple[int, ...]
    exponent: int  # NOTE: len = 2 ** exponent
    required: int  # NOTE: max history required to be held
    limit: int  # NOTE: multiple of required to hold
    times: np.ndarray
    values: np.ndarray  # NOTE: flattened 1D array
    mut: HistoryMutable

    @classmethod
    def new_mutable(cls):
        return HistoryMutable(0)

    @classmethod
    def new(
        cls,
        v: V,
        required: int | None,
        limit: int = 1,
    ):
        if required is None:
            required = 0
        if not isinstance(v, np.ndarray):
            shape = (1,)
            dtype = np.dtype(type(v))
        else:
            shape = v.shape
            dtype = v.dtype
        if np.issubdtype(dtype, np.datetime64):
            dtype = np.int64
        exponent = 0
        return cls(
            shape,
            exponent,
            required,
            limit,
            np.empty((), dtype=dtype),
            np.empty((), dtype=dtype),
            mut=cls.new_mutable(),
        )

    @property
    def length(self) -> int:
        return 1

    @property
    def size(self):
        return np.prod(np.array(self.shape))


class HistoryInterface(abc.ABC):
    shape: tuple[int, ...]
    exponent: int  # NOTE: len = 2 ** exponent
    required: int  # NOTE: max history required to be held
    occupied: int  # NOTE: current index occupied up to
    times: np.ndarray
    values: np.ndarray  # NOTE: flattened 1D array
    mut: HistoryMutable

    @property
    @abc.abstractmethod
    def dtype(self): ...

    @abc.abstractmethod
    def append(self, v: V, t: float): ...

    @abc.abstractmethod
    def last_n_before(
        self, n: int, t: float
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def last_n_between(
        self, n: int, l: float, r: float
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def last_before(self, t: float) -> V: ...

    @abc.abstractmethod
    def last_between(self, l: float, r: float) -> V: ...

    @abc.abstractmethod
    def last_t(self) -> float: ...


#  ------------------


class Last_0D(LastKW, HistoryInterface):

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        raise ValueError(self)


class History_0D(HistoryKW, HistoryInterface):
    pass


class Last_1D(LastKW, HistoryInterface):

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        raise ValueError(self)


class History_1D(HistoryKW, HistoryInterface):
    pass


class Last_2D(LastKW, HistoryInterface):

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        raise ValueError(self)


class History_2D(HistoryKW, HistoryInterface):
    pass


#  ------------------

@overload
def append_d0(
    occupied: int,
    required: int,
    limit: int,
    vs: np.ndarray,
    ts: np.ndarray,
    t: float,
    v: V,
    truncate: Literal[True] = True,
) -> int: ...

@overload
def append_d0(
    occupied: int,
    required: int,
    limit: int,
    vs: np.ndarray,
    ts: np.ndarray,
    t: float,
    v: V,
    truncate: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]: ...

def append_d0(
    occupied: int,
    required: int,
    limit: int,
    vs: np.ndarray,
    ts: np.ndarray,
    t: float,
    v: V,
    truncate: bool = True,
) -> int | tuple[int, np.ndarray, np.ndarray]:
    if truncate and occupied == limit * required:
        sl = slice((limit - 1) * required, limit * required)
        ts[:required] = ts[sl]
        vs[:required] = vs[sl]
        occupied = required
    elif occupied == limit * required:
        ts = np.hstack((
            ts, np.empty_like(ts),
        ))
        vs = np.hstack((
            vs, np.empty_like(vs),
        ))
    ts[occupied] = t
    vs[occupied] = v
    occupied += 1
    if not truncate:
        return occupied, ts, vs
    return occupied


def append_nd(
    occupied: int,
    required: int,
    limit: int,
    size: int,
    vs: np.ndarray,
    ts: np.ndarray,
    t: float,
    v: np.ndarray,
    truncate: bool = False,
) -> int:
    if truncate and occupied == limit * required:
        sl = slice((limit - 1) * required, limit * required)
        ts[:required] = ts[sl]
        sl = slice(
            (limit - 1) * required * size,
            limit * required * size,
        )
        vs[: required * size] = vs[sl]
        occupied = required
    elif occupied == limit * required:
        ts = np.hstack((
            ts, np.empty_like(ts),
        ))
        vs = np.hstack((
            vs, np.empty_like(vs),
        ))
    ts[occupied] = t
    vs[occupied * size : (occupied + 1) * size] = v
    occupied += 1
    return occupied


#  ------------------


class History_Null(HistoryKW, HistoryInterface):

    def dtype(self):
        return None

    def append(self, v: V, t: float):
        return

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        raise ValueError(self)

    def last_before(self, t: float):
        raise ValueError(self)

    def last_between(self, l: float, r: float):
        raise ValueError(self)

    def last_t(self):
        return ValueError(self)


#  ------------------


class History_D0_F64(History_0D):

    @property
    def dtype(self):
        return np.float64

    def append(self, v: float, t: float):
        self.mut.occupied = append_d0(
            self.mut.occupied,
            self.required,
            self.limit,
            self.values,
            self.times,
            t,
            v,
        )

    def last_n_before(self, n: int, t: float):
        return algos.last_n_before(
            self.values,
            self.times,
            n,
            t,
            self.mut.occupied,
            self.exponent,
        )

    def last_n_between(self, n: int, l: float, r: float):
        return algos.last_n_between(
            self.values,
            self.times,
            n,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        )

    def last_before(
        self, t: float, allow_nan: bool = True
    ) -> float | None:
        f = algos.last_before if allow_nan else algos.last_before_not_nan
        return f(
            self.values,
            self.times,
            t,
            self.mut.occupied,
            self.exponent,
        )

    def last_between(self, l: float, r: float) -> float | None:
        return algos.last_between(
            self.values,
            self.times,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        )

    def last_t(self) -> float:
        occupied = self.mut.occupied
        if occupied == 0:
            return -1
        return self.times[occupied - 1]

@dataclass
class Unbounded_D0_F64_Mut(HistoryMutable):
    limit: int | None
    vs: np.ndarray
    ts: np.ndarray
    exponent: int

class Unbounded_D0_F64(History_D0_F64):
    mut: Unbounded_D0_F64_Mut

    @property
    def dtype(self):
        return np.float64

    @classmethod
    def new_mutable(cls):
        return Unbounded_D0_F64_Mut(
            0, None, np.empty(1), np.empty(1), 0
        )

    def append(self, v: float, t: float):
        if self.mut.limit is None:
            self.mut.vs = self.values
            self.mut.ts = self.times
            self.mut.limit = self.limit
            self.mut.exponent = int(np.log2(self.required))
        self.mut.occupied, self.mut.ts, self.mut.vs = append_d0(
            self.mut.occupied,
            self.required,
            self.mut.limit,
            self.mut.vs,
            self.mut.ts,
            t,
            v,
            truncate=False,
        )
        if self.mut.occupied > self.mut.limit * self.required:
            self.mut.limit *= 2
            self.mut.exponent += 1

    def last_n_before(self, n: int, t: float):
        return algos.last_n_before(
            self.mut.vs,
            self.mut.ts,
            n,
            t,
            self.mut.occupied,
            self.mut.exponent,
        )

    def last_n_between(self, n: int, l: float, r: float):
        return algos.last_n_between(
            self.mut.vs,
            self.mut.ts,
            n,
            l,
            r,
            self.mut.occupied,
            self.mut.exponent,
        )

    def last_before(
        self, t: float, allow_nan: bool = True
    ) -> float | None:
        f = algos.last_before if allow_nan else algos.last_before_not_nan
        return f(
            self.mut.vs,
            self.mut.ts,
            t,
            self.mut.occupied,
            self.mut.exponent,
        )

    def last_between(self, l: float, r: float) -> float | None:
        return algos.last_between(
            self.mut.vs,
            self.mut.ts,
            l,
            r,
            self.mut.occupied,
            self.mut.exponent,
        )

    def last_t(self) -> float:
        occupied = self.mut.occupied
        if occupied == 0:
            return -1
        return self.times[occupied - 1]


@dataclass
class Last_D0_F64_Mut(HistoryMutable):
    t: float
    v: float


class Last_D0_F64(History_D0_F64):
    mut: Last_D0_F64_Mut

    @classmethod
    def new_mutable(cls):
        return Last_D0_F64_Mut(0, -1, 0)

    @property
    def dtype(self):
        return np.float64

    def append(self, v: float, t: float):
        self.mut.t = t
        self.mut.v = v

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        return ValueError(self)

    def last_before(self, t: float):
        assert self.mut.t <= t, (self, t)
        if self.mut.t == -1:
            raise ValueError(self, t)
        return self.mut.v

    def last_between(self, l: float, r: float):
        assert self.mut.t <= r, (self, r)
        last_t = self.mut.t
        if last_t < l:
            return np.nan
        return self.mut.v

    def last_t(self) -> float:
        return self.mut.t


#  ------------------


class History_D0_Date(History_0D):

    @property
    def dtype(self):
        return np.datetime64

    def append(self, v: dt.date, t: float):
        v_np = np.datetime64(v, "s").astype(np.int64)
        self.mut.occupied = append_d0(
            self.mut.occupied,
            self.required,
            self.limit,
            self.values,
            self.times,
            t,
            v_np,
            # np.datetime64,
        )

    def last_n_before(self, n: int, t: float):
        return algos.last_n_before(
            self.values,
            self.times,
            n,
            t,
            self.mut.occupied,
            self.exponent,
        ).astype("datetime64[s]").astype("M8[D]")

    def last_n_between(self, n: int, l: float, r: float):
        return algos.last_n_between(
            self.values,
            self.times,
            n,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        ).astype("datetime64[s]").astype("M8[D]")

    def last_before(self, t: float) -> dt.date | None:
        res = np.datetime64(int(algos.last_before(
            self.values,
            self.times,
            t,
            self.mut.occupied,
            self.exponent,
        )), "s")
        if res is None:
            return res
        return cast(
            dt.date,
            (cast(np.datetime64, res))
            .astype("M8[D]")
            .astype("O"),
        )

    def last_between(self, l: float, r: float):
        res = algos.last_between(
            self.values,
            self.times,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        )
        if np.isnan(res):
            return None
        elif res is None:
            return res
        return cast(
            dt.date,
            np.datetime64(int(res), "s")
            .astype("M8[D]")
            .astype("O"),
        )

    def last_t(self) -> float:
        occupied = self.mut.occupied
        if occupied == 0:
            return -1
        return self.times[occupied - 1]

@dataclass
class Unbounded_D0_Date_Mut(HistoryMutable):
    limit: int | None
    vs: np.ndarray
    ts: np.ndarray
    exponent: int

class Unbounded_D0_Date(History_D0_Date):
    mut: Unbounded_D0_Date_Mut

    @property
    def dtype(self):
        return np.datetime64

    @classmethod
    def new_mutable(cls):
        return Unbounded_D0_Date_Mut(0, None, np.empty(1), np.empty(1), 0)

    def append(self, v: dt.date, t: float):
        if self.mut.limit is None:
            self.mut.limit = self.limit
            self.mut.vs = self.values
            self.mut.ts = self.times
            self.mut.exponent = int(np.log2(self.required))
        v_np = np.datetime64(v, "s").astype(np.int64)
        self.mut.occupied, self.mut.ts, self.mut.vs = append_d0(
            self.mut.occupied,
            self.required,
            self.mut.limit,
            self.mut.vs,
            self.mut.ts,
            t,
            v_np,
            # np.datetime64,
            truncate=False,
        )
        if self.mut.occupied > self.mut.limit * self.required:
            self.mut.limit *= 2
            self.mut.exponent += 1

    def last_n_before(self, n: int, t: float):
        return algos.last_n_before(
            self.mut.vs,
            self.mut.ts,
            n,
            t,
            self.mut.occupied,
            self.mut.exponent,
        ).astype("datetime64[s]").astype("M8[D]")

    def last_n_between(self, n: int, l: float, r: float):
        return algos.last_n_between(
            self.mut.vs,
            self.mut.ts,
            n,
            l,
            r,
            self.mut.occupied,
            self.mut.exponent,
        ).astype("datetime64[s]").astype("M8[D]")

    def last_before(self, t: float) -> dt.date | None:
        res = np.datetime64(int(algos.last_before(
            self.mut.vs,
            self.mut.ts,
            t,
            self.mut.occupied,
            self.mut.exponent,
        )), "s")
        if res is None:
            return res
        return cast(
            dt.date,
            (cast(np.datetime64, res))
            .astype("M8[D]")
            .astype("O"),
        )

    def last_between(self, l: float, r: float):
        res = algos.last_between(
            self.mut.vs,
            self.mut.ts,
            l,
            r,
            self.mut.occupied,
            self.mut.exponent,
        )
        if np.isnan(res):
            return None
        elif res is None:
            return res
        return cast(
            dt.date,
            np.datetime64(int(res), "s")
            .astype("M8[D]")
            .astype("O"),
        )

    def last_t(self) -> float:
        occupied = self.mut.occupied
        if occupied == 0:
            return -1
        return self.times[occupied - 1]


@dataclass
class Last_D0_Date_Mut(HistoryMutable):
    t: float
    v: dt.date


class Last_D0_Date(History_D0_Date):
    mut: Last_D0_Date_Mut

    @classmethod
    def new_mutable(cls):
        return Last_D0_Date_Mut(0, -1, dt.date(2000, 1, 1))

    @property
    def dtype(self):
        return np.datetime64

    def append(self, v: dt.date, t: float):
        self.mut.t = t
        self.mut.v = v

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        return ValueError(self)

    def last_before(self, t: float):
        assert self.mut.t <= t, (self, t)
        if self.mut.t == -1:
            raise ValueError(self, t)
        return self.mut.v

    def last_between(self, l: float, r: float):
        assert self.mut.t <= r, (self, r)
        last_t = self.mut.t
        if last_t < l:
            return np.nan
        return self.mut.v

    def last_t(self) -> float:
        return self.mut.t


# TODO: bool, int, datetime, string (<U5, |S5 etc. for fixed, StringDType for arbitrary, astype("U5") for casting between)

#  ------------------


class History_D1_Date(History_1D):

    # TODO: int64 seconds and back

    @property
    def dtype(self):
        return np.datetime64

    def append(self, v: np.ndarray, t: float):
        v = v.astype("datetime64[s]").astype(np.int64)
        self.mut.occupied = append_d0(
            self.mut.occupied,
            self.required,
            self.limit,
            self.values,
            self.times,
            t,
            v.reshape((self.size,)),
        )

    def last_n_before(self, n: int, t: float):
        return algos.last_n_before_nd(
            self.values,
            self.times,
            self.size,
            n,
            t,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape).astype("datetime64[s]")

    def last_n_between(self, n: int, l: float, r: float):
        return algos.last_n_between_nd(
            self.values,
            self.times,
            self.size,
            n,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape).astype("datetime64[s]")

    def last_before(self, t: float):
        return algos.last_before_nd(
            self.values,
            self.times,
            self.size,
            t,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape).astype("datetime64[s]")

    def last_between(self, l: float, r: float):
        return algos.last_between_nd(
            self.values,
            self.times,
            self.size,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape).astype("datetime64[s]")

    def last_t(self) -> float:
        occupied = self.mut.occupied
        if occupied == 0:
            return -1
        return self.times[occupied - 1]

def np_to_date_0d(v: np.datetime64) -> dt.date:
    return (
        v.astype("M8[D]").astype("O")
    )

def np_to_date_1d(v: np.ndarray) -> list[dt.date]:
    return list(
        v.astype("M8[D]")
        .astype("O")
    )

@dataclass
class Last_D1_Date_Mut(HistoryMutable):
    t: float
    v: np.ndarray


class Last_D1_Date(History_D1_Date):
    mut: Last_D1_Date_Mut

    @classmethod
    def new_mutable(cls):
        return Last_D1_Date_Mut(0, -1, np.empty((1,)))

    @property
    def dtype(self):
        return np.datetime64

    def append(self, v: np.ndarray, t: float):
        self.mut.t = t
        self.mut.v = v

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        return ValueError(self)

    def last_before(self, t: float):
        assert self.mut.t <= t, (self, t)
        if self.mut.t == -1:
            raise ValueError(self, t)
        return self.mut.v

    def last_between(self, l: float, r: float):
        assert self.mut.t <= r, (self, r)
        last_t = self.mut.t
        if last_t < l:
            return np.nan
        return self.mut.v

    def last_t(self) -> float:
        return self.mut.t


class History_D1_F64(History_1D):

    @property
    def dtype(self):
        return np.float64

    def append(self, v: np.ndarray, t: float):
        self.mut.occupied = append_d0(
            self.mut.occupied,
            self.required,
            self.limit,
            self.values,
            self.times,
            t,
            v.reshape((self.size,)),
        )

    def last_n_before(self, n: int, t: float):
        return algos.last_n_before_nd(
            self.values,
            self.times,
            self.size,
            n,
            t,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape)

    def last_n_between(self, n: int, l: float, r: float):
        return algos.last_n_between_nd(
            self.values,
            self.times,
            self.size,
            n,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape)

    def last_before(self, t: float):
        return algos.last_before_nd(
            self.values,
            self.times,
            self.size,
            t,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape)

    def last_between(self, l: float, r: float):
        return algos.last_between_nd(
            self.values,
            self.times,
            self.size,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape)

    def last_t(self) -> float:
        occupied = self.mut.occupied
        if occupied == 0:
            return -1
        return self.times[occupied - 1]


@dataclass
class Last_D1_F64_Mut(HistoryMutable):
    t: float
    v: np.ndarray


class Last_D1_F64(History_D1_F64):
    mut: Last_D1_F64_Mut

    @classmethod
    def new_mutable(cls):
        return Last_D1_F64_Mut(0, -1, np.empty((1,)))

    @property
    def dtype(self):
        return np.float64

    def append(self, v: np.ndarray, t: float):
        self.mut.t = t
        self.mut.v = v

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        return ValueError(self)

    def last_before(self, t: float):
        assert self.mut.t <= t, (self, t)
        if self.mut.t == -1:
            raise ValueError(self, t)
        return self.mut.v

    def last_between(self, l: float, r: float):
        assert self.mut.t <= r, (self, r)
        last_t = self.mut.t
        if last_t < l:
            return np.nan
        return self.mut.v

    def last_t(self) -> float:
        return self.mut.t


#  ------------------


class History_D2_F64(History_2D):

    @property
    def dtype(self):
        return np.float64

    def append(self, v: np.ndarray, t: float):
        self.mut.occupied = append_d0(
            self.mut.occupied,
            self.required,
            self.limit,
            self.values,
            self.times,
            t,
            v.reshape((self.size,)),
        )

    def last_n_before(self, n: int, t: float):
        return algos.last_n_before_nd(
            self.values,
            self.times,
            self.size,
            n,
            t,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape)

    def last_n_between(self, n: int, l: float, r: float):
        return algos.last_n_between_nd(
            self.values,
            self.times,
            self.size,
            n,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape)

    def last_before(self, t: float):
        return algos.last_before_nd(
            self.values,
            self.times,
            self.size,
            t,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape)

    def last_between(self, l: float, r: float):
        return algos.last_between_nd(
            self.values,
            self.times,
            self.size,
            l,
            r,
            self.mut.occupied,
            self.exponent,
        ).reshape(self.shape)

    def last_t(self) -> float:
        occupied = self.mut.occupied
        if occupied == 0:
            return -1
        return self.times[occupied - 1]


@dataclass
class Last_D2_F64_Mut(HistoryMutable):
    t: float
    v: np.ndarray


class Last_D2_F64(History_D2_F64):
    mut: Last_D2_F64_Mut

    @classmethod
    def new_mutable(cls):
        return Last_D2_F64_Mut(
            0,
            -1,
            np.empty(
                (
                    1,
                    1,
                )
            ),
        )

    @property
    def dtype(self):
        return np.float64

    def append(self, v: np.ndarray, t: float):
        self.mut.t = t
        self.mut.v = v

    def last_n_before(self, n: int, t: float):
        raise ValueError(self)

    def last_n_between(self, n: int, l: float, r: float):
        return ValueError(self)

    def last_before(self, t: float):
        assert self.mut.t <= t, (self, t)
        if self.mut.t == -1:
            raise ValueError(self, t)
        return self.mut.v

    def last_between(self, l: float, r: float):
        assert self.mut.t <= r, (self, r)
        last_t = self.mut.t
        if last_t < l:
            return np.nan
        return self.mut.v

    def last_t(self) -> float:
        return self.mut.t


#  ------------------
#
class Unbounded:

    D0_Date = Unbounded_D0_Date
    D0_F64 = Unbounded_D0_F64

class Last:

    D0_Date = Last_D0_Date
    Scalar_Date = Last_D0_Date

    D0_F64 = Last_D0_F64
    Scalar_F64 = Last_D0_F64

    D1_Date = Last_D1_Date
    Vector_Date = Last_D1_Date

    D1_F64 = Last_D1_F64
    Vector_F64 = Last_D1_F64

    D2_F64 = Last_D2_F64
    Matrix_F64 = Last_D2_F64


class History:
    Any = HistoryInterface

    D0_Date = History_D0_Date
    Scalar_Date = History_D0_Date

    D0_F64 = History_D0_F64
    Scalar_F64 = History_D0_F64

    D1_Date = History_D1_Date
    Vector_Date = History_D1_Date

    D1_F64 = History_D1_F64
    Vector_F64 = History_D1_F64

    D2_F64 = History_D2_F64
    Matrix_F64 = History_D2_F64

    Null = History_Null
    null = History_Null.new(np.zeros(0), 0, 2)


#  ------------------
