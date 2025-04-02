import operator
from functools import reduce
from typing import NamedTuple, Sequence, Union, ClassVar

import numpy

#  ------------------

# TODO: prefix with float, int, bool (ArrayRow even multi d could in theory acc vertically, unpacking row fields into cols?)

np = numpy.ndarray

np_f64 = numpy.float64

np_1D = numpy.ndarray[tuple[int], numpy.dtype[np_f64]]
np_2D = numpy.ndarray[tuple[int, int], numpy.dtype[np_f64]]

Shape = int | tuple[int, int] | tuple[int, int, int]

#  ------------------


class ArrayAny(NamedTuple):
    """
    shape: tuple[int, ...]
    size: int
    n: int
    incr: int
    method: str
    raw: numpy.ndarray
    """

    shape: tuple[int, ...]
    size: int
    n: int
    incr: int
    method: str
    raw: numpy.ndarray | list

    @property
    def last(self):
        raise ValueError(self)

Any = ArrayAny

class ArrayList(ArrayAny):

    shape: tuple[int, ...]
    size: int
    n: int
    incr: int
    method: str
    raw: list

class ArrayND(ArrayAny):

    shape: tuple[int, ...]
    size: int
    n: int
    incr: int
    method: str
    raw: numpy.ndarray



#  ------------------


class ArrayNull(ArrayND):

    @classmethod
    def new(cls, incr: int = 2, method: str = "exp"):
        return cls(
            shape=(),
            size=1,
            n=0,
            incr=incr,
            method=method,
            raw=numpy.empty(0),
        )


Null = ArrayNull

#  ------------------


def needs_resize(
    arr: numpy.ndarray,
    v_size: int,
    n_curr: int,
    n_new: int,
):
    required: int = (n_curr + n_new) * v_size
    return required >= arr.size


# TODO: numba jit
def resize_lin(
    arr: numpy.ndarray,
    v_size: int,
    n_curr: int,
    # n_new: int,
    n_incr: int,
):
    # v_size: int = reduce(operator.mul, v_shape, 1)
    # required: int = (n_curr + n_new) * v_size
    # if required <= arr.size:
    #     return arr
    res = numpy.empty((arr.size + n_incr))
    for i in range(n_curr):
        # TODO if n was the number of instances, we'd multiply by size, but given it coutns the number of filled cells in the vector, we just iterate directly
        res[i] = arr[i]
    return res


# TODO: numba jit
def resize_exp(
    arr: numpy.ndarray,
    v_size: int,
    n_curr: int,
    # n_new: int,
    n_incr: int,
):
    # required: int = (n_curr + n_new) * v_size
    # if required <= arr.size:
    #     return arr
    res = numpy.empty((arr.size * n_incr))
    for i in range(n_curr):
        res[i] = arr[i]
    return res


# TODO: factor out the update / write, so we can then call them in the call method of the relevant shape node

#  ------------------


def array_v_size(*shape: int):
    return reduce(operator.mul, shape, 1)


#  ------------------

class ArrayObject(ArrayList):

    DIMS: ClassVar[int] = 0

    @classmethod
    def new(cls, incr: int = 2, method: str = "exp"):
        return cls(
            shape=(),
            size=1,
            n=0,
            incr=incr,
            method=method,
            raw=[],
        )


    def add(self, v: float):
        self.raw.append(v)
        self = self._replace(n=self.n + 1)
        return self


    @property
    def data(self):
        return self.raw[: self.n]

    @property
    def last(self):
        if self.n == 0:
            return None
        return self.raw[self.n - 1]

Object = ArrayObject

class Array(ArrayND):

    DIMS: ClassVar[int] = 0

    @classmethod
    def new(cls, incr: int = 2, method: str = "exp"):
        return cls(
            shape=(),
            size=1,
            n=0,
            incr=incr,
            method=method,
            raw=numpy.empty(32),
        )

    def needs_resize(self, n):
        return needs_resize(self.raw, self.size, self.n, n)

    def resize(self):
        # TODO; swithc on method
        raw = resize_exp(
            self.raw, self.size, self.n, self.incr
        )
        return self._replace(
            raw=raw,
        )

    def add(self, v: float):
        if self.needs_resize(1):
            self = self.resize()
        self.raw[self.n] = v
        self = self._replace(n=self.n + 1)
        return self

    def add_many(self, v: numpy.ndarray):
        assert len(v.shape) == 1, v.shape
        n: int = v.size
        while self.needs_resize(n):
            self = self.resize()
        for i in range(n):
            self.raw[self.n + i] = v[i]
        self = self._replace(n=self.n + 1)
        return self

    @property
    def data(self):
        return self.raw[: self.n]

    @property
    def last(self):
        if self.n == 0:
            return None
        return self.raw[self.n - 1]


D0 = Scalar = Array


class Array1D(ArrayND):
    """
    the naming convention is in terms of the array items (ie. the data will be 2D for Array1D)
    """

    DIMS: ClassVar[int] = 1

    # TODO: possibly change to incremental rather than exponential re-sizing

    @classmethod
    def new(cls, incr: int = 2, method: str = "exp"):
        return cls(
            shape=(),
            size=0,
            n=0,
            incr=incr,
            method=method,
            raw=numpy.empty(32),
        )

    def needs_resize(self, n):
        return needs_resize(self.raw, self.size, self.n, n)

    def resize(self):
        # TODO; swithc on method
        raw=resize_exp(
            self.raw, self.size, self.n, self.incr
        )
        return self._replace(
            raw=raw
        )

    def add(self, v: numpy.ndarray):
        assert len(v.shape) == 1, v.shape
        if self.shape == ():
            self = self._replace(
                shape=v.shape,
                size=v.size,
            )
        assert v.shape == self.shape, (self.shape, v.shape)
        while self.needs_resize(1):
            self = self.resize()
        for i in range(v.size):
            try:
                self.raw[self.n + i] = v[i]
            except:
                raise ValueError(self)
        self = self._replace(n=self.n + v.size)
        return self

    def add_many(self, v: numpy.ndarray):
        assert len(v.shape) == 2, v.shape
        if self.shape == ():
            self = self._replace(
                shape=v[:-1].shape,
                size=v.size,
            )
        assert v.shape[:-1] == self.shape, (
            self.shape,
            v.shape,
        )
        while self.needs_resize(v.shape[0]):
            self = self.resize()
        v_flat = numpy.ravel(v)
        for i in range(v_flat.size):
            self.raw[self.n + i] = v_flat[i]
        self = self._replace(n=self.n + v_flat.size)
        return self

    @property
    def data(self):
        (width,) = self.shape
        data = self.raw[: self.n]
        return numpy.reshape(
            data, (int(len(data) / width), width)
        )


D1 = Vec = Vector = Array1D

#  ------------------

class Array2D(ArrayND):
    
    DIMS: ClassVar[int] = 2

    @classmethod
    def new(cls, incr: int = 2, method: str = "exp"):
        return cls(
            shape=(),
            size=1,
            n=0,
            incr=incr,
            method=method,
            raw=numpy.empty(32),
        )

D2 = Matrix = Mat = Array2D

#  ------------------


class ArrayRow(ArrayND):

    # allocate a c array of given size
    # fill with structs

    def add(self, v):
        return


#  ------------------
