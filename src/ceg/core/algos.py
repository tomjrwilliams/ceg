from typing import TypeVar, Callable
import functools

import numpy as np
import numba as nb

from frozendict import frozendict

#  ------------------

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

#  ------------------


def tuple_append_none(
    v: tuple[T | None, ...]
) -> tuple[T | None, ...]:
    return v + (None,)


def frozendict_append_tuple(
    d: frozendict[K, tuple[V, ...]], k: K, v: V
) -> frozendict[K, tuple[V, ...]]:
    return d.set(k, d.get(k, ()) + (v,))


def fold_star(acc, f: Callable, it):
    return functools.reduce(
        lambda ac, v: f(ac, *v), it, acc
    )


def set_tuple(
    t: tuple[T, ...], i: int, v: T, default: T
) -> tuple[T, ...]:
    # if already equal return
    if i > len(t):
        t = (
            t
            + tuple(
                (default for _ in range(i - len(t) - 1))
            )
            + (v,)
        )
    # elif t[i] is v or t[i] == v:
    #     pass
    else:
        t = tuple((*t[:i], v, *t[i + 1 :]))
    return t


#  ------------------

sig_last_before = nb.float64(
    nb.float64[:],
    nb.float64[:],
    nb.float64,
    nb.int64,
    nb.int64,
)
# NOTE: somehow slower with njit and sig?


def last_before_np(
    v: np.ndarray,
    t: np.ndarray,
    before: float,
    occupied: int,
    exponent: int,
):
    if v[0] > t:
        return np.nan
    return v[:occupied][t[:occupied] <= before][-1]


@nb.jit(fastmath=True)
def last_before_naive(
    v: np.ndarray,
    t: np.ndarray,
    before: float,
    occupied: int,
    exponent: int,
):
    """
    >>> EXPON = 3
    >>> round = lambda v: np.round(v, 4)
    >>> last_before = last_before_naive
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> list(round(vs))
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_before(
    ...         vs,
    ...         vs,
    ...         2,
    ...         occupied=1,
    ...         exponent=EXPON,
    ...     )
    ... )
    0.0
    >>> round(
    ...     last_before(
    ...         vs,
    ...         vs,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    1.1429
    >>> round(
    ...     last_before(
    ...         vs,
    ...         vs,
    ...         3,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    1.1429
    >>> round(
    ...     last_before(
    ...         vs,
    ...         vs,
    ...         8,
    ...         occupied=8,
    ...         exponent=EXPON,
    ...     )
    ... )
    8.0
    """
    if t[0] > before:
        return np.nan
    for i in range(occupied):
        ix = occupied - (1 + i)
        if t[ix] <= before:
            return v[ix]
    return np.nan


@nb.jit(fastmath=True)
def last_ix_before(
    t: np.ndarray,
    before: float,
    occupied: int,
    exponent: int,
):
    if t[0] > before:
        return -1
    pow = 1
    ix = step = int(2 ** (exponent - 1))
    while pow < exponent:
        tt = t[ix - 1]
        pow += 1
        step = step // 2
        if tt > before or ix >= occupied:
            ix -= step
        else:
            ix += step
    while t[ix] > before or ix >= occupied:
        ix -= 1
        if ix == -1:
            return -1
    return ix


@nb.jit(fastmath=True)
def last_before(
    v: np.ndarray,
    t: np.ndarray,
    before: float,
    occupied: int,
    exponent: int,
):
    """
    >>> EXPON = 3
    >>> round = lambda v: np.round(v, 4)
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> list(round(vs))
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_before(
    ...         vs,
    ...         vs,
    ...         2,
    ...         occupied=1,
    ...         exponent=EXPON,
    ...     )
    ... )
    0.0
    >>> round(
    ...     last_before(
    ...         vs,
    ...         vs,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    1.1429
    >>> round(
    ...     last_before(
    ...         vs,
    ...         vs,
    ...         3,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    1.1429
    >>> round(
    ...     last_before(
    ...         vs,
    ...         vs,
    ...         8,
    ...         occupied=8,
    ...         exponent=EXPON,
    ...     )
    ... )
    8.0
    """
    if t[0] > before:
        return np.nan
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return np.nan
    return v[ix]

@nb.jit(fastmath=True)
def last_before_not_nan(
    v: np.ndarray,
    t: np.ndarray,
    before: float,
    occupied: int,
    exponent: int,
):
    if t[0] > before:
        return np.nan
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return np.nan
    while np.isnan(v[ix]) and ix >= 0:
        ix -= 1
    if ix == -1:
        return np.nan
    return v[ix]

# TODO: parallel=true?

@nb.jit(fastmath=True)
def last_between(
    v: np.ndarray,
    t: np.ndarray,
    after: float,
    before: float,
    occupied: int,
    exponent: int,
):
    """
    >>> EXPON = 3
    >>> round = lambda v: np.round(v, 4)
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> list(round(vs))
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_between(
    ...         vs,
    ...         vs,
    ...         -1.0,
    ...         2,
    ...         occupied=1,
    ...         exponent=EXPON,
    ...     )
    ... )
    0.0
    >>> round(
    ...     last_between(
    ...         vs,
    ...         vs,
    ...         1,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    1.1429
    >>> round(
    ...     last_between(
    ...         vs,
    ...         vs,
    ...         1.2,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    nan
    >>> round(
    ...     last_between(
    ...         vs,
    ...         vs,
    ...         6.8,
    ...         7.5,
    ...         occupied=8,
    ...         exponent=EXPON,
    ...     )
    ... )
    6.8571
    >>> round(
    ...     last_between(
    ...         vs,
    ...         vs,
    ...         7,
    ...         7.5,
    ...         occupied=8,
    ...         exponent=EXPON,
    ...     )
    ... )
    nan
    """
    if t[0] > before:
        return np.nan
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return np.nan
    if t[ix] < after:
        return np.nan
    return v[ix]


def last_before_equiv():
    """
    >>> is_eq = (
    ...     lambda v0, v1: (
    ...         np.isnan(v0) and np.isnan(v1)
    ...     )
    ...     or v0 == v1
    ... )
    >>> vs = {
    ...     p: np.linspace(0, 2**p, 2**p)
    ...     for p in range(3, 18, 3)
    ... }
    >>> for p, v in vs.items():
    ...     n = 2**p
    ...     assert all(
    ...         (
    ...             is_eq(
    ...                 last_before_naive(
    ...                     v, v, i, offset, p
    ...                 ),
    ...                 last_before(
    ...                     v, v, i, offset, p
    ...                 ),
    ...             )
    ...             for i in range(n)
    ...             for offset in [
    ...                 int(n / 10),
    ...                 int(n / 3),
    ...                 int(n * 0.8),
    ...             ]
    ...         )
    ...     )
    ...
    """


#  ------------------


@nb.jit(fastmath=True)
def last_n_before_naive(
    v: np.ndarray,
    t: np.ndarray,
    n: int,
    before: float,
    occupied: int,
    exponent: int,
):
    res = np.empty(n, dtype=v.dtype)
    res[:] = np.nan
    if t[0] > before:
        return res
    for i in range(occupied):
        ix = occupied - (1 + i)
        if t[ix] <= before:
            ix += 1
            if ix >= n:
                res[:] = v[ix - n : ix]
            else:
                res[-ix:] = v[:ix]
            break
    return res



@nb.jit(fastmath=True)
def last_n_before(
    v: np.ndarray,
    t: np.ndarray,
    n: int,
    before: float,
    occupied: int,
    exponent: int,
):
    """
    >>> EXPON = 3
    >>> round = lambda v: list(np.round(v, 4))
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> round(vs)
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_n_before(
    ...         vs,
    ...         vs,
    ...         3,
    ...         3,
    ...         occupied=3,
    ...         exponent=EXPON,
    ...     )
    ... )
    [0.0, 1.1429, 2.2857]
    >>> round(
    ...     last_n_before(
    ...         vs,
    ...         vs,
    ...         3,
    ...         3,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    [nan, 0.0, 1.1429]
    >>> round(
    ...     last_n_before(
    ...         vs,
    ...         vs,
    ...         3,
    ...         2,
    ...         occupied=3,
    ...         exponent=EXPON,
    ...     )
    ... )
    [nan, 0.0, 1.1429]
    """
    res = np.empty(n, dtype=v.dtype)
    res[:] = np.nan
    if t[0] > before:
        return res
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return res
    ix += 1
    if ix >= n:
        res[:] = v[ix - n : ix]
    else:
        res[-ix:] = v[:ix]
    return res


def last_n_before_np(
    v: np.ndarray,
    t: np.ndarray,
    n: int,
    before: float,
    occupied: int,
    exponent: int,
):
    res = v[:occupied][t[:occupied] <= before][-n:]
    l = len(res)
    if l == n:
        return res
    nulls = np.empty(n - l)
    nulls[:] = np.nan
    return np.concatenate((nulls, res))


@nb.jit(fastmath=True)
def last_n_between(
    v: np.ndarray,
    t: np.ndarray,
    n: int,
    after: float,
    before: float,
    occupied: int,
    exponent: int,
):
    """
    >>> EXPON = 3
    >>> round = lambda v: list(np.round(v, 4))
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> round(vs)
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_n_between(
    ...         vs,
    ...         vs,
    ...         3,
    ...         -1.0,
    ...         3,
    ...         occupied=3,
    ...         exponent=EXPON,
    ...     )
    ... )
    [0.0, 1.1429, 2.2857]
    >>> round(
    ...     last_n_between(
    ...         vs,
    ...         vs,
    ...         3,
    ...         1,
    ...         3,
    ...         occupied=3,
    ...         exponent=EXPON,
    ...     )
    ... )
    [nan, 1.1429, 2.2857]
    """
    res = np.empty(n, dtype=v.dtype)
    res[:] = np.nan
    if t[0] > before:
        return res
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return res
    if t[ix] < after:
        return res
    ix += 1
    while t[ix - n] < after:
        n -= 1
        if n == 0:
            return res
    if ix >= n:
        res[-n:] = v[ix - n : ix]
    else:
        res[-ix:] = v[:ix]
    return res


#  ------------------


@nb.jit(fastmath=True)
def last_before_nd(
    v: np.ndarray,
    t: np.ndarray,
    size: int,
    before: float,
    occupied: int,
    exponent: int,
):
    """
    >>> EXPON = 3
    >>> round = lambda v: list(np.round(v, 4))
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> round(vs)
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_before_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         2,
    ...         occupied=1,
    ...         exponent=EXPON,
    ...     )
    ... )
    [0.0, 1.1429]
    >>> round(
    ...     last_before_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    [2.2857, 3.4286]
    >>> round(
    ...     last_before_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         3,
    ...         occupied=3,
    ...         exponent=EXPON,
    ...     )
    ... )
    [4.5714, 5.7143]
    """
    res = np.empty(size, dtype=v.dtype)
    res[:] = np.nan
    if t[0] > before:
        return res
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return res
    return v[ix * size : (ix + 1) * size]


@nb.jit(fastmath=True)
def last_between_nd(
    v: np.ndarray,
    t: np.ndarray,
    size: int,
    after: float,
    before: float,
    occupied: int,
    exponent: int,
):
    """
    >>> EXPON = 3
    >>> round = lambda v: list(np.round(v, 4))
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> round(vs)
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_between_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         1.0,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    [2.2857, 3.4286]
    >>> round(
    ...     last_between_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         1.5,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    [nan, nan]
    """
    res = np.empty(size, dtype=v.dtype)
    res[:] = np.nan
    if t[0] > before:
        return res
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return res
    if t[ix] < after:
        return res
    return v[ix * size : (ix + 1) * size]


@nb.jit(fastmath=True)
def last_n_before_nd(
    v: np.ndarray,
    t: np.ndarray,
    size: int,
    n: int,
    before: float,
    occupied: int,
    exponent: int,
):  # TODO: write the test with n, size variables (clearer)
    """
    >>> EXPON = 3
    >>> round = lambda v: list(np.round(v, 4))
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> round(vs)
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_n_before_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         2,
    ...         2,
    ...         occupied=1,
    ...         exponent=EXPON,
    ...     )
    ... )
    [nan, nan, 0.0, 1.1429]
    >>> round(
    ...     last_n_before_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         2,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    [0.0, 1.1429, 2.2857, 3.4286]
    >>> round(
    ...     last_n_before_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         2,
    ...         3,
    ...         occupied=3,
    ...         exponent=EXPON,
    ...     )
    ... )
    [2.2857, 3.4286, 4.5714, 5.7143]
    """
    res = np.empty(size * n, dtype=v.dtype)
    res[:] = np.nan
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return res
    ix += 1
    if ix >= n:
        res[:] = v[(ix - n) * size : ix * size]
    else:
        res[-ix * size :] = v[: ix * size]
    return res


@nb.jit(fastmath=True)
def last_n_between_nd(
    v: np.ndarray,
    t: np.ndarray,
    size: int,
    n: int,
    after: float,
    before: float,
    occupied: int,
    exponent: int,
):
    """
    >>> EXPON = 3
    >>> round = lambda v: list(np.round(v, 4))
    >>> vs = np.linspace(0, 2**EXPON, 2**EXPON)
    >>> assert vs.shape == (8,)
    >>> round(vs)
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(
    ...     last_n_between_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         2,
    ...         0,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    [0.0, 1.1429, 2.2857, 3.4286]
    >>> round(
    ...     last_n_between_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         2,
    ...         1.0,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    [nan, nan, 2.2857, 3.4286]
    >>> round(
    ...     last_n_between_nd(
    ...         vs,
    ...         vs,
    ...         2,
    ...         2,
    ...         1.5,
    ...         2,
    ...         occupied=2,
    ...         exponent=EXPON,
    ...     )
    ... )
    [nan, nan, nan, nan]
    """
    res = np.empty(size * n, dtype=v.dtype)
    res[:] = np.nan
    ix = last_ix_before(t, before, occupied, exponent)
    if ix == -1:
        return res
    if t[ix] < after:
        return res
    ix += 1
    while t[ix - n] < after:
        n -= 1
        if n == 0:
            return res
    if ix >= n:
        res[-n * size :] = v[(ix - n) * size : ix * size]
    else:
        res[-ix * size :] = v[: ix * size]
    return res

#  ------------------
