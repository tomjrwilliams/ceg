
import numpy as np
import numba as nb


sig_last_before = nb.float64(
    nb.float64[:], nb.float64[:], nb.float64, nb.int64, nb.int64
)
# NOTE: somehow slower with njit and sig?

@nb.jit(fastmath=True)
def last_before_naive(
    v: np.ndarray, 
    t: np.ndarray, 
    at: float, 
    occupied: int,
    pow_size: int,
):
    """
    >>> POWER = 3
    >>> round = lambda v: np.round(v, 4)
    >>> last_before = last_before_naive
    >>> vs = np.linspace(0, 2 ** POWER, 2 ** POWER)
    >>> assert vs.shape == (8,)
    >>> list(round(vs))
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(last_before(vs, vs, 2, 1, POWER))
    0.0
    >>> round(last_before(vs, vs, 2, 2, POWER))
    1.1429
    >>> round(last_before(vs, vs, 3, 2, POWER))
    1.1429
    >>> round(last_before(vs, vs, 8, 8, POWER))
    8.0
    """
    if t[0] > at:
        return np.NAN
    for i in range(occupied):
        ix = occupied-(1 + i)
        if t[ix] <= at:
            return v[ix]
    return np.NAN

@nb.jit(fastmath=True)
def last_before(
    v: np.ndarray, 
    t: np.ndarray, 
    at: float, 
    occupied: int,
    pow_size: int
):
    """
    >>> POWER = 3
    >>> round = lambda v: np.round(v, 4)
    >>> vs = np.linspace(0, 2 ** POWER, 2 ** POWER)
    >>> assert vs.shape == (8,)
    >>> list(round(vs))
    [0.0, 1.1429, 2.2857, 3.4286, 4.5714, 5.7143, 6.8571, 8.0]
    >>> round(last_before(vs, vs, 2, 1, POWER))
    0.0
    >>> round(last_before(vs, vs, 2, 2, POWER))
    1.1429
    >>> round(last_before(vs, vs, 3, 2, POWER))
    1.1429
    >>> round(last_before(vs, vs, 8, 8, POWER))
    8.0
    """
    if t[0] > at:
        return np.NAN
    l = v.shape[0]
    pow = 1
    ix = step = l // 2
    while pow < pow_size:
        tt = t[ix-1]
        pow += 1
        step = step // 2
        if tt > at or ix >= occupied:
            ix -= step
        else:
            ix += step
    while t[ix] > at or ix >= occupied:
        ix -= 1
        if ix == -1:
            return np.NAN
    return v[ix]

def last_before_equiv():
    """
    >>> is_eq = lambda v0, v1: (
    ...     np.isnan(v0) and np.isnan(v1) 
    ... ) or v0 == v1
    >>> vs = {
    ...     p: np.linspace(0, 2 ** p, 2 ** p)
    ...     for p in range(3, 18, 3)
    ... }
    >>> for p, v in vs.items():
    ...     n = 2 ** p
    ...     assert all((
    ...         is_eq(
    ...             last_before_naive(v, v, i, offset, p),
    ...             last_before(v, v, i, offset, p),
    ...         ) 
    ...         for i in range(n) 
    ...         for offset in [int(n / 10), int(n/3), int(n * .8)]
    ...     ))
    """