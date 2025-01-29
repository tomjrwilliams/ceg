import functools
from functools import reduce, partial
import operator

from frozendict import frozendict
from typing import TypeVar, Generic, Optional, Callable

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def identity(*v):
    return


class Immutable(Generic[T]):

    raw: tuple[T, ...]
    acc: frozendict[int, T]

    def __init__(self, raw, acc=frozendict()):
        self.raw = raw
        self.acc = acc

    # @classmethod
    # def new(cls, v: tuple[T, ...]):
    #     return cls(v, ) # type: ignore

    def __iter__(self):
        self = self.flush(self.acc)
        return self.raw.__iter__()

    def flush(self, acc=None):
        acc = self.acc if acc is None else acc
        raw = list(self.raw)
        any(
            map(
                partial(operator.setitem, raw),
                acc.keys(),
                acc.values(),
            )
        )
        return Immutable(tuple(raw), frozendict())

    def set(self, i: int, v: T):
        acc = self.acc.set(i, v)
        if len(acc) > len(self.raw) / 4:
            return self.flush(acc)
        return Immutable(self.raw, acc)


def add_none(
    v: tuple[Optional[T], ...]
) -> tuple[Optional[T], ...]:
    return v + (None,)


def set_tuple_add(
    d: frozendict[K, tuple[V, ...]], k: K, v: V
) -> frozendict[K, tuple[V, ...]]:
    return d.set(k, d.get(k, ()) + (v,))


def fold_star(acc, f: Callable, it):
    return functools.reduce(
        lambda ac, v: f(ac, *v), it, acc
    )


def set_tuple_star(t: tuple[T, ...], iv) -> tuple[T, ...]:
    return set_tuple(t, *iv)


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


if __name__ == "__main__":
    import random
    import time

    now = time.perf_counter

    N = 1000

    tup = tuple(range(N))
    imm = Immutable(tup)

    tup_ts = []
    imm_ts = []

    ROUNDS = 100

    for _ in range(ROUNDS):
        vs = random.sample(tup, k=N)
        idx = random.choices(list(range(N)), k=int(N / 3))

        for i in idx:
            n = now()
            tup = set_tuple(tup, i, vs[i], -1)
            for v in tup:
                pass
            tup_ts.append(now() - n)

            n = now()
            imm = imm.set(i, vs[i])
            for v in imm:
                pass
            imm_ts.append(now() - n)

    print("tup", sum(tup_ts))
    print("imm", sum(imm_ts))
