# add
# div
# sub
# mul
# pow, log

from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from ..core import Graph, Node, Ref, Event, Loop, Defn, define, steps

#  ------------------

class ratio_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64


class ratio(ratio_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     rat = bind(ratio.new(r, r))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, ratio_kw
    )

    @classmethod
    def new(
        cls, l: Ref.Scalar_F64, r: Ref.Scalar_F64
    ):
        return cls(cls.DEF.name, l=l, r=r)

    def __call__(
        self, event: Event, graph: Graph
    ):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        if np.isnan(l) or np.isnan(r):
            return np.NAN
        return l / r

#  ------------------

class pct_diff_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64
    shift: tuple[int | None, ...] | None
    m: float | None = None


class pct_diff(pct_diff_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     rat = bind(ratio.new(r, r))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, pct_diff_kw
    )

    @classmethod
    def new(
        cls, l: Ref.Scalar_F64, r: Ref.Scalar_F64, shift: tuple[int | None, ...] | None = None, m: float | None = None,
    ):
        return cls(cls.DEF.name, l=l, r=r, shift=shift, m=m)

    def __call__(
        self, event: Event, graph: Graph
    ):
        # i_l = i_r = 1
        # if self.shift is not None:
        #     l_shift, r_shift = self.shift
        #     if l_shift is not None:
        #         i_l = i_l + l_shift
        #     if r_shift is not None:
        #         i_r = i_r + r_shift
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        if np.isnan(l) or np.isnan(r):
            return np.NAN

        return ((l / r) - 1) * (1 if self.m is None else self.m)

#  ------------------

class subtract_kw(NamedTuple):
    type: str
    #
    l: Ref.Scalar_F64
    r: Ref.Scalar_F64


class subtract(subtract_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     rat = bind(ratio.new(r, r))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, ratio_kw
    )

    @classmethod
    def new(
        cls, l: Ref.Scalar_F64, r: Ref.Scalar_F64
    ):
        return cls(cls.DEF.name, l=l, r=r)

    def __call__(
        self, event: Event, graph: Graph
    ):
        l = self.l.history(graph).last_before(event.t)
        r = self.r.history(graph).last_before(event.t)
        if np.isnan(l) or np.isnan(r):
            return np.NAN
        return l - r

diff = subtract