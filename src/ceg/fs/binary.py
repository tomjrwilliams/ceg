# add
# div
# sub
# mul
# pow, log

from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from .. import core

#  ------------------

class ratio_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    l: core.Ref.Col
    r: core.Ref.Col


class ratio(ratio_kw, core.Node.Col):
    """
    scalar mean (optional rolling window)
    v: core.Ref.Col
    window: float | None
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     rat = bind(ratio.new(r, r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, ratio_kw
    )

    @classmethod
    def new(
        cls, l: core.Ref.Col, r: core.Ref.Col
    ):
        return cls(*cls.args(), l=l, r=r)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        l = graph.select(self.l, event, t=False, i = -1)
        r = graph.select(self.r, event, t= False, i = -1)
        if np.isnan(l) or np.isnan(r):
            return np.NAN
        return l / r

#  ------------------

class pct_diff_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    l: core.Ref.Col
    r: core.Ref.Col
    shift: tuple[int | None, ...] | None
    m: float | None = None


class pct_diff(pct_diff_kw, core.Node.Col):
    """
    scalar mean (optional rolling window)
    v: core.Ref.Col
    window: float | None
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     rat = bind(ratio.new(r, r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, pct_diff_kw
    )

    @classmethod
    def new(
        cls, l: core.Ref.Col, r: core.Ref.Col, shift: tuple[int | None, ...] | None = None, m: float | None = None,
    ):
        return cls(*cls.args(), l=l, r=r, shift=shift, m=m)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        # i_l = i_r = 1
        # if self.shift is not None:
        #     l_shift, r_shift = self.shift
        #     if l_shift is not None:
        #         i_l = i_l + l_shift
        #     if r_shift is not None:
        #         i_r = i_r + r_shift
        l = graph.select(self.l, event, t=False, i = -1)
        r = graph.select(self.r, event, t= False, i = -1)
        if np.isnan(l) or np.isnan(r):
            return np.NAN

        return ((l / r) - 1) * (1 if self.m is None else self.m)

#  ------------------

class subtract_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    l: core.Ref.Col
    r: core.Ref.Col


class subtract(subtract_kw, core.Node.Col):
    """
    scalar mean (optional rolling window)
    v: core.Ref.Col
    window: float | None
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     rat = bind(ratio.new(r, r))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, ratio_kw
    )

    @classmethod
    def new(
        cls, l: core.Ref.Col, r: core.Ref.Col
    ):
        return cls(*cls.args(), l=l, r=r)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        l = graph.select(self.l, event, t=False, i = -1)
        r = graph.select(self.r, event, t= False, i = -1)
        if np.isnan(l) or np.isnan(r):
            return np.NAN
        return l - r

diff = subtract