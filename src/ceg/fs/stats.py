from typing import NamedTuple, ClassVar
import numpy

from .. import core

#  ------------------


class mean_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col
    window: float | None


class mean(mean_kw, core.Node.Col):
    """
    scalar mean (optional rolling window)
    v: core.Ref.Col
    window: float | None
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g = core.Graph.new()
    >>> loop = core.loop.Fixed(1)
    >>> g, r = g.bind(None, ref=core.Ref.Col)
    >>> g, r = g.bind(
    ...     rand.gaussian.new(r).sync(v=loop),
    ...     ref=r,
    ... )
    >>> g, mu = g.bind(mean.new(r))
    >>> g, mu_3 = g.bind(mean.new(r, window=3))
    >>> g, (*es, e) = g.steps(
    ...     core.Event(0, r), n=18
    ... )
    >>> _, rs = core.mask(r, e, g.data)
    >>> _, mus = core.mask(mu, e, g.data)
    >>> _, mu_3s = core.mask(mu_3, e, g.data)
    >>> list(numpy.round(rs, 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(mus, 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(numpy.round(mu_3s, 2))
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, mean_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col, window: float | None = None
    ):
        return cls(*cls.args(), v=v, window=window)

    def __call__(self, event: core.Event, data: core.Data):
        t, v = core.mask(self.v, event, data)
        if self.window is None:
            return numpy.nanmean(v)
        return numpy.nanmean(v[t > event.t - self.window])


#  ------------------
