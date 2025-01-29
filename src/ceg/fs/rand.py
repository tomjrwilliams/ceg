from typing import NamedTuple, ClassVar
from .. import core

import numpy

rng = numpy.random.default_rng(42069)

#  ------------------


class gaussian_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col
    mean: float
    std: float
    # TODO: seed?


class gaussian(gaussian_kw, core.Node.Col):
    """
    gaussian noise (pass v=self to get random walk)
    mean: float
    std: float
    >>> g = core.Graph.new()
    >>> g, r = g.bind(None, ref=core.Ref.Col)
    >>> g, r = g.bind(
    ...     gaussian.new(r).sync(
    ...         v=core.loop.Fixed(1)
    ...     ),
    ...     ref=r,
    ... )
    >>> g, (*es, e) = g.steps(core.Event(0, r), n = 5)
    >>> ts, vs = core.mask(r, e, g.data)
    >>> numpy.round(vs, 2)
    array([-1.03, -0.42,  0.33, -0.56,  0.05])
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, gaussian_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        return cls(*cls.args(), v, mean=mean, std=std)

    def __call__(self, event: core.Event, data: core.Data):
        ts, vs = core.mask(self.v, event, data)
        step = rng.normal(self.mean, self.std, size=None)
        if not len(vs):
            return step
        return vs[-1] + step


#  ------------------
