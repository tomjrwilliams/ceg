from typing import NamedTuple, ClassVar
from .. import core

import numpy

#  ------------------

RNG = {}


def rng(seed: int, reset: bool = False):
    if seed in RNG and not reset:
        return RNG[seed]
    gen = numpy.random.default_rng(seed)
    RNG[seed] = gen
    return gen


#  ------------------


class gaussian_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col
    mean: float
    std: float
    seed: int


class gaussian(gaussian_kw, core.Node.Col):
    """
    gaussian noise (pass v=self to get random walk)
    mean: float
    std: float
    >>> g = core.Graph.new()
    >>> loop = core.loop.Fixed(1)
    >>> g, r = g.bind(None, ref=core.Ref.Col)
    >>> g, r = g.bind(
    ...     gaussian.new(r).sync(v=loop),
    ...     ref=r,
    ... )
    >>> g, (*es, e) = g.steps(
    ...     core.Event(0, r), n=6
    ... )
    >>> ts, vs = core.mask(r, e, g.data)
    >>> list(numpy.round(vs, 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
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
        seed: int = 0,
    ):
        return cls(
            *cls.args(), v, mean=mean, std=std, seed=seed
        )

    def __call__(self, event: core.Event, data: core.Data):
        ts, vs = core.mask(self.v, event, data)
        step = rng(self.seed).normal(
            self.mean, self.std, size=None
        )
        if not len(vs):
            return step
        return vs[-1] + step


#  ------------------
