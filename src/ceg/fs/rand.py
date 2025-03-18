from typing import NamedTuple, ClassVar, overload, Literal
from .. import core

import numpy

#  ------------------

RNG = {}


@overload
def rng(
    seed: int, reset: Literal[False] = False
) -> numpy.random.Generator: ...


@overload
def rng(seed: int, reset: Literal[True] = True) -> None: ...


def rng(seed: int, reset: bool = False):
    if seed in RNG and not reset:
        return RNG[seed]
    gen = numpy.random.default_rng(seed)
    RNG[seed] = gen
    if reset:
        return
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
    >>> rng(seed=0, reset=True)
    >>> loop = core.loop.Fixed(1)
    >>> g = core.Graph.new()
    >>> with g.implicit() as (bind, done):
    ...     r = bind(None, ref=core.Ref.Col)
    ...     r = bind(
    ...         gaussian.new(r).sync(v=loop),
    ...         ref=r,
    ...     )
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=6)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
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

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        vs = graph.select(self.v, event)
        step = rng(self.seed).normal(
            self.mean, self.std, size=None
        )
        if not len(vs):
            return step
        return vs[-1] + step