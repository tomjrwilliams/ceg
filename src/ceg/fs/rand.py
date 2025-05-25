from typing import NamedTuple, ClassVar, overload, Literal, cast

from ..core import Graph, Node, Ref, Event, Loop, Defn, define, steps

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
    #
    v: Ref.D0_F64
    mean: float
    std: float
    seed: int


class gaussian(gaussian_kw, Node.D0_F64):
    """
    gaussian noise (pass v=self to get random walk)
    mean: float
    std: float
    >>> rng(seed=0, reset=True)
    >>> loop = loop.Fixed(1)
    >>> g = Graph.new()
    >>> g, r = gaussian.bind(g)
    >>> g, es = g.steps(Event(0, r), n=6)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    """

    DEF: ClassVar[Defn] = define(
        Node.D0_F64, gaussian_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.D0_F64,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
    ):
        return cls(
            cls.DEF.name, v, mean=mean, std=std, seed=seed
        )

    @classmethod
    def bind(
        cls,
        g: Graph,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        step=1.,
        using: TPlugin | tuple[TPlugin, ...] | None = None,
        # TODO: using
    ):
        loop = loop.Fixed(step)
        with g.implicit() as (bind, done):
            r = bind(None, Ref.Object, using)
            r = cast(Ref.D0_F64, r)
            r = bind(
                cls.new(r, mean, std, seed).sync(v=loop),
                r,
                None,
            )
            g = done()
        r = cast(Ref.D0_F64, r)
        return g, r

    def __call__(
        self, event: Event, graph: Graph
    ):
        vs = graph.select(self.v, event)
        step = rng(self.seed).normal(
            self.mean, self.std, size=None
        )
        if not len(vs):
            return step
        return vs[-1] + step