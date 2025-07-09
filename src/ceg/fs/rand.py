from typing import (
    NamedTuple,
    ClassVar,
    overload,
    Literal,
    cast,
)

from ..core import (
    Graph,
    Node,
    Ref,
    Event,
    Loop,
    dataclass,
    define,
    steps,
)

import numpy as np

#  ------------------


RNG = {}


@overload
def rng(
    seed: int, reset: Literal[False] = False
) -> np.random.Generator: ...


@overload
def rng(seed: int, reset: Literal[True] = True) -> None: ...


def rng(seed: int, reset: bool = False):
    if seed in RNG and not reset:
        return RNG[seed]
    gen = np.random.default_rng(seed)
    RNG[seed] = gen
    if reset:
        return
    return gen


#  ------------------


@dataclass(frozen=True)
class gaussian(Node.D0_F64):
    """
    gaussian noise (pass v=self to get random walk)
    mean: float
    std: float
    >>> rng(seed=0, reset=True)
    >>> g = Graph.new()
    >>> g, r = g.bind(
    ...     gaussian.new(), when=Loop.every(1)
    ... )
    >>> for g, e, t in steps(
    ...     g, Event.zero(r), n=3, iter=True
    ... )():
    ...     print(
    ...         np.round(
    ...             r.history(g).last_before(t), 4
    ...         )
    ...     )
    0.1257
    -0.1321
    0.6404
    >>> rng(seed=0, reset=True)
    >>> g = Graph.new()
    >>> g, r = gaussian.walk(g)
    >>> for g, e, t in steps(
    ...     g, Event.zero(r), n=3, iter=True
    ... )():
    ...     print(
    ...         np.round(
    ...             r.history(g).last_before(t), 4
    ...         )
    ...     )
    0.1257
    -0.0064
    0.634
    """
    type: str
    #
    mean: float
    std: float
    seed: int
    v: Ref.D0_F64 | None

    @staticmethod
    def new(
        # cls,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        v: Ref.D0_F64 | None = None,
    ):
        return gaussian("gaussian", mean=mean, std=std, seed=seed, v=v)
    
    @classmethod
    def walk(
        cls,
        g: Graph,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        step=1.0,
        keep: int = 2,
    ):
        # TODO: pass in walk start value
        g, r = g.bind(None, Ref.Scalar_F64)
        g, r = gaussian.new(
            mean, std, seed, v=r.select(last=keep)
        ).pipe(g.bind, r, Loop.Const.new(step))
        return g, cast(Ref.Scalar_F64, r)

    def __call__(self, event: Event, graph: Graph):
        step = rng(self.seed).normal(
            self.mean, self.std, size=None
        )
        if event.prev is None or self.v is None:
            return step
        v = self.v.history(graph).last_before(event.t)
        if v is None:
            return step
        return v + step


@dataclass(frozen=True)
class gaussian_1d(Node.D1_F64):
    """
    gaussian noise vector (pass v=self to get random walk)
    mean: float
    std: float
    >>> rng(seed=0, reset=True)
    >>> g = Graph.new()
    >>> g, r = g.bind(
    ...     gaussian_1d.new((2,)),
    ...     when=Loop.every(1),
    ... )
    >>> for g, e, t in steps(
    ...     g, Event.zero(r), n=3, iter=True
    ... )():
    ...     print(
    ...         np.round(
    ...             r.history(g).last_before(t), 4
    ...         )
    ...     )
    [ 0.1257 -0.1321]
    [0.6404 0.1049]
    [-0.5357  0.3616]
    >>> rng(seed=0, reset=True)
    >>> g = Graph.new()
    >>> g, r = gaussian_1d.walk(g, (2,))
    >>> for g, e, t in steps(
    ...     g, Event.zero(r), n=3, iter=True
    ... )():
    ...     print(
    ...         np.round(
    ...             r.history(g).last_before(t), 4
    ...         )
    ...     )
    [ 0.1257 -0.1321]
    [ 0.7662 -0.0272]
    [0.2305 0.3344]
    """
    type: str
    #
    shape: tuple[int]
    mean: float
    std: float
    seed: int
    v: Ref.D1_F64 | None

    @staticmethod
    def new(
        # cls,
        shape: tuple[int],
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        v: Ref.D1_F64 | None = None,
    ):
        return gaussian_1d(
            "gaussian_1d",
            shape,
            mean=mean,
            std=std,
            seed=seed,
            v=v,
        )

    @classmethod
    def walk(
        cls,
        g: Graph,
        shape: tuple[int],
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        step=1.0,
        keep: int = 1,
    ):
        g, r = g.bind(None, Ref.Vector_F64)
        g, r = cls.new(
            shape, mean, std, seed, v=r.select(last=keep)
        ).pipe(g.bind, r, Loop.Const.new(step))
        return g, cast(Ref.Vector_F64, r)

    def __call__(self, event: Event, graph: Graph):
        step = rng(self.seed).normal(
            self.mean, self.std, size=self.shape
        )
        if event.prev is None or self.v is None:
            return step
        v = self.v.history(graph).last_before(event.t)
        return v + step

gaussian_vec = gaussian_1d


@dataclass(frozen=True)
class gaussian_2d(Node.D2_F64):
    """
    gaussian noise matrix (pass v=self to get random walk)
    mean: float
    std: float
    >>> rng(seed=0, reset=True)
    >>> g = Graph.new()
    >>> g, r = g.bind(
    ...     gaussian_2d.new((2, 2)),
    ...     when=Loop.every(1),
    ... )
    >>> for g, e, t in steps(
    ...     g, Event.zero(r), n=3, iter=True
    ... )():
    ...     print(
    ...         np.round(
    ...             r.history(g).last_before(t), 4
    ...         )
    ...     )
    [[ 0.1257 -0.1321]
     [ 0.6404  0.1049]]
    [[-0.5357  0.3616]
     [ 1.304   0.9471]]
    [[-0.7037 -1.2654]
     [-0.6233  0.0413]]
    >>> rng(seed=0, reset=True)
    >>> g = Graph.new()
    >>> g, r = gaussian_2d.walk(g, (2, 2))
    >>> for g, e, t in steps(
    ...     g, Event.zero(r), n=3, iter=True
    ... )():
    ...     print(
    ...         np.round(
    ...             r.history(g).last_before(t), 4
    ...         )
    ...     )
    [[ 0.1257 -0.1321]
     [ 0.6404  0.1049]]
    [[-0.4099  0.2295]
     [ 1.9444  1.052 ]]
    [[-1.1137 -1.0359]
     [ 1.3211  1.0933]]
    """

    type: str
    #
    shape: tuple[int, int]
    mean: float
    std: float
    seed: int
    v: Ref.D2_F64 | None

    @staticmethod
    def new(
        # cls,
        shape: tuple[int, int],
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        v: Ref.D2_F64 | None = None,
    ):
        return gaussian_2d(
            "gaussian_2d",
            shape,
            mean=mean,
            std=std,
            seed=seed,
            v=v,
        )

    @classmethod
    def walk(
        cls,
        g: Graph,
        shape: tuple[int, int],
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        step=1.0,
        keep: int = 1,
    ):
        g, r = g.bind(None, Ref.Matrix_F64)
        g, r = cls.new(
            shape, mean, std, seed, v=r.select(last=keep)
        ).pipe(g.bind, r, Loop.Const.new(step))
        return g, cast(Ref.Matrix_F64, r)

    def __call__(self, event: Event, graph: Graph):
        step = rng(self.seed).normal(
            self.mean, self.std, size=self.shape
        )
        if event.prev is None or self.v is None:
            return step
        v = self.v.history(graph).last_before(event.t)
        return v + step

gaussian_mat = gaussian_2d