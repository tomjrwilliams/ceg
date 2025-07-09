from typing import NamedTuple, ClassVar, cast

import numpy as np
import scipy

from ..core import (
    Graph,
    Node,
    Ref,
    Event,
    Loop,
    dataclass,
    define,
    steps,
    batches,
    ByDate,
)
from ..num.pca import *

#  ------------------

@dataclass(frozen=True)
class pca(Node.D1_F64_D2_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> when = Loop.every(1)
    >>> g, r0 = g.bind(
    ...     rand.gaussian.new(), when=when, keep=3
    ... )
    >>> g, r1 = g.bind(
    ...     rand.gaussian.new(), when=when, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(
    ...         pca.new(
    ...             (r0, r1), window=3, factors=1
    ...         )
    ...     )
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     eig = r2.history(g, slot=0).last_before(t)[0]
    ...     vec = r2.history(g, slot=1).last_before(t)[:, 0]
    ...     print(v0, v1, np.round(eig, 2), np.round(vec, 2))
    0.13 -0.13 nan [nan nan]
    0.64 0.1 0.66 [0.99 0.12]
    -0.54 0.36 0.86 [-0.97  0.24]
    1.3 0.95 1.74 [-0.87 -0.49]
    -0.7 -1.27 2.12 [-0.69 -0.72]
    """

    type: str
    #
    vs: tuple[Ref.Scalar_F64, ...]
    window: int
    factors: int
    mus: tuple[Ref.Scalar_F64, ...] | None
    signs: tuple[int | None] | None
    centre: bool

    @staticmethod
    def new(
        # cls,
        vs: tuple[Ref.Scalar_F64, ...],
        window: int,
        factors: int,
        mus: tuple[Ref.Scalar_F64, ...] | None = None,
        signs: tuple[int | None] | None = None,
        centre: bool = False,
    ):
        return pca(
            "pca",
            vs=vs,
            window=window,
            factors=factors,
            mus=mus,
            signs=signs,
            centre=centre,
        )

    def __call__(self, event: Event, graph: Graph):
        vs = [
            v.history(graph).last_n_before(
                self.window, event.t
            )
            for v in self.vs
        ]
        if self.mus is not None:
            mus = [
                mu.history(graph).last_before(event.t)
                for mu in self.mus
            ]
        else:
            mus = None
        
        e, U = svd_pca(
            vs,
            mus=mus,
            centre=self.centre,
            signs=self.signs,
            keep=self.factors
        )

        return e, U


@dataclass(frozen=True)
class pca_v(Node.D1_F64_D2_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> when = Loop.every(1)
    >>> g, r0 = g.bind(
    ...     rand.gaussian.new(), when=when, keep=3
    ... )
    >>> g, r1 = g.bind(
    ...     rand.gaussian.new(), when=when, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r2 = bind(
    ...         pca.new(
    ...             (r0, r1), window=3, factors=1
    ...         )
    ...     )
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     eig = r2.history(g, slot=0).last_before(t)[0]
    ...     vec = r2.history(g, slot=1).last_before(t)[:, 0]
    ...     print(v0, v1, np.round(eig, 2), np.round(vec, 2))
    0.13 -0.13 nan [nan nan]
    0.64 0.1 0.66 [0.99 0.12]
    -0.54 0.36 0.86 [-0.97  0.24]
    1.3 0.95 1.74 [-0.87 -0.49]
    -0.7 -1.27 2.12 [-0.69 -0.72]
    """

    type: str
    #
    vs: Ref.Vector_F64
    window: int
    factors: int
    mus: tuple[Ref.Scalar_F64, ...] | None
    signs: tuple[int | None] | None
    centre: bool

    @classmethod
    def month_end(
        cls, 
        g: Graph, 
        vs: Ref.Vector_F64,
        d: Ref.Scalar_Date,
        window: int,
        factors: int,
        mus: tuple[Ref.Scalar_F64, ...] | None = None,
        signs: tuple[int | None] | None = None,
        centre: bool = False,
    ):
        n = cls.new(
            vs=vs.select(window),
            window=window,
            factors=factors,
            mus=mus,
            signs=signs,
            centre=centre,
        )
        return g.bind(
            n,
            #  when=ByDate.month_end(d)
        )

    @staticmethod
    def new(
        # cls,
        vs: Ref.Vector_F64,
        window: int,
        factors: int,
        mus: tuple[Ref.Scalar_F64, ...] | None = None,
        signs: tuple[int | None] | None = None,
        centre: bool = False,
    ):
        return pca_v(
            "pca_v",
            vs=vs,
            window=window,
            factors=factors,
            mus=mus,
            signs=signs,
            centre=centre,
        )

    def __call__(self, event: Event, graph: Graph):
        vs = self.vs.history(graph).last_n_before(
            self.window, event.t
        )
        
        if self.mus is not None:
            mus = [
                mu.history(graph).last_before(event.t)
                for mu in self.mus
            ]
        else:
            mus = None
        
        e, U = svd_pca(
            vs,
            mus=mus,
            centre=self.centre,
            signs=self.signs,
            keep=self.factors
        )

        return e, U

#  ------------------


# TODO: pca_reg, pca_ew

# optionally fitting with some smoothing to previous value




#  ------------------

# 