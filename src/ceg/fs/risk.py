from typing import NamedTuple, ClassVar, cast

import numpy as np
import scipy

from ..core import (
    Graph,
    Node,
    Ref,
    Event,
    Loop,
    Defn,
    define,
    steps,
    batches,
)

#  ------------------


class pnl_linear_kw(NamedTuple):
    type: str
    #
    pos: Ref.Scalar_F64
    px: Ref.Scalar_F64

    scale: Ref.Scalar_F64 | None

    # TODO: just make this Ref.ref_f64 - dont need to implement for every type?

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(cls, pos: Ref.Scalar_F64, px: Ref.Scalar_F64, scale: Ref.Scalar_F64 | None = None):
        return pnl_linear("pnl_linear", pos = pos.select(4), px=px.select(4), scale=scale if scale is None else scale.select(4))


class pnl_linear(pnl_linear_kw, Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(pnl_linear.new(r0, r0))
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    1.13 1.13
    1.99 3.12
    3.63 6.75
    4.74 10.37
    5.2 13.58
    """

    DEF: ClassVar[Defn] = define.node(Node.Scalar_F64, pnl_linear_kw)

    bind = define.bind_from_new(pnl_linear_kw.new, pnl_linear_kw.ref)

    def __call__(self, event: Event, graph: Graph):

        if event.prev is None:
            return np.nan

        px_hist = self.px.history(graph)
        pos_hist = self.pos.history(graph)

        scale_prev = np.nan
        if self.scale is not None:
            scale_hist = self.scale.history(graph)
            scale_prev = scale_hist.last_before(event.prev.t, allow_nan=False)

        if scale_prev is None or np.isnan(scale_prev):
            scale_prev = 1

        pos_prev = pos_hist.last_before(event.prev.t, allow_nan=False)
        
        px_curr = px_hist.last_before(event.t)
        px_prev = px_hist.last_before(event.prev.t, allow_nan=False)

        if pos_prev is None or px_curr is None or px_prev is None:
            return np.nan
        
        rx = px_curr - px_prev
        return rx * (pos_prev / scale_prev)

#  ------------------
