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

from .rolling import ewm_kwargs, alpha

#  ------------------

# class xo: pass the three components individually

#  ------------------



class xo_mean_ew_rms_ew_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    span_f: float
    span_s: float
    span_v: float
    rx: str

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.D0_F64_4:
        return Ref.d0_f64_4(i, slot=slot)

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        span_f: float,
        span_s: float,
        span_v: float,
        rx: str = "pct"
    ):
        return xo_mean_ew_rms_ew(
            "xo_mean_ew_rms_ew", 
            v=v, 
            span_f=span_f,
            span_s=span_s,
            span_v=span_v,
            rx=rx
        )


class xo_mean_ew_rms_ew(xo_mean_ew_rms_ew_kw, Node.D0_F64_4):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(rms_ew.new(r0, span=4),keep = 2)
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
    1.99 1.53
    3.63 2.59
    4.74 3.61
    5.2 4.32
    """

    DEF: ClassVar[Defn] = define.node(Node.D0_F64_4, xo_mean_ew_rms_ew_kw)
    bind = define.bind_from_new(xo_mean_ew_rms_ew_kw.new, xo_mean_ew_rms_ew_kw.ref)

    def __call__(self, event: Event, graph: Graph):

        v = self.v.history(graph).last_before(event.t)

        if event.prev is None:
            if v is None:
                return v
            return np.square(v)

        rf: Ref.D0_F64_4 = cast(Ref.D0_F64_4, event.prev.ref)

        prev_v = self.v.history(graph).last_before(event.prev.t)

        prev_f = rf.history(graph, slot = 1).last_before(event.prev.t)
        prev_s = rf.history(graph, slot = 2).last_before(event.prev.t)
        prev_vol = rf.history(graph, slot = 3).last_before(event.prev.t)

        if v is None or prev_v is None:
            return None, prev_f, prev_s, prev_vol

        # TODO: pass in the abs adjusted for pnl series and rms
        # to get in dollars

        # TODO: or rebase the pct_rx, cum prod
        # and do on that in pct
        # it's the signal anyway, doesnt need to be in dollars

        if (
            prev_f is None
            or prev_s is None
            or prev_vol is None
        ):
            return 0, v, v, np.square(v)
        
        alpha_f = alpha(self.span_f)
        alpha_s = alpha(self.span_s)
        alpha_v = alpha(self.span_v)

        if np.isnan(v):
            return v, prev_f, prev_s, prev_vol
    
        prev_sq = np.square(prev_vol)

        if self.rx == "pct":
            v_sq = np.square((v / prev_v) - 1)
        elif self.rx == "abs":
            v_sq = np.square(v - prev_v)
        else:
            raise ValueError(self)

        mu_f = ((1 - alpha_f) * prev_f) + (alpha_f * v)
        mu_s = ((1 - alpha_s) * prev_s) + (alpha_s * v)

        vol = np.sqrt(
            ((1 - alpha_v) * prev_sq) + (alpha_v * v_sq)
        )

        return (mu_f - mu_s) / vol, mu_f, mu_s, vol

#  ------------------
