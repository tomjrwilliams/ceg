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
)

from .rolling import alpha

#  ------------------

# class xo: pass the three components individually

#  ------------------




@dataclass(frozen=True)
class xo_mean_ew_rms_ew(Node.D0_F64_4):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(xo_mean_ew_rms_ew.new(
    ...         r0, span_f=4, span_s=12, span_v=6
    ...     ), keep = 2)
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v10 = round(
    ...         r1.history(g, slot=0).last_before(t), 2
    ...     )
    ...     v11 = round(
    ...         r1.history(g, slot=1).last_before(t), 2
    ...     )
    ...     v12 = round(
    ...         r1.history(g, slot=2).last_before(t), 2
    ...     )
    ...     v13 = round(
    ...         r1.history(g, slot=3).last_before(t), 2
    ...     )
    ...     print(v0, v10, v11, v12, v13)
    1.13 0.0 1.13 1.13 1.13
    1.99 0.2 1.47 1.26 1.06
    3.63 0.57 2.34 1.62 1.25
    4.74 0.99 3.3 2.1 1.21
    5.2 1.4 4.06 2.58 1.05
    """

    type: str
    
    v: Ref.Scalar_F64
    span_f: float
    span_s: float
    span_v: float
    rx: str

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        span_f: float,
        span_s: float,
        span_v: float,
        rx: str = "abs"
    ):
        return xo_mean_ew_rms_ew(
            "xo_mean_ew_rms_ew", 
            v=v, 
            span_f=span_f,
            span_s=span_s,
            span_v=span_v,
            rx=rx
        )

    bind = define.bind_from_new(new, Node.D0_F64_4.ref)

    def __call__(self, event: Event, graph: Graph):

        v = self.v.history(graph).last_before(event.t)

        if event.prev is None:
            if v is None:
                return None
            return 0., v, v, np.abs(v)

        rf: Ref.D0_F64_4 = cast(Ref.D0_F64_4, event.prev.ref)

        prev_v = self.v.history(graph).last_before(event.prev.t)

        prev_f = rf.history(graph, slot = 1).last_before(event.prev.t)
        prev_s = rf.history(graph, slot = 2).last_before(event.prev.t)
        prev_vol = rf.history(graph, slot = 3).last_before(event.prev.t)

        if v is None or prev_v is None:
            return None

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
            return 0., v, v, np.abs(v)
        
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

        xo = (mu_f - mu_s) / vol

        return xo, mu_f, mu_s, vol

#  ------------------
