from typing import NamedTuple, ClassVar, cast

import datetime as dt

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
#  ------------------

@dataclass(frozen=True)
class pos_linear(Node.D0_F64_3):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(pos_linear.new(r0))
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=2, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g, slot = 0).last_before(t), 2
    ...     )
    ...     print(v0, v1)
    1.13 nan
    1.99 0.5
    3.63 0.28
    4.74 0.21
    5.2 0.19
    """

    type: str
    #
    scale: Ref.Scalar_F64

    d: Ref.Scalar_Date | None
    signal: Ref.Scalar_F64 | None

    const: float | None

    lower: float | None
    upper: float | None
    delta: float | None

    # TODO: min / max clip
    # TODO: scale needs inverting or not

    freq: str | None

    @classmethod
    def new(cls, scale: Ref.Scalar_F64, d: Ref.Scalar_Date | None = None, signal: Ref.Scalar_F64 | None=None, upper: float | None = None, lower: float | None = None, delta: float | None = None, freq: str | None = None, const: float | None = None):
        return pos_linear("pos_linear", signal=signal, scale=scale, d=d, upper = upper, lower = lower, delta=delta, freq=freq, const=const)

    bind = define.bind_from_new(new, Node.D0_F64_3.ref)

    def __call__(self, event: Event, graph: Graph):

        if event.prev is None:
            return np.nan, np.nan, np.nan

        if self.signal is not None:
            sig_hist = self.signal.history(graph)
            v_sig = sig_hist.last_before(event.t)
        else:
            v_sig = self.const or 1.

        scl_hist = self.scale.history(graph)

        v_scl = scl_hist.last_before(event.t)

        if self.d is not None:
            d_hist = self.d.history(graph)
            d = d_hist.last_before(event.t)
            d_prev = d_hist.last_before(event.prev.t)

            assert isinstance(d, dt.date), self
            assert isinstance(d_prev, dt.date), self
        else:
            d = d_prev = None

        ref = cast(Ref.D0_F64_3, event.ref)

        hist_self_pos = ref.history(graph, slot=0)
        hist_self_sig = ref.history(graph, slot=1)
        hist_self_scl = ref.history(graph, slot=2)
        
        prev_pos = hist_self_pos.last_before(event.prev.t, allow_nan=False, strict=False)
        prev_sig = hist_self_sig.last_before(event.prev.t, allow_nan=False, strict=False)
        prev_scl = hist_self_scl.last_before(event.prev.t, allow_nan=False, strict=False)

        if v_sig is None or v_scl is None:
            return prev_pos, prev_sig, prev_scl

        v_sig = np.clip(v_sig, a_min = -3., a_max=3.)

        if prev_pos is None or prev_sig is None or prev_scl is None:
            return v_sig / v_scl, v_sig, v_scl

        if np.isnan(v_scl):
            v_scl = prev_scl

        if np.isnan(v_sig):
            v_sig = prev_sig
        
        new_scl = None
        
        if self.freq is None:
            new_scl = scl_hist.last_before(event.t, allow_nan=False)
        else:
            assert d is not None, self
            assert d_prev is not None, self
            if self.freq == "M":
                if d.month != d_prev.month:
                    new_scl = scl_hist.last_before(event.t, allow_nan=False)
                    assert new_scl is not None, self
            elif self.freq == "D15":
                if d.day == 15:
                    new_scl = scl_hist.last_before(event.t, allow_nan=False)
                    assert new_scl is not None, self
            else:
                raise ValueError(self)
        
        if new_scl is None and not np.isnan(prev_scl):
            new_scl = prev_scl
        elif new_scl is None:
            new_scl = np.nan

        if self.lower is None and self.upper is None and self.freq is None:
            return v_sig / new_scl, v_sig, new_scl

        new_sig = None

        if self.delta is not None:
            delta = self.delta

            if abs(v_sig - prev_sig) > delta:
                new_sig = v_sig

        if self.upper is not None:
            upper: float = self.upper
            if v_sig > upper and prev_sig < upper:
                new_sig = v_sig
            
        if self.lower is not None:
            lower: float = self.lower
            if v_sig < lower and prev_sig > lower:
                new_sig = v_sig

        if new_sig is None and not np.isnan(prev_sig):
            new_sig = prev_sig
        elif new_sig is None:
            new_sig = v_sig

        new_sig = np.clip(new_sig, a_min = -3., a_max=3.)
        # TODO: freq can indicate in string if vol, sig or both?

        return new_sig / new_scl, new_sig, new_scl

#  ------------------



@dataclass(frozen=True)
class pnl_linear(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(pos_linear.new(r0))
    ...     r2 = bind(pnl_linear.new(r1, r0))
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g, slot=0).last_before(t), 2
    ...     )
    ...     v2 = round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    1.13 nan nan
    1.99 0.5 nan
    3.63 0.28 0.82
    4.74 0.21 0.3
    5.2 0.19 0.1
    """

    type: str
    #
    pos: Ref.D0_F64_3
    px: Ref.Scalar_F64

    @classmethod
    def new(cls, pos: Ref.D0_F64_3, px: Ref.Scalar_F64, ):
        return pnl_linear("pnl_linear", pos = pos.select(4), px=px.select(4))

    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):

        if event.prev is None: # return none?
            return np.nan

        px_hist = self.px.history(graph)
        pos_hist = self.pos.history(graph, slot = 0)
        pos_prev = pos_hist.last_before(
            event.prev.t, allow_nan=False, strict=False
        )
        # TODO: if nan -> prev = 0?
        # or try, if throw, prev = 0
        
        px_curr = px_hist.last_before(event.t)
        px_prev = px_hist.last_before(event.prev.t, allow_nan=False, strict=False)

        if pos_prev is None or px_curr is None or px_prev is None:
            return np.nan
        
        rx = px_curr - px_prev
        return rx * pos_prev

#  ------------------
