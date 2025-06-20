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


class pos_linear_kw(NamedTuple):
    type: str
    #
    signal: Ref.Scalar_F64
    scale: Ref.Scalar_F64

    d: Ref.Scalar_Date

    lower: float | None
    upper: float | None
    delta: float | None

    freq: str | None

    # TODO: just make this Ref.ref_f64 - dont need to implement for every type?

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.D0_F64_3:
        return Ref.d0_f64_3(i, slot=slot)

    @classmethod
    def new(cls, signal: Ref.Scalar_F64, scale: Ref.Scalar_F64, d: Ref.Scalar_Date, upper: float | None = None, lower: float | None = None, delta: float | None = None, freq: str | None = None):
        return pos_linear("pos_linear", signal=signal, scale=scale, d=d, upper = upper, lower = lower, delta=delta, freq=freq)


class pos_linear(pos_linear_kw, Node.D0_F64_3):
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

    DEF: ClassVar[Defn] = define.node(Node.D0_F64_3, pos_linear_kw)

    bind = define.bind_from_new(pos_linear_kw.new, pos_linear_kw.ref)

    def __call__(self, event: Event, graph: Graph):

        if event.prev is None:
            return np.nan, np.nan, np.nan

        sig_hist = self.signal.history(graph)
        scl_hist = self.scale.history(graph)
        d_hist = self.d.history(graph)

        v_sig = sig_hist.last_before(event.t)
        v_scl = scl_hist.last_before(event.t)

        d = d_hist.last_before(event.t)
        d_prev = d_hist.last_before(event.prev.t)

        assert d is not None, self
        assert d_prev is not None, self

        ref = cast(Ref.D0_F64_3, event.ref)

        hist_self_pos = ref.history(graph, slot=0)
        hist_self_sig = ref.history(graph, slot=1)
        hist_self_scl = ref.history(graph, slot=2)
        
        prev_pos = hist_self_pos.last_before(event.prev.t, allow_nan=False)
        prev_sig = hist_self_sig.last_before(event.prev.t, allow_nan=False)
        prev_scl = hist_self_scl.last_before(event.prev.t, allow_nan=False)

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
            pass
        elif self.freq == "M":
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

class pnl_linear_kw(NamedTuple):
    type: str
    #
    pos: Ref.D0_F64_3
    px: Ref.Scalar_F64

    # TODO: just make this Ref.ref_f64 - dont need to implement for every type?

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> Ref.Scalar_F64:
        return Ref.d0_f64(i, slot=slot)

    @classmethod
    def new(cls, pos: Ref.D0_F64_3, px: Ref.Scalar_F64, ):
        return pnl_linear("pnl_linear", pos = pos.select(4), px=px.select(4))


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

        # TODO: pos is actually signal

        if event.prev is None:
            return np.nan

        px_hist = self.px.history(graph)
        pos_hist = self.pos.history(graph, slot = 0)
        pos_prev = pos_hist.last_before(event.prev.t, allow_nan=False)
        
        px_curr = px_hist.last_before(event.t)
        px_prev = px_hist.last_before(event.prev.t, allow_nan=False)

        if pos_prev is None or px_curr is None or px_prev is None:
            return np.nan
        
        rx = px_curr - px_prev
        return rx * pos_prev

#  ------------------

# TODO: would be interesting to do pca on the spectrum
# take the high and low min / max 
# and then do pca on each bound to the next (as they're strictly ordered)