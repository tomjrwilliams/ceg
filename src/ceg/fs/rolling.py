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

#  ------------------


@dataclass(frozen=True)
class sum(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(sum.new(r0, window=3))
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

    type: str
    #
    v: Ref.Scalar_F64
    window: int

    @classmethod
    def new(cls, v: Ref.Scalar_F64, window: int):
        return cls("sum", v=v, window=window)

    def __call__(self, event: Event, graph: Graph):
        return np.nansum(
            self.v.history(graph).last_n_before(
                self.window, event.t
            )
        )

#  ------------------



@dataclass(frozen=True)
class mean(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(mean.new(r0, window=3))
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
    1.99 1.56
    3.63 2.25
    4.74 3.46
    5.2 4.53
    """

    type: str
    #
    v: Ref.Scalar_F64
    window: int

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        window: int,
    ):
        return cls("mean", v=v, window=window)

    def __call__(self, event: Event, graph: Graph):
        return np.nanmean(
            self.v.history(graph).last_n_before(
                self.window, event.t
            )
        )


#  ------------------


@dataclass(frozen=True)
class mean_w(Node.Scalar_F64):

    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            "mean_w",
            v=v,
        )

    def __call__(self, event: Event, graph: Graph):
        raise ValueError(self)


#  ------------------

def alpha(span: float):
    return 2 / (span + 1)

def ewm_kwargs(
    span: float | None,
    alpha: float | None,
):
    kw = dict(span=span, alpha=alpha)
    if alpha is None:
        if span is None:
            raise ValueError(kw)
        alpha = 2 / (span + 1)
    if span is None:
        if alpha is None:
            raise ValueError(kw)
        span = (2 / alpha) - 1
    if 2 / (span + 1) != alpha:
        raise ValueError(kw)
    return dict(span=span, alpha=alpha)

@dataclass(frozen=True)
class mean_ew(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(mean_ew.new(r0, span=4),keep = 2)
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
    1.99 1.47
    3.63 2.34
    4.74 3.3
    5.2 4.06
    """
    type: str
    #
    v: Ref.Scalar_F64
    span: float | None
    alpha: float | None

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        span: float | None=None,
        alpha: float | None=None,
    ):
        return mean_ew("mean_ew", v=v, **ewm_kwargs(
            span=span, alpha=alpha
        ))
    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        if event.prev is None:
            return v
        rf: Ref.Scalar_F64 = cast(Ref.Scalar_F64, event.ref)
        prev = rf.history(graph).last_before(event.prev.t, allow_nan=False)
        if prev is None or np.isnan(prev):
            return v
        elif v is None or np.isnan(v):
            return prev
        alpha = self.alpha
        if alpha is None:
            raise ValueError(self)
        return ((1 - alpha) * prev) + (alpha * v)


#  ------------------




@dataclass(frozen=True)
class std(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(std.new(r0, window=3))
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
    1.13 0.0
    1.99 0.43
    3.63 1.04
    4.74 1.13
    5.2 0.66
    """

    type: str
    #
    v: Ref.Scalar_F64
    window: int

    @classmethod
    def new(cls, v: Ref.Scalar_F64, window: int):
        return std("std", v=v, window=window)

    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        # TODO: if only one observation, return nan?
        return np.nanstd(
            self.v.history(graph).last_n_before(
                self.window, event.t
            )
        )


#  ------------------



@dataclass(frozen=True)
class std_w(Node.Scalar_F64):
    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            "std_w",
            v=v,
        )

    def __call__(self, event: Event, graph: Graph):
        raise ValueError(self)


#  ------------------



@dataclass(frozen=True)
class std_ew(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(mean_ew.new(r0, span=4), keep = 2)
    ...     r2 = bind(std_ew.new(r0, r1, span=4), keep = 2)
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v2 = round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    1.13 1.13 0.0
    1.99 1.47 0.33
    3.63 2.34 0.86
    4.74 3.3 1.13
    5.2 4.06 1.13
    """

    type: str
    #
    v: Ref.Scalar_F64
    mu: Ref.Scalar_F64
    span: float | None
    alpha: float | None

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        mu: Ref.Scalar_F64,
        span: float | None=None,
        alpha: float | None=None,
    ):
        return std_ew("std_ew", v=v, mu=mu, **ewm_kwargs(
            span=span, alpha=alpha
        ))
    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        mu = self.mu.history(graph).last_before(event.t)
        if v is None or mu is None:
            return v
        if event.prev is None:
            return v - mu
        rf: Ref.Scalar_F64 = cast(Ref.Scalar_F64, event.ref)
        prev = rf.history(graph).last_before(event.prev.t, allow_nan=False)
        if prev is None or np.isnan(prev):
            return v
        alpha = self.alpha
        if alpha is None:
            raise ValueError(self)
        v = np.square(v - mu)
        prev = np.square(prev)
        return np.sqrt(
            ((1 - alpha) * prev) + (alpha * v)
        )


#  ------------------


@dataclass(frozen=True)
class rms(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(rms.new(r0, window=3))
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
    1.99 1.62
    3.63 2.48
    4.74 3.63
    5.2 4.57
    """

    type: str
    #
    v: Ref.Scalar_F64
    window: int

    @classmethod
    def new(cls, v: Ref.Scalar_F64, window: int):
        return rms("rms", v=v, window=window)

    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):

        return np.sqrt(
            np.nanmean(
                np.square(
                    self.v.history(graph).last_n_before(
                        self.window, event.t
                    )
                )
            )
        )


#  ------------------


@dataclass(frozen=True)
class rms_w(Node.Scalar_F64):

    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            "rms_w",
            v=v,
        )

    def __call__(self, event: Event, graph: Graph):
        raise ValueError(self)


#  ------------------


@dataclass(frozen=True)
class rms_ew(Node.Scalar_F64):
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
    type: str
    #
    v: Ref.Scalar_F64
    span: float | None
    alpha: float | None
    b: float

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        span: float | None=None,
        alpha: float | None=None,
        b: float = 1.,
    ):
        return rms_ew("rms_ew", v=v, b=b, **ewm_kwargs(
            span=span, alpha=alpha
        ))

    bind = define.bind_from_new(new, Node.Scalar_F64.ref)

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        if event.prev is None:
            if v is None:
                return v
            return np.abs(v) * self.b
        rf: Ref.Scalar_F64 = cast(Ref.Scalar_F64, event.ref)
        prev = rf.history(graph).last_before(event.prev.t, allow_nan=False)
        if prev is None or np.isnan(prev) or v is None or np.isnan(v):
            if v is None:
                return v
            return np.abs(v) * self.b
        alpha = self.alpha
        if alpha is None:
            raise ValueError(self)

        # TODO: re-fill the nans at plot stage

        if np.isnan(v):
            return prev
        prev_sq = np.square(prev / self.b)
        v_sq = np.square(v)
        
        return np.sqrt(
            ((1 - alpha) * prev_sq) + (alpha * v_sq)
        ) * self.b

#  ------------------




@dataclass(frozen=True)
class max( Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=0.2, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(max.new(r0, window=3))
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
    0.33 0.33
    0.39 0.39
    1.23 1.23
    1.54 1.54
    1.2 1.54
    """
    type: str
    #
    v: Ref.Scalar_F64
    window: int

    @classmethod
    def new(cls, v: Ref.Scalar_F64, window: int):
        return cls("max", v=v, window=window)

    def __call__(self, event: Event, graph: Graph):
        return np.nanmax(self.v.history(graph).last_n_before(
            self.window, event.t
        ))


#  ------------------



@dataclass(frozen=True)
class max_w(Node.Scalar_F64):

    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            "max_w",
            v=v,
        )

    def __call__(self, event: Event, graph: Graph):
        raise ValueError(self)


#  ------------------



@dataclass(frozen=True)
class max_ew(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=0.2, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(mean_ew.new(r0, span=4),keep = 2)
    ...     r2 = bind(max_ew.new(r0, r1, span=4),keep = 2)
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v2 = round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    0.33 0.33 0.33
    0.39 0.35 0.39
    1.23 0.71 1.23
    1.54 1.04 1.54
    1.2 1.1 1.37
    """

    type: str
    #
    v: Ref.Scalar_F64
    mu: Ref.Scalar_F64 | None
    span: float | None
    alpha: float | None

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        mu: Ref.Scalar_F64 | None = None,
        span: float | None=None,
        alpha: float | None=None,
    ):
        return max_ew("max_ew", v=v, mu=mu, **ewm_kwargs(
            span=span, alpha=alpha
        ))

    bind = define.bind_from_new(
        new, Node.Scalar_F64.ref
    )

    def __call__(self, event: Event, graph: Graph):
        alpha = self.alpha
        v = self.v.history(graph).last_before(event.t)
        if event.prev is None:
            return v
        rf: Ref.Scalar_F64 = cast(Ref.Scalar_F64, event.ref)
        prev = rf.history(graph).last_before(event.prev.t, allow_nan=False)
        if v is None or np.isnan(v):
            return np.nan
        if prev is None or np.isnan(prev):
            return v
        alpha = self.alpha
        if alpha is None:
            raise ValueError(self)
        if v > prev:
            return v
        if self.mu is None:
            return (1 - alpha) * prev
        mu = self.mu.history(graph).last_before(event.t, allow_nan=False)
        if mu is None or np.isnan(mu):
            return (1 - alpha) * prev
        return ((1 - alpha) * prev) + (alpha * mu)

#  ------------------




@dataclass(frozen=True)
class min(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(min.new(r0, window=3))
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
    -0.07 -0.07
    -0.41 -0.41
    0.03 -0.41
    -0.06 -0.41
    -0.8 -0.8
    """

    type: str
    #
    v: Ref.Scalar_F64
    window: int

    @classmethod
    def new(cls, v: Ref.Scalar_F64, window: int):
        return cls("min", v=v, window=window)

    def __call__(self, event: Event, graph: Graph):
        return np.nanmin(
            self.v.history(graph).last_n_before(
                self.window, event.t
            )
        )

#  ------------------



@dataclass(frozen=True)
class min_w(Node.Scalar_F64):
    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            "min_w",
            v=v,
        )

    def __call__(self, event: Event, graph: Graph):
        raise ValueError(self)


#  ------------------



@dataclass(frozen=True)
class min_ew(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(mean_ew.new(r0, span=4),keep = 2)
    ...     r2 = bind(min_ew.new(r0, r1, span=4),keep = 2)
    ...     g = done()
    ...
    >>> for g, es, t in batches(
    ...     g, Event.zero(r0), n=5, g=3, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     v2 = round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    -0.07 -0.07 -0.07
    -0.41 -0.21 -0.41
    0.03 -0.11 -0.29
    -0.06 -0.09 -0.21
    -0.8 -0.37 -0.8
    """

    type: str
    #
    v: Ref.Scalar_F64
    mu: Ref.Scalar_F64 | None
    span: float | None
    alpha: float | None

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        mu: Ref.Scalar_F64 | None = None,
        span: float | None=None,
        alpha: float | None=None,
    ):
        return cls("min_ew", v=v, mu=mu, **ewm_kwargs(
            span=span, alpha=alpha
        ))
    bind = define.bind_from_new(
        new, Node.Scalar_F64.ref
    )

    def __call__(self, event: Event, graph: Graph):
        v = self.v.history(graph).last_before(event.t)
        if event.prev is None:
            return v
        rf: Ref.Scalar_F64 = cast(Ref.Scalar_F64, event.ref)
        # TODO: fix allow nan......
        prev = rf.history(graph).last_before(event.prev.t, allow_nan=False)
        if v is None or np.isnan(v):
            return np.nan
        if prev is None or np.isnan(prev):
            return v
        alpha = self.alpha
        if alpha is None:
            raise ValueError(self)
        if v < prev:
            return v
        if self.mu is None:
            return (1 - alpha) * prev
        mu = self.mu.history(graph).last_before(event.t, allow_nan=False)
        if mu is None or np.isnan(mu):
            return (1 - alpha) * prev
        return ((1 - alpha) * prev) + (alpha * mu)

#  ------------------

# hinge (as per turn, av delta around point), incl w, ew
# possibly with an optional non linear transf (eg. sq, exp?)

# variance

# skew, kurtosis

# levy

#  ------------------



@dataclass(frozen=True)
class skew(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(skew.new(r0, window=3))
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
    -0.07 nan
    -0.41 nan
    0.03 -0.53
    -0.06 -0.58
    -0.8 -0.67
    """

    type: str
    #
    v: Ref.Scalar_F64
    window: int

    @classmethod
    def new(cls, v: Ref.Scalar_F64, window: int):
        return cls("skew", v=v, window=window)

    def __call__(self, event: Event, graph: Graph):
        return scipy.stats.skew(
            self.v.history(graph).last_n_before(
                self.window, event.t
            )
        )

@dataclass(frozen=True)
class kurtosis(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(kurtosis.new(r0, window=3))
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
    -0.07 nan
    -0.41 nan
    0.03 -1.5
    -0.06 -1.5
    -0.8 -1.5
    """

    type: str
    #
    v: Ref.Scalar_F64
    window: int

    @classmethod
    def new(cls, v: Ref.Scalar_F64, window: int):
        return cls("kurtosis", v=v, window=window)

    def __call__(self, event: Event, graph: Graph):
        return scipy.stats.kurtosis(
            self.v.history(graph).last_n_before(
                self.window, event.t
            )
        )

#  ------------------


@dataclass(frozen=True)
class quantile(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=-0.2, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     r1 = bind(quantile.new(r0, 0.5, window=3))
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
    -0.07 -0.07
    -0.41 -0.24
    0.03 -0.07
    -0.06 -0.06
    -0.8 -0.06
    """

    type: str
    #
    v: Ref.Scalar_F64
    q: float
    window: int

    @classmethod
    def new(cls, v: Ref.Scalar_F64, q: float, window: int):
        return cls("quantile", v=v,q=q, window=window)

    def __call__(self, event: Event, graph: Graph):
        return np.nanquantile(
            self.v.history(graph).last_n_before(
                self.window, event.t
            ),
            self.q
        )

#  ------------------



@dataclass(frozen=True)
class quantile_w(Node.Scalar_F64):

    type: str
    #
    v: Ref.Scalar_F64

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            "quantile_w",
            v=v,
        )

    def __call__(self, event: Event, graph: Graph):
        raise ValueError(self)


#  ------------------



@dataclass(frozen=True)
class quantile_ew(Node.Scalar_F64):
    type: str
    #
    v: Ref.Scalar_F64
    span: float | None
    alpha: float | None

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
        span: float | None=None,
        alpha: float | None=None,
    ):
        return cls("quantile_ew", v=v, **ewm_kwargs(
            span=span, alpha=alpha
        ))

    def __call__(self, event: Event, graph: Graph):
        raise ValueError(self)

# TODO: manually generate our own exp weights

#  ------------------



@dataclass(frozen=True)
class cov(Node.Scalar_F64):
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
    ...     r2 = bind(cov.new(r0, r1, 3))
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
    ...     v2 = round(
    ...         r2.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, v2)
    0.13 -0.13 0.0
    0.64 0.1 0.03
    -0.54 0.36 -0.06
    1.3 0.95 0.15
    -0.7 -1.27 0.64
    """

    type: str
    #
    v1: Ref.Scalar_F64
    v2: Ref.Scalar_F64
    window: int
    mu_1: Ref.Scalar_F64 | None
    mu_2: Ref.Scalar_F64 | None

    @classmethod
    def new(
        cls,
        v1: Ref.Scalar_F64,
        v2: Ref.Scalar_F64,
        window: int,
        mu_1: Ref.Scalar_F64 | None = None,
        mu_2: Ref.Scalar_F64 | None = None,
    ):
        return cls(
            "cov",
            v1=v1,
            v2=v2,
            window=window,
            mu_1=mu_1,
            mu_2=mu_2,
        )

    def __call__(self, event: Event, graph: Graph):
        v1 = self.v1.history(graph).last_n_before(
            self.window, event.t
        )
        v2 = self.v2.history(graph).last_n_before(
            self.window, event.t
        )

        # TODO: if only one observation, return nan?
        mu_1 = None
        if self.mu_1 is not None:
            mu_1 = self.mu_1.history(graph).last_before(
                event.t
            )
        if mu_1 is None:
            mu_1 = np.nanmean(v1)

        if self.mu_2 is not None:
            mu_2 = self.mu_2.history(graph).last_before(
                event.t
            )
        mu_2 = None
        if mu_2 is None:
            mu_2 = np.nanmean(v2)

        res = np.nanmean(
            (v1 - float(mu_1)) * (v2 - float(mu_2))
        )

        return res

# TODO: cov ew

#  ------------------



@dataclass(frozen=True)
class corr( Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> when = Loop.every(1)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> g, r1 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     s0 = bind(std.new(r0, 3))
    ...     s1 = bind(std.new(r1, 3))
    ...     r2 = bind(cov.new(r0, r1, 3))
    ...     co = bind(corr.new(r2, s0, s1))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=6, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     vco = round(
    ...         co.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, vco)
    1.13 0.87 nan
    2.77 1.97 1.0
    3.23 3.33 0.93
    5.53 5.28 0.97
    5.83 5.02 0.97
    """

    type: str
    #
    cov: Ref.Scalar_F64
    v1: Ref.Scalar_F64 # vol
    v2: Ref.Scalar_F64 # vol

    @classmethod
    def new(
        cls,
        cov: Ref.Scalar_F64,
        v1: Ref.Scalar_F64,
        v2: Ref.Scalar_F64,
    ):
        return cls(
            "corr",
            cov=cov,
            v1=v1,
            v2=v2,
        )

    def __call__(self, event: Event, graph: Graph):
        cov = self.cov.history(graph).last_before(event.t)
        v1 = self.v1.history(graph).last_before(event.t)
        v2 = self.v2.history(graph).last_before(event.t)
        if cov is None or v1 is None or v2 is None:
            return None
        if v1 == 0 or v2 == 0 or np.isnan(v1) or np.isnan(v2) or np.isnan(cov):
            return np.nan
        return cov / (v1 * v2)

# /TODO: corr ew


class beta_kw(NamedTuple):
    type: str
    #
    corr: Ref.Scalar_F64
    v1: Ref.Scalar_F64 # vol
    v2: Ref.Scalar_F64 # vol


@dataclass(frozen=True)
class beta(Node.Scalar_F64):
    """
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> when = Loop.every(1)
    >>> g, r0 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> g, r1 = rand.gaussian.walk(
    ...     g, mean=1, keep=3
    ... )
    >>> with g.implicit() as (bind, done):
    ...     s0 = bind(std.new(r0, 3))
    ...     s1 = bind(std.new(r1, 3))
    ...     r2 = bind(cov.new(r0, r1, 3))
    ...     co = bind(corr.new(r2, s0, s1))
    ...     b = bind(beta.new(co, s0, s1))
    ...     g = done()
    ...
    >>> es = [Event.zero(r0), Event.zero(r1)]
    >>> for g, es, t in batches(
    ...     g, *es, n=5, g=7, iter=True
    ... )():
    ...     v0 = round(
    ...         r0.history(g).last_before(t), 2
    ...     )
    ...     v1 = round(
    ...         r1.history(g).last_before(t), 2
    ...     )
    ...     vb = round(
    ...         b.history(g).last_before(t), 2
    ...     )
    ...     print(v0, v1, vb)
    1.13 0.87 nan
    2.77 1.97 1.48
    3.23 3.33 0.83
    5.53 5.28 0.86
    5.83 5.02 1.31
    """

    type: str
    #
    corr: Ref.Scalar_F64
    v1: Ref.Scalar_F64 # vol
    v2: Ref.Scalar_F64 # vol

    @classmethod
    def new(
        cls,
        corr: Ref.Scalar_F64,
        v1: Ref.Scalar_F64,
        v2: Ref.Scalar_F64,
    ):
        return cls(
            "beta",
            corr=corr,
            v1=v1,
            v2=v2,
        )

    def __call__(self, event: Event, graph: Graph):
        corr = self.corr.history(graph).last_before(event.t)
        v1 = self.v1.history(graph).last_before(event.t)
        v2 = self.v2.history(graph).last_before(event.t)
        if corr is None or v1 is None or v2 is None:
            return None
        # v1 = stock, v2 = market
        return (corr * v1) / v2

# TODO; beta  ew

# TODO: spearman

#  ------------------

