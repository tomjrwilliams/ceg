from typing import NamedTuple, ClassVar
import numpy

from .. import core

#  ------------------


class mean_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col
    window: float | None


class mean(mean_kw, core.Node.Col):
    """
    scalar mean (optional rolling window)
    v: core.Ref.Col
    window: float | None
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> loop = core.loop.Fixed(1)
    >>> with g.implicit() as (bind, done):
    ...     r = bind(None, ref=core.Ref.Col)
    ...     r = bind(
    ...         rand.gaussian.new(r).sync(v=loop),
    ...         ref=r,
    ...     )
    ...     mu = bind(mean.new(r))
    ...     mu_3 = bind(mean.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, mean_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col, window: float | None = None
    ):
        return cls(*cls.args(), v=v, window=window)

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        if self.window is None:
            return numpy.nanmean(
                graph.select(self.v, event)
            )
        t, v = graph.select(self.v, event, t=True)
        return numpy.nanmean(v[t > event.t - self.window])


#  ------------------


class mean_w_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class mean_w(mean_w_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, mean_w_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------


class mean_ew_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class mean_ew(mean_ew_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, mean_ew_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------


class std_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class std(std_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, std_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------


class std_w_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class std_w(std_w_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, std_w_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------


class std_ew_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class std_ew(std_ew_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, std_ew_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------


class rms_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class rms(rms_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, rms_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------


class rms_w_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class rms_w(rms_w_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, rms_w_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------


class rms_ew_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class rms_ew(rms_ew_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, rms_ew_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------

# hinge (as per turn, av delta around point), incl w, ew

# variance

# quantile

# iqr

# max, min incl w, ew

# skew, kurtosis

#  ------------------


class ex_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col


class ex(ex_kw, core.Node.Col):

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, ex_kw
    )

    @classmethod
    def new(
        cls,
        v: core.Ref.Col,
    ):
        return cls(
            *cls.args(),
            v=v,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        pass


#  ------------------

# corr, cov, levy, beta (1D)

#  ------------------
