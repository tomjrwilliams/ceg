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
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> loop = core.loop.Fixed(1)
    >>> g = core.Graph.new()
    >>> g, r = g.bind(None, ref=core.Ref.Col)
    >>> g, r = g.bind(
    ...     rand.gaussian.new(r).sync(v=loop),
    ...     ref=r,
    ... )
    >>> g, mu = g.bind(mean.new(r))
    >>> g, mu_3 = g.bind(mean.new(r, window=3))
    >>> g, (*_, e) = g.steps(
    ...     core.Event(0, r), n=18
    ... )
    >>> list(numpy.round(g.select(r, e), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, e), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(numpy.round(g.select(mu_3, e), 2))
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
