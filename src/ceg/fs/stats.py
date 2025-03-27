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
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
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
        window = event.t + 1 if self.window is None else self.window
        t, v = graph.select(self.v, event, t=True)
        return numpy.nanmean(v[t > event.t - window])


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


class cov_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v1: core.Ref.Col
    v2: core.Ref.Col
    window: float | None
    mu_1: core.Ref.Col | None
    mu_2: core.Ref.Col | None


class cov(cov_kw, core.Node.Col):
    """
    scalar mean (optional rolling window)
    v: core.Ref.Col
    window: float | None
    >>> g = core.Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
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
        core.Node.Col, cov_kw
    )

    @classmethod
    def new(
        cls,
        v1: core.Ref.Col,
        v2: core.Ref.Col,
        window: float | None = None,
        mu_1: core.Ref.Col | None = None,
        mu_2: core.Ref.Col | None = None,
    ):
        return cls(
            *cls.args(),
            v1=v1,
            v2=v2,
            window=window,
            mu_1=mu_1,
            mu_2=mu_2,
        )

    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        window = event.t + 1 if self.window is None else self.window
        # TODO: assert aligned?
        t1, v1 = graph.select(self.v1, event, t=True)
        t2, v2 = graph.select(self.v2, event, t=True)
        if self.mu_1 is None:
            mu_1 = numpy.nanmean(v1[t1 > event.t - window])
        else:
            mu_1 = graph.select(self.mu_1, event)[-1]
        if self.mu_2 is None:
            mu_2 = numpy.nanmean(v2[t2 > event.t - window])
        else:
            mu_2 = graph.select(self.mu_2, event)[-1]
        return numpy.nanmean(
            (v1 - mu_1) * (v2 - mu_2)
            # assume elementwise?
        )

#  ------------------
