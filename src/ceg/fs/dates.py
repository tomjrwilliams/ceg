from typing import NamedTuple, ClassVar, overload, Literal, cast
from .. import core

import numpy

#  ------------------

import datetime as dt

class daily_kw(NamedTuple):
    type: str
    schedule: core.Schedule
    #
    v: core.Ref.Col
    start: dt.date
    end: dt.date

class daily(daily_kw, core.Node.Object):
    """
    todo: loop shoudl terminate at end
    >>> loop = core.loop.Fixed(1)
    >>> g = core.Graph.new()
    >>> with g.implicit() as (bind, done):
    ...     r = bind(None, ref=core.Ref.Col)
    ...     r = bind(
    ...         gaussian.new(r).sync(v=loop),
    ...         ref=r,
    ...     )
    ...     g = done()
    ...
    >>> g, es = g.steps(core.Event(0, r), n=6)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    """

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Object, daily_kw
    )

    @classmethod
    def new(
        cls, self, start: dt.date, end: dt.date
    ):
        return cls(
            *cls.args(), v=self, start=start, end=end
        )

    @classmethod
    def bind(
        cls,
        g: core.Graph,
        start: dt.date,
        end: dt.date,
        step=1.,
        using: core.TPlugin | tuple[core.TPlugin, ...] | None = None,
        # TODO: using
    ):
        loop = core.loop.FixedUntilDate(step, end)
        with g.implicit() as (bind, done):
            r = bind(None, core.Ref.Object, using)
            r = bind(
                cls.new(r, start, end).sync(v=loop),
                r,
                None,
            )
            g = done()
        r = cast(core.Ref.Object, r)
        return g, r


    def __call__(
        self, event: core.Event, graph: core.Graph
    ):
        vs = graph.select(self.v, event)
        if not len(vs):
            return self.start
        v = vs[-1] + dt.timedelta(days=1)
        if v > self.end:
            raise ValueError(self, v)
        return v