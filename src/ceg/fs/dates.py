from typing import (
    NamedTuple,
    ClassVar,
    overload,
    Literal,
    cast,
)

from ..core import (
    Graph,
    Node,
    Ref,
    Event,
    Loop,
    Defn,
    define,
    steps,
)

#  ------------------

import datetime as dt


class daily_kw(NamedTuple):
    type: str
    #
    prev: Ref.Scalar_Date
    start: dt.date
    end: dt.date


class daily(daily_kw, Node.Scalar_Date):
    """
    >>> start = dt.date(2025, 1, 1)
    >>> end = dt.date(2025, 1, 6)
    >>> g, r = Graph.new().pipe(
    ...     daily.loop, start, end
    ... )
    >>> g, e, t = g.pipe(
    ...     steps, Event.zero(r), n=6
    ... ).last()
    >>> e.ref.history(g).last_before(t)
    datetime.date(2025, 1, 6)
    """

    DEF: ClassVar[Defn] = define(Node.Scalar_Date, daily_kw)

    @classmethod
    def new(
        cls,
        self: Ref.Scalar_Date,
        start: dt.date,
        end: dt.date,
    ):
        return cls(
            cls.DEF.name, prev=self, start=start, end=end
        )

    @classmethod
    def loop(
        cls,
        g: Graph,
        start: dt.date,
        end: dt.date,
        step=1.0,
        keep: int = 1,
    ):
        g, r = g.bind(None, Ref.Scalar_Date)
        g, r = cls.new(
            r.select(last=keep), start, end
        ).pipe(g.bind, r, Loop.UntilDate.new(step, end, r))
        return g, cast(Ref.Scalar_Date, r)

    def __call__(self, event: Event, graph: Graph):
        h = self.prev.history(graph, strict=False)
        if h is None:
            if event.ref.eq(self.prev):
                return self.start
            raise ValueError(dict(self=self, h=h))
        d = h.last_before(event.t)
        assert d is not None, self
        d = d + dt.timedelta(days=1)
        if d > self.end:
            raise ValueError(self, d)
        return d

    # NOTE: alt loop implementation that also type hints nicely : )
    # with g.implicit() as (bind, done):
    #     r = bind(None, core.Ref.Scalar_Date)
    #     r = bind(
    #         cls.new(r.select(last=keep), start, end),
    #         r,
    #         when=core.Loop.UntilDate.new(step, end, r)
    #     )
    #     g = done()
