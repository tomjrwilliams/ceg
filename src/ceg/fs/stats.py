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
    """

    # >>> g = core.Graph.new()
    # >>> g, ref = g.bind(mean.new())

    DEF: ClassVar[core.Defn] = core.define(
        core.Node.Col, mean_kw
    )

    @classmethod
    def new(
        cls, v: core.Ref.Col, window: float | None = None
    ):
        return cls(*cls.args(), v=v, window=window)

    def __call__(self, event: core.Event, data: core.Data):
        t, v = core.mask(self.v, event, data)
        if self.window is None:
            return numpy.nanmean(v)
        return numpy.nanmean(v[t <= event.t - self.window])


#  ------------------
