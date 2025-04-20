from typing import NamedTuple, ClassVar, Type

from frozendict import frozendict

from .. import Ref
from ..types import *
from ..types import Sync

# TODO: separate guard types
import datetime as dt
import numpy
import numpy as np

rng = numpy.random.default_rng(42069)


class Loop(Sync):
    """ """

    pass

#  ------------------


class EveryKw(NamedTuple):
    key: str

class Every(EveryKw, Sync):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        vs = getattr(node, self.key)
        if isinstance(vs, Ref.Any):
            vs = [vs]
        if all_series(
            graph, vs, lambda e: e.t.last == event.t
        ):
            return event._replace(ref=ref)
        return

#  ------------------


class FixedKw(NamedTuple):
    step: float


class Fixed(FixedKw, Loop):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        assert event.ref == ref, (self, node, ref, event)
        return event._replace(t=event.t + self.step)

class FixedUntilDateKw(NamedTuple):
    step: float
    until: dt.date


class FixedUntilDate(FixedUntilDateKw, Loop):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        assert event.ref == ref, (self, node, ref, event)
        vs = graph.select(ref, event, t = False)
        if len(vs) and vs[-1] == self.until:
            return None
        return event._replace(t=event.t + self.step)


class RandKw(NamedTuple):
    dist: str
    params: tuple[float, ...]


class Rand(RandKw, Loop):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        assert event.ref == ref, (self, node, ref, event)
        if self.dist == "normal":
            step = rng.normal(*self.params, size=None)
        else:
            raise ValueError(self)
        return event._replace(t=event.t + step)


#  ------------------

class WeekStartKw(NamedTuple):
    dx: Ref.Object

class WeekStart(WeekStartKw, Loop):
    pass

class WeekEndKw(NamedTuple):
    dx: Ref.Object

class WeekEnd(WeekEndKw, Loop):
    pass

class MonthStartKw(NamedTuple):
    dx: Ref.Object

class MonthStart(MonthStartKw, Loop):
    pass

class MonthEndKw(NamedTuple):
    dx: Ref.Object
    # take calendar?
    # also filter where returns are none - or we don't know that ahead of time? that could be a bad system, so that we couldn't execute is valid?


class MonthEnd(MonthEndKw, Loop):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        v_last = graph.select(self.dx, event, t=False, i=-1)
        assert isinstance(v_last, dt.date), v_last
        if (v_last + dt.timedelta(days=1)).day == 1:
            if all_series(
                graph, params.keys(), lambda e: e.t.last == event.t
            ):
                return event._replace(ref=ref)
        return None

class QuarterStartKw(NamedTuple):
    dx: Ref.Object

class QuarterStart(QuarterStartKw, Loop):
    pass

class QuarterEndKw(NamedTuple):
    dx: Ref.Object

class QuarterEnd(QuarterEndKw, Loop):
    pass

class YearStartKw(NamedTuple):
    dx: Ref.Object

class YearStart(YearStartKw, Loop):
    pass

class YearEndKw(NamedTuple):
    dx: Ref.Object

class YearEnd(YearEndKw, Loop):
    pass

#  ------------------

class ScenarioStartKw(NamedTuple):
    step: float
    scenarios: frozendict[str, tuple[dt.date, dt.date]]

class ScenarioStart(ScenarioStartKw, Loop):
    pass

class ScenarioEndKw(NamedTuple):
    step: float
    scenarios: frozendict[str, tuple[dt.date, dt.date]]

class ScenarioEnd(ScenarioEndKw, Loop):
    pass

#  ------------------

class SignChangeKw(NamedTuple):
    step: float
    # optional within a given window rather than just prev?
    # TODO: treatment of zero (include, pos, neg, skip)

class SignChange(SignChangeKw, Loop):
    
    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        graph: GraphLike,
    ):
        # TODO: see above

        assert event.ref == ref, (self, node, ref, event)
        # v is dates?
        v = graph.select(ref, event, t = False)
        v_last = v[-1]
        if v_last == np.NAN:
            return None
        v = v[:-1]
        for vv in v[::-1]:
            if vv == np.NAN:
                continue
            if v_last > 0 and vv > 0:
                continue
            return event._replace(t=event.t + self.step)
        return None

# NOTE: for eg. cooling off, probably best to do that with a node (ie. a sign_change node, and then a second sign_change_w_cool_off) and then just use the sign change 

#  ------------------
