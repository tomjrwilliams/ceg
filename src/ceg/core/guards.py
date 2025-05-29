from __future__ import annotations

import logging
import abc
from typing import (
    Generic,
    ClassVar,
    Type,
    NamedTuple,
    TypeVar,
)

from dataclasses import dataclass
from heapq import heapify, heappush, heappop

import datetime as dt

from frozendict import frozendict
import numpy as np

from .refs import Ref, R, GraphInterface
from .nodes import Event, Node, N

#  ------------------

logger = logging.Logger(__file__)


@dataclass
class GuardMutable:
    prev: Event | None = None


class GuardInterface(abc.ABC, Generic[N]):

    @abc.abstractproperty
    def mut(self) -> GuardMutable: ...

    @property
    def prev(self):
        return self.mut.prev

    @classmethod
    @abc.abstractclassmethod
    def new(cls, *args, **kwargs) -> GuardInterface: ...

    @abc.abstractmethod
    def init(
        self,
        ref: Ref.Any,
        params: frozendict[int, tuple[str, ...]],
    ) -> GuardInterface: ...

    @abc.abstractmethod
    def next(
        self,
        event: Event,
        ref: Ref.Any,
        node: N,
        graph: GraphInterface,
    ) -> Event | None: ...

    def set_prev(
        self,
        event: Event,
        trigger: Event | None,
    ):
        if (
            self.mut.prev is None
            and trigger is not None
            and event.ref == trigger.ref
        ):
            self.mut.prev = trigger
        event = event._replace(
            prev=(
                None
                if self.mut.prev is None
                else self.mut.prev._replace(prev=None)
            )
        )
        self.mut.prev = event
        return event


G = TypeVar("G", bound=GuardInterface)

#  ------------------


class AnyN(NamedTuple):
    pass


class AllReadyKW(NamedTuple):
    mut: GuardMutable
    params: tuple[int, ...]
    ts: list[float]
    queue: list[tuple[float, int]]


class AllReady(AllReadyKW, GuardInterface[N]):

    @classmethod
    def new(cls) -> AllReady:
        return AllReady(GuardMutable(), (), [], [])  # type: ignore

    def init(
        self,
        ref: Ref.Any,
        params: frozendict[int, tuple[str, ...]],
    ) -> AllReady:
        ts = []
        queue = []
        heapify(ts)
        heapify(queue)
        return self._replace(
            params=tuple(params.keys()),
            ts=ts,
            queue=queue,
        )

    def next(
        self,
        event: Event,
        ref: Ref.Any,
        node: N,
        graph: GraphInterface,
    ) -> Event | None:
        # NOTE: ref is of the node, event is of param
        # TODO: now we have prev
        # assume we always push a final event, can simply wait for prev != event.t, fire event(t=prev)? but then events arent strictly ordered?
        t = event.t
        i = event.ref.i
        if not len(self.ts) or t > self.ts[-1]:
            heappush(self.ts, t)
            for p in self.params:
                heappush(self.queue, (t, p))
        t_next = self.ts[0]
        (t_next_, i_next) = self.queue[0]
        assert t_next == t_next_, (t_next, t_next_)
        if t_next == event.t and i_next == i:
            heappop(self.queue)
        if not len(self.queue) or self.queue[0][0] > t_next:
            heappop(self.ts)
            return self.set_prev(Event.new(t, ref), event)
        return None


#  ------------------


class LoopConstKw(NamedTuple):
    mut: GuardMutable
    step: float


class LoopConst(LoopConstKw, GuardInterface[N]):

    @classmethod
    def new(cls, step: float) -> LoopConst:
        return LoopConst(GuardMutable(), step)

    def init(
        self,
        ref: Ref.Any,
        params: frozendict[int, tuple[str, ...]],
    ) -> LoopConst:
        return self

    def next(
        self,
        event: Event,
        ref: Ref.Any,
        node: N,
        graph: GraphInterface,
    ) -> Event | None:
        assert event.ref.eq(ref), (self, node, ref, event)
        return self.set_prev(
            event._replace(
                t=event.t + self.step,
            ),
            event,
        )


class LoopUntilDateKw(NamedTuple):
    mut: GuardMutable
    step: float
    until: dt.date
    date: Ref.Scalar_Date


class LoopUntilDate(LoopUntilDateKw, GuardInterface[N]):

    @classmethod
    def new(
        cls,
        step: float,
        until: dt.date,
        date: Ref.Scalar_Date,
    ) -> LoopUntilDate:
        return LoopUntilDate(
            GuardMutable(), step, until, date
        )

    def init(
        self,
        ref: Ref.Any,
        params: frozendict[int, tuple[str, ...]],
    ) -> LoopUntilDate:
        return self

    def next(
        self,
        event: Event,
        ref: Ref.Any,
        node: N,
        graph: GraphInterface,
    ) -> Event | None:
        assert event.ref.eq(ref), (self, node, ref, event)
        h = self.date.history(graph, strict=False)
        if h is None:
            assert event.t == 0, event
            return self.set_prev(event, None)
        d = h.last_before(event.t)
        if d == self.until:
            return None
        return self.set_prev(
            event._replace(t=event.t + self.step), event
        )


class LoopRandKw(NamedTuple):
    mut: GuardMutable
    dist: str
    params: tuple[float, ...]


rng = np.random.default_rng(66642666)


class LoopRand(LoopRandKw, GuardInterface[N]):

    @classmethod
    def new(
        cls, dist: str, params: tuple[float, ...]
    ) -> LoopRand:
        return LoopRand(GuardMutable(), dist, params)

    def init(
        self,
        ref: Ref.Any,
        params: frozendict[int, tuple[str, ...]],
    ) -> LoopRand:
        return self

    def next(
        self,
        event: Event,
        ref: Ref.Any,
        node: N,
        graph: GraphInterface,
    ) -> Event | None:
        assert event.ref.eq(ref), (self, node, ref, event)
        if self.dist == "normal":
            step = rng.normal(*self.params, size=None)
        else:
            raise ValueError(self)
        return self.set_prev(
            event._replace(t=event.t + step), event
        )


#  ------------------


class ByDateKW(NamedTuple):
    mut: GuardMutable
    date: Ref.Scalar_Date
    last: int | None

    @classmethod
    def new(cls, date: Ref.Scalar_Date):
        return cls(GuardMutable(), date, None)

    def init(
        self,
        ref: Ref.Any,
        params: frozendict[int, tuple[str, ...]],
    ):
        if not len(params):
            return self
        return self._replace(last=max(*params.keys))


class WeekStart(ByDateKW, GuardInterface[N]):
    pass


class WeekEnd(ByDateKW, GuardInterface[N]):
    pass


class MonthStart(ByDateKW, GuardInterface[N]):
    pass


class MonthEnd(ByDateKW, GuardInterface[N]):

    def next(
        self,
        event: Event,
        ref: Ref.Any,
        node: N,
        graph: GraphInterface,
    ) -> Event | None:
        if (
            self.last is not None
            and self.last != event.ref.i
        ):
            # TODO: only fire on the last param (?) so we know all are ready? only works if all fire on all t?
            return None
        d = self.date.history(graph).last_before(event.t)
        assert isinstance(d, dt.date), d
        if (d + dt.timedelta(days=1)).day == 1:
            # if all_series(
            #     graph, params.keys(), lambda e: e.t.last == event.t
            # ):
            return self.set_prev(
                event._replace(ref=ref), event
            )
        return None


class QuarterStart(ByDateKW, GuardInterface[N]):
    pass


class QuarterEnd(ByDateKW, GuardInterface[N]):
    pass


class YearStart(ByDateKW, GuardInterface[N]):
    pass


class YearEnd(ByDateKW, GuardInterface[N]):
    pass


#  ------------------

# class ScenarioStartKw(NamedTuple):
#     step: float
#     scenarios: frozendict[str, tuple[dt.date, dt.date]]

# class ScenarioStart(ScenarioStartKw, GuardInterface[N]):
#     pass

# class ScenarioEndKw(NamedTuple):
#     step: float
#     scenarios: frozendict[str, tuple[dt.date, dt.date]]

# class ScenarioEnd(ScenarioEndKw, GuardInterface[N]):
#     pass

#  ------------------


class ByValueKW(NamedTuple):
    mut: GuardMutable
    value: Ref.Any
    last: int | None
    # optional within a given window rather than just prev?
    # TODO: treatment of zero (include, pos, neg, skip)

    @classmethod
    def new(cls, value: Ref.Any):
        return cls(GuardMutable(), value, None)

    def init(
        self,
        ref: Ref.Any,
        params: frozendict[int, tuple[str, ...]],
    ):
        if not len(params):
            return self
        return self._replace(last=max(*params.keys))


class SignChange(ByValueKW, GuardInterface[N]):

    def next(
        self,
        event: Event,
        ref: Ref.Any,
        node: N,
        graph: GraphInterface,
    ) -> Event | None:
        if (
            self.last is not None
            and self.last != event.ref.i
        ):
            # TODO: only fire on the last param (?) so we know all are ready? only works if all fire on all t?
            return None
        vs = self.value.history(graph).last_n_before(
            2, event.t
        )
        v0 = vs[-1]
        v1 = vs[0]
        # assume we dont emit nan
        if np.isnan(v0) or np.isnan(v1):
            return None
        elif v0 > 0 and v1 < 0:
            return self.set_prev(
                event._replace(ref=ref), event
            )
        elif v1 > 0 and v0 < 0:
            return self.set_prev(
                event._replace(ref=ref), event
            )
        return None


# NOTE: for eg. cooling off, probably best to do that with a node (ie. a sign_change node, and then a second sign_change_w_cool_off) and then just use the sign change

#  ------------------


class Guard:
    Any = GuardInterface

    AllReady = AllReady


class Loop:
    Const = LoopConst
    Rand = LoopRand
    UntilDate = LoopUntilDate

    @classmethod
    def every(cls, step: float):
        return LoopConst.new(step)


class ByDate:
    MonthEnd = MonthEnd


class ByValue:
    SignChange = SignChange


#  ------------------
