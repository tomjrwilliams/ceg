from __future__ import annotations

from typing import (
    NamedTuple,
    ClassVar,
    Type,
    Optional,
    Callable,
    Iterable,
    Any,
    ParamSpec,
    Concatenate,
    TypeVar,
    Generic,
    Protocol,
    overload,
)
from typing_extensions import Self

P = ParamSpec("P")

from frozendict import frozendict

from .Array import Shape

from . import Array
from . import Ref
from . import Series

#  ------------------


class Defn(NamedTuple):
    name: str

    params: tuple[str, ...]


#  ------------------


class Event(NamedTuple):
    """
    t: int
    ref: Ref.Any
    """

    # means both fire and fired, dependending on context
    # means literally just "node i at time t"
    t: int
    ref: Ref.Any


#  ------------------


class Sync(NamedTuple):

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,
        event: Event,
        params: frozendict[int, tuple[str, ...]],
        data: Data,
    ):
        raise ValueError(self)


class Schedule(NamedTuple):
    """
    sync: frozendict[str, Sync]
    """

    sync: frozendict[str, Sync]

    @classmethod
    def new(
        cls,
        sync: frozendict[str, Sync] = frozendict(),  # type: ignore
    ):
        return cls(sync)

    def next(
        self,
        node: NodeND,
        ref: Ref.Any,  # ref to self
        event: Event,  # ref might not be self (eg. might be param)
        ustream: UStream,
        data: Data,
    ) -> Event | list[Event] | None:
        return next_event(
            self,
            node,
            ref,
            event,
            ustream,
            data,
        )


def next_event(
    schedule: Schedule,
    node: NodeND,
    ref: Ref.Any,  # ref to self
    event: Event,  # ref might not be self (eg. might be param)
    ustream: UStream,
    data: Data,
):
    params = ustream[ref.i]
    for k in params.get(ref.i, ()):
        sync = schedule.sync.get(k)
        if sync is None:
            continue
        res = sync.next(node, ref, event, params, data)
        if res is not None:
            return res
    t = event.t
    assert ref.i not in params, (node, ref, params)
    # as will never fire
    if all_series(
        params.keys(), data, lambda e: e.t.last == t
    ):
        return event._replace(ref=ref)
    return None


#  ------------------

# NOTE: direct writing the results will be more efficient as not alloc an unnecessary temporary result array
# but more error prone
# and harder to have plguins mask over res values
# so probably do use that for now

UStream = tuple[frozendict[int, tuple[str, ...]], ...]
DStream = frozendict[int, tuple[int, ...]]

Data = tuple[Series.Any, ...]


@overload
def mask(
    ref: Ref.Col, t: float | Event, data: Data
) -> tuple[Array.np_1D, Array.np_1D]: ...


@overload
def mask(
    ref: Ref.Col1D, t: float | Event, data: Data
) -> tuple[Array.np_1D, Array.np_2D]: ...


def mask(ref: Ref.Any, t: float | Event, data: Data):
    return series(ref, data).mask(
        t if isinstance(t, float) else t.t
    )


# null Series
SeriesToBool = Callable[[Series.Any], bool]


def series(ref: Ref.Any | int, data: Data):
    if isinstance(ref, int):
        return data[ref]
    return data[ref.i]


def is_series(
    ref: Ref.Any | int,
    data: Data,
    f: SeriesToBool,
):
    return f(series(ref, data))


def all_series(
    refs: Iterable[Ref.Any | int],
    data: Data,
    f: SeriesToBool,
):
    return all((f(series(ref, data)) for ref in refs))


Schedules = frozendict[str, Schedule]

sync_null = Schedule.new()

#  ------------------


class NodeKW(NamedTuple):
    type: str
    schedule: Schedule


# NOTE: separate out the namedtuple def because it doesn't separate out the classvars (so tries to create fields for below)
class NodeND(NodeKW):

    DEF: ClassVar[Defn] = Defn("NULL", ())
    REF: ClassVar[Type[Ref.Any]] = Ref.Any
    SERIES: ClassVar[Type[Series.Any]] = Series.Any

    @classmethod
    def args(cls) -> tuple[str, Schedule]:
        return cls.DEF.name, sync_null

    def sync(self, **kwargs: Sync):
        return self._replace(
            schedule=self.schedule._replace(
                sync=self.schedule.sync | kwargs
            )
        )

    def __call__(self, event: Event, data: Data) -> Any:
        raise ValueError(self)

    # TODO: if attr isn't needed, shape can be just *path: int?
    def ref(
        self,
        i: int,
        attr: str | None = None,
        shape: Shape | None = None,
    ) -> Ref.Any:
        return self.REF(i, attr, shape)


#  ------------------

Nodes = tuple[NodeND, ...]

N = TypeVar("N", bound=NodeND)

#  ------------------

__all__ = [
    # "Sync",
    "Schedule",
    "next_event",
    "Event",
    "Defn",
    "UStream",
    "DStream",
    "Data",
    "SeriesToBool",
    "series",
    "mask",
    "is_series",
    "all_series",
    "NodeND",
    "Nodes",
]
