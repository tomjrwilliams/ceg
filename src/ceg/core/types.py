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
    Literal,
)
from typing_extensions import Self

import numpy as np

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
        graph: GraphLike,
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
        graph: GraphLike,
    ) -> Event | list[Event] | None:
        return next_event(
            self,
            node,
            ref,
            event,
            graph,
        )


#  ------------------


null_schedule = Schedule.new()


def next_event(
    schedule: Schedule,
    node: NodeND,
    ref: Ref.Any,  # ref to self
    event: Event,  # ref might not be self (eg. might be param)
    graph: GraphLike,
):
    params = graph.ustream[ref.i]

    for k in params.get(event.ref.i, ()):
        sync = schedule.sync.get(k)
        if sync is None:
            continue
        res = sync.next(node, ref, event, params, graph)
        # if res is not None:
        return res
    
    if len(schedule.sync):
        return None

    t = event.t
    # assert ref.i not in params, (node, ref, params)
    # as will never fire
    if all_series(
        graph, params.keys(), lambda e: e.t.last == t
    ):
        return event._replace(ref=ref)
    return None


#  ------------------

# NOTE: direct writing the results will be more efficient as not alloc an unnecessary temporary result array
# but more error prone
# and harder to have plguins mask over res values
# so probably do use that for now

#  ------------------

UStream = tuple[frozendict[int, tuple[str, ...]], ...]
DStream = frozendict[int, tuple[int, ...]]

Data = tuple[Series.Any, ...]

#  ------------------

# TODO: need to pass in the slice / index we want
# as that's always the next step

# can go all the way through to the array

# also need overloads / diff methods
# for one, many (many nested, ...), many concat (optional reshape), etc.

# with also flags for include nan, fill forward, backward, etc.


@overload
def select(
    graph: GraphLike,
    ref: Ref.Object,
    at: float | Event,
    t: Literal[False] = False,
    i: int | slice | None = None,
    where: dict[str, Callable] | None = None,
    null: bool | str = True,
) -> list: ...


@overload
def select(
    graph: GraphLike,
    ref: Ref.Object,
    at: float | Event,
    t: Literal[True] = True,
    i: int | slice | None = None,
    where: dict[str, Callable] | None = None,
    null: bool | str = True,
) -> tuple[list[float], list]: ...


@overload
def select(
    graph: GraphLike,
    ref: Ref.Col,
    at: float | Event,
    t: Literal[False] = False,
    i: int | slice | None = None,
    where: dict[str, Callable] | None = None,
    null: bool | str = True,
) -> Array.np_1D: ...


@overload
def select(
    graph: GraphLike,
    ref: Ref.Col,
    at: float | Event,
    t: Literal[True] = True,
    i: int | slice | None = None,
    where: dict[str, Callable] | None = None,
    null: bool | str = True,
) -> tuple[Array.np_1D, Array.np_1D]: ...


@overload
def select(
    graph: GraphLike,
    ref: Ref.Col1D,
    at: float | Event,
    t: Literal[False] = False,
    i: int | slice | None = None,
    where: dict[str, Callable] | None = None,
    null: bool | str = True,
) -> Array.np_2D: ...


@overload
def select(
    graph: GraphLike,
    ref: Ref.Col1D,
    at: float | Event,
    t: Literal[True] = True,
    i: int | slice | None = None,
    where: dict[str, Callable] | None = None,
    null: bool | str = True,
) -> tuple[Array.np_1D, Array.np_2D]: ...


def select(
    graph: GraphLike,
    ref: Ref.Any | tuple[Ref.Any, ...],
    at: float | Event,
    t: bool = False,
    i: int | slice | None = None,
    where: dict[str, Callable] | None = None,
    null: bool | str = True,
):
    if isinstance(ref, Ref.Any):
        res = series(graph, ref).select(
            at if isinstance(at, float) else at.t, t=t, i=i, where=where, null=null
        )
        if t:
            t, v = res
            assert len(t) == len(v), dict(t=len(t), v = len(v), node=graph.nodes[ref.i])
        return res
    # TODO: need align if on diff t?
    res = tuple((
        series(graph, r).select(
            at if isinstance(at, float) else at.t,
            t=t, 
            i=None if not null else i, 
            where=where,
            null = True if not null else null
        )
        for r in ref
    ))
    if not null and t:
        not_null = np.logical_not(
            np.all([np.isnan(v) for t, v in res], axis=0)
        )
        return tuple((
            (t[not_null], v[not_null]) for t, v in res
        ))
    elif not null:
        not_null = np.logical_not(
            np.all([np.isnan(v) for v in res], axis=0)
        )
        return tuple((v[not_null] for v in res))
    return res

def mask(
    graph: GraphLike,
    ref: Ref.Any,
    at: float | Event,
    where: dict[str, Callable] | None = None,
    null: bool | str = True,
) -> Array.np_1D:
    return series(graph, ref).mask(
        at if isinstance(at, float) else at.t, where=where, null=null
    )


# TODO: implementations for other dtypes (eg. as jax, pd, polars, etc. - can arguably be a kwarg as well not a separate func?)

#  ------------------

# null Series
SeriesToBool = Callable[[Series.Any], bool]


# overlods on ref type -> series type


def series(
    graph: GraphLike, ref: Ref.Any | int
) -> Series.Any:
    data = graph.data
    if isinstance(ref, int):
        return data[ref]
    return data[ref.i]


def is_series(
    graph: GraphLike,
    ref: Ref.Any | int,
    f: SeriesToBool,
):
    return f(series(graph, ref))


def all_series(
    graph: GraphLike,
    refs: Iterable[Ref.Any | int],
    f: SeriesToBool,
):
    return all((f(series(graph, ref)) for ref in refs))


class GraphLike(Protocol):

    @property
    def nodes(self) -> Nodes: ...

    @property
    def data(self) -> Data: ...

    @property
    def ustream(self) -> UStream: ...

    @property
    def dstream(self) -> DStream: ...

    series = series
    is_series = is_series
    all_series = all_series

    select = select
    mask = mask


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
        return cls.DEF.name, null_schedule

    def sync(self, **kwargs: Sync):
        return self._replace(
            schedule=self.schedule._replace(
                sync=self.schedule.sync | kwargs
            )
        )

    def __call__(
        self, event: Event, graph: GraphLike
    ) -> Any:
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
    "GraphLike",
    "SeriesToBool",
    "series",
    "is_series",
    "all_series",
    "select",
    "mask",
    "NodeND",
    "Nodes",
]
