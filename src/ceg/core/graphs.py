from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    NamedTuple,
    ClassVar,
    Type,
    TypeVar,
    Any,
    Optional,
    Callable,
    Iterable,
    Generator,
    Iterator,
    get_type_hints,
    get_origin,
    get_args,
    cast,
)
from heapq import heapify, heappush, heappop

from collections import defaultdict

import itertools
import functools

from frozendict import frozendict

from .types import *
from .immut import *

from . import Array
from . import Ref
from . import Series
from . import Node

#  ------------------

T = TypeVar("T")

#  ------------------


def define(
    t: Type[Node.Any],
    t_kw: Type[NamedTuple],
):

    params = tuple(yield_param_keys(t_kw))

    assert t_kw.__name__[-3:].lower() == "_kw", t_kw
    name = t_kw.__name__[:-3]

    return Defn(
        name=name,
        params=params,  # the keys
        # dims, oritentation, etc. (from t)
    )


#  ------------------


def rec_yield_param(k, v: Ref.Any | Iterable | Any):
    if isinstance(v, Ref.Any):
        yield (k, v.i)
    elif isinstance(v, Iterable):
        yield from rec_yield_params(k, v)


def rec_yield_params(k: str, v: Iterable):
    if isinstance(v, dict):
        yield from rec_yield_params(k, v.keys())
        yield from rec_yield_params(k, v.values())
    elif isinstance(v, Iterable):
        _ = list(map(partial(rec_yield_param, k), v))


def yield_params(
    node: Node.Any,
) -> Iterator[tuple[str, int]]:
    for k in node.DEF.params:
        v = getattr(node, k)
        yield from rec_yield_param(k, v)


def rec_yield_hint_types(hint):
    try:
        o = get_origin(hint)
        yield o
    except:
        pass
    try:
        args = get_args(hint)
        for a in args:
            yield from rec_yield_hint_types(a)
    except:
        pass
    yield hint


def yield_param_keys(t_kw: Type[NamedTuple]):
    for k, h in get_type_hints(t_kw).items():
        for h in rec_yield_hint_types(h):
            if not isinstance(h, type):
                continue
            if issubclass(h, Ref.Any):
                yield k
                break


#  ------------------


class Scope(NamedTuple):
    pass


class Plugin(NamedTuple):

    def before(
        self,
        graph: Graph,
        node: Node.Any,
        event: Event,
        scope: Scope,
    ):
        # eg. cache might replace with a dummy node here that does nothing
        return node

    def after(
        self,
        graph: Graph,
        res,
        node: Node.Any,
        event: Event,
        scope: Scope,
    ):
        # eg. cache might replace the dummy res with real data
        return res, node


def use_plugins(
    graph: Graph,
    node: Node.Any,
    ref: Ref.Any,
    using: Plugin | tuple[Plugin, ...] | None,
):
    return graph


#  ------------------


# TODO: null node and series would mean not having to constantly assert not none just for the type checker


class GraphKW(NamedTuple):

    queue: list[Event]  # heapify
    nodes: Nodes
    index: frozendict[Node.Any, int]
    ustream: UStream  # params
    dstream: DStream  # dependents
    data: Data
    plugins: frozendict[Plugin, Scope]


class Graph(GraphKW, GraphLike):
    """
    queue: list
    nodes: tuple
    guards: list
    index: dict
    ustream: tuple # params
    dstream: dict # dependents
    rows: tuple
    cols: tuple
    """

    # TODO: plugin is the key
    # and the acc is a generic accumulator object
    # that

    @classmethod
    def new(cls):
        queue = []
        heapify(queue)
        index: frozendict[Node.Any, int] = frozendict()  # type: ignore
        dstream: frozendict[int, tuple[int, ...]] = frozendict()  # type: ignore
        plugins: frozendict[Plugin, Scope] = frozendict()  # type: ignore
        return cls(
            queue=queue,
            nodes=(),
            index=index,
            ustream=(),
            dstream=dstream,
            data=(),
            plugins=plugins,
        )

    def bind(
        self,
        node: Node.Any | None = None,
        ref: Ref.Any | Type[Ref.Any] | None = None,
        using: Plugin | tuple[Plugin, ...] | None = None,
    ):
        return bind(self, node=node, ref=ref, using=using)

    def step(self, *events: Event) -> tuple[Graph, Event]:
        return step(self, *events)

    def steps(self, *events: Event, n: int = 1):
        return steps(self, *events, n=n)


#  ------------------


def init_node(
    node: Node.Any,
    ref: Ref.Any,
    nodes: Nodes,
    ustream: UStream,
    dstream: DStream,
    data: Data,
) -> tuple[Nodes, UStream, DStream, Data]:
    i = ref.i
    nodes = set_tuple(nodes, i, node, Node.null)
    acc: defaultdict[int, list[str]] = defaultdict(list)
    for k, p in yield_params(node):
        acc[p].append(k)
    params: frozendict[int, tuple[str, ...]] = frozendict(
        zip(acc.keys(), map(tuple, acc.values()))
    )  # type: ignore
    ustream = set_tuple(
        ustream, i, params, frozendict()  # type: ignore
    )
    dstream = fold_star(
        dstream,
        set_tuple_add,
        zip(params, itertools.repeat(ref.i, len(params))),
    )
    data = set_tuple(
        data, i, node.SERIES.new(), Series.null
    )
    return nodes, ustream, dstream, data


#  ------------------


def bind(
    graph: Graph,
    node: Node.Any | None = None,
    ref: Ref.Any | Type[Ref.Any] | None = None,
    using: Plugin | tuple[Plugin, ...] | None = None,
) -> tuple[Graph, Ref.Any]:
    # TODO: node= int to prealloc many ref
    # TODO: kwrg for only return graph (eg. if pre alloc ref and want to fold over)

    i: int
    res: Ref.Any
    ustream: UStream
    #
    (
        queue,
        nodes,
        index,
        ustream,
        dstream,
        data,
        plugins,
    ) = graph
    #
    if node is None:
        assert isinstance(ref, type), ref
        assert issubclass(ref, Ref.Any), ref
        #
        i = len(graph.nodes)
        res = ref.new(i)
        #
        nodes = nodes + (Node.null,)
        ustream = ustream + (frozendict(),)  # type: ignore
        data = data + (Series.null,)
    elif isinstance(node, Node.Any) and ref is None:
        i = index.get(node, len(graph.nodes))
        res = node.ref(i)
        if i == len(graph.nodes):
            nodes, ustream, dstream, data = init_node(
                node, res, nodes, ustream, dstream, data
            )  # type: ignore
    elif isinstance(node, Node.Any) and isinstance(
        ref, Ref.Any
    ):
        i = index.get(node, ref.i)
        assert i == ref.i, (node, ref, i)
        res = ref
        if nodes[i] == Node.null:
            nodes, ustream, dstream, data = init_node(
                node, res, nodes, ustream, dstream, data
            )  # type: ignore
    else:
        raise ValueError(node, ref)

    graph = Graph(
        queue,
        nodes,
        index,
        ustream,
        dstream,
        data,
        plugins,
    )

    if isinstance(node, Node.Any):
        graph = use_plugins(graph, node, res, using)

    return graph, res


#  ------------------


def step(
    graph: Graph, *events: Event
) -> tuple[Graph, Event]:
    (
        queue,
        nodes,
        index,
        ustream,
        dstream,
        data,
        plugins,
    ) = graph

    for e in events:
        heappush(queue, e)

    event = heappop(queue)

    try:
        t, ref = event
    except:
        raise ValueError(event)
    node = nodes[ref.i]

    assert node is not None, ref

    for p, sc in plugins.items():
        node = p.before(graph, node, event, sc)

    res = node(event, graph)

    for p, sc in plugins.items():
        res, node = p.after(graph, res, node, event, sc)

    s = series(graph, ref)

    if isinstance(s, Series.Null):
        s = node.SERIES.new()
        assert not isinstance(s, Series.Null), s
        data = set_tuple(data, ref.i, s, Series.null)

    data: Data = set_tuple(
        data, ref.i, s.append(t, res), Series.null
    )

    graph = Graph(
        queue,
        nodes,
        index,
        ustream,
        dstream,
        data,
        plugins,
    )

    for i in dstream.get(ref.i, ()):
        nd = nodes[i]
        assert nd is not None, ref

        e = nd.schedule.next(
            nd,
            nd.ref(i),
            event,
            graph,
        )

        if e is None:
            pass
        elif isinstance(e, Event):
            heappush(queue, e)
        elif isinstance(e, Iterable):
            _ = list(map(partial(heappush, queue), e))
        else:
            raise ValueError(e)

    return graph, event


def steps(
    graph: Graph, *events: Event, n: int = 1
) -> tuple[Graph, tuple[Event, ...]]:
    es: list[Event | None] = [None for _ in range(n)]
    for i in range(n):
        if i == 0:
            graph, e = step(graph, *events)
        else:
            graph, e = step(graph)
        es[i] = e
    return graph, tuple(cast(list[Event], es))


#  ------------------

# until can be a user level code, loop step however you please


# the graph is cyclic

# so we need to avoid cases where we just cycle the same region and don't loop any others?

# hence the queue for changes to be dealt with
