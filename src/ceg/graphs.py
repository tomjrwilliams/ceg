
# cyclic event graph (is the name)
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    NamedTuple,
    ClassVar,
    Type,
    TypeVar,
    Optional,
    Callable,
    Iterable,
    Generator,
    Iterator,
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
    t: Type[Node.Any], t_kw: Type[NamedTuple], 
):

    params = tuple(yield_param_keys(t_kw))

    assert t_kw.__name__[-3:].lower() == "_kw", t_kw
    name = t_kw.__name__[:-3]

    return Defn(
        name=name,
        params=params, # the keys
    )

#  ------------------

def yield_params(node: Node.Any) -> Iterator[tuple[str, int]]:
    # yield from recursively any fields that are ref
    yield from []

def yield_param_keys(t_kw: Type[NamedTuple]):
    # yield from recursively any fields that are ref
    yield from []

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
    graph: Graph, node: Node.Any, ref: Ref.Any, using: Plugin | tuple[Plugin, ...] | None):
    return graph

#  ------------------


# TODO: null node and series would mean not having to constantly assert not none just for the type checker

class Graph(NamedTuple):
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
    queue: list[Event] # heapify
    nodes: Nodes
    index: frozendict[Node.Any, int]
    ustream: UStream # params
    dstream: DStream # dependents
    data: Data
    plugins: frozendict[Plugin, Scope]

    # TODO: plugin is the key
    # and the acc is a generic accumulator object
    # that 

    @classmethod
    def new(cls):
        queue = []
        heapify(queue)
        index: frozendict[Node.Any, int] = frozendict() # type: ignore
        dstream: frozendict[int, tuple[int, ...]] = frozendict() # type: ignore
        plugins: frozendict[Plugin, Scope] = frozendict() # type: ignore
        return cls(
            queue=queue,
            nodes=(),
            index=index,
            ustream=(),
            dstream=dstream,
            data=(),
            plugins=plugins,
        )

    # overloads on ref dims

    def series(self, ref: Ref.Any, t: float):
        s = self.data[ref.i]
        assert s is not None, (ref)
        return s.mask(t)

    def bind(
        self,
        node: Node.Any | None = None,
        ref: Ref.Any | Type[Ref.Any] | None=None,
        using: Plugin | tuple[Plugin, ...] | None = None,
    ):
        return bind(
            self, node=node, ref=ref, using=using
        )

    def step(self) -> Graph:
        return step(self)


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
    for k, i in yield_params(node):
        acc[i].append(k)
    params: frozendict[int, tuple[str, ...]] = frozendict(zip(
        acc.keys(),
        map(tuple, acc.values())
    )) # type: ignore
    ustream = set_tuple(
        ustream, i, params,
        frozendict() # type: ignore
    )
    dstream = fold_star(
        dstream, set_tuple_add, zip(
            params,
            itertools.repeat(ref.i, len(params))
        )
    )
    data = set_tuple(
        data, i, node.SERIES.new(), Series.null
    )
    return nodes, ustream, dstream, data

#  ------------------


def bind(
    graph: Graph,
    node: Node.Any | None = None,
    ref: Ref.Any | Type[Ref.Any] | None=None,
    using: Plugin | tuple[Plugin, ...] | None = None,
):
    # TODO: node= optional int to prealloc many ref
    #ref = int allowed as well as a ref type?
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
        ustream = ustream + (frozendict(),) # type: ignore
        data = data + (Series.null,)
    elif isinstance(node, Node.Any) and ref is None:
        i = index.get(node, len(graph.nodes))
        res = node.ref(i)
        if i == len(graph.nodes):
            nodes, ustream, dstream, data = init_node(
                node,
                res,
                nodes, ustream, dstream, data
            ) # type: ignore
    elif isinstance(node, Node.Any) and isinstance(ref, Ref.Any):
        i = index.get(node, ref.i)
        assert i == ref.i, (node, ref, i)
        res = ref
        if nodes[i] is None:
            nodes, ustream, dstream, data = init_node(
                node,
                res,
                nodes, ustream, dstream, data
            ) # type: ignore
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


def step(graph: Graph):
    (
        queue,
        nodes,
        index,
        ustream,
        dstream,
        data,
        plugins,
    ) = graph

    event = heappop(queue)

    t, ref = event
    node = nodes[ref.i]

    assert node is not None, ref

    n = node

    for p, sc in plugins.items():
        n = p.before(graph, n, event, sc)
    
    res = n(event, data)

    for p, sc in plugins.items():
        res, n = p.after(graph, res, node, event, sc)

    s = series(ref, data)
    if isinstance(s, Series.Null):
        s = node.SERIES.new()
        assert not isinstance(s, Series.Null), s
        data = set_tuple(
            data, ref.i, s, Series.null
        )

    data: Data = set_tuple(
        data, ref.i, s.append(t, res), Series.null
    )

    for i in dstream.get(ref.i, ()):
        n = nodes[i]
        assert n is not None, ref

        e = n.schedule.next(
            n, n.ref(i), event, ustream, data
        )

        if e is None:
            pass
        elif isinstance(e, Event):
            heappush(queue, e)
        elif isinstance(e, Iterable):
            _ = list(map(partial(heappush, queue), e))
        else:
            raise ValueError(e)
        
        # do we need a tie break on n (incr global event counter?)
    
    graph = Graph(
        queue,
        nodes,
        index,
        ustream,
        dstream,
        data,
        plugins,
    )
    return graph


#  ------------------





# until can be a user level code, loop step however you please




# the graph is cyclic

# so we need to avoid cases where we just cycle the same region and don't loop any others? 

# hence the queue for changes to be dealt with
