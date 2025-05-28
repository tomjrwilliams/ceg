import logging
from typing import NamedTuple, Iterable, cast, Callable, overload, Literal, Generator, Iterator

from functools import partial

from heapq import heapify, heappush, heappop

from .histories import History
from .refs import R, Ref
from .nodes import N, Node, Ref, Event
from .graphs import Graph

from .algos import set_tuple

logger = logging.Logger(__file__)

#  ------------------

class Step(NamedTuple):
    event: Event
    node: Node.Any

# TODO: namedutple with the event, node
# so can do eg. step_until(t)
# eg. where t is the graph or whatever

class GraphEvent(NamedTuple):
    graph: Graph
    event: Event

class GraphEvents(NamedTuple):
    graph: Graph
    events: tuple[Event, ...]

    def last(self) -> GraphEvent:
        return GraphEvent(self.graph, self.events[-1])

class GraphBatches(NamedTuple):
    graph: Graph
    events: tuple[tuple[Event, ...], ...]

    def last(self) -> GraphEvent:
        return GraphEvent(self.graph, self.events[-1][-1])

class GraphUntilTrigger(NamedTuple):
    graph: Graph
    events: tuple[Event, ...]
    trigger: Event | None

def step(
    graph: Graph, *events: Event
) -> tuple[Graph, Event | None]:
    
    queue = graph.queue
    nodes = graph.nodes
    data = graph.data
    required = graph.required

    for e in events:
        heappush(queue, e)

    try:
        event = heappop(queue)
    except IndexError:
        # implies done?
        return graph, None

    try:
        t, ref, prev = event
    except:
        raise ValueError(event)

    node = nodes[ref.i]
    assert node is not None, ref

    res = node(event, graph)
    hist = data[ref.i]

    if isinstance(hist, History.Null):
        hist = Ref.history(ref, res, required.get(ref.i))
        data = set_tuple(data, ref.i, hist, History.null)
        graph = graph._replace(data=data)

    if not isinstance(res, tuple):
        assert isinstance(hist, History.Any), hist
        hist.append(res, t)
    else:
        assert isinstance(hist, tuple), hist
        for v, h in zip(res, hist):
            h.append(v, t)

    dstream = graph.dstream
    guards = graph.guards

    for i in dstream.get(ref.i, (
        (ref.i,) if not len(graph.ustream[ref.i]) else ()
    )):
        nd = nodes[i]
        assert nd is not None, ref
        
        e = guards[i].next(
            event,
            nd.ref(i),
            nd,
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

    return GraphEvent(graph, event)

def iter_batches(
    graph: Graph, 
    *events: Event, 
    n: int = 1,
    g: int = 1,
):
    for i in range(n):
        acc: list[Event | None] = [None for _ in range(g)]
        for ii in range(g):
            if i == 0 and ii == 0:
                graph, e = step(graph, *events)
            else:
                graph, e = step(graph)
            if e is None:
                acc = acc[:ii]
                break
            acc[ii] = e
        yield GraphEvents(graph, cast(tuple[Event, ...], tuple(acc)))

def batches(
    graph: Graph, 
    *events: Event, 
    n: int = 1,
    g: int = 1,
    iter: bool = False,
):
    if iter:
        return lambda: iter_batches(graph, *events, n=n, g=g)
    e = None
    es: list[list[Event|None]] = [[None for _ in range(g)] for _ in range(n)]
    for i in range(n):
        acc = es[i]
        for ii in range(g):
            if i == 0 and ii == 0:
                graph, e = step(graph, *events)
            else:
                graph, e = step(graph)
            if e is None:
                es[i] = acc[:ii]
                break
            acc[ii] = e
        if e is None:
            break
    return GraphBatches(
        graph, 
        cast(
            tuple[tuple[Event, ...], ...], 
            tuple(map(tuple, es))
        )
    )

def iter_steps(
    graph: Graph, 
    *events: Event, 
    n: int = 1,
):
    for i in range(n):
        if i == 0:
            graph, e = step(graph, *events)
        else:
            graph, e = step(graph)
        if e is None:
            break
        yield GraphEvent(graph, e)

def steps(
    graph: Graph, 
    *events: Event, 
    n: int = 1,
    iter: bool = False,
):
    if iter:
        return lambda: iter_steps(graph, *events, n=n)
    es: list[Event | None] = [None for _ in range(n)]
    for i in range(n):
        if i == 0:
            graph, e = step(graph, *events)
        else:
            graph, e = step(graph)
        if e is None:
            es = es[:i]
            break
        es[i] = e
    return GraphEvents(graph, tuple(cast(list[Event], es)))

def step_until(
    graph: Graph,
    f: Callable[[Graph, Event], bool],
    *events: Event,
):
    # TODO: concat with the given events
    es = []
    graph, e = step(graph, *events)
    if e is None:
        return graph, None, (e,)
    while e is not None and not f(graph, e):
        es.append(e)
        graph, e = step(graph)
    if e is None:
        return GraphUntilTrigger(graph,tuple(es), None)
    # if len(refs):
    # TODO until each of those not just all events
    trigger = e
    while e is not None and e.t <= trigger.t:
        es.append(e)
        graph, e = step(graph)
    if e is not None:
        es.append(e)
    return GraphUntilTrigger(graph, tuple(es), trigger)


def step_until_done(
    graph: Graph, 
    *events: Event,
):
    es = []
    graph, e = step(graph, *events)
    while e is not None:
        es.append(e)
        graph, e = step(graph)
    return GraphEvents(graph, tuple(es))

#  ------------------