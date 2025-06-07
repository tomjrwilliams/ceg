import logging
from typing import (
    NamedTuple,
    Iterable,
    cast,
    Callable,
    overload,
    Literal,
    Generator,
    Iterator,
)

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
    event: Event | None
    t: float | None


class GraphEvents(NamedTuple):
    graph: Graph
    events: tuple[Event, ...]
    t: float | None

    def last(self) -> GraphEvent:
        return GraphEvent(
            self.graph, self.events[-1], self.t
        )


class GraphBatches(NamedTuple):
    graph: Graph
    events: tuple[tuple[Event, ...], ...]
    t: tuple[float, ...]

    def last(self) -> GraphEvent:
        return GraphEvent(
            self.graph,
            self.events[-1][-1],
            self.events[-1][-1].t,
        )


class GraphUntilTrigger(NamedTuple):
    graph: Graph
    events: tuple[Event, ...]
    trigger: Event | None
    t: float | None


def step(graph: Graph, *events: Event) -> GraphEvent:

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
        return GraphEvent(graph, None, None)

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

    for i in dstream.get(ref.i, ()) + (
        (ref.i,) if not len(graph.ustream[ref.i]) else ()
    ):
        nd = nodes[i]
        assert nd is not None, ref

        try:
            nd_ref = nd.ref(i)
        except:
            raise ValueError(nd)

        e = guards[i].next(
            event,
            nd_ref,
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

    return GraphEvent(graph, event, event.t)


def iter_batches(
    graph: Graph,
    *events: Event,
    n: int = 1,
    g: int = 1,
):
    t = None
    for i in range(n):
        acc: list[Event | None] = [None for _ in range(g)]
        for ii in range(g):
            if i == 0 and ii == 0:
                graph, e, _ = step(graph, *events)
            else:
                graph, e, _ = step(graph)
            if e is None:
                acc = acc[:ii]
                break
            t = e.t
            acc[ii] = e
        if not len(acc):
            return
        yield GraphEvents(
            graph, cast(tuple[Event, ...], tuple(acc)), t
        )


def batches(
    graph: Graph,
    *events: Event,
    n: int = 1,
    g: int = 1,
    iter: bool = False,
):
    if iter:
        return lambda: iter_batches(
            graph, *events, n=n, g=g
        )
    e = None
    t = None
    es: list[list[Event | None]] = [
        [None for _ in range(g)] for _ in range(n)
    ]
    for i in range(n):
        acc = es[i]
        for ii in range(g):
            if i == 0 and ii == 0:
                graph, e, _ = step(graph, *events)
            else:
                graph, e, _ = step(graph)
            if e is None:
                es[i] = acc[:ii]
                break
            t = e.t
            acc[ii] = e
        if e is None:
            break
    es_res = cast(
        tuple[tuple[Event, ...], ...],
        tuple(map(tuple, [_es for _es in es if len(_es)])),
    )
    return GraphBatches(
        graph, es_res, tuple([_es[-1].t for _es in es_res])
    )


def iter_steps(
    graph: Graph,
    *events: Event,
    n: int = 1,
):
    for i in range(n):
        if i == 0:
            graph, e, _ = step(graph, *events)
        else:
            graph, e, _ = step(graph)
        if e is None:
            break
        t = e.t
        yield GraphEvent(graph, e, t)


def steps(
    graph: Graph,
    *events: Event,
    n: int = 1,
    iter: bool = False,
):
    if iter:
        return lambda: iter_steps(graph, *events, n=n)
    t = None
    es: list[Event | None] = [None for _ in range(n)]
    for i in range(n):
        if i == 0:
            graph, e, _ = step(graph, *events)
        else:
            graph, e, _ = step(graph)
        if e is None:
            es = es[:i]
            break
        t = e.t
        es[i] = e
    return GraphEvents(
        graph, tuple(cast(list[Event], es)), t
    )

def until_triggered(
    f: Callable[[Graph, Event], bool],
    graph: Graph,
    event: Event,
    next: bool = False,
    ii: int = 0,
):
    if not next:
        return f(graph, event)
    if not len(graph.queue):
        return True
    if ii == 0:
        return False
    return f(graph, graph.queue[0])

def step_until(
    graph: Graph,
    f: Callable[[Graph, Event], bool],
    *events: Event,
    next: bool = False,
):
    # TODO: concat with the given events
    t = None
    es = []
    graph, e, _ = step(graph, *events)
    if e is None:
        return graph, None, (e,)
    while e is not None and not until_triggered(f, graph, e, next):
        es.append(e)
        graph, e, _ = step(graph)
    if e is None:
        return GraphUntilTrigger(graph, tuple(es), None, t)
    t = e.t
    # if len(refs):
    # TODO until each of those not just all events
    trigger = e
    while e is not None and e.t <= trigger.t:
        es.append(e)
        graph, e, _ = step(graph)
    if e is not None:
        es.append(e)
        t = e.t
    return GraphUntilTrigger(graph, tuple(es), trigger, t)

def iter_batch_until(
    graph: Graph,
    f: Callable[[Graph, Event], bool],
    *events: Event,
    n: int = 1,
    next: bool = False,
):
    e = None
    t = None
    es: list[list[Event | None]] = [
        [] for _ in range(n)
    ]
    for i in range(n):
        acc = es[i]
        ii = 0
        while (
            i == 0 and ii == 0
        ) or (
            e is not None
            and not until_triggered(f, graph, e, next, ii)
        ):
            if i == 0 and ii == 0:
                graph, e, _ = step(graph, *events)
            else:
                graph, e, _ = step(graph)
            ii += 1
            if e is None:
                break
            t = e.t
            acc.append(e)
            # TODO: and all at the same t? (but only if not next)
        if e is None:
            break
        if len(acc):
            acc = cast(list[Event], acc)
            yield GraphEvents(
                graph, tuple(acc), acc[-1].t
            )
        
def batch_until(
    graph: Graph,
    f: Callable[[Graph, Event], bool],
    *events: Event,
    n: int = 1,
    iter: bool = False,
    next: bool = False,
):
    if iter:
        return lambda: iter_batch_until(
            graph, f, *events, n=n, next=next
        )
    e = None
    t = None
    es: list[list[Event | None]] = [
        [] for _ in range(n)
    ]
    for i in range(n):
        acc = es[i]
        ii = 0
        while (
            i == 0 and ii == 0
        ) or (
            e is not None
            and not until_triggered(f, graph, e, next)
        ):
            if i == 0 and ii == 0:
                graph, e, _ = step(graph, *events)
            else:
                graph, e, _ = step(graph)
            ii += 1
            if e is None:
                break
            t = e.t
            acc.append(e)
            # TODO: and all at the same t? (but only if not next)
        if e is None:
            break
    es_res = cast(
        tuple[tuple[Event, ...], ...],
        tuple(map(tuple, [_es for _es in es if len(_es)])),
    )
    return GraphBatches(
        graph, es_res, tuple([_es[-1].t for _es in es_res])
    )
# TODO: batch until

def step_until_done(
    graph: Graph,
    *events: Event,
):
    t = None
    es = []
    graph, e, _ = step(graph, *events)
    while e is not None:
        es.append(e)
        t = e.t
        graph, e, _ = step(graph)
    return GraphEvents(graph, tuple(es), t)


#  ------------------
