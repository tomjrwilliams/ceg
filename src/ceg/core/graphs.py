from __future__ import annotations

import logging
import abc
from typing import (
    NamedTuple,
    Type,
    ParamSpec,
    Concatenate,
    TypeVar,
    Callable,
    Literal,
    Protocol,
    Generic,
    overload,
)
from heapq import heapify, heappush, heappop

import itertools
from functools import wraps

import contextlib
from collections import defaultdict


from frozendict import frozendict

from .algos import (
    set_tuple,
    frozendict_append_tuple,
    fold_star,
)
from .histories import History
from .refs import Ref, Scope, Data, R
from .nodes import (
    N,
    Node,
    Defn,
    Event,
    yield_params,
    yield_param_keys,
    NodeInterface,
)
from .guards import Guard, Ready, Loop

from functools import wraps
from inspect import Parameter, signature, Signature
logger = logging.Logger(__file__)

#  ------------------

T = TypeVar("T")

#  ------------------


# TODO: use pydnatic dataclas

# check that you can subglass graph to extend the discriminated union

# with new nodes, guards

# discriminate easily enough on type field, add to guard, similar new and define syntax

class Key(NamedTuple):
    pass


class Value(NamedTuple):
    pass


TNodes = tuple[Node.Any, ...]
TGuards = tuple[Guard.Any, ...]

UStream = tuple[frozendict[int, tuple[str, ...]], ...]
DStream = frozendict[int, tuple[int, ...]]

State = frozendict[Key, Value]


class GraphKW(NamedTuple):

    queue: list[Event]  # heapify
    nodes: TNodes
    guards: TGuards
    index: frozendict[Node.Any, int]
    ustream: UStream  # params
    dstream: DStream  # dependents
    required: frozendict[int, int]
    data: Data
    state: State


F = ParamSpec("F")
FRes = TypeVar("FRes")


class Graph(GraphKW):
    """
    queue: list
    nodes: tuple
    guards: list
    index: dict
    ustream: tuple # params
    dstream: dict # dependents
    required: dict
    data: Data
    state: State
    """

    def pipe(
        self,
        f: Callable[Concatenate[Graph, F], FRes],
        *args: F.args,
        **kwargs: F.kwargs,
    ) -> FRes:
        return f(self, *args, **kwargs)

    @classmethod
    def new(cls):
        queue = []
        heapify(queue)
        index: frozendict[Node.Any, int] = frozendict()  # type: ignore
        dstream: frozendict[int, tuple[int, ...]] = frozendict()  # type: ignore
        return cls(
            queue=queue,
            nodes=(),
            guards=(),
            index=index,
            ustream=(),
            dstream=dstream,
            required=frozendict(),  # type: ignore
            data=(),
            state=frozendict(),  # type: ignore
        )

    def with_state(self, key: Key, value: Value) -> Graph:
        return self._replace(
            state=self.state.set(key, value)
        )

    def bind(
        self,
        node: Node.Any[R, N] | None = None,
        ref: R | Type[R] | None = None,
        when: Guard.Any[N] | None = None,
        keep: int | None = 1,
    ) -> tuple[Graph, R]:
        return bind(
            self, node=node, ref=ref, when=when, keep=keep
        )

    @contextlib.contextmanager
    def mutable(
        self, partition: bool | Ref.Any | None = None
    ):
        ctxt, is_done = graph_context(
            self, partition=partition, mutable=True
        )
        try:
            yield ctxt
        finally:
            assert is_done(), ctxt

    @contextlib.contextmanager
    def implicit(
        self, partition: bool | Ref.Any | None = None
    ):
        ctxt, is_done = graph_context(
            self, partition=partition, mutable=False
        )
        try:
            yield ctxt
        finally:
            assert is_done(), ctxt


class MutableBind(Protocol):

    def __call__(
        self,
        node: Node.Any[R, N] | None = None,
        ref: R | Type[R] | None = None,
        when: Guard.Any[N] | None = None,
        keep: int | None = 1,
    ) -> R: ...


class ImplicitBind(Protocol):

    def __call__(
        self,
        node: Node.Any[R, N] | None = None,
        ref: R | Type[R] | None = None,
        when: Guard.Any[N] | None = None,
        keep: int | None = 1,
    ) -> R: ...


class MutableContext(NamedTuple):
    bind: MutableBind
    state: Callable[[], Graph]
    update: Callable[[Graph], None]
    done: Callable[[], Graph]


class ImplicitContext(NamedTuple):
    bind: ImplicitBind
    done: Callable[[], Graph]
    # TODO: partition function for nested partitions?


@overload
def graph_context(
    g: Graph,
    partition: bool | Ref.Any | None = None,
    mutable: Literal[True] = True,
) -> tuple[MutableContext, Callable[[], bool]]: ...


@overload
def graph_context(
    g: Graph,
    partition: bool | Ref.Any | None = None,
    mutable: Literal[False] = False,
) -> tuple[ImplicitContext, Callable[[], bool]]: ...


def graph_context(
    g: Graph,
    partition: bool | Ref.Any | None = None,
    mutable: bool = True,
) -> tuple[
    MutableContext | ImplicitContext,
    Callable[[], bool],
]:
    # TODO: alternatively partition can be an existing ref

    DONE: bool = False

    def bind(
        node: Node.Any[R, N] | None = None,
        ref: R | Type[R] | None = None,
        when: Guard.Any[N] | None = None,
        keep: int | None = 1,
    ) -> R:
        nonlocal g
        nonlocal DONE
        assert not DONE, (node, ref)
        # TODO: partition
        g, res = g.bind(
            node=node, ref=ref, when=when, keep=keep
        )
        return res

    def state() -> Graph:
        nonlocal g
        nonlocal DONE
        assert not DONE, DONE
        return g

    def update(g_new: Graph):
        nonlocal g
        nonlocal DONE
        assert not DONE, DONE
        g = g_new
        return

    def done():
        nonlocal g
        nonlocal DONE
        assert not DONE, DONE
        DONE = True
        return g

    def is_done():
        return DONE

    if mutable:
        ctxt = MutableContext(bind, state, update, done)
    else:
        ctxt = ImplicitContext(bind, done)

    return ctxt, is_done


#  ------------------


def init_node(
    node: Node.Any[R, N],
    ref: R,
    nodes: TNodes,
    guards: TGuards,
    ustream: UStream,
    dstream: DStream,
    data: Data,
    when: Guard.Any[N] | None = None,
) -> tuple[
    TNodes, TGuards, UStream, DStream, Data, dict[int, int]
]:
    i = ref.i
    nodes = set_tuple(nodes, i, node, Node.null)
    acc: defaultdict[int, list[str]] = defaultdict(list)

    required: dict[int, int] = {}

    for k, p, sc in yield_params(node):
        acc[p].append(k)
        if sc is None:
            continue
        elif required.get(p) is None:
            required[p] = sc.required
        elif sc.required > required[p]:
            required[p] = sc.required

    # print(node)
    # print(node.DEF.params)
    # print(acc)

    # TODO: if any fields are named ref, throw

    params: frozendict[int, tuple[str, ...]] = frozendict(
        zip(acc.keys(), map(tuple, acc.values()))
    )  # type: ignore
    ustream = set_tuple(
        ustream, i, params, frozendict()  # type: ignore
    )

    dstream = fold_star(
        dstream,
        frozendict_append_tuple,
        zip(params, itertools.repeat(ref.i, len(params))),
    )
    data = set_tuple(data, i, History.null, History.null)
    # init is on first value (need to know shape, not necessarily fixed at runtime)
    
    if when is None:
        when = Ready.All.new().init(ref, params)
    else:
        when = when.init(ref, params)

    guards = set_tuple(
        guards, i, when, Ready.All.new()
    )
    return nodes, guards, ustream, dstream, data, required


#  ------------------

# TODO: possible to have a generic namedtuple return?
# so we can then do GraphRef.pipe(...)
# and have the type hints understand?


def bind(
    graph: Graph,
    node: Node.Any[R, N] | None = None,
    ref: R | Type[R] | None = None,
    when: Guard.Any[N] | None = None,
    keep: int | None = 1,
) -> tuple[Graph, R]:
    # TODO: node= int to prealloc many ref ?
    # TODO: option to only return graph (eg. if pre alloc ref and want to fold over)?
    # TODO: if partition given, assume can batch cross partitions
    # ie. within partitions we assume have to be sequential

    i: int
    res: R
    ustream: UStream

    #
    (
        queue,
        nodes,
        guards,
        index,
        ustream,
        dstream,
        required,
        data,
        state,
    ) = graph
    #

    # TODO: unpack and bind state
    # where state is passed as arg on the node itself
    # so same as param iteration?

    req = {}
    if node is None:
        assert isinstance(ref, type), ref
        assert issubclass(ref, Ref.Any), ref
        #
        i = len(graph.nodes)
        res = ref.new(i)

        nodes = nodes + (Node.null,)
        ustream = ustream + (frozendict(),)  # type: ignore
        data = data + (History.null,)
    elif isinstance(node, Node.Any) and ref is None:
        i = index.get(node, len(graph.nodes))
        res = node.ref(i)
        if i == len(graph.nodes):
            (
                nodes,
                guards,
                ustream,
                dstream,
                data,
                req,
            ) = init_node(
                node,
                res,
                nodes,
                guards,
                ustream,
                dstream,
                data,
                when=when,
            )  # type: ignore
    elif isinstance(node, Node.Any) and isinstance(
        ref, Ref.Any
    ):
        i = index.get(node, ref.i)
        assert i == ref.i, (node, ref, i)
        res = ref
        if nodes[i] == Node.null:
            (
                nodes,
                guards,
                ustream,
                dstream,
                data,
                req,
            ) = init_node(
                node,
                res,
                nodes,
                guards,
                ustream,
                dstream,
                data,
                when=when,
            )  # type: ignore
    else:
        raise ValueError(type(node), type(ref), node, ref)

    if keep is not None and req.get(i, 0) < keep:
        req = req | {i: keep}

    required = required | {
        p: r
        for p, r in req.items()
        if p not in required or required.get(p, 0) < r
    }

    graph = Graph(
        queue,
        nodes,
        guards,
        index,
        ustream,
        dstream,
        required,
        data,
        state,
    )

    return graph, res


#  ------------------

P = ParamSpec("P")
O = TypeVar("O", bound = Node.Any)
R = TypeVar("R", bound = Ref.Any)

class fs:
    pass

Fs = TypeVar("Fs", bound=fs)

# class HasNew(Generic[O, P, R, Fs]):
# class HasNew(Generic[O, P, R]):

#     # def __call__(self, NAME: str, *args: P.args, **kwargs: P.kwargs) -> O: ...

#     # @classmethod
#     # def fs(cls) -> Type[Fs]: ...

#     @classmethod
#     def ref(cls, i: int | Ref.Any, slot: int | None = None) -> R: ...

#     @classmethod
#     def new(cls, *args: P.args, **kwargs: P.kwargs) -> O: ...

#     @classmethod
#     def bind(cls, g: Graph, *args: P.args, **kwargs: P.kwargs) -> tuple[Graph, R]:
#         g, r = g.bind(node=cls.new(*args, **kwargs))
#         return g, cls.ref(r)

def prepend_argument(
    f,
    f_wrapped: Callable[P, PRes],
    name: str,
    t: Type,
) -> Callable[P, PRes]:
    sig = signature(f)
    params = [
        Parameter(
            name,
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=t
        )
    ] + list(sig.parameters.values()) + [
        Parameter("keep", Parameter.KEYWORD_ONLY, annotation=int | None)
    ]
    f_wrapped.__signature__ = sig.replace(parameters=params)
    f_wrapped.__annotations__ = {
        **{name: t},
        **f_wrapped.__annotations__,
        **{"keep": int | None}
    }
    return f_wrapped

P = ParamSpec("P")
PRes = TypeVar("PRes")

class Annotation(NamedTuple):
    type: str

class define:

    fs = fs
    # bind_from_new = HasNew

    Annotation = Annotation

    @staticmethod
    def annotation(type: str):
        return Annotation(type)

    @staticmethod
    def node(
        t: Type[NodeInterface],
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

    @staticmethod
    def bind_from_new(
        new: Callable[P, Node.Any],
        ref: Callable[..., R]
    ) -> Callable[
        Concatenate[Graph, P], 
        tuple[Graph, R]
    ]:
        @wraps(new)
        def bind(
            # cls,
            g: Graph, 
            *args: P.args, 
            **kwargs: P.kwargs,
        ) -> tuple[Graph, R]:
            keep = kwargs.pop("keep", 1)
            g, r = g.bind(
                node=new(*args, **kwargs), 
                keep=keep # type: ignore
            )
            return g, ref(r)
        return prepend_argument(
            new,
            bind,
            "g",
            Graph,
        )

#  ------------------
