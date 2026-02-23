
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Type,
    Generic,
    Callable,
    TypeVar,
    overload,
)

from dataclasses import replace
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass



T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

I0 = TypeVar("I0")
I1 = TypeVar("I1")
I2 = TypeVar("I2")

@dataclass(frozen=True)
class Graph:
    nodes: list[Node]

    def with_node(self, node: Node):
        self.nodes.append(node)
        return self


@dataclass(frozen=True)
class R1(Generic[T1]):
    i: int

    def value(self, graph: Graph) -> T1:
        node = graph.nodes[self.i]
        assert isinstance(node, (N0x1, N1x1)), node
        if isinstance(node, N0x1):
            res = node.f()
        else:
            raise ValueError((node, self))
        assert isinstance(res, node.t)
        return res


@dataclass(frozen=True)
class R2(Generic[T1, T2]):
    i: int

@dataclass(frozen=True)
class Node:
    pass

@dataclass(frozen=True)
class N0x1(Node, Generic[T1]):
    ref: R1[T1]
    f: Callable[[], T1]
    t: Type[T1]
    inputs: None = None

    @classmethod
    def Ref(cls, i: int) -> R1[T1]:
        return R1(i)
    
    @classmethod
    def bind(
        cls,
        graph: Graph,
        f: Callable[[], T1],
        t: Type[T1],
    ):
        ref = cls.Ref(len(graph.nodes))
        graph = graph.with_node(cls(ref, f, t, None))
        return graph, ref

@dataclass(frozen=True)
class N0x2(Node, Generic[T1, T2]):
    ref: R2[T1, T2]
    f: Callable[[], tuple[T1, T2]]
    t: tuple[Type[T1], Type[T2]]
    inputs: None = None

    @classmethod
    def Ref(cls, i: int) -> R2[T1, T2]:
        return R2(i)

    @classmethod
    def bind(
        cls,
        graph: Graph,
        f: Callable[[], tuple[T1, T2]],
        t: tuple[Type[T1], Type[T2]],
    ):
        ref = cls.Ref(len(graph.nodes))
        graph = graph.with_node(cls(ref, f, t, None))
        return graph, ref

@dataclass(frozen=True)
class N1x1(Node, Generic[I1, T1]):
    ref: R1[T1]
    f: Callable[[I1], T1]
    t: Type[T1]
    inputs: tuple[R1[I1]] 

    @classmethod
    def Ref(cls, i: int) -> R1[T1]:
        return R1(i)

    @classmethod
    def bind(
        cls,
        graph: Graph,
        f: Callable[[I1], T1],
        t: Type[T1],
        i1: R1[I1]
    ):
        ref = cls.Ref(len(graph.nodes))
        graph = graph.with_node(cls(ref, f, t, (i1,)))
        return graph, ref


@dataclass(frozen=True)
class N1x2(Node, Generic[I1, T1, T2]):
    ref: R2[T1, T2]
    f: Callable[[I1], tuple[T1, T2]]
    t: tuple[Type[T1], Type[T2]]
    inputs: tuple[R1[I1]]

    @classmethod
    def Ref(cls, i: int) -> R2[T1, T2]:
        return R2(i)

    @classmethod
    def bind(
        cls,
        graph: Graph,
        f: Callable[[I1], tuple[T1, T2]],
        t: tuple[Type[T1], Type[T2]],
        i1: R1[I1],
    ):
        ref = cls.Ref(len(graph.nodes))
        graph = graph.with_node(cls(ref, f, t, (i1,)))
        return graph, ref

class N:
    
    @dataclass(frozen=True)
    class F64(N0x1[float]):
        
        @classmethod
        def Ref(cls, i: int):
            return R.R64(i)
        
    # TODO: sub type bind

class R:
    
    @dataclass(frozen=True)
    class R64(R1[float]):
        pass

g = Graph([])
g, r0 = N.F64.bind(g, lambda: 1., float)
v0 = r0.value(g)


# TODO: as part of agent modelling

# they can propose to replace a given output with a new kind

# or input with another

# which then may change other values in the graph (eg. scaling relationships, or just parameters)


# params may also just change on their own (or also as part of a directed learning process?)


# fs are callable which means they can hold params themselves, for intermediate updates


# then the agents need to 'act' using the graph (or sub graphs?) per period
# as well as deciding to try and adapt the graph