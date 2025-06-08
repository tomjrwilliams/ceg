
from typing import Protocol, ParamSpec, Type, TypeVar, Callable, Concatenate, cast, Generic, NamedTuple, ClassVar

from .graphs import Graph, Node, Ref


# def def_bind(
#     f_new: Callable[P, N],
#     t_ref: Type[R]
# ) -> Callable[Concatenate[Graph, P], tuple[Graph, R]]:
#     def bind(g: Graph, *args: P.args, **kwargs: P.kwargs) -> tuple[Graph, R]:
#         g, r = g.bind(node = f_new(*args, **kwargs))
#         return g, cast(t_ref, r)
#     return bind
