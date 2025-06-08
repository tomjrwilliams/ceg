
from typing import Protocol, ParamSpec, Type, TypeVar, Callable, Concatenate, cast, Generic, NamedTuple, ClassVar

from .graphs import Graph, Node, Ref

P = ParamSpec("P")
N = TypeVar("N", bound = Node.Any)
R = TypeVar("R", bound = Ref.Any)

# def def_bind(
#     f_new: Callable[P, N],
#     t_ref: Type[R]
# ) -> Callable[Concatenate[Graph, P], tuple[Graph, R]]:
#     def bind(g: Graph, *args: P.args, **kwargs: P.kwargs) -> tuple[Graph, R]:
#         g, r = g.bind(node = f_new(*args, **kwargs))
#         return g, cast(t_ref, r)
#     return bind

class fs:
    pass

F = TypeVar("F", bound=fs)

class HasNew(Generic[N, P, R, F]):

    def __call__(self, NAME: str, *args: P.args, **kwargs: P.kwargs) -> N: ...

    @classmethod
    def fs(cls) -> Type[F]: ...

    @classmethod
    def ref(cls, i: int | Ref.Any, slot: int | None = None) -> R: ...

    @classmethod
    def new(cls, *args: P.args, **kwargs: P.kwargs) -> N: ...

    @classmethod
    def bind(cls, g: Graph, *args: P.args, **kwargs: P.kwargs) -> tuple[Graph, R]:
        g, r = g.bind(node=cls.new(*args, **kwargs))
        return g, cls.ref(r)

def bind_from_new(
    new: Callable[P, N],
    ref: Callable[[Ref.Any], R],
    fs: Type[F]
) -> Callable[
    [Type[N]], HasNew[N, P, R, F]
]:
    def decorator(cls):
        class cls_new(cls, HasNew):

            @classmethod
            def fs(cls) -> Type[F]:
                return fs

        return cls_new
    return decorator
