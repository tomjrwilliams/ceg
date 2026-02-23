import operator
from inspect import Parameter
from typing import (
    TypeVar,
    TypeGuard,
    Type,
    Any,
    Callable,
    Union,
    Sequence,
    get_origin,
    overload
)
from types import UnionType
from functools import partial

EMPTY = Parameter.empty()
Empty = Parameter.empty

V = TypeVar("V")
T = TypeVar("T")
C = TypeVar("C")

class ValueShouldBe(ValueError):
    # TODO: fields for args
    # TODO: informative error message template
    pass

class ValueShouldBePredicate(ValueShouldBe):
    pass

class ValueShouldBeEqual(ValueShouldBe):
    pass

class ValueShouldBeType(ValueShouldBe):
    pass

class ValueShouldBeNone(ValueShouldBe):
    pass

class ValueShouldBeNotNone(ValueShouldBe):
    pass

class ValueShouldBeNotEmpty(ValueShouldBe):
    pass

def is_none(value: V | None) -> TypeGuard[None]:
    return value is None

def not_none(value: V | None) -> TypeGuard[V]:
    return value is not None

def is_empty(value: Any) -> TypeGuard[Empty]:
    return isinstance(value, Empty)

def not_empty(value: V | Empty) -> TypeGuard[V]:
    return not is_empty(value)

def is_type(value: V | Type[T]) -> TypeGuard[Type[T]]:
    if isinstance(value, type):
        return True
    elif get_origin(value) is UnionType:
        return True
    return False    

def not_type(value: V | Type[T]) -> TypeGuard[V]:
    return not is_type(value)

# def not_type(value: V | Type[T]) -> TypeGuard[V]:
#     return not isinstance(value, type)

def assert_is_instance(
    value: Any,
    t: Type[T],
) -> T:
    if not isinstance(value, t):
        raise ValueShouldBeType(value, t)
    return value

def assert_is_not_none(value: V | None) -> V:
    if value is None:
        raise ValueShouldBeNotNone(value)
    return value

def assert_is_none(value: Any | None) -> None:
    if value is not None:
        raise ValueShouldBeNone(value)
    return value

def assert_not_empty(value: V | Empty) -> V:
    if not_empty(value):
        return value
    raise ValueShouldBeNotEmpty(value)

def assert_comparison(
    value: V,
    to: C,
    using: Callable[[V, C], bool]
) -> V:
    if not using(value, to):
        raise ValueShouldBePredicate(
            value, to, using
        )
    return value

def is_comparator_for(
    comp: Callable[[Any, Any], bool] | None,
    to: C,
) -> TypeGuard[Callable[[Any, C], bool]]:
    return comp is not None

@overload
def assert_is(
    value: V | None,
    type_or_value: None,
    comparator: None=None,
) -> None: ...

@overload
def assert_is(
    value: V,
    type_or_value: None,
    comparator: Callable[[V, T], bool],
) -> V: ...
    
@overload
def assert_is(
    value: V | None,
    type_or_value: Empty = EMPTY,
    comparator: None = None,
) -> V: ...

@overload
def assert_is(
    value: V,
    type_or_value: Type[T],
    comparator: None = None,
) -> T: ...

@overload
def assert_is(
    value: V,
    type_or_value: Type[T],
    comparator: Callable[[V, type[T]], bool],
) -> V: ...

@overload
def assert_is(
    value: V,
    type_or_value: T,
    comparator: Callable[[V, T], bool] | None = None,
) -> V: ...

# TODO: separate type and value kwargs, easier fo rpartial applicaiton

def assert_is(
    value: V | None,
    type_or_value: T | Type[T] | None | Empty = EMPTY,
    comparator: (
        Callable[[V, T], bool]
        | Callable[[V, Type[T]], bool]
        | None
    ) = None,
):
    """
    >>> i: int = 1
    >>> i_opt: int | None = 1
    >>> v0: None = assert_is(i, None) 
    >>> v1: int = assert_is(i_opt)
    >>> v2: float = assert_is(i_opt, float)
    >>> v3: type[int] = assert_is(int, float, issubclass)
    >>> v4: int = assert_is(i, 1)
    >>> v5: int = assert_is(i, 1, operator.le)
    """
    if is_empty(type_or_value):
        return assert_is_not_none(value)
    elif is_type(type_or_value):
        if is_none(comparator):
            return assert_is_instance(value, type_or_value)
        t: Type[T] = type_or_value
        if is_comparator_for(comparator, type_or_value):
            return assert_comparison(value, t, comparator)
        raise ValueError() # eh?
    elif type_or_value is None:
        if is_comparator_for(comparator, None):
            return assert_comparison(value, None, comparator)
        return assert_is_none(value)
    else:
        value = assert_is_not_none(value)
        v: T = assert_not_empty(type_or_value)
        if is_comparator_for(comparator, v):
            return assert_comparison(value, v, comparator)
        elif value != v:
            raise ValueShouldBeEqual(value, v)
    return value

# def assert_are(
#     value: Sequence[V] | None,
#     type_or_value: T | Type[T] | None | Empty = EMPTY,
#     comparator: (
#         Callable[[V, T], bool]
#         | Callable[[V, Type[T]], bool]
#         | None
#     ) = None,
# ):
#     val = all(map(
#         partial(
#             assert_is,
#             type_or_value=type_or_value, 
#             comparator=comparator,
#         ),
#         assert_is(value),
#     ))