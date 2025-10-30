from typing import TypeVar

_ = TypeVar("_", bound=object)
T = TypeVar("T")
ID = TypeVar("ID", covariant=False, contravariant=False)
DtI = TypeVar("DtI", covariant=False, contravariant=False)
KeyI = TypeVar("KeyI", covariant=False, contravariant=False)
ValI = TypeVar("ValI", covariant=False, contravariant=False)
DtO = TypeVar("DtO", covariant=False)
KeyO = TypeVar("KeyO", covariant=False)
ValO = TypeVar("ValO", covariant=False)
ValT = TypeVar("ValT", covariant=False)

def not_none(v: T | None) -> T:
    assert v is not None, v
    return v