from typing import TypeVar

ID = TypeVar("ID", covariant=False, contravariant=False)
KeyI = TypeVar("KeyI", covariant=False, contravariant=False)
ValI = TypeVar("ValI", covariant=False, contravariant=False)
KeyO = TypeVar("KeyO", covariant=False)
ValO = TypeVar("ValO", covariant=False)

_ = TypeVar("_", bound=object)

