from typing import TypeVar, Type
from dataclasses import fields, MISSING

from .validation import assert_is

T = TypeVar("T")


def field_default(c, key: str, t: Type[T]) -> T | None:
    fs = fields(c)
    for f in fs:
        if f.name == key:
            v = f.default
            if v is MISSING or v is None:
                return None
            return assert_is(v, t)
    return None
