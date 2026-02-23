from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, TypeVar, cast, overload

# TODO: typed replace
from dataclasses import replace

import numpy as np

from frozendict import frozendict

from pydantic_core import CoreSchema, core_schema
from pydantic import GetCoreSchemaHandler, TypeAdapter

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


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

def only_index_not_none(
    *vs: Any | None, strict: bool = True
) -> int | None:
    i_res: int | None = None
    for i, v in enumerate(vs):
        if v is not None:
            if i_res is None:
                i_res = i
            else:
                raise ValueError(vs)
    if strict and i_res is None:
        raise ValueError(vs)
    return i_res


class ndarray(np.ndarray):
    """
    pydantic enhanced np.ndarray wrapper
    """

    @classmethod
    def cast(cls, v: np.ndarray) -> ndarray:
        return cls(v)

    @classmethod
    def validate(cls, v: Any, info: core_schema.ValidationInfo):
        if isinstance(v, np.ndarray):
            return v
        elif isinstance(v, Iterable):
            return np.array(v)
        else:
            raise ValueError((v, type(v)))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.chain_schema(
            [
                core_schema.with_info_plain_validator_function(
                    cls.validate,
                ),
            ]
        )


import polars
class pl:

    class Series(polars.Series):
        """
        pydantic enhanced pl.Series wrapper
        """

        @classmethod
        def cast(cls, v: polars.Series) -> pl.Series:
            return cls(v)

        @classmethod
        def validate(cls, v: Any, info: core_schema.ValidationInfo):
            if isinstance(v, pl.Series):
                return v
            # elif isinstance(v, Iterable):
            #     return np.array(v)
            else:
                raise ValueError((v, type(v)))

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            return core_schema.chain_schema(
                [
                    core_schema.with_info_plain_validator_function(
                        cls.validate,
                    ),
                ]
            )
    class DataFrame(polars.DataFrame):
        """
        pydantic enhanced pl.DataFrame wrapper
        """

        @classmethod
        def cast(cls, v: polars.DataFrame) -> pl.DataFrame:
            return cls(v)
        
        @classmethod
        def cast_opt(cls, v: polars.DataFrame | None) -> pl.DataFrame | None:
            if v is None:
                return v
            return cls(v)

        @classmethod
        def validate(cls, v: Any, info: core_schema.ValidationInfo):
            if isinstance(v, pl.DataFrame):
                return v
            # elif isinstance(v, Iterable):
            #     return np.array(v)
            else:
                raise ValueError((v, type(v)))

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            return core_schema.chain_schema(
                [
                    core_schema.with_info_plain_validator_function(
                        cls.validate,
                    ),
                ]
            )
        
class frozen:
    
    class dict(frozendict[KeyO, ValO]):
        """
        pydantic enhanced frozendict
        """

        @classmethod
        def cast(cls, v: frozendict[KeyO, ValO]) -> frozen.dict[KeyO, ValO]:
            return cls(v)

        @classmethod
        def validate(cls, v: Any, info: core_schema.ValidationInfo):
            if isinstance(v, frozendict):
                return v
            elif isinstance(v, dict):
                return frozendict(v)
            else:
                raise ValueError((v, type(v)))

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            return core_schema.chain_schema(
                [
                    core_schema.with_info_plain_validator_function(
                        cls.validate,
                    ),
                ]
            )
      