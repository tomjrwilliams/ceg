
from typing import TYPE_CHECKING, Any, Iterable

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

class ndarray(np.ndarray):
    """
    pydantic enhanced np.ndarray wrapper
    """

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