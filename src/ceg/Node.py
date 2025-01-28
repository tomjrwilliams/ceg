
from typing import NamedTuple, ClassVar, Type

from . import Ref
from . import Array
from . import Series

from .types import *

from frozendict import frozendict

Any = NodeND

class NodeNull(NodeND):
    
    @classmethod
    def new(cls):
        return cls(*cls.args())

Null = NodeNull
null = Null.new()

class NodeCol1D(NodeND):
    REF: ClassVar[Type[Ref.Any]] = Ref.Col1D
    SERIES: ClassVar[Type[Series.Any]] = Series.Col1D

Col1D = NodeCol1D
