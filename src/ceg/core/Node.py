from typing import NamedTuple, ClassVar, Type

from . import Ref
from . import Array
from . import Series

from .types import *

Any = NodeND


class NodeNull(NodeND):

    @classmethod
    def new(cls):
        return cls(*cls.args())


Null = NodeNull
null = Null.new()


class NodeCol(NodeND):
    REF: ClassVar[Type[Ref.Any]] = Ref.Col
    SERIES: ClassVar[Type[Series.Any]] = Series.Col


Col = NodeCol


class NodeCol1D(NodeND):
    REF: ClassVar[Type[Ref.Any]] = Ref.Col1D
    SERIES: ClassVar[Type[Series.Any]] = Series.Col1D


Col1D = NodeCol1D

class NodeCol2D(NodeND):
    REF: ClassVar[Type[Ref.Any]] = Ref.Col2D
    SERIES: ClassVar[Type[Series.Any]] = Series.Col2D


Col2D = NodeCol2D