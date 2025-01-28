from typing import NamedTuple

from .Array import Shape

class Ref(NamedTuple):
    i: int
    attr: str | None
    shape: Shape | None

    @classmethod
    def new(cls, i: int, attr: str | None=None, shape: Shape | None=None):
        return cls(i, attr, shape)

Any = Ref

class RefColND(Ref):
    pass

class RefRowND(Ref):
    pass

ColND = RefColND
RowND = RefRowND

class Ref0D(Ref):
    pass

class Ref1D(Ref):
    pass

class Ref2D(Ref):
    pass

class Ref3D(Ref):
    pass

D0 = Ref0D
D1 = Ref1D
D2 = Ref2D
D3 = Ref3D

class RefFloatND(Ref):
    pass

class RefIntND(Ref):
    pass

class RefBoolND(Ref):
    pass

FloatND = RefFloatND
IntND = RefIntND
BoolND = RefBoolND


class RefCol(RefColND, Ref0D):
    pass
class RefCol1D(RefColND, Ref1D):
    pass
class RefCol2D(RefColND, Ref2D):
    pass
class RefCol3D(RefColND, Ref3D):
    pass

Col = RefCol
Col1D = RefCol1D
Col2D = RefCol2D
Col3D = RefCol3D

class RefInt(RefIntND, RefCol):
    pass
class RefInt1D(RefIntND, RefCol1D):
    pass
class RefInt2D(RefIntND, RefCol2D):
    pass
class RefInt3D(RefIntND, RefCol3D):
    pass

class RefBool(RefBoolND, RefCol):
    pass
class RefBool1D(RefBoolND, RefCol1D):
    pass
class RefBool2D(RefBoolND, RefCol2D):
    pass
class RefBool3D(RefBoolND, RefCol3D):
    pass

class RefFloat(RefFloatND, RefCol):
    pass
class RefFloat1D(RefFloatND, RefCol1D):
    pass
class RefFloat2D(RefFloatND, RefCol2D):
    pass
class RefFloat3D(RefFloatND, RefCol3D):
    pass

class RefRow(RefRowND, Ref0D):
    pass
class RefRow1D(RefRowND, Ref1D):
    pass
class RefRow2D(RefRowND, Ref2D):
    pass
class RefRow3D(RefColND, Ref3D):
    pass