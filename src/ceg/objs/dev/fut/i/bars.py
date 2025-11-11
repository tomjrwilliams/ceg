
# @dataclass(frozen=True)
# class BarFutIntraDay(Object):
#     parent: BarFut | str
#     version: int = 0
#     type: Literal["BarFutIntraDay"] = "BarFutIntraDay"

#     # TODO: freq? for which bar sampling freq

#     @classmethod
#     def new(
#         cls,
#         parent: BarFut | EngineRoot,
#         path = "i",
#     ):
#         if isinstance(parent, EngineRoot):
#             bars = Bar.new(parent)
#             parent = BarFut.new(bars)
#         return cls(
#             path=f"{parent.path}.{path}",
#             parent=parent,
#         )
