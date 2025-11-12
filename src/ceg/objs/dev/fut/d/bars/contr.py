
# @dataclass(frozen=True)
# class FutDailyBarContractSnap(FutDailyBarSnap):
#     type: Literal["FutDailyBarContractSnap"] = "FutDailyBarContractSnap"

# @dataclass(frozen=True)
# class FutDailyBarContract(Object):
#     parent: FutDailyBar

#     symbol: str
#     field: str
#     adjust: str | None = None
#     expiry: str | None = None
#     fx: str | None = None  # ccy to convert to
#     source: str | None = None

#     version: int = 0
#     type: Literal["FutDailyBarContract"] = "FutDailyBarContract"

#     @classmethod
#     def new(
#         cls,
#         parent: FutDailyBar,
#         symbol: str,
#         expiry: str,
#         field: str,
#         adjust: str | None = None,
#         fx: str | None = None,
#         source: str | None = None,
#         path = "con",
#     ):
#         return cls(
#             path=f"{parent.path}-{path}",
#             parent=parent,
#             symbol=symbol,
#             expiry=expiry,
#             field=field,
#             adjust=adjust,
#             fx=fx,
#             source=source,
#         )

#     def value_key_val(self, domain: Domain) -> ValueKeyVal:
#         # TODO: as for ordinal, but specific expiry
#         raise ValueError(self)

#     def depends(self, domain: Domain) -> list[Dependency]:
#         # TODO: as for ordinal, but specific expiry
#         return []

#     def handlers(self, event: Event, domain: Domain) -> list[Handler]:
#         # TODO: as for ordinal, but specific expiry
#         return []