

#  ------------------

# https://interactivebrokers.github.io/tws-api/contract_details.html

# either 

FIELD_MAP = {
    "contract_id": "conId",
    "id": "conId",
    "type": "secType",
    "sec_id": "secId",
    "sec_id_type": "secIdType",
    "symbol_local": "localSymbol",
    "primary_exchange": "primaryExchange",
    "expiry": "lastTradeDateOrContractMonth",
    "include_expired": "includeExpired",
}


def contract(
    **kwargs,
):
    contr = Contract()
    for k, v in kwargs.items():
        if v is None:
            continue
        setattr(contr, FIELD_MAP.get(k, k), v)

    # contract.symbol = symbol
    # contract.secType = sec_type
    # contract.currency = currency
    # contract.exchange = exchange
    # contract.lastTradeDateOrContractMonth = contract_last_trade_or_month

    return contr

#  ------------------

class Contract(NamedTuple):
    """
    id: int
    type: str
    symbol: str
    underlying: str
    currency: str
    exchange: str
    """

    id: int | None
    type: str
    symbol: str
    currency: str | None
    exchange: str | None
    # underlying: str | None
    primary_exchange: str | None # for uniquely identifiying where many traded; echange = SMART primary = ARCA, say
    expiry: str | None # futures
    last_trade: str | None
    local_symbol: str | None # also futures eg. FGBL MAR 23
    strike: float | None
    right: str | None # C or P
    multiplier: str | None #
    sec_id: str | None
    sec_id_type: str | None # eg. FIGI
    description: str | None # for search
    # issuer_id: str | None 
    include_expired: bool | None

    @classmethod
    def new(
        cls,
        type: str,
        symbol: str,
        currency: str | None = None,
        exchange: str | None = None,
        id: int | None = None,
        # underlying: str | None=None,
        primary_exchange: str | None=None,
        expiry: str | None=None,
        last_trade: str | None=None,
        local_symbol: str | None=None,
        strike: float | None=None,
        right: str | None=None,
        multiplier: str | None=None,
        sec_id: str | None=None,
        sec_id_type: str | None=None,
        description: str | None=None,
        include_expired: bool | None=None,
    ):
        return cls(
            id=id,
            type=type,
            symbol=symbol,
            currency=currency,
            exchange=exchange,
            # underlying=underlying,
            primary_exchange=primary_exchange,
            expiry=expiry,
            last_trade=last_trade,
            local_symbol=local_symbol,
            strike=strike,
            right=right,
            multiplier=multiplier,
            sec_id=sec_id,
            sec_id_type=sec_id_type,
            description=description,
            include_expired=include_expired,
        )

table_contracts = sql.Table.new(
    "contracts",
    id=int,
    type=str,
    symbol=str,
    currency=str,
    exchange=str,
    # underlying=str ,
    primary_exchange=str ,
    expiry=str ,
    last_trade=str,
    local_symbol=str ,
    strike=float,
    right=str ,
    multiplier=str ,
    sec_id=str ,
    sec_id_type=str ,
    description=str ,
    include_expired=bool,
    primary_key=(
        "id",
    ),
)

# TODO: reqContractDetails endpoint


# int 	ConId [get, set]
#  	The unique IB contract identifier.

# string 	Symbol [get, set]
#  	The underlying's asset symbol.

# string 	SecType [get, set]
#  	The security's type: STK - stock (or ETF) OPT - option FUT - future IND - index FOP - futures option CASH - forex pair BAG - combo WAR - warrant BOND- bond CMDTY- commodity NEWS- news FUND- mutual fund.

# string 	LastTradeDateOrContractMonth [get, set]
#  	The contract's last trading day or contract month (for Options and Futures). Strings with format YYYYMM will be interpreted as the Contract Month whereas YYYYMMDD will be interpreted as Last Trading Day.

# string 	LastTradeDate [get, set]
#  	The contract's last trading day.

# double 	Strike [get, set]
#  	The option's strike price.

# string 	Right [get, set]
#  	Either Put or Call (i.e. Options). Valid values are P, PUT, C, CALL.

# string 	Multiplier [get, set]
#  	The instrument's multiplier (i.e. options, futures).

# string 	Exchange [get, set]
#  	The destination exchange.

# string 	Currency [get, set]
#  	The underlying's currency.

# string 	LocalSymbol [get, set]
#  	The contract's symbol within its primary exchange For options, this will be the OCC symbol.

# string 	PrimaryExch [get, set]
#  	The contract's primary exchange. For smart routed contracts, used to define contract in case of ambiguity. Should be defined as native exchange of contract For exchanges which contain a period in name, will only be part of exchange name prior to period, i.e. ENEXT for ENEXT.BE.

# string 	TradingClass [get, set]
#  	The trading class name for this contract. Available in TWS contract description window as well. For example, GBL Dec '13 future's trading class is "FGBL".

# bool 	IncludeExpired [get, set]
#  	If set to true, contract details requests and historical data queries can be performed pertaining to expired futures contracts. Expired options or other instrument types are not available.

# string 	SecIdType [get, set]
#  	Security's identifier when querying contract's details or placing orders ISIN - Example: Apple: US0378331005 CUSIP - Example: Apple: 037833100.

# string 	SecId [get, set]
#  	Identifier of the security type. More...

# string 	Description [get, set]
#  	Description of the contract.

# string 	IssuerId [get, set]
#  	IssuerId of the contract.

# string 	ComboLegsDescription [get, set]
#  	Description of the combo legs.