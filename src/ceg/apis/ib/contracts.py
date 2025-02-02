
from ibapi.contract import Contract

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

#  ------------------

def contract(
    **kwargs,
):
    contr = Contract()
    for k, v in kwargs.items():
        setattr(contr, k, v)

    # contract.symbol = symbol
    # contract.secType = sec_type
    # contract.currency = currency
    # contract.exchange = exchange
    # contract.lastTradeDateOrContractMonth = contract_last_trade_or_month

    return contr
    
#  ------------------
