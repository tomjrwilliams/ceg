
from __future__ import annotations

#  ------------------

from ibapi.contract import Contract

import datetime

from .utils import datetime_to_str
from .api import Requests, connect
from .contracts import contract

#  ------------------

# sqlite db

# contract:
# id on the internal id, merge in other fields as given

# query:
# contract, asof, start, end, step, etc.

# price:
# date time (for non daily bars?), contract, rth, method (trades vs adjusted vs mid vs bid vs ask), duration_step, query_id, high, low, open, close

#  ------------------

# TODO:

# query the db for the range asked for, and send requests for the date / times not found
# how to mask over weekends? non trading days?
# ie. what does it return for days where there isn't data?
# ideally, we fill as null (which is a valid value)

#  ------------------

# TODO: so we need to send enough extra context with the request down into the queue for the API to log against an id
# for it to be able to store all of the above as required

#  ------------------

# ADJUSTED_LAST or TRADES have volume: whatToShow

# https://interactivebrokers.github.io/tws-api/historical_bars.html


def get_daily(
    # :contract
    contr: dict | Contract,
    # :request fields
    end: datetime.datetime | datetime.date | str,
    duration: str, # replace to strat date, infer the duration from that
    bar_size: str, # this is just always 1 D (caching methodology is totally different for intra-day, so needs a diff func)
    bar_method: str,
    use_rth=1,
    # connection fields
    instance: int = 69,
    timeout: float = 3.,
):
    conn = connect(instance)
    
    # assert isinstance(contr, dict), contr
    if isinstance(contr, dict):
        contr = contract(**contr)

    if not isinstance(end, str):
        end = datetime_to_str(end)

    expects = int(duration.split(" ")[0])

    # steps:
    # TODO: check for any queries at all with done not set
    # run through the done check flow (ie. n_results = n_expected, fill none in gaps)

    # TODO: so on load, look fro synthetics overlappnig with range
    # build queries between gaps as required

    # TODO: store query, with start end contractid, rth, bar_method, query_asof, and expected size

    # TODO: on final entry, set query done
    # if so, fill date gaps between start end with none

    # price pkey is date,con_id,query_id

    # then have two price types: one the historic, stored per query
    # and one maintained for the final synthetic aggregated query (s)
    # so we can even do queries against particular historic queries without even joining to a new table

    # TODO: check for synthetic queries with earliest start, and rhs within the query range
    # merge query ranges (into synthetic)

    # TODO: look for any now within / overlapping new range
    # merge as required (deleting this time)

    # TODO: then finalyl should be able to do a single query on the price table, join the final query id, with the given date range
    
    requests, req = Requests.new().bind(
        "reqHistoricalData",
        expects=expects,
        timeout=timeout,
        contract=contr,
        endDateTime=end,
        durationString=duration,
        barSizeSetting=bar_size,
        whatToShow=bar_method,
        useRTH = use_rth,
        formatDate=1,
        keepUpToDate=0,
        chartOptions = []
    )

    res = req.run(conn)

    return

# python ../../ceg/src/ceg/apis/ib.py

# IYJ (ISHARES)
# XLKS (INVESCO)
# SX6P (STOXX)

# cboe msci indices?
# cme s&p indices?
# dj global?

#  ------------------

