
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


def get_contract(
    # :contract
    contr: dict | Contract,
    # :request fields
    end: datetime.datetime | datetime.date | str,
    duration: str,
    bar_size: str,
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

