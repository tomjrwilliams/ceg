from __future__ import annotations

#  ------------------

from typing import ClassVar, NamedTuple, Literal, overload

import datetime
from functools import partial, lru_cache

import polars
import numpy

from ..utils import datetime_to_str
from ..db import DB, Contract, Query, Bar
from ..api import Requests, connect
from ..contracts import contract

from .... import core

#  ------------------

# ADJUSTED_LAST or TRADES have volume: whatToShow

# https://interactivebrokers.github.io/tws-api/historical_bars.html


def dates_between(start, end):
    yield start
    d = start
    while d < end:
        d += ONE_DAY
        yield d
    return

#  ------------------

ONE_DAY = datetime.timedelta(days=1)

BARS_SCHEMA = {
    "date": polars.Date,
    "open": polars.Float64,
    "high": polars.Float64,
    "low": polars.Float64,
    "close": polars.Float64,
    "volume": polars.Float64,
    "wap": polars.Float64,
}

BAR_FIELD_INDICES = {
    k: i - 1
    for i, k in enumerate(BARS_SCHEMA)
    if k != "date"
}


#  ------------------

def get_daily_level(
    fp: str,
    con: dict | Contract,
    start: datetime.date,
    end: datetime.date,
    bar_method: str,
    field: str | int,
    use_rth: bool = True,
    df: bool = False,
    at: datetime.date | None = None,
):
    if df:
        assert isinstance(field, str), field
        return get_daily_bars(
            fp,
            con,
            start,
            end,
            bar_method,
            use_rth=use_rth,
            # TODO: just rth?
            df=True,
            at=at,
        ).select("date", field)
    i = (
        field if isinstance(field, int)
        else BAR_FIELD_INDICES[field]
    )
    return get_daily_bars(
        fp,
        con,
        start,
        end,
        bar_method,
        use_rth=use_rth,
        df=df,
        at=at,
    )[:, i]

get_daily_open = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["open"]
)
get_daily_high = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["high"]
)
get_daily_low = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["low"]
)
get_daily_close = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["close"]
)
get_daily_volume = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["volume"]
)
get_daily_wap = partial(
    get_daily_level,
    field=BAR_FIELD_INDICES["wap"]
)

#  ------------------

class CacheKey(NamedTuple):
    fp: str
    id: int
    bar_method: str
    use_rth: bool
    
CACHE: dict[
    CacheKey, 
    tuple[
        datetime.date, 
        datetime.date,
        core.Array.np_2D
    ]
] = {}

@overload
def get_daily_bars(
    fp: str,
    con: dict | Contract,
    start: datetime.date,
    end: datetime.date,
    bar_method: str,
    use_rth: bool = True,
    df: Literal[False] = False,
    at: Literal[None] = None,
) -> core.Array.np_2D: ...

@overload
def get_daily_bars(
    fp: str,
    con: dict | Contract,
    start: datetime.date,
    end: datetime.date,
    bar_method: str,
    use_rth: bool = True,
    df: Literal[False] = False,
    at: datetime.date = None,
) -> core.Array.np_1D: ...

@overload
def get_daily_bars(
    fp: str,
    con: dict | Contract,
    start: datetime.date,
    end: datetime.date,
    bar_method: str,
    use_rth: bool = True,
    df: Literal[True] = True,
    at: datetime.date | None = None,
) -> polars.DataFrame: ...

def get_daily_bars(
    fp: str,
    con: dict | Contract,
    start: datetime.date,
    end: datetime.date,
    bar_method: str,
    use_rth: bool = True,
    df: bool = False,
    at: datetime.date | None = None,
):
    id: int
    if isinstance(con, dict):
        id = con["id"]
    else:
        id = con.id
    key = CacheKey(
        fp, id, bar_method, use_rth
    )
    if key not in CACHE:
        cache_end = end
        cache_start = end + ONE_DAY
        res = numpy.empty((0, 6))
    else:
        cache_start, cache_end, res = CACHE[key]
        
    if start < cache_start:
        bars: polars.DataFrame = req_daily_bars(
            fp,
            con,
            start,
            cache_start - ONE_DAY,
            bar_method=bar_method,
            use_rth=use_rth,
            df=True,
        )
        res = numpy.vstack((
            bars.select(
                "open",
                "high",
                "low",
                "close",
                "volume",
                "wap",
            ).to_numpy(),
            res,
        ))
        cache_start = start
    if end > cache_end:
        bars: polars.DataFrame = req_daily_bars(
            fp,
            con,
            end + ONE_DAY,
            cache_end,
            bar_method=bar_method,
            use_rth=use_rth,
            df=True,
        )
        res = numpy.vstack((
            bars.select(*BARS_SCHEMA.keys())
            .to_numpy()[:, 1:],
            res,
        ))
        cache_end = end
    if (
        cache_start == start
        or cache_end == end
    ):
        CACHE[key] = (
            cache_start,
            cache_end,
            res,
        )
    if at is not None:
        i_l = (at - cache_start).days
        i_r = i_l + 1
    else:
        i_l = (start - cache_start).days
        i_r = (cache_end - end).days

    if at is not None:
        res = res[i_l]
    elif i_r > 0:
        res = res[i_l:-i_r,:]
    else:
        res = res[i_l:]
        
    if df:
        return polars.DataFrame(
            res.T,
            schema = {
                "open": polars.Float64,
                "high": polars.Float64,
                "low": polars.Float64,
                "close": polars.Float64,
                "volume": polars.Float64,
                "wap": polars.Float64,
            }
        ).with_columns(
            polars.date_range(
                start, end
            ).alias("date")
        ).select(*BARS_SCHEMA.keys())
    return res

#  ------------------

@overload
def req_daily_bars(
    fp: str,
    # :contract
    con: dict | Contract,
    # :request fields
    start: datetime.date,
    end: datetime.date,
    bar_method: str,
    use_rth: bool = True,
    # connection fields
    instance: int = 69,
    timeout: float = 3.0,
    df: Literal[False] = False,
) -> list[Bar]: ...

@overload
def req_daily_bars(
    fp: str,
    # :contract
    con: dict | Contract,
    # :request fields
    start: datetime.date,
    end: datetime.date,
    bar_method: str,
    use_rth: bool = True,
    # connection fields
    instance: int = 69,
    timeout: float = 3.0,
    df: Literal[True] = True,
) -> polars.DataFrame: ...

def req_daily_bars(
    fp: str,
    # :contract
    con: dict | Contract,
    # :request fields
    start: datetime.date,
    end: datetime.date,
    bar_method: str,
    use_rth: bool = True,
    # connection fields
    instance: int = 69,
    timeout: float = 3.0,
    df: bool = False,
):
    conn = None
    db = DB(fp)

    print(con)

    db.contracts.create(db.fp)
    db.contract_dates.create(db.fp)
    db.queries.create(db.fp)
    db.bars.create(db.fp)
    db.history.create(db.fp)

    contr = db.insert_contract(con)
    contract_id = contr.id

    contr_start = db.get_start(contract_id)

    if contr_start is None:

        conn = connect(fp, instance)

        requests, req = Requests.new().bind(
            "reqHeadTimeStamp",
            expects=1,
            timeout=timeout,
            contract=contract(**contr._asdict()),
            whatToShow=bar_method,
            useRTH=0,
            formatDate=1,
        )

        conn.queue_i.put(
            ("contract_id", req.id, contract_id)
        )

        _ = req.run(conn, done=False)

    contr_start = db.get_start(contract_id)
    assert contr_start is not None, contr

    old_start = start
    if start < contr_start:
        start = contr_start

    if start >= end:
        queries = []
        required = []
    else:
        queries = db.get_queries(
            fields=None,
            where=dict(
                contract_id=f"={contract_id}",
                # TODO: this is one case where the encoding would be useful, capturing that int maps to bool
            ),
        )

        queries = [finalise_query(db, q) for q in queries]

        required = required_queries(
            db,
            start,
            end,
            queries,
            #
            contract_id=contract_id,
            rth=use_rth,
            method=bar_method,
            asof=datetime.date.today(),
            bound=0,
            done=None,
        )

    if len(required) and conn is None:
        conn = connect(fp, instance)

    for i, q in enumerate(required):
        assert conn is not None, contr

        print("required", q.start, q.end)

        query_id = q.id

        delta = (q.end - q.start).days

        duration = f"{delta} D"
        bar_size = "1 day"

        expects = delta

        requests, req = Requests.new(offset=1).bind(
            "reqHistoricalData",
            expects=expects,
            timeout=timeout,
            contract=contract(**contr._asdict()),
            endDateTime=datetime_to_str(end),
            durationString=duration,
            barSizeSetting=bar_size,
            whatToShow=bar_method,
            useRTH=int(use_rth),
            formatDate=1,
            keepUpToDate=0,
            chartOptions=[],
        )
        conn.queue_i.put(("query_id", req.id, query_id))
        conn.queue_i.put(
            ("contract_id", req.id, contract_id)
        )

        if len(required) and i == len(required) - 1:
            res = requests.run(conn)

    # if not len(required):
    #     conn.queue_i.put("EXIT")

    required = [finalise_query(db, q) for q in required]

    # merged = merge_queries(db, queries, required)

    res = query_bars(db, contr, start, end)

    if not len(res) or old_start < res[0].date:
        if not len(res):
            null_end = end
        else:
            null_end = res[0].date + datetime.timedelta(
                days=-1
            )
        res = [
            Bar(
                contract_id,
                d,
                query_id=None,  # type: ignore
                open=None,  # type: ignore
                high=None,  # type: ignore
                low=None,  # type: ignore
                close=None,  # type: ignore
                volume=None,  # type: ignore
                wap=None,  # type: ignore
            )
            for d in dates_between(old_start, null_end)
        ] + res

    if df:
        return polars.DataFrame([r._asdict() for r in res])
    return res


def required_queries(
    db: DB,
    start: datetime.date,
    end: datetime.date,
    queries: list[Query],
    batch: int = 20,
    **kwargs,
) -> list[Query]:
    queries = sorted(queries, key=lambda q: q.start)
    lhs = start
    required = []
    for q in queries:
        if not q.done:
            continue
        if q.start <= lhs and q.end >= end:
            return required
        elif q.start <= lhs:
            lhs = q.end
        elif q.start > lhs and q.start <= end:
            required.append((lhs, q.start))
            lhs = q.end
    if lhs < end:
        required.append((lhs, end))
    qs = []
    for s, e in required:
        lhs: datetime.date = s
        rhs = None
        while rhs is None or rhs < e:
            rhs = min(
                [e, lhs + datetime.timedelta(days=batch)]
            )
            qs.append(
                db.insert_query(
                    Query(
                        id=None,  # type: ignore
                        start=lhs,
                        end=rhs,
                        expected=(e - s).days,
                        **kwargs,
                    )
                )
            )
            lhs = rhs
            if rhs == e:
                break
    return qs


def finalise_query(db: DB, query: Query):
    if query.done is not None:
        return query
    bar_dates = db.get_bars(
        {"date": datetime.date},
        where=dict(
            contract_id=f"={query.contract_id}",
            query_id=f"={query.id}",
        ),
        historic=True,
    )
    bar_ds = [d["date"] for d in bar_dates]
    query = query._replace(
        bound=len(bar_dates),
        done=len(bar_dates) == query.expected,
        start=min([query.start] + bar_ds),
        end=max([query.end] + bar_ds),
    )
    if not query.done:
        return query
    for d in dates_between(query.start, query.end):
        if d in bar_ds:
            continue
        db.insert_bar(
            Bar(
                contract_id=query.contract_id,
                date=d,
                query_id=query.id,
                open=None,  # type: ignore
                high=None,  # type: ignore
                low=None,  # type: ignore
                close=None,  # type: ignore
                volume=None,  # type: ignore
                wap=None,  # type: ignore
            ),
            historic=False,
        )
    return db.insert_query(query)


def query_bars(
    db: DB,
    # queries: list[Query],
    contract: Contract,
    start: datetime.date,
    end: datetime.date,
):
    return db.get_bars(
        fields=None,
        where=dict(
            contract_id=f"={contract.id}",
            date=f" BETWEEN {start.toordinal()} AND {end.toordinal()}",
            # TODO: this conversion like bool should behandled by sql helper module
        ),
        order=dict(date="ASC"),
        historic=False,
    )

#  ------------------
