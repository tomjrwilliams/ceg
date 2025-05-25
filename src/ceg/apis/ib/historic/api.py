from __future__ import annotations

#  ------------------

from typing import ClassVar, NamedTuple, Literal, overload

import datetime
from functools import partial, lru_cache

import polars
import numpy

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

#  ------------------

@overload
def req_daily_bars(
    fp: str,
    # :contract
    con: Contract,
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
    con: Contract,
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
    con: Contract,
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

    db.contracts.create(db.fp)
    db.contract_dates.create(db.fp)
    db.queries.create(db.fp)
    db.bars.create(db.fp)
    db.history.create(db.fp)

    if con.type=="IND":
        bar_method="TRADES"

    try:
        if con.id is None:
            contrs = db.list_contracts(**{
                k: v for k, v in con._asdict().items()
                if v is not None
            })

            if not len(contrs):
                conn = connect(fp, instance)
                
                requests, req = Requests.new().bind(
                    "reqContractDetails",
                    expects=-1,
                    timeout=timeout,
                    contract=contract(**con._asdict()),
                )

                _ = req.run(conn, done=False)

        contrs = db.list_contracts(**{
            k: v for k, v in con._asdict().items()
            if v is not None
        })

        assert len(contrs) == 1, contrs
        contr = contrs[0]

        contract_id = contr.id
        assert contract_id is not None, contract_id
        contr_start = db.get_start(contract_id)

        if contr_start is None:

            if conn is None:
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

            # for q in queries:
            #     print(q)

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

        requests = Requests.new()

        for i, q in enumerate(required):
            assert conn is not None, contr

            print(
                "required",
                con.type,
                con.symbol,
                con.exchange,
                q.start,
                q.end,
            )

            query_id = q.id

            delta = (q.end - q.start).days

            duration = f"{delta} D"
            bar_size = "1 day"

            expects = delta

            requests, req = requests.bind(
                "reqHistoricalData",
                expects=expects,
                timeout=timeout,
                contract=contract(**contr._asdict()),
                endDateTime=(
                    end.strftime("%Y%m%d-%H:%M:%S")
                    if contr.type != "CONTFUT"
                    else ""
                ),
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

        if conn is not None:
            conn.queue_i.put("EXIT")
        if df:
            return polars.DataFrame([r._asdict() for r in res])
        return res

    except Exception as e:
        if conn is not None:
            conn.queue_i.put("EXIT")
        raise e


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
                        expected=(rhs - lhs).days,
                        **kwargs,
                    )
                )
            )
            lhs = rhs
            if rhs == e:
                break
    return qs


def finalise_query(db: DB, query: Query):
    if query.done:
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
