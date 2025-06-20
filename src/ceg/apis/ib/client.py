from __future__ import annotations

import threading
import queue
import time
import datetime

from typing import NamedTuple
from frozendict import frozendict

from collections import defaultdict

from ibapi.client import EClient
from ibapi.wrapper import EWrapper

from .prices.db import DB

#  ------------------

# TODO: just the base interrupt, message passing fraemwork

class API(EWrapper, EClient):
    def __init__(
        self, db_fp: str, q_i: queue.Queue, q_o: queue.Queue
    ):
        EClient.__init__(self, self)

        self.db_fp = db_fp

        self.query_ids = {}
        self.contract_ids = {}

        self.q_i = q_i
        self.q_o = q_o

        self.req_done = {}
        self.n_messages = defaultdict(int)

    def keyboardInterrupt(self):
        # self.nKeybIntHard += 5
        raise SystemExit()

    def check_queue_in(self):
        message = None
        try:
            message = self.q_i.get(timeout=0.01)
        except:
            pass
        if message == "EXIT":
            raise SystemExit()
        if isinstance(message, tuple):
            # print(message)
            m = message[0]
            if m == "query_id":
                self.query_ids[message[1]] = message[2]
            elif m == "contract_id":
                self.contract_ids[message[1]] = message[2]
            else:
                raise ValueError(message)
        return message is not None

    def msgLoopTmo(self):
        while self.check_queue_in():
            pass
        return

    def msgLoopRec(self):
        while self.check_queue_in():
            pass

    def headTimestamp(self, reqId, headTimestamp: str):

        self.check_queue_in()

        self.n_messages[reqId] += 1

        contract_id = self.contract_ids.get(reqId)

        assert contract_id is not None, (
            reqId,
            self.contract_ids,
        )

        d = headTimestamp.split("-")[0]
        y = d[:4]
        m = d[4:6]
        d = d[6:8]

        dt = datetime.date(int(y), int(m), int(d))

        print(dt)

        db = DB(self.db_fp)
        db.insert_start(contract_id, dt)

        self.q_o.put(
            {
                "id": reqId,
                "n": self.n_messages[reqId],
                "done": True,
            }
        )

    def historicalData(self, reqId, bar):
        self.check_queue_in()

        self.n_messages[reqId] += 1

        # TODO: at some point can pass a mpaping from reqid to contract and quyery dowb the pipe, for now we only have one per instance

        query_id = self.query_ids.get(reqId)
        contract_id = self.contract_ids.get(reqId)

        assert query_id is not None, (reqId, self.query_ids)
        assert contract_id is not None, (
            reqId,
            self.contract_ids,
        )

        d = bar.date
        y = d[:4]
        m = d[4:6]
        d = d[6:8]

        bar = {
            "contract_id": contract_id,
            "date": datetime.date(int(y), int(m), int(d)),
            "query_id": query_id,
            "open": bar.open,  # TODO: what is null value?
            "high": bar.high,  # TODO: what is null value?
            "low": bar.low,  # TODO: what is null value?
            "close": bar.close,  # TODO: what is null value?
            "volume": self.nullify(bar.volume),
            "wap": self.nullify(bar.wap),
        }
        print(bar)

        db = DB(self.db_fp)

        _ = db.insert_bar(bar)
        # _ = db.insert_bar(bar, historic=False)
        # _ = db.insert_bar(bar, historic=True)

        self.q_o.put(
            {
                "id": reqId,
                "n": self.n_messages[reqId],
                "done": False,
            }
        )

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self.check_queue_in()
        self.req_done[reqId] = True
        self.q_o.put(
            {
                "id": reqId,
                "done": True,
                "start": start,
                "end": end,
                "n": None,
            }
        )

    def contractDetails(self, reqId, contractDetails):
        det = contractDetails

        self.check_queue_in()
        self.n_messages[reqId] += 1

        con = det.contract

        con_dict = {
            "id": con.conId,
            "type": con.secType,
            "symbol": con.symbol,
            "currency": con.currency,
            "exchange": con.exchange,
            # "underlying": con,
            "primary_exchange": con.primaryExchange,
            "expiry": con.lastTradeDateOrContractMonth,
            "last_trade": con.lastTradeDate,
            "local_symbol": con.localSymbol,
            "strike": con.strike,
            "right": con.right,
            "multiplier": con.multiplier,
            "sec_id": con.secId,
            "sec_id_type": con.secIdType,
            "description": con.description,
            "include_expired": con.includeExpired,
        }
        db = DB(self.db_fp)
        res = db.insert_contract(con_dict)

        print(res)

        # TODO: valid exchanges, long name, contract month, industry, category, subcategory, trading hours, liquid hours, time zone id, last trade time, real expiration date
        # (in the details not the contract)

        self.q_o.put(
            {
                "id": reqId,
                "n": self.n_messages[reqId],
                "done": False,
            }
        )

    def contractDetailsEnd(self, reqId):
        self.check_queue_in()
        self.req_done[reqId] = True
        self.q_o.put(
            {
                "id": reqId,
                "done": True,
                "n": None,
            }
        )

    def nullify(self, v, null_if=-1):
        if v == null_if:
            return None
        if isinstance(v, decimal.Decimal):
            v = float(v)
            if v == 0:
                return None
        return v

import decimal

class Request(NamedTuple):
    id: int
    method: str
    expects: int
    timeout: float
    kwargs: dict

    def run(self, conn: Connection, done: bool = True):
        res = None
        api, thread, q_i, q_o = conn
        getattr(api, self.method)(
            self.id, *self.kwargs.values()
        )
        while True:
            try:
                res = q_o.get(timeout=self.timeout)
            except queue.Empty:
                if done:
                    q_i.put("EXIT")
                    thread.join()
                break
            if (
                res["id"] == self.id
                and (
                    res["done"] or res["n"] == self.expects
                )
            ):
                if done:
                    q_i.put("EXIT")
                    thread.join()
                break
        return res

class Requests(NamedTuple):
    offset: int
    kwargs: frozendict[int, Request]
    results: dict[int, list]

    @classmethod
    def new(cls, offset = 0):
        return cls(offset, frozendict(), {})  # type: ignore

    def bind(
        self,
        method: str,
        expects: int,
        timeout: float = 3.0,
        **kwargs,
    ):
        i_req = self.offset + len(self.kwargs)
        req = Request(
            i_req, method, expects, timeout, kwargs
        )
        return (
            self._replace(
                kwargs=self.kwargs.set(i_req, req)
            ),
            req,
        )

    def run(self, conn: Connection):
        res = None
        api, thread, q_i, q_o = conn
        timeout = 3.0
        expects = {}
        for req_id, req in self.kwargs.items():
            getattr(api, req.method)(
                req.id, *req.kwargs.values()
            )
            if req.timeout > timeout:
                timeout = req.timeout
            expects[req_id] = req.expects
        results = defaultdict(int)
        done = {}
        while True:
            try:
                res = q_o.get(timeout=timeout)
            except queue.Empty:
                q_i.put("EXIT")
                thread.join()
                break
            req_id = res["id"]
            done[req_id] = done.get(req_id, False) or res["done"]
            results[req_id] += 1
            if all(
                [
                    (results.get(id) == e) or (done.get(id))
                    for id, e in expects.items()
                ]
            ):
                q_i.put("EXIT")
                thread.join()
                break
        return results


#  ------------------


class Connection(NamedTuple):
    api: API
    thread: threading.Thread
    queue_i: queue.Queue
    queue_o: queue.Queue

from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

def connect(db: str, id: int):

    q_i = queue.Queue()
    q_o = queue.Queue()

    api = API(db, q_i, q_o)
    
    out = StringIO()

    with redirect_stdout(out) as _, redirect_stderr(out) as _:
        api.connect(host="127.0.0.1", port=7496, clientId=id)

        def run():
            api.run()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        time.sleep(1)

    out_s = out.getvalue()
    exit = False

    if "Couldn't connect to TWS." in out_s:
        raise ValueError(out_s)
    elif not "Market data farm connection is OK" in out_s:
        exit = True

    if exit:
        q_i.put("EXIT")
        thread.join()

        raise ValueError(out_s)

    conn = Connection(api, thread, q_i, q_o)

    return conn


#  ------------------
