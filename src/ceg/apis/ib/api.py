
from __future__ import annotations

import threading
import queue
import time

from typing import NamedTuple
from frozendict import frozendict

from collections import defaultdict

from ibapi.client import EClient
from ibapi.wrapper import EWrapper

#  ------------------

class API(EWrapper, EClient):
    def __init__(self, q_i: queue.Queue, q_o: queue.Queue):
        EClient.__init__(self, self)
        self.q_i = q_i
        self.q_o = q_o

        self.n_messages = defaultdict(int)

    def keyboardInterrupt(self):
        # self.nKeybIntHard += 5
        raise SystemExit()

    def msgLoopTmo(self):
        message = None
        try:
            message = self.q_i.get(timeout=0.01)
        except:
            pass
        if message == "EXIT":
            raise SystemExit()
        # intended to be overloaded
        pass

    def msgLoopRec(self):
        # intended to be overloaded
        pass

    def historicalData(self, reqId, bar):
        self.n_messages[reqId] += 1

        # TODO: write both the query 
        # TODO: the contract
        # TODO: and the results to a sqlite db

        print(bar)
        self.q_o.put({"id": reqId, "n": self.n_messages[reqId]})

class Request(NamedTuple):
    id: int
    method: str
    expects: int
    timeout: float
    kwargs: dict

    def run(self, conn: Connection):
        res = None
        api, thread, q_i, q_o = conn
        getattr(api, self.method)(self.id, *self.kwargs.values())
        while True:
            try:
                res = q_o.get(timeout=self.timeout)
            except queue.Empty:
                q_i.put("EXIT")
                thread.join()
                break
            if res["id"] == self.id and res["n"] == self.expects:
                q_i.put("EXIT")
                thread.join()
                break
        return res

class Requests(NamedTuple):
    kwargs: frozendict[int, Request]
    results: dict[int, list]

    @classmethod
    def new(cls):
        return cls(
            frozendict(), # type: ignore
            {}
        )

    def bind(
        self,
        method: str,
        expects: int,
        timeout: float = 3.,
        **kwargs
    ):
        i_req = len(self.kwargs)
        req = Request(
            i_req, method, expects, timeout, kwargs
        )
        return self._replace(
            kwargs=self.kwargs.set(i_req, req)
        ), req

#  ------------------

class Connection(NamedTuple):
    api: API
    thread: threading.Thread
    queue_i: queue.Queue
    queue_o: queue.Queue

def connect(id: int):

    q_i = queue.Queue()
    q_o = queue.Queue()

    api = API(q_i, q_o)
    api.connect(host="127.0.0.1", port=7496, clientId=id)

    def run():
        api.run()
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    time.sleep(1)

    conn = Connection(api, thread, q_i, q_o)

    return conn

#  ------------------
