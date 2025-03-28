import datetime

from types import SimpleNamespace
from typing import NamedTuple, ClassVar, overload, Literal, Protocol

from . import sql

#  ------------------

class StringNamespace(SimpleNamespace):

    def __getattribute__(self, k: str) -> str:
        return SimpleNamespace.__getattribute__(self, k)

T = TYPES = StringNamespace(
    FX="CASH",
    CRYPT="CRYPTO",
    EQ="STK",
    STOCK="STK",
    INDEX="IND",
    CFD="CFD",
    FUT="FUT",
    GENERIC="CONTFUT",
    CHAIN="FUT+CONTFUT",
    OPTION="OPT",
    FUT_OPT="FOP",
    BOND="BOND",
    MUTUAL="FUND",
    CO="CMDTY",
    WARRANT="WAR",
    STRUCT="IOPT",
)
E = EXCH = EXCHANGES = StringNamespace(
    ARCA="ARCA",
    BATS="BATS",
    CBOE="CBOE",
    #
    SMART="SMART",
)
C = CURRENCY = StringNamespace(
    USD="USD"
)


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

class Query(NamedTuple):
    """
    id: int
    contract_id: int
    rth: bool
    method: str
    asof: datetime.date
    start: datetime.date
    end: datetime.date
    expected: int
    bound: int
    done: bool
    """

    id: int
    contract_id: int
    rth: bool
    method: str
    asof: datetime.date
    start: datetime.date
    end: datetime.date
    expected: int
    bound: int
    done: bool


class Bar(NamedTuple):
    """
    contract_id: int
    date: datetime.date
    query_id: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    wap: float
    """

    contract_id: int
    date: datetime.date
    query_id: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    wap: float


#  ------------------

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
table_contract_dates = sql.Table.new(
    "contract_dates",
    contract_id=int,
    start=datetime.date,
    end=datetime.date,
    primary_key=("contract_id",),
)

table_queries = sql.Table.new(
    "queries",
    contract_id=int,
    rth=bool,
    method=str,
    asof=datetime.date,
    start=datetime.date,
    end=datetime.date,
    expected=int,
    bound=int,
    done=bool,
).with_id()

table_bars = sql.Table.new(
    "bars",
    contract_id=int,
    date=datetime.date,
    query_id=int,
    open=float,
    high=float,
    low=float,
    close=float,
    volume=float,
    wap=float,
    primary_key=(
        "contract_id",
        "date",
    ),
    # bar_count = float, ?
)

table_historic = table_bars._replace(
    name="history",
    primary_key=(
        "contract_id",
        "query_id",
        "date",
    ),
)

#  ------------------


def drop_prefix(d):
    return {k.split(".")[-1]: v for k, v in d.items()}


class DB_KW(NamedTuple):
    fp: str


class DB(DB_KW):

    contracts: ClassVar[sql.Table] = table_contracts
    contract_dates: ClassVar[sql.Table] = (
        table_contract_dates
    )
    queries: ClassVar[sql.Table] = table_queries
    bars: ClassVar[sql.Table] = table_bars
    history: ClassVar[sql.Table] = table_historic

    def list_contracts(self, **kwargs) -> list[Contract]:
        cls = type(self)
        res = sql.select(
            self.fp,
            {cls.contracts: dict(cls.contracts.schema)},
            where={
                cls.contracts: {
                    k: (
                        f"={v}"
                        if not isinstance(v, str)
                        else f'="{v}"'
                    )
                    for k, v in kwargs.items()
                }
            },
        )
        return [
            Contract(**drop_prefix(d))
            for d in res
        ]

    def get_contract(self, id: int) -> Contract | None:
        cls = type(self)
        res = sql.select(
            self.fp,
            {cls.contracts: dict(cls.contracts.schema)},
            where={cls.contracts: dict(id=f"={id}")},
        )
        if len(res) == 0:
            res = None
        elif len(res) == 1:
            res = res[0]
        else:
            raise ValueError("Duplicates:", res)
        if res is None:
            return res
        return Contract(**drop_prefix(res))  # type: ignore

    def insert_contract(self, con: dict | Contract):
        cls = type(self)
        if isinstance(con, Contract):
            con = con._asdict()
        id: int | None = con.get("id")
        if id is None:
            raise ValueError(con)
        d = self.get_contract(id)
        if d is not None:
            con = {**d._asdict(), **con}
        # print(con)
        sql.insert(
            self.fp,
            cls.contracts,
            [con],
            if_exists="REPLACE",
        )
        return Contract(**con)

    def get_start(self, id: int) -> datetime.date | None:
        cls = type(self)
        res = sql.select(
            self.fp,
            {
                cls.contract_dates: dict(
                    start=datetime.date,
                )
            },
            where={
                cls.contract_dates: dict(
                    contract_id=f"={id}"
                )
            },
        )
        if len(res) == 0:
            res = None
        elif len(res) == 1:
            res = res[0]
        else:
            raise ValueError("Duplicates:", res)
        if res is None:
            return res
        return res["contract_dates.start"]

    def insert_start(
        self,
        id: int,
        start: datetime.date,
        end: datetime.date | None = None,
    ):
        cls = type(self)
        # TODO: this will overwrite end if not given but previously stored!
        sql.insert(
            self.fp,
            cls.contract_dates,
            [
                dict(
                    contract_id=id,
                    start=start,
                    end=end,
                )
            ],
            if_exists="REPLACE",
        )

    @overload
    def get_queries(
        self,
        fields: dict,
        where=None,
        order=None,
        parser=None,
        one: Literal[False] = False,
    ) -> list[dict]: ...

    @overload
    def get_queries(
        self,
        fields: Literal[None],
        where=None,
        order=None,
        parser=None,
        one: Literal[False] = False,
    ) -> list[Query]: ...

    @overload
    def get_queries(
        self,
        fields: dict,
        where=None,
        order=None,
        parser=None,
        one: Literal[True] = True,
    ) -> dict | None: ...

    @overload
    def get_queries(
        self,
        fields: Literal[None],
        where=None,
        order=None,
        parser=None,
        one: Literal[True] = True,
    ) -> Query | None: ...

    def get_queries(
        self,
        fields: dict | None,
        where=None,
        order=None,
        parser=None,
        one: bool = False,
    ):
        cls = type(self)
        t = cls.queries
        res = sql.select(
            self.fp,
            {
                t: (
                    dict(cls.queries.schema)
                    if fields is None
                    else fields
                )
            },
            where=(where if where is None else {t: where}),
            order=(order if order is None else {t: order}),
            parser=(
                parser if parser is None else {t: parser}
            ),
        )
        if one and not len(res):
            return None
        elif fields is None and one:
            return Query(**drop_prefix(res[0]))
        elif one:
            return drop_prefix(res[0])
        elif fields is None:
            return [Query(**drop_prefix(r)) for r in res]
        return [drop_prefix(r) for r in res]

    @overload
    def get_bars(
        self,
        fields: dict,
        where=None,
        order=None,
        parser=None,
        one: Literal[False] = False,
        historic: bool = False,
    ) -> list[dict]: ...

    @overload
    def get_bars(
        self,
        fields: Literal[None] = None,
        where=None,
        order=None,
        parser=None,
        one: Literal[False] = False,
        historic: bool = False,
    ) -> list[Bar]: ...

    @overload
    def get_bars(
        self,
        fields: dict,
        where=None,
        order=None,
        parser=None,
        one: Literal[True] = True,
        historic: bool = False,
    ) -> dict | None: ...

    @overload
    def get_bars(
        self,
        fields: Literal[None] = None,
        where=None,
        order=None,
        parser=None,
        one: Literal[True] = True,
        historic: bool = False,
    ) -> Bar | None: ...

    def get_bars(
        self,
        fields=None,
        where=None,
        order=None,
        parser=None,
        one: bool = False,
        historic: bool = False,
    ):
        cls = type(self)
        t = cls.bars if not historic else self.history
        res = sql.select(
            self.fp,
            {
                t: (
                    dict(t.schema)
                    if fields is None
                    else fields
                )
            },
            where=(where if where is None else {t: where}),
            order=(order if order is None else {t: order}),
            parser=(
                parser if parser is None else {t: parser}
            ),
        )
        if one and not len(res):
            return None
        elif fields is None and one:
            return Bar(**drop_prefix(res[0]))
        elif one:
            return drop_prefix(res[0])
        elif fields is None:
            return [Bar(**drop_prefix(r)) for r in res]
        return [drop_prefix(r) for r in res]

    def insert_query(self, query: dict | Query):
        if isinstance(query, Query):
            query = query._asdict()
        cls = type(self)
        # print(query)
        # if query["id"] is None:
        #     query.pop("id")
        sql.insert(
            self.fp,
            cls.queries,
            [query],
            if_exists="REPLACE",
        )
        q = Query(**query)
        if query["id"] is None:
            q_id = self.get_queries(
                fields=dict(id=int),
                where=dict(
                    contract_id=f"={q.contract_id}",
                    rth=f"={int(q.rth)}",
                    method=f"='{q.method}'",
                    asof=f"={q.asof.toordinal()}",
                    start=f"={q.start.toordinal()}",
                    end=f"={q.end.toordinal()}",
                ),
                one=True,
            )
            assert q_id is not None, query
            q = q._replace(id=q_id["id"])
        # print(q)
        return q

    def insert_bar(
        self,
        bar: dict | Bar,
        historic: bool = True,
    ):
        if isinstance(bar, Bar):
            bar = bar._asdict()
        assert "date" in bar, bar
        assert "contract_id" in bar, bar
        assert "query_id" in bar, bar
        cls = type(self)
        # print(bar)
        if historic:
            sql.insert(
                self.fp,
                self.history,
                [bar],
                if_exists="REPLACE",
            )
        sql.insert(
            self.fp,
            cls.bars,
            [bar],
            if_exists="REPLACE",
        )
        return Bar(**bar)


# python ../../ceg/src/ceg/apis
