from __future__ import annotations
import datetime
import sqlite3
import contextlib
from typing import (
    Callable,
    Type,
    NamedTuple,
    overload,
    Literal,
    Any,
)
from dataclasses import dataclass
from frozendict import frozendict

from pprint import pp

#  ------------------

TEST_DB = "./__local__/test.db"


def identity(v):
    return v


def types(d):
    if isinstance(d, dict):
        return {k: types(v) for k, v in d.items()}
    elif isinstance(d, tuple):
        return tuple([types(v) for v in d])
    elif isinstance(d, list):
        return [types(v) for v in d]
    return type(d)


# INT
# INTEGER
# TINYINT
# SMALLINT
# MEDIUMINT
# BIGINT
# UNSIGNED BIG INT
# INT2
# INT8
#  -> INTEGER
# CHARACTER(20)
# VARCHAR(255)
# VARYING CHARACTER(255)
# NCHAR(55)
# NATIVE CHARACTER(70)
# NVARCHAR(100)
# TEXT
# CLOB
#  -> TEXT
# BLOB
#  -> BLOB
# REAL
# DOUBLE
# DOUBLE PRECISION
# FLOAT
# -> REAL
# NUMERIC
# DECIMAL(10,5)
# BOOLEAN
# DATE
# DATETIME
#  -> NUMERIC

TYPE_NAME = {
    bool: "INTEGER",
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    datetime.datetime: "REAL", # DATETIME? NUMERIC?
    datetime.date: "INTEGER",
    bytes: "BLOB",
}
TYPE_FORMAT = {
    bool: lambda v: str(int(v)),
    int: str,
    float: str,
    str: str,
    datetime.datetime: lambda v: str(v.timestamp()),
    datetime.date: lambda v: str(v.toordinal()),
    bytes: lambda v: v,
}
TYPE_PARSE = {
    bool: bool,
    int: int,
    float: float,
    str: str,
    datetime.datetime: lambda v: datetime.datetime.fromtimestamp(v),
    datetime.date: lambda v: datetime.date.fromordinal(v),
    bytes: lambda v: v,
    None: lambda b: b,
}


#  ------------------


@dataclass(frozen=True, eq=False)
class Column:
    table: TTable
    key: str

    t: Type | Callable | None = None

    def __str__(self):
        table = str(self.table)
        return f"{table}.{self.key}"

    def __eq__(self, val) -> Comparator:
        return eq(self, val)

    def __neq__(self, val) -> Comparator:
        return ne(self, val)

    def __gt__(self, val) -> Comparator:
        return gt(self, val)

    def __lt__(self, val) -> Comparator:
        return lt(self, val)

    def __ge__(self, val) -> Comparator:
        return ge(self, val)

    def __le__(self, val) -> Comparator:
        return le(self, val)


@dataclass(frozen=True)
class Comparator:
    col: Column
    val: str | Column | Any | None = None

    def comp(self):
        # used for custom string comparators
        assert isinstance(self.val, str), self
        return self.val

    def fval(self):
        if isinstance(self.val, bytes):
            return memoryview(self.val)
        if isinstance(self.val, str):
            return f"'{self.val}'"
        return self.val

    def sql(self):
        assert self.val is not None, self
        return f"{self.col}{self.comp()}"


@dataclass(frozen=True)
class eq(Comparator):

    def comp(self):
        return f"={self.fval()}"


@dataclass(frozen=True)
class ne(Comparator):

    def comp(self):
        return f"!={self.fval()}"


@dataclass(frozen=True)
class gt(Comparator):

    def comp(self):
        return f">{self.fval()}"


@dataclass(frozen=True)
class lt(Comparator):

    def comp(self):
        return f"<{self.fval()}"


@dataclass(frozen=True)
class ge(Comparator):

    def comp(self):
        return f">={self.fval()}"


@dataclass(frozen=True)
class le(Comparator):

    def comp(self):
        return f"<={self.fval()}"


class where:
    str = Comparator
    eq = eq
    ne = ne
    gt = gt
    lt = lt
    ge = ge
    le = le


@dataclass(frozen=True)
class Order:
    col: Column
    direction: str

    def sql(self):
        return f"{self.col} {self.direction}"


@dataclass(frozen=True)
class asc(Order):
    direction: str = "ASC"


@dataclass(frozen=True)
class desc(Order):
    direction: str = "DESC"


class order:
    asc = asc
    desc = desc


#  ------------------


class Table(NamedTuple):
    """
    name: str
    schema: dict[str, Type]
    primary_key: tuple[str, ...] | None
    unique: tuple[str | tuple[str, ...], ...] | None
    not_null: tuple[str, ...] | None
    flags: dict | None
    >>> t = Table.new("tab").create(TEST_DB)
    >>> _ = list(map(pp, t.pragma.values()))
    {'name': 'id',
     'type': 'INTEGER',
     'flags': '',
     'not_null': False,
     'default': None,
     'primary_key': True,
     'primary_key_index': 0}
    """

    name: str
    schema: frozendict[str, Type]
    primary_key: tuple[str, ...]
    unique: tuple[str | tuple[str, ...], ...] | None
    # TODO: Default values
    not_null: tuple[str, ...] | None
    flags: frozendict | None
    id: str | None

    db: str | None = None
    pragma: dict | None = None
    aliased: str | None = None

    def __str__(self):
        if self.aliased is not None:
            return self.aliased
        return self.name

    @classmethod
    def new(
        cls,
        name: str,
        primary_key: str | tuple[str, ...] = "id",
        unique: tuple[str | tuple[str, ...], ...] | None = None,
        not_null: tuple[str, ...] | None = None,
        flags: dict | None = None,
        # id: str | None = None,
        **schema: Type,
        # TODO: Defaults
    ):
        id = None
        if isinstance(primary_key, str):
            if primary_key in schema:
                pass
            else:
                # assume auto increment
                flag = {primary_key: "PRIMARY KEY AUTOINCREMENT"}
                schema = {primary_key: int} | schema
                flags = (
                    flag | flags if flags is not None else flag
                )
            id = primary_key
            primary_key = (primary_key,)
        return cls(
            name=name,
            schema=frozendict(schema),  # type: ignore
            primary_key=primary_key,
            unique=unique,
            not_null=not_null,
            flags=(
                flags if flags is None else frozendict(flags)
            ),
            id=id,
            db=None,
            pragma=None,
            aliased=None,
        )

    def alias(self, name: str):
        return self._replace(aliased=name)

    def col(self, k: str, t: Type | Callable | None = None):
        assert k in self.schema, (self, k)
        return Column(self, k, t=t)

    def create(self, db: str):
        (
            table,
            schema,
            primary_key,
            unique,
            not_null,
            flags,
            id,
            _,
            _,
            _,
        ) = self
        if flags is None:
            pass
        elif flags.get(id) == "PRIMARY KEY AUTOINCREMENT":
            primary_key = None
        pragma = create_table(
            db,
            table=table,
            schema=dict(schema),
            primary_key=primary_key,
            unique=unique,
            not_null=not_null,
            flags=(flags if flags is None else dict(flags)),
        )
        return self._replace(db=db, pragma=pragma)

    def insert(
        self,
        values: list[dict] = [],
        if_exists: str | None = None,
        format: dict[str, Callable] | None = None,
        **value,
    ):
        """
        >>> t = Table.new("tab", val=float).create(TEST_DB)
        >>> ids = t.insert(val=1.0)
        >>> types(ids)
        [<class 'int'>]
        """
        if len(values) == 0:
            assert len(value), value
            values = [value]
        assert self.db is not None, self
        return insert(
            self.db,
            self,
            values,
            if_exists=if_exists,
            format=format,
        )

    def select(
        self,
        fields: Column | list[Column],
        joins: Comparator | list[Comparator] | None = None,
        where: Comparator | list[Comparator] | None = None,
        order: Order | list[Order] | None = None,
        limit: int | None = None,
    ):
        """
        >>> t = Table.new("tab", val=float).create(TEST_DB)
        >>> rs = t.select(
        ...     t.col("id"),
        ...     where=t.col("val")==1.
        ... )
        >>> types(rs)
        [{'tab.id': <class 'int'>}]
        """
        assert self.db is not None, self
        return select(
            self.db,
            fields,
            joins=joins,
            where=where,
            order=order,
            limit=limit,
        )

    def update(
        self,
        joins: Comparator | list[Comparator] | None = None,
        where: Comparator | list[Comparator] | None = None,
        **values: Any,
    ):
        """
        >>> t = Table.new("tab", val=float).create(TEST_DB)
        >>> val = t.col("val")
        >>> ids_0 = t.insert(val=4.)
        >>> ids_1 = t.update(val=5.0, where=val==4.)
        >>> assert ids_0 == ids_1
        >>> rs = t.select(
        ...     t.col("id"),
        ...     where=val==5.
        ... )
        >>> types(rs)
        [{'tab.id': <class 'int'>}]
        """
        assert self.db is not None, self
        return update(
            self.db,
            self,
            values=values,
            joins=joins,
            where=where,
        )

    def delete(
        self,
        joins: Comparator | list[Comparator] | None = None,
        where: Comparator | list[Comparator] | None = None,
        id: str | None = None,
    ):
        """
        >>> t = Table.new("tab", val=float).create(TEST_DB)
        >>> val = t.col("val")
        >>> ids_0 = t.insert(val=3.)
        >>> ids_1 = t.select(t.col("id"), where=val == 3.)
        >>> assert ids_1[0]["tab.id"] == ids_0[0]
        >>> ids_2 = t.delete(where=val== 3.)
        >>> assert ids_0 == ids_2
        >>> t.select(t.col("id"), where=val == 3.)
        []
        """
        assert self.db is not None, self
        return delete(
            self.db,
            self,
            joins=joins,
            where=where,
            id=id,
        )


#  ------------------

TTable = str | Table


def table_str(t: TTable):
    if isinstance(t, Table):
        return t.name
    assert isinstance(t, str), t
    return t


#  ------------------

import pathlib


@contextlib.contextmanager
def connect(fp, read_only: bool = False):
    fp = f"file:{fp}"
    if read_only and pathlib.Path(fp).exists():
        fp = fp + "?mode=ro"
    try:
        con = sqlite3.connect(fp, uri=True)
    except:
        raise ValueError(fp)
    try:
        with con as cur:
            yield cur
    finally:
        con.close()


#  ------------------


def field_str(f, t, flags, not_null):
    return (
        (f"{f} {TYPE_NAME.get(t, t)} ")
        + ("" if f not in flags else flags.get(f, "") + " ")
        + (
            ""
            if not_null is None or f not in not_null
            else "NOT NULL "
        )
    )


def create_table(
    db: str,
    table: str,
    schema: dict[str, Type],
    primary_key: tuple[str, ...] | None = None,
    unique: tuple[str | tuple[str, ...], ...] | None = None,
    # TODO: Defaults
    not_null: tuple[str, ...] | None = None,
    flags: dict | None = None,
):
    if not_null is None:
        not_null = ()
    if flags is None:
        flags = {}
    s = ", ".join(
        [
            field_str(f, t, flags, not_null)
            for f, t in schema.items()
        ]
    )
    if primary_key is not None:
        pk = ", ".join(primary_key)
        s = s + f", PRIMARY KEY({pk})"
    if unique is not None:
        s = s + ", " + ", ".join(
            ["UNIQUE(" + (
                u if isinstance(u, str) else ",".join(u)
            ) + ")" for u in unique]
        )
    with connect(db) as cur:
        try:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ({s})"
            )
            cols = cur.execute(
                f"; PRAGMA table_info({table})"
            ).fetchall()
        except:
            raise ValueError(s)
    cols = parse_table_info(cols)
    if len(cols) >= len(schema):
        return {k: c for k, c in cols.items() if k in schema}
    for k, t in schema.items():
        if k not in cols:
            s = f"ALTER TABLE {table} "
            s = (
                s
                + f"ADD COLUMN {field_str(k, t, flags, not_null)};"
            )
            if k in not_null:
                uniq_key = f"uniq_{table}_{k}"
                s = (
                    s
                    + f"CREATE UNIQUE INDEX {uniq_key} on {table}({k});"
                )
            with connect(db) as cur:
                try:
                    cur.execute(s)
                    cols = cur.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                except:
                    raise ValueError(s)
    return parse_table_info(cols)


def parse_table_info(cols):
    return {
        name: dict(
            name=name,
            type=t.split(" ")[0],
            flags=" ".join(t.split(" ")[1:]),
            not_null=bool(notnull),
            default=deflt,
            primary_key=pkey_index > 0,
            primary_key_index=(
                None if pkey_index == 0 else pkey_index - 1
            ),
        )
        for _, name, t, notnull, deflt, pkey_index in cols
    }


#  ------------------


def format_values(
    d: dict,
    format: dict[str, Callable] | None = None,
):
    if format is None:
        return tuple(
            (
                TYPE_FORMAT.get(type(v), identity)(v)
                for k, v in d.items()
            )
        )
    return tuple(
        (
            format.get(k, TYPE_FORMAT.get(type(v), identity))(v)
            for k, v in d.items()
        )
    )


def fields_parser(
    fields: list[Column],
) -> dict[str, Type | Callable]:
    return {
        str(f): (
            f.t
            if f.t is not None
            else (
                TYPE_PARSE[None]
                if isinstance(f.table, str)
                else TYPE_PARSE.get(
                    f.table.schema.get(f.key),
                    f.table.schema.get(f.key),
                )
            )
        )
        for f in fields
    }


# for select / delete
def add_fields_str(
    s: str,
    fields: list[Column],
):
    s += f", ".join([str(f) for f in fields])
    return s


def join_str(join: Comparator):
    assert isinstance(join, where.eq), join
    assert isinstance(join.val, Column), join
    t = join.val.table
    return f" JOIN {t} ON {join.sql()}"


def add_joins_str(
    s: str,
    joins: Comparator | list[Comparator] | None = None,
):
    if isinstance(joins, Comparator):
        s = s + "\n " + join_str(joins)
    elif isinstance(joins, list):
        s = s + "\n ".join(map(join_str, joins))
    return s


def add_where_str(
    s: str,
    where: Comparator | list[Comparator] | None = None,
):
    if isinstance(where, Comparator):
        s = s + f" WHERE {where.sql()}"
    elif isinstance(where, list) and len(where):
        s = (
            s
            + " WHERE "
            + "\n AND ".join([cond.sql() for cond in where])
        )
    return s


def add_order_str(
    s: str,
    order: Order | list[Order] | None = None,
):
    if isinstance(order, Order):
        s += f" ORDER BY {order.sql()}"
    elif isinstance(order, list) and len(order):
        s = (
            s
            + " ORDER BY "
            + "\n".join([o.sql() for o in order])
        )
    return s


#  ------------------


def insert(
    db: str,
    table: TTable,
    values: list[dict],
    if_exists: str | None = None,
    format: dict[str, Callable] | None = None,
    id: str | None = None,
):
    # TODO: assert dict keys match table schema if given?
    # NOTE: replace will set to default / null any columns not given in the passed data

    if if_exists is not None:
        assert if_exists in {
            "IGNORE",
            "REPLACE",
        }, if_exists
        if_exists = f"OR {if_exists} "
    else:
        if_exists = ""
    id = id if isinstance(table, str) else table.id
    ks = ",".join(values[0].keys())
    phs = ",".join(["?" for _ in values[0].keys()])
    s = f"INSERT {if_exists}INTO {table} ({ks}) VALUES({phs})"
    if id is not None:
        s += f" RETURNING {id}"
    with connect(db) as cur:
        try:
            res = cur.executemany(
                s,
                [format_values(d, format) for d in values],
            ).fetchall()
        except:
            raise ValueError(s)
    return [r[0] for r in res]


def select(
    db: str,
    fields: Column | list[Column],
    joins: Comparator | list[Comparator] | None = None,
    where: Comparator | list[Comparator] | None = None,
    order: Order | list[Order] | None = None,
    limit: int | None = None,
):
    s = "SELECT "

    if isinstance(fields, Column):
        fields = [fields]

    s = add_fields_str(s, fields)

    s = s + f" FROM {fields[0].table}"

    s = add_joins_str(s, joins)
    s = add_where_str(s, where)
    s = add_order_str(s, order)

    if limit is not None:
        s += f" LIMIT {limit}"

    with connect(db, read_only=True) as cur:
        try:
            q = cur.execute(s)
        except:
            raise ValueError(s)
        res: list[tuple] = q.fetchall()

    parser = fields_parser(fields)
    parser_keys = list(parser.keys())

    return [
        {k: parser[k](v) for k, v in zip(parser_keys, r)}
        for r in res
    ]


def update(
    db: str,
    table: TTable,
    values: dict[str, Any],
    joins: Comparator | list[Comparator] | None = None,
    where: Comparator | list[Comparator] | None = None,
    format: dict[str, Callable] | None = None,
    id: str | None = None,
) -> list[int]:
    id = id if isinstance(table, str) else table.id
    table = table if isinstance(table, str) else table.name

    s = f"UPDATE {table}"

    if not len(values):
        return []

    s += " SET "

    for k, v in values.items():
        s += f"{k}=?, "
    vs = format_values(values, format)

    s = s[:-2]

    if joins is not None:
        s = s + f" FROM {table}"

    s = add_joins_str(s, joins)
    s = add_where_str(s, where)

    s += f" RETURNING {id}"

    with connect(db, read_only=True) as cur:
        try:
            q = cur.execute(s, vs)
        except:
            raise ValueError(s)
        res = q.fetchall()

    return [r[0] for r in res]


def delete(
    db: str,
    table: TTable,
    joins: Comparator | list[Comparator] | None = None,
    where: Comparator | list[Comparator] | None = None,
    id: str | None = None,
) -> list[int]:

    s = f"DELETE FROM {table}"

    id = id if isinstance(table, str) else table.id

    s = add_joins_str(s, joins)
    s = add_where_str(s, where)

    s += f" RETURNING {id}"

    with connect(db) as cur:
        res = cur.execute(s).fetchall()

    return [r[0] for r in res]


#  ------------------
