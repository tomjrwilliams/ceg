import datetime
import sqlite3
import contextlib
from typing import (
    Callable,
    Type,
    NamedTuple,
    overload,
    Literal,
)

from frozendict import frozendict

from pprint import pp

#  ------------------


def identity(v):
    return v


TYPE_NAME = {
    bool: "INTEGER",
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    datetime.date: "INTEGER",
}
TYPE_FORMAT = {
    bool: lambda v: str(int(v)),
    int: str,
    float: str,
    str: str,
    datetime.date: lambda v: str(v.toordinal()),
}
TYPE_PARSE = {
    bool: bool,
    int: int,
    float: float,
    str: str,
    datetime.date: lambda v: datetime.date.fromordinal(v),
}

#  ------------------


class Table(NamedTuple):
    """
    name: str
    schema: dict[str, Type]
    primary_key: tuple[str, ...] | None
    unique: tuple[str | tuple[str, ...], ...] | None
    not_null: tuple[str, ...] | None
    flags: dict | None
    >>> t = Table.new("test").with_id()
    >>> _ = list(
    ...     map(
    ...         pp,
    ...         t.create(
    ...             "__local__/test.db"
    ...         ).values(),
    ...     )
    ... )
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
    primary_key: tuple[str, ...] | None
    unique: tuple[str | tuple[str, ...], ...] | None
    # TODO: Defaults
    not_null: tuple[str, ...] | None
    flags: frozendict | None
    id: str | None

    def __str__(self):
        return self.name

    def with_id(self, key: str = "id", auto_increment=True):
        flag = frozendict(
            {key: "PRIMARY KEY AUTOINCREMENT"}
        )
        assert (
            self.primary_key is None
            or self.primary_key == ()
        ), self
        return self._replace(
            schema=frozendict({key: int}) | self.schema,
            primary_key=None,
            flags=(
                flag | self.flags
                if self.flags is not None and auto_increment
                else flag if auto_increment else self.flags
            ),
            id=key,
        )

    @classmethod
    def new(
        cls,
        name: str,
        primary_key: tuple[str, ...] | None = None,
        unique: (
            tuple[str | tuple[str, ...], ...] | None
        ) = None,
        not_null: tuple[str, ...] | None = None,
        flags: dict | None = None,
        # id: str | None = None,
        **schema: Type,
        # TODO: Defaults
    ):
        return cls(
            name=name,
            schema=frozendict(schema),  # type: ignore
            primary_key=primary_key,
            unique=unique,
            not_null=not_null,
            flags=(
                flags
                if flags is None
                else frozendict(flags)
            ),
            id=None,
        )

    def create(self, db: str):
        (
            table,
            schema,
            primary_key,
            unique,
            not_null,
            flags,
            _,
        ) = self
        return create_table(
            db,
            table=table,
            schema=dict(schema),
            primary_key=primary_key,
            unique=unique,
            not_null=not_null,
            flags=(flags if flags is None else dict(flags)),
        )

    def insert(
        self,
        db: str,
        values: list[dict],
        # format: dict | None = None,
        if_exists: str | None = None,
    ):
        return insert(
            db,
            self,
            values,
            # format=format,
            if_exists=if_exists,
        )


#  ------------------


class Query(NamedTuple):
    acc: dict[str, dict[str, Type]]

    def select(self, table: Table, *keys: str):
        return self._replace(
            acc={
                **self.acc,
                **{
                    table.name: {
                        k: table.schema[k] for k in keys
                    }
                },
            }
        )


query = Query({})

# TODO: join, where, order

# eg. where returns field objects that have gt, eq etc. for generating the appropriate cond (for the dict representation below)

# join similarly, order i guess methods for asc and desc

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
        s = s + ", ".join(
            [f"UNIQUE(" + ",".join(u) + ")" for u in unique]
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
        return {
            k: c for k, c in cols.items() if k in schema
        }
    s = ""
    for k, t in schema.items():
        if k not in cols:
            s = s + f"ALTER TABLE {table} "
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
    assert len(s), dict(schema=schema, cols=cols)
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


def insert(
    db: str,
    table: TTable,
    values: list[dict],
    if_exists: str | None = None,
    format: dict[str, Callable] | None = None,
    id: str | None = None,
):
    """
    >>> db = "__local__/test.db"
    >>> t = Table.new("test", val=float).with_id()
    >>> _ = t.create(db)
    >>> t.insert(db, [dict(val=1.0)])
    """
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
    with connect(db) as cur:
        try:
            cur.executemany(
                s,
                [format_values(d, format) for d in values],
            )
        except:
            raise ValueError(s)
    return


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
            format.get(
                k, TYPE_FORMAT.get(type(v), identity)
            )(v)
            for k, v in d.items()
        )
    )


# if id is not None and id not in values[0].keys():
#     phs = "?," + phs
#     ks = f"{id},{ks}"
#     values = [
#         {id: None} | d for d in values
#     ]

#  ------------------


# for select / delete
def add_fields_str(
    s: str,
    fields: dict[TTable, dict[str, Type]],
):
    schema = {}
    for table, fs in fields.items():
        s = s + ", ".join(
            [
                ".".join([str(table), f])
                for f, t in fs.items()
            ]
        )
        schema = {
            **schema,
            **{f"{table}.{f}": t for f, t in fs.items()},
        }
    return s, schema


def _join_str(on: dict[TTable, str]):
    assert len(on) == 2, on
    return f" = ".join(
        [".".join([str(t), f]) for t, f in on.items()]
    )


def add_joins_str(
    s: str,
    joins: dict[TTable, dict[TTable, str]] | None = None,
):
    if joins is not None:
        s = s + "\n ".join(
            [
                f" JOIN {j} ON {_join_str(on)}"
                for j, on in joins.items()
            ]
        )
    return s


def add_where_str(
    s: str,
    where: dict[TTable, dict[str, str]] | None = None,
):
    if where is not None:
        s = (
            s
            + " WHERE "
            + "\n AND ".join(
                [
                    " AND ".join(
                        [
                            f"{table}.{f} {cond}"
                            for f, cond in conds.items()
                        ]
                    )
                    for table, conds in where.items()
                ]
            )
        )
    return s


def add_order_str(
    s: str,
    order: dict[TTable, dict[str, str]] | None = None,
):
    if order is not None:
        s = (
            s
            + " ORDER BY "
            + "\n".join(
                [
                    ", ".join(
                        [
                            f"{table}.{f} {direc}"
                            for f, direc in direcs.items()
                        ]
                    )
                    for table, direcs in order.items()
                ]
            )
        )
    return s


def select(
    db: str,
    # query
    fields: dict[TTable, dict[str, Type]],
    joins: dict[TTable, dict[TTable, str]] | None = None,
    where: dict[TTable, dict[str, str]] | None = None,
    order: dict[TTable, dict[str, str]] | None = None,
    # results
    parser: dict[TTable, dict[str, Callable]] | None = None,
):
    """
    >>> db = "__local__/test.db"
    >>> t = Table.new("test", val=float).with_id()
    >>> _ = t.create(db)
    >>> query = query.select(t, "id").acc
    >>> where = {t: dict(val="=1.0")}
    >>> rs = select(db, query, where=where)
    >>> [
    ...     {k: type(v) for k, v in r.items()}
    ...     for r in rs
    ... ]
    [{'test.id': <class 'int'>}]
    """
    s = "SELECT "

    s, schema = add_fields_str(s, fields)

    s = s + f" FROM {list(fields.keys())[0]}"

    s = add_joins_str(s, joins)
    s = add_where_str(s, where)
    s = add_order_str(s, order)

    with connect(db, read_only=True) as cur:
        q = cur.execute(s)
        res = q.fetchall()

    if parser is None:
        parser = {}

    # TODO: map
    res = [
        {
            k: parse_res(parser, schema, k, v)
            for k, v in zip(schema.keys(), r)
        }
        for r in res
    ]

    return res


def parse_res(parser, schema, k, v):
    # if k.split(".")[-1] in {"start", "end", "asof", "date"}:
    #     print(k, v)
    if v is None:
        return v
    return parser.get(
        k, TYPE_PARSE.get(schema[k], schema[k])
    )(
        v
    )  # type: ignore


#  ------------------


def delete(
    db: str,
    table: TTable,
    joins: dict[TTable, dict[TTable, str]] | None = None,
    where: dict[TTable, dict[str, str]] | None = None,
):
    """
    >>> db = "__local__/test.db"
    >>> t = Table.new("test", val=float).with_id()
    >>> _ = t.create(db)
    >>> query = query.select(t, "id").acc
    >>> where = {t: dict(val="=1.0")}
    >>> delete(db, t, where=where)
    >>> select(db, query, where=where)
    []
    """

    s = f"DELETE FROM {table}"

    s = add_joins_str(s, joins)
    s = add_where_str(s, where)

    with connect(db) as cur:
        cur.execute(s)

    return


#  ------------------
