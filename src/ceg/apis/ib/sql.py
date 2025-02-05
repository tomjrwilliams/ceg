import sqlite3
import contextlib
from typing import Callable, Type, NamedTuple

#  ------------------

def identity(v):
    return v

TYPE_NAME = {
    int: "INTEGER"
}
TYPE_FORMAT = {
    int: str,
}
TYPE_PARSE = {
    int: int
}

#  ------------------

class Table(NamedTuple):
    name: str
    schema: dict[str, Type]
    primary_key: tuple[str, ...] | None
    unique: tuple[str | tuple[str, ...], ...] | None
    # TODO: Defaults
    not_null: tuple[str, ...] | None
    flags: dict | None

    def __str__(self):
        return self.name

    @classmethod
    def new(
        cls,
        name: str,
        primary_key: tuple[str, ...] | None = None,
        unique: tuple[str | tuple[str, ...], ...] | None = None,
        # TODO: Defaults
        not_null: tuple[str, ...] | None = None,
        flags: dict | None = None,
        **schema: Type,
    ):
        return cls(
            name=name,
            schema=schema,
            primary_key=primary_key,
            unique=unique,
            not_null=not_null,
            flags=flags,
        )
    
    def create(self, db: str):
        (
            table,
            schema,
            primary_key,
            unique,
            not_null,
            flags,
        ) = self
        return create_table(
            db,
            table=table,
            schema=schema,
            primary_key=primary_key,
            unique=unique,
            not_null=not_null,
            flags=flags,
        )

class Query(NamedTuple):
    acc: dict[str, dict[str, Type]]

    def select(
        self,
        table: Table,
        *keys: str
    ):
        return self._replace(acc = {
            **self.acc,
            **{table.name: {
                k: table.schema[k]
                for k in keys
            }}
        })

query = Query({})

#  ------------------

TTable = str | Table

def table_str(t: TTable):
    if isinstance(t, Table):
        return t.name
    assert isinstance(t, str), t
    return t

#  ------------------

@contextlib.contextmanager
def connect(fp, read_only: bool = False):
    fp = f"file:{fp}"
    if read_only:
        fp = fp + "?mode=ro"
    con = sqlite3.connect(fp, uri=True)
    try:
        yield con
    finally:
        con.close()

#  ------------------

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
    if flags is None:
        flags = {}
    fs = ", ".join([
        (
            f"{f} {TYPE_NAME.get(t, t)} "
        ) + (
            "" if f not in flags else flags.get(f, "") + " "
        ) + (
            "" 
            if not_null is None or f not in not_null 
            else "NOT NULL "
        )
        for f, t in schema.items()
    ])
    if primary_key is not None:
        pk = ", ".join(primary_key)
        fs = fs + f", PRIMARY_KEY({pk})"
    if unique is not None:
        fs = fs + ", ".join([
            f"UNIQUE(" + ",".join(u) + ")"
            for u in unique
        ])
    with connect(db) as con:
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {table}({fs})"
        )
    return

#  ------------------

def insert(
    db: str,
    table: TTable,
    values: list[dict],
    format: dict | None = None,
    if_exists: str | None = None,
):
    # TODO: assert dict keys match table schema if given?
    # NOTE: replace will set to default / null any columns not given in the passed data
    if format is None:
        format = TYPE_FORMAT
    with connect(db) as con:
        if if_exists is not None:
            assert if_exists in {"IGNORE", "REPLACE"}, if_exists
            if_exists = f"OR {if_exists}"
        else:
            if_exists = ""
        con.execute(
            f"INSERT {if_exists} INTO {table} " + "\n".join([
                ", ".join([
                    format.get(
                        k, TYPE_FORMAT[type(v)]
                    )(v)
                    for k, v in d.items()
                ]) for d in values
            ])
        )
    return

#  ------------------

# for select / delete
def add_fields_str(
    s: str,
    fields: dict[TTable, dict[str, Type]],
):
    schema = {}
    for table, fs in fields.items():
        s = s + ", ".join([
            ".".join([str(table), f])
            for f, t in fs.items()
        ])
        schema = {
            **schema,
            **{
                f"{table}.{f}": t
                for f, t in fs.items()
            }
        }
    return s, schema

def _join_str(on: dict[TTable, str]):
    assert len(on) == 2, on
    return f" = ".join([
        ".".join([str(t), f])
        for t, f in on.items()
    ])

def add_joins_str(
    s: str,
    joins: dict[TTable, dict[TTable, str]] | None = None,
):
    if joins is not None:
        s = s + "\n ".join([
            f" JOIN {j} ON {_join_str(on)}"
            for j, on in joins.items()
        ])
    return s


def add_where_str(
    s: str,
    where: dict[TTable, dict[str, str]] | None = None,
):
    if where is not None:
        s = s + " WHERE " + "\n AND ".join([
            " AND ".join([
                f"{table}.{f} {cond}"
                for f, cond in conds.items()
            ])
            for table, conds in where.items()
        ])
    return s

def add_order_str(
    s: str,
    order: dict[TTable, dict[str, str]] | None = None,
):
    if order is not None:
        s = s + " ORDER BY " + "\n".join([
            ", ".join([
                f"{table}.{f} {direc}"
                for f, direc in direcs.items()
            ])
            for table, direcs in order.items()
        ])
    return s

def select(
    db: str,
    # query
    fields: dict[TTable, dict[str, Type]],
    joins: dict[TTable, dict[TTable, str]] | None = None,
    where: dict[TTable, dict[str, str]] | None = None,
    order: dict[TTable, dict[str, str]] | None = None,
    # results
    one: bool = False,
    parser: dict[TTable, dict[str, Callable]] | None = None,
):
    s = "SELECT "

    s, schema = add_fields_str(s, fields)

    s = s + f" FROM {list(fields.keys())[0]}"

    s = add_joins_str(s, joins)
    s = add_where_str(s, where)
    s = add_order_str(s, order)
    
    with connect(db, read_only=True) as con:
        q = con.execute(s)
        if one:
            res = q.fetchone()
            res = (res,)
        else:
            res = q.fetchall()

    if parser is None:
        parser = {}

    # TODO: map
    res = [
        {
            k: parser.get(k, TYPE_PARSE.get(
                schema[k], schema[k]
            ))(v) # type: ignore
            for k, v in zip(schema.keys(), r)
        } for r in res
    ]

    if one:
        return res[0]
    return res

#  ------------------

def delete(
    db: str,
    fields: dict[TTable, dict[str, Type]],
    joins: dict[TTable, dict[TTable, str]] | None = None,
    where: dict[TTable, dict[str, str]] | None = None,
):
    
    s = "SELECT "

    s, schema = add_fields_str(s, fields)

    s = s + f" FROM {list(fields.keys())[0]}"

    s = add_joins_str(s, joins)
    s = add_where_str(s, where)

    with connect(db, read_only=True) as con:
        con.execute(s)
    
    return

#  ------------------
