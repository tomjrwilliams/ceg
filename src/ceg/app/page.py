from typing import cast, Iterable, Callable, get_type_hints, get_args, get_origin, Type, NamedTuple, Union, Optional
import types
import datetime as dt
import math
import numpy as np
import polars as pl
import streamlit as st

from frozendict import frozendict

from ceg.app.nav import Shared, Page

import ceg

def repr_type(t: Type):
    s = str(t)
    if s.endswith(".args"):
        return "*args"
    elif s.endswith(".kwargs"):
        return "**kwargs"
    try:
        s = s.split("<class '")[1].split("'>")[0]
    except:
        return s
    s = s.replace("datetime.", "")
    return s.split(".")[-1].replace("Ref_", "").strip()

class Annot(NamedTuple):
    t: Type | tuple[Type, ...]
    optional: bool
    union: bool

    def repr(self):
        if self.union:
            s = "|".join([repr_type(t) for t in self.t])
        else:
            s = repr_type(self.t)
        if self.optional:
            return f"{s}|N"
        return s

class Signature(NamedTuple):
    annots: frozendict[str, Annot]
    depth: int
    f: Callable

    # TODO: allow explicitly pass
    # and in construction allow masking certain kwargs
    # eg. field for daily_close, bar etc.

Universe = frozendict[str, ceg.Node.Any | Callable]
Signatures = frozendict[str, Signature]

def signatures(
    universe: Universe
) -> Signatures:
    res = cast(Signatures, frozendict())
    for key, f in universe.items():
        depth = 0
        annots = {}
        if isinstance(f, tuple):
            f, r = f
            hints = get_type_hints(r)
        else:
            hints = get_type_hints(f)
        returns = hints.pop("return", None)
        for k, h in hints.items():
            origin = get_origin(h)
            if (
                origin is types.UnionType
                or origin is Union
            ):
                args = get_args(h)
                if type(None) in args:
                    args = tuple((
                        a for a in args if a is not type(None)
                    ))
                    if len(args) == 1:
                        annots[k] = Annot(
                            args[0], optional=True, union=False
                        )
                    else:
                        annots[k] = Annot(
                            args,
                            optional=True,
                            union=True,
                        )
                else:
                    annots[k] = Annot(args, optional=False, union=True)
            else:
                annots[k] = Annot(h, False, False)
        res = res.set(key, Signature(
            cast(frozendict[str, Annot], frozendict(annots)),
            depth,
            f
        ))
    return cast(Signatures, res)

def signatures_df(sigs: Signatures) -> pl.DataFrame:
    return pl.DataFrame([
        {
            **{"func": k},
            **{
                f"kw-{i}": f"{k}:{annot.repr()}"
                for i, (k, annot)
                in enumerate(sig.annots.items())
            }
        }
        for k, sig in sigs.items()
    ])

def empty_cell(cell):
    if cell is None:
        return True
    if isinstance(cell, str):
        return len(cell) == 0
    return False

def empty_row(
    row: dict, 
    *keys: str, 
    method: Callable[[Iterable[bool]], bool] = any
):
    return method([empty_cell(row[k]) for k in keys])

def label_padding(df: pl.DataFrame):
    return (
        0 if not len(df) else 2 + int(math.log(len(df), 10))
    )

TYPES = {
    "bool": bool,
    "int": int,
    "float": float,
}

def parse_date(v: str):
    if v.isnumeric() and len(v) == 4:
        y = int(v)
        return dt.date(y, 1, 1)
    elif v.isnumeric() and len(v) == 8:
        y = int(v[4:])
        m = int(v[4:6])
        d = int(v[6:])
        return dt.date(y, m, d)
    vs = v.split("-")
    if len(vs) == 2:
        y, m = v
        return dt.date(int(y), int(m), 1)
    elif len(vs) == 3:
        y, m, d = v
        return dt.date(int(y), int(m), int(d))
    else:
        raise ValueError(v)

def parse_kwarg(
    kw: str, 
    refs: dict[str, ceg.Ref.Any],
    annots: frozendict[str, Annot],
):
    # TODO: pass in the type so we can infer from sig
    k, v = kw.split("=")
    if ":" in k:
        k, t = k.split(":")
        if t == "ref":
            v = refs[v]
        elif t == "date":
            v = parse_date(v)
        else:
            v = TYPES[t](v)
    elif k in annots:
        t = annots[k]
        if t.union:
            for tt in t.t:
                try:
                    if tt is dt.date:
                        v = parse_date(v)
                    else:
                        v = tt(v)
                    return k, v
                except:
                    pass
        else:
            assert isinstance(t.t, type), t
            try:
                if t.t is dt.date:
                    v = parse_date(v)
                else:
                    v = t.t(v)
            except:
                raise ValueError(v)
    elif v.isnumeric():
        v = float(v)
    return k, v

def kwargs_ready(
    vs: list[str | None]
) -> tuple[bool, list[str]]:
    try:
        end = vs.index(".")
        return True, cast(list[str], vs[:end])
    except:
        if all([v is not None for v in vs]):
            return True, cast(list[str], vs)
        return False, []

def parse_kwargs(
    row: dict[str, str | None],
    refs: dict[str, ceg.Ref.Any],
    sig: Signature
):
    # TODO: pass in the type so we can infer from sig
    vs = [v for k, v in row.items() if k.startswith("kw-")]
    ready, vs = kwargs_ready(vs)
    if not ready:
        return {}
    return dict((parse_kwarg(v, refs, sig.annots) for v in vs))

def rows_to_refs(
    g: ceg.Graph,
    df: pl.DataFrame,
    refs: dict[str, ceg.Ref.Any],
    universe: Universe,
    sigs: Signatures,
    keep: dict[str, int] = {}
):
    es = []
    for i, row in enumerate(df.iter_rows(named=True)):
        if empty_row(row, "label", "func"):
            continue

        init = row.pop("i")

        func_name = row["func"]
        if func_name not in universe:
            raise ValueError(f"Not found: {func_name}")

        label = row["label"]

        sig = sigs[func_name]
        func = sig.f

        kwargs = parse_kwargs(row, refs, sig)

        if not len(kwargs):
            continue

        if "keep" not in kwargs and label in keep:
            kwargs["keep"] = keep[label]
        
        try:
            if sig.depth:
                res = func(g)(**kwargs) # type: ignore
            else:
                res = func(g, **kwargs) # type: ignore
                # TODO: check that the first arg is always g?
        except Exception as e:
            raise ValueError((str(e), kwargs))

        g, r = cast(tuple[ceg.Graph, ceg.Ref.Any], res)

        # i += offset
        # label = f"{str(i).rjust(pad, '0')}-{row['label']}"
        refs[label] = r

        if init:
            es.append(ceg.Event.zero(r))

    return g, refs, es

class DynamicKw(NamedTuple):
    name: str
    shared: Shared
    universe: Universe

class Dynamic(DynamicKw, Page):

    def run(self):

        sigs = signatures(self.universe)
        sigs_df = signatures_df(sigs)

        st.text("universe:")
        st.dataframe(sigs_df)

        schema_model = pl.DataFrame(schema={
            "label": pl.String,
            "i": pl.Boolean,
            **sigs_df.schema,
        })

        st.text("model:")
        df_model = cast(
            pl.DataFrame, 
            st.data_editor(schema_model, num_rows="dynamic")
        )

        st.text("init:")
        schema_plot = pl.DataFrame(schema={
            "label": pl.String,
            "x": pl.Boolean,
            "y": pl.Boolean,
            "y2": pl.Boolean,
        })
        df_plot = cast(
            pl.DataFrame,
            st.data_editor(schema_plot, num_rows="dynamic")
        )

        steps = 100 # TODO param on page

        g = ceg.Graph.new()

        g, refs, es = g.pipe(
            rows_to_refs, 
            df_model,
            refs={},
            universe=self.universe,
            sigs=sigs,
            keep={
                label: steps
                for label in df_plot.get_column("label")
            }
        )

        if not len(refs) or not len(es):
            return

        g, es, ts = ceg.batches(g, *es, n = steps, g = len(refs))

        data_plot = pl.DataFrame({
            label: refs[label].history(g).last_n_before(steps, ts[-1])
            for label in df_plot.get_column("label")
        })

        x_label = df_plot.filter(pl.col("x")).get_column("label")
        if len(x_label) == 1:
            st.line_chart(data=data_plot, x = x_label[0])
        elif not len(x_label):
            st.line_chart(data_plot)
        else:
            raise ValueError(df_plot)
