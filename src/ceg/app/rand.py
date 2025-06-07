from typing import cast, Iterable, Callable

import streamlit as st

import math
import numpy as np
import polars as pl

from ceg.app.nav import Page

import ceg

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

def rows_to_refs_rand(
    g: ceg.Graph,
    df: pl.DataFrame,
    step: float,
    keep: int,
):
    # pad = df.pipe(label_padding)
    refs: dict[str, ceg.Ref.Scalar_F64] = {}

    for i, row in enumerate(df.iter_rows(named=True)):
        if empty_row(row, "label", "mean", "sigma"):
            continue
        elif row["walk"]:
            g, r = g.pipe(
                ceg.fs.rand.gaussian.walk,
                mean=row["mean"],
                std=row["sigma"],
                seed=1 or row["seed"],
                step=step,
                keep=keep,
            )
        else:
            g, r = g.bind(ceg.fs.rand.gaussian.new(
                mean=row["mean"],
                std=row["sigma"],
                seed=1 or row["seed"],
            ), when=ceg.Loop.every(step), keep=keep)
        
        # label = f"{str(i).rjust(pad, '0')}-{row['label']}"
        refs[row["label"]] = r
        
    return g, refs

TYPES = {
    "bool": bool,
    "int": int,
    "float": float,
}

def parse_kwarg(kw: str, refs: dict[str, ceg.Ref.Scalar_F64]):
    # TODO: pass in the type so we can infer from sig
    k, v = kw.split("=")
    if ":" in v:
        v, t = v.split(":")
        if t == "ref":
            v = refs[v]
        else:
            v = TYPES[t](v)
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
    refs: dict[str, ceg.Ref.Scalar_F64]
):
    # TODO: pass in the type so we can infer from sig
    vs = [v for k, v in row.items() if k.startswith("kw-")]
    ready, vs = kwargs_ready(vs)
    if not ready:
        return {}
    return dict((parse_kwarg(v, refs) for v in vs))

def rows_to_refs_rolling(
    g: ceg.Graph,
    df: pl.DataFrame,
    refs: dict[str, ceg.Ref.Scalar_F64],
    keep: int,
):
    pad = df.pipe(label_padding)

    for i, row in enumerate(df.iter_rows(named=True)):
        if empty_row(row, "label", "func"):
            continue
        kwargs = parse_kwargs(row, refs)
        if not len(kwargs):
            continue
        node: ceg.Node.Scalar_F64 = getattr(
            ceg.fs.rolling, row["func"]
        )
        g, r = g.bind(
            node.new(**kwargs), keep=keep,
        )

        # i += offset
        # label = f"{str(i).rjust(pad, '0')}-{row['label']}"
        refs[row["label"]] = r

    return g, refs

class Gaussian(Page):
    def run(self):

        # TODO: reversion param
        schema_rand = pl.DataFrame(schema = {
            "label": pl.String,
            "mean": pl.Float64,
            "sigma": pl.Float64,
            "walk": pl.Boolean,
            "seed": pl.Int32,
        })
        st.text("rand:")
        df_rand = cast(
            pl.DataFrame, 
            st.data_editor(schema_rand, num_rows="dynamic")
        )

        schema_rolling = pl.DataFrame(schema={
            "label": pl.String,
            "func": pl.String,
            "kw-0": pl.String,
            "kw-1": pl.String,
            "kw-2": pl.String,
            "kw-3": pl.String,
            "kw-4": pl.String,
        })
        st.text("rand:")
        df_rolling = cast(
            pl.DataFrame,
            st.data_editor(schema_rolling, num_rows="dynamic")
        )

        steps = 100
        step = 1.

        g = ceg.Graph.new()

        g, refs = g.pipe(
            rows_to_refs_rand, 
            df_rand,
            step=step,
            keep=steps,
        )

        if not len(refs):
            return

        es = [ceg.Event.zero(r) for r in refs.values()]

        g, refs = g.pipe(
            rows_to_refs_rolling,
            df_rolling,
            refs,
            keep=steps,
        )

        g, es, ts = ceg.batches(g, *es, n = 100, g = len(refs))

        st.line_chart(pl.DataFrame({
            r_label: r.history(g).last_n_before(steps, ts[-1])
            for r_label, r in refs.items()
        }))
