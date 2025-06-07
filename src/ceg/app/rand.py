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

class Gaussian(Page):
    def run(self):
        schema = pl.DataFrame(schema = {
            "label": pl.String,
            "mean": pl.Float64,
            "sigma": pl.Float64,
            "walk": pl.Boolean,
            "seed": pl.Int32,
        })

        # TODO: reversion 

        df = cast(
            pl.DataFrame, 
            st.data_editor(schema, num_rows="dynamic")
        )
        pad = (
            0 if not len(df) else 1 + int(math.log(len(df), 10))
        )

        steps = 100
        step = 1.

        g = ceg.Graph.new()
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
                    keep=steps,
                )
            else:
                g, r = g.bind(ceg.fs.rand.gaussian.new(
                    mean=row["mean"],
                    std=row["sigma"],
                    seed=1 or row["seed"],
                ), when=ceg.Loop.every(step), keep=steps)
            
            label = f"{str(i).rjust(pad, '0')}-{row['label']}"
            refs[label] = r
        
        if not len(refs):
            return

        es = [ceg.Event.zero(r) for r in refs.values()]
        g, es, ts = ceg.batches(g, *es, n = 100, g = len(refs))

        st.line_chart(pl.DataFrame({
            r_label: r.history(g).last_n_before(steps, ts[-1])
            for r_label, r in refs.items()
        }))
