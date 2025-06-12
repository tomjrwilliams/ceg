from typing import cast, Type, NamedTuple, Any
import types
import datetime as dt
import math
import numpy as np
import polars as pl
import streamlit as st

from frozendict import frozendict

from ceg.app.nav import Shared, Page

import ceg

#  ------------------

class DataFrame(NamedTuple):
    name: str
    data: pl.DataFrame
    editable: bool

    label: str | None

    @classmethod
    def new(
        cls,
        name: str,
        data: pl.DataFrame,
        editable: bool=False,
        label: str | None = None,
    ):
        return cls(name, data, editable, label)

    @classmethod
    def empty(
        cls,
        name: str,
        schema: dict[str, Type[pl.DataType]],
        editable: bool=False,
        label: str | None = None,
    ):
        return cls(
            name=name,
            data=pl.DataFrame(schema=schema),
            editable=editable,
            label=label
        )

    @property
    def schema(self):
        return self.data.schema

    def add(self, dfs: dict[str, pl.DataFrame]):
        if self.label is not None:
            st.text(f"{self.label}:")
        if self.editable:
            dfs[self.name] = cast(
                pl.DataFrame,
                st.data_editor(
                    self.data,
                    num_rows="dynamic",
                )
            )
        else:
            st.dataframe(self.data)
            dfs[self.name] = self.data
        return dfs

class Transformation:
    
    def apply(
        self,
        page: Page,
        g: ceg.Graph,
        refs: dict[str, ceg.Ref.Any],
        es: list[ceg.Event],
        dfs: dict[str, pl.DataFrame],
        shared: frozendict[str, Any],
    ):
        return g, refs, es, shared


class DynamicKw(NamedTuple):
    name: str
    shared: Shared
    dfs: tuple[DataFrame, ...] = ()
    tfs: tuple[Transformation, ...] = ()


class Dynamic(DynamicKw, Page):

    @classmethod
    def new(
        cls,
        name: str,
        shared: Shared,
        dfs: tuple[DataFrame, ...] = (),
        tfs: tuple[Transformation, ...] = (),
    ):
        return cls(name, shared, dfs, tfs)

    def run(self):
        shared = self.shared
        dfs: dict[str, pl.DataFrame] = {}

        run = st.toggle("run", False)

        for df in self.dfs:
            df.add(dfs)
        
        g = ceg.Graph.new()

        refs: dict[str, ceg.Ref.Any] = {}
        es: list[ceg.Event] = []

        if run:
            for tf in self.tfs:
                g, refs, es, shared = tf.apply(
                    self,
                    g,
                    refs,
                    es,
                    dfs,
                    shared,
                )

#  ------------------
