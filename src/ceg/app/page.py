from typing import cast, Type, NamedTuple, Any, Callable
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
    page: str
    name: str
    data: pl.DataFrame
    editable: bool

    label: str | None

    @classmethod
    def new(
        cls,
        page: str,
        name: str,
        data: pl.DataFrame,
        editable: bool=False,
        label: str | None = None,
    ):
        return cls(page, name, data, editable, label)

    @classmethod
    def empty(
        cls,
        page: str,
        name: str,
        schema: dict[str, Type[pl.DataType]],
        editable: bool=False,
        label: str | None = None,
    ):
        return cls(
            page=page,
            name=name,
            data=pl.DataFrame(schema=schema),
            editable=editable,
            label=label
        )

    @property
    def schema(self):
        return self.data.schema

    @property
    def full_name(self):
        return f"{self.page}-{self.name}"

    def add(self, dfs: dict[str, pl.DataFrame], on_change: Callable | None = None):
        if self.label is not None:
            st.text(f"{self.label}:")
        if self.editable:
            if self.full_name not in st.session_state:
                st.session_state[self.full_name] = self.data
            dfs[self.name] = cast(
                pl.DataFrame,
                st.data_editor(
                    st.session_state[self.full_name],
                    num_rows="dynamic",
                    on_change=on_change
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

    dfs_data: dict[str, pl.DataFrame] = {}


class Dynamic(DynamicKw, Page):

    @classmethod
    def new(
        cls,
        name: str,
        shared: Shared,
        dfs: tuple[DataFrame, ...] = (),
        tfs: tuple[Transformation, ...] = (),
    ):
        return cls(name, shared, dfs, tfs, {})

    @property
    def active(self):
        return f"{self.name}-active"

    @property
    def steps(self):
        return f"{self.name}-steps"

    def run(self):
        if self.active not in st.session_state:
            st.session_state[self.active] = False

        st.toggle(
            "run", 
            st.session_state[self.active],
            key=self.active,
        )

        for df in self.dfs:
            df.add(self.dfs_data)

        shared = self.shared

        if st.session_state[self.active]:
            # here we do graphs?
            g = ceg.Graph.new()

            refs: dict[str, ceg.Ref.Any] = {}
            es: list[ceg.Event] = []

            for tf in self.tfs:
                g, refs, es, shared = tf.apply(
                    self,
                    g,
                    refs,
                    es,
                    self.dfs_data,
                    shared
                )


#  ------------------
