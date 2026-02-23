from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Type,
    Any,
    Callable,
    Literal,
    overload,
    cast,

)
from pathlib import Path
import datetime as dt
import numpy as np
import polars as pl

from .... import apis
st = apis.st.library()

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ..variables import *
from .core import *
from ....apis import t

@dataclass(frozen=True)
class Table(Component):
    key: str
    df: t.pl.DataFrame

    def render(self):
        st.table(self.df)
        return self

from ....apis.st import cache

@cache()
def read_df(fp: str):
    return pl.read_parquet(fp)

@dataclass(frozen=True)
class CachedDataFrame(Component):
    key: str
    fp: str
    schema: dict[str, Type[pl.DataType]]

    df: t.pl.DataFrame | None = None
    widths: dict[str, str | float] | None = None
    
    @property
    def dirty(self) -> StateVariable:
        return StateVariable(f"{self.key}_dirty")
    
    @property
    def data(self) -> StateVariable:
        return StateVariable(f"{self.key}_data")

    def load(self, cached: bool = True):
        df_init = None

        fp = Path(self.fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        if fp.exists():
            if cached:
                df_init = read_df(self.fp)
            else:
                df_init = pl.read_parquet(self.fp)
        else:
            df_init = None

        return df_init

    def merge_schema(self, df_init: pl.DataFrame | None):

        if df_init is None:
            df_init = pl.DataFrame(schema=self.schema)

        for k, dtype in self.schema.items():
            if k not in df_init.schema:
                df_init = df_init.with_columns(
                    pl.lit(None)
                    .alias(k)
                    .cast(dtype)
                )

        return df_init

    def render(self) -> CachedDataFrame:

        # def on_change_df():
        #     self.dirty.set(True)
    
        df_init = None

        if not self.data.exists():
            df_init = self.load(cached=False)
        else:
            df_init = self.data.get(t=pl.DataFrame)

        df_init = self.merge_schema(df_init)

        column_configs = None
        if self.widths is not None:
            column_configs = {}
            for k, w in self.widths.items():
                col = st.column_config.Column(width=w) # type: ignore
                column_configs[k] = col

        df = cast(pl.DataFrame, st.data_editor(
            df_init,
            num_rows="dynamic",
            # on_change=on_change_df,
            key=self.dirty.key,
            column_config=column_configs,
        ))

        if not df.equals(df_init):
            self.data.set(df)
            df.write_parquet(self.fp)
            st.rerun() # NOTE: needed, but moves focus?

        # self.data.set(df)
        # if self.dirty.get(False):
        #     if len(df):
        #         df.write_parquet(self.fp)
        #     self.dirty.set(False)

        return replace(self, df=t.pl.DataFrame.cast(df))