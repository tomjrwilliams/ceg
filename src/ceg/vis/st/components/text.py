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
import json
from pathlib import Path
import datetime as dt
import numpy as np
import polars as pl
import streamlit as st

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ..variables import *
from .core import *
from ....apis import t


@dataclass(frozen=True)
class CachedText(Component):
    key: str
    fp: str
    label: str

    @property
    def dirty(self) -> StateVariable:
        return StateVariable(f"{self.key}_dirty")
    
    @property
    def data(self) -> StateVariable:
        return StateVariable(f"{self.key}_data")

    def render(self) -> CachedText:

        # def on_change_df():
        #     self.dirty.set(True)
    
        v_init = None

        if not self.data.exists():
            fp = Path(self.fp)
            fp.parent.mkdir(exist_ok=True, parents=True)
            if fp.exists():
                with fp.open('r') as f:
                    v_init = json.load(f)["value"]
                # TODO: cast to schema?
                self.data.set(v_init)
            else:
                v_init = None
        else:
            v_init = self.data.get(t=str)

        v = st.text_input(
            self.label,
            key=self.dirty.key,
            value=v_init,
        )

        if not v == v_init:
            self.data.set(v)
            with Path(self.fp).open('w+') as f:
                json.dump({"value": v}, f)
            st.rerun() # NOTE: needed, but moves focus?

        return self