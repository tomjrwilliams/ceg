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


@dataclass(frozen=True)
class DateRange(Component):
    key: str

    d0: dt.date
    d1: dt.date

    input_labels: tuple[str, str] = ("start:", "end:")
    slider_label: str = "date:"

    sync: str | None = None

    @property
    def version(self) -> VersionVariable:
        key = self.key if self.sync is None else self.sync
        return VersionVariable(f"{key}_version")
    
    @property
    def input_0(self) -> StateVariable:
        return StateVariable(f"{self.key}_input_0")
    
    @property
    def input_1(self) -> StateVariable:
        return StateVariable(f"{self.key}_input_1")
    
    @property
    def slider(self) -> StateVariable:
        return StateVariable(f"{self.key}_slider")

    @property
    def date_0(self) -> VersionedVariable:
        key = self.key if self.sync is None else self.sync
        return VersionedVariable(
            f"{key}_date_0", self.version
        )
    
    @property
    def date_1(self) -> VersionedVariable:
        key = self.key if self.sync is None else self.sync
        return VersionedVariable(
            f"{key}_date_1", self.version
        )

    def render(self) -> DateRange:
        d0 = self.d0
        d1 = self.d1

        def on_change_input():
            self.version.incr()
            self.date_0.set(self.input_0.get())
            self.date_1.set(self.input_1.get())
        
        def on_change_slider():
            self.version.incr()
            d0, d1 = self.slider.get()
            self.date_0.set(d0)
            self.date_1.set(d1)

        if not self.date_0.exists():
            self.date_0.set(d0)
            self.date_1.set(d1)
            
        c1, c2 = st.columns(2)

        with c1:
            d0_i = st.empty()
            d0_i.date_input(
                self.input_labels[0],
                self.date_0.get(d0),
                on_change=on_change_input,
                key=self.input_0.key
            )
        with c2:
            d1_i = st.empty()
            d1_i.date_input(
                self.input_labels[1],
                self.date_1.get(d1),
                on_change=on_change_input,
                key=self.input_1.key
            )
        
        slider = st.empty()
        slider.slider(
            self.slider_label,
            d0,
            d1,
            (self.date_0.get(d0), self.date_1.get(d1)),
            on_change=on_change_slider,
            key=self.slider.key
        )

        return self
