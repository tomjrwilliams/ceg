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

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ..variables import *
from .core import *
from ....apis import t

from ....apis.st import cache, library

st = library()


@cache()
def read_text(fp: Path):
    with fp.open("r") as f:
        v_init = json.load(f)["value"]


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

    def load(
        self,
        cached: bool = True,
        set: bool = False,
    ):
        fp = Path(self.fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        if fp.exists():
            if cached:
                with fp.open("r") as f:
                    v_init = json.load(f)["value"]
            else:
                v_init = read_text(fp)
            if set:
                self.data.set(v_init)
        else:
            v_init = None

        return v_init

    def render(self) -> CachedText:

        # def on_change_df():
        #     self.dirty.set(True)

        v_init = None

        if not self.data.exists():
            v_init = self.load(cached=False, set=True)
        else:
            v_init = self.data.get(t=str)

        v = st.text_input(
            self.label,
            key=self.dirty.key,
            value=v_init,
        )

        if not v == v_init:
            self.data.set(v)
            with Path(self.fp).open("w+") as f:
                json.dump({"value": v}, f)
            st.rerun()  # NOTE: needed, but moves focus?

        return self
