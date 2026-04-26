from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Annotated,
)
import datetime as dt

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

import polars as pl


def parse_lag(s: str):
    n, *segs = s.split("D")
    return dt.timedelta(days=int(n))


@dataclass
class FreqElement:
    unit: str
    n: int


@dataclass(frozen=True)
class Freq:
    els: list[FreqElement]


def calc_sample_limit(s: str, freq: str):
    s_d = "D" in s
    s_w = "W" in s
    s_m = "M" in s
    if freq == "a":
        return 1 if s_d or s_w else 2
    elif freq == "sa":
        return 2 if s_m else 1
    elif freq == "q":
        return 4 if s_m else 1
    elif freq == "m":
        return 12 if s_m else 1 if s_d else 2

    # TODO: if week specified, then only once per month
    # unless m, then once per year
    # if d specified, once per week

    # etc.

    # (unless multi)

    elif freq == "bw":
        return 2 if s_d else 4 if s_w else 28
    elif freq == "w":
        return 2 if s_d else 6 if s_w else 56
    elif freq == "d":
        return 10 if s_d else 30 if s_w else 380
    else:
        raise ValueError((s, freq))


def rebalance_mask(s: str, dt: str = "date"):
    # TODO: range selection, set selection
    # TODO: every n (second month, etc.)

    d = pl.col(dt)

    w = d.dt.week()
    m = d.dt.month()
    y = d.dt.year()

    m_start = d.dt.month_start()
    m_weekday = m_start.dt.weekday()

    w_in_m = ((d - m_start).dt.total_days() / 7).floor() + 1

    exprs = []
    n_d = n_w = n_m = None

    if "D" in s:
        n_d, s = s.split("D")
        n_d = int(n_d)
        # TODO: many?
        exprs.append(d.dt.weekday() == n_d)
    if "W" in s:
        n_w, s = s.split("W")
        n_w = int(n_w)
        exprs.append(w_in_m == n_w)
        if "D" not in s:
            exprs.append(d.dt.weekday() == 3)
    if "M" in s:
        n_m, s = s.split("M")
        n_m = int(n_m)
        exprs.append(m == n_m)

    return pl.all_horizontal(*exprs)
