from typing import NamedTuple, cast
import os
import datetime as dt
import zipfile

import contextlib
from frozendict import frozendict

from io import BytesIO

from functools import wraps, lru_cache

import polars
import pathlib

from ..utils import s3

# TZ = US Eastern

class Folders:
    ETF="etf"
    FUT="futures"
    FUTC="individual_contracts_archive"
    # TODO: _update
    FX="fx"
    IND="index"

class SuffixMap(NamedTuple):
    file: frozendict[str, str]
    folder: frozendict[str, str]

class Suffix:
    ratio = "ratio"
    abs = "abs"
    raw = "raw"

suffix_maps = cast(frozendict[str, SuffixMap], frozendict({
    Folders.ETF: SuffixMap(
        file=frozendict(empty=""),
        folder=frozendict(
            adjsplitdiv="adjsplitdiv",
            # adjsplit=""
            # unadj
        )
    ),
    Folders.FUT: SuffixMap(
        file=frozendict(
            ratio="continuous_ratio_adjusted",
            abs="continuous_absolute_adjusted",
            raw="continuous_unadjusted",
        ),
        folder=frozendict(
            ratio="contin_adj_ratio",
            abs="contin_adj_absolute",
            raw="contin_UNadj",
        ),
    ),
}))

# etf suffixes:
# adjsplitdiv

# generic suffixes:
# contin_adj_absolute
# contin_adj_ratio
# contin_UNadj

# other:
# continuous_contract_dates

@contextlib.contextmanager
def open_file(
    parent: str,
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full", # full
    freq: str = "1day", # 1day
):
    suffix_folder = None
    suffix_file = None
    
    if suffix is not None:
        suffixes: SuffixMap = suffix_maps[folder]
        suffix_folder = suffixes.folder.get(suffix)
        suffix_file = suffixes.file.get(suffix)

    if folder == Folders.ETF:
        folder = f"{folder}_{symbol[0]}"

    folder_stem = f"{folder}_{snap}_{freq}"
    if suffix_folder is not None:
        folder_stem = f"{folder_stem}_{suffix_folder}"
    
    dp = pathlib.Path(parent)
    fps = []

    for fp in dp.iterdir():
        if not fp.suffix == ".zip":
            continue
        if fp.name.startswith(folder_stem):
            fps.append(fp)
    
    assert len(fps) == 1, fps
    fp = fps[0]

    file_stem = f"{symbol}_{snap}_{freq}"
    if suffix_file is not None:
        file_stem = f"{file_stem}_{suffix_file}"

    with zipfile.ZipFile(fp) as zip:

        zip_names = zip.namelist()
        match_names = []

        for name in zip_names:
            if name.startswith(file_stem) or name == file_stem:
                match_names.append(name)

        assert len(match_names) == 1, dict(
            symbols=[
                n.split("_")[0]
                for n in zip_names
            ],
            snap=snap,
            freq=freq,
            suffix=suffix_file
        )

        name = match_names[0]
        with zip.open(name, mode="r") as f:
            yield f.read()

import io

def _read_daily_file(
    f: bytes | io.TextIOWrapper,
    start: dt.date | None,
    end: dt.date | None,
) -> polars.DataFrame:
    # TODO: maybe stream filter / binary search the rows in the date range
    df = polars.read_csv(
        f,
        has_header=False,
        schema_overrides=dict(
            column_1=polars.Date,
            column_2=polars.Float64,
            column_3=polars.Float64,
            column_4=polars.Float64,
            column_5=polars.Float64,
        ),
        try_parse_dates=True,
    )
    df = df.rename({
        "column_1": "date",
        "column_2": "open",
        "column_3": "high",
        "column_4": "low",
        "column_5": "close",
    })
    if "column_6" in df:
        df = df.rename({
            "column_6": "volume"
        })
    else:
        df = df.with_columns(
            polars.lit(None)
            .cast(polars.Float64)
            .alias("volume")
        )
    if "column_7" in df:
        df = df.rename({
            "column_7": "open_interest"
        })
    else:
        df = df.with_columns(
            polars.lit(None)
            .cast(polars.Float64)
            .alias("open_interest")
        )
    assert "date_right" not in df.schema, df.schema
    return df.pipe(filter_start_end, start, end)

def read_daily_file(
    fp: str | bytes,
    start: dt.date | None=None,
    end: dt.date | None=None,
) -> polars.DataFrame:

    if isinstance(fp, str):
        with pathlib.Path(fp).open('r') as f:
            df = _read_daily_file(
                f=f, start=start, end=end
            )
    else:
        df = _read_daily_file(
            f=fp, start=start, end=end
        )
    
    return df

def filter_start_end(
    df: polars.DataFrame,
    start: dt.date | None=None,
    end: dt.date | None=None,
):
    if start is not None:
        df = df.filter(
            polars.col("date") >= start
        )
    else:
        start = df.get_column("date").min() # type: ignore
    if end is not None:
        df = df.filter(
            polars.col("date") <= end
        )
    else:
        end = df.get_column("date").max() # type: ignore
    assert start is not None, start
    assert end is not None, end
    df = polars.select(
        polars.date_range(start, end)
        .alias("date")
    ).join(
        df,
        how="left",
        on="date",
    )
    assert "date_right" not in df.schema, df.schema
    return df

# TODO: wrap the file in a aws caching if flag set
# then just store the (relevant) files on AWS

# simple script to sync to AWS

def read_file(
    parent: str,
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full", # full
    freq: str = "1day", # 1day
    #
    start: dt.date | None=None,
    end: dt.date | None=None,
):
    if freq == "1day":
        if os.environ.get("DATA_SOURCE_FRD") == "AWS":
            return read_file_s3(
                folder,
                symbol,
                suffix,
                snap=snap,
                freq=freq,
            ).pipe(filter_start_end, start, end)
        with open_file(
            parent=parent,
            folder=folder,
            symbol=symbol,
            snap=snap,
            freq=freq,
            suffix=suffix,
        ) as f:
            return read_daily_file(
                f, start=start, end=end
            )
    else:
        raise ValueError(freq)

def get_bucket():
    tz_offset = os.environ.get("TIMEZONE_OFFSET", -480)
    offset = int(tz_offset)
    if offset < -180 or offset > 180:
        return "data-frd-sg"
    return "data-frd-uk"

def write_file_s3(
    parent: str,
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full", # full
    freq: str = "1day", # 1day
):
    bucket = get_bucket()
    key = f"{snap}/{freq}/{folder}/{symbol}/{suffix}"
    df = read_file(
        parent=parent,
        folder=folder,
        symbol=symbol,
        suffix=suffix,
        snap=snap,
        freq=freq,
    )
    return s3.put_object(
        bucket,
        key,
        df.serialize(),
    )

def cache():
    if os.environ.get("STREAMLIT") == "true":
        def st_decorator(f):
            import streamlit as st
            return st.cache_data(f)
        return st_decorator
    else:
        def lru_decorator(f):
            return lru_cache(maxsize=512)(f)
        return lru_decorator

@cache()
def read_file_s3(
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full", # full
    freq: str = "1day", # 1day
    #
):
    bucket = get_bucket()
    key = f"{snap}/{freq}/{folder}/{symbol}/{suffix}"
    bs: bytes = s3.get_object(bucket, key)
    return polars.DataFrame.deserialize(BytesIO(bs))