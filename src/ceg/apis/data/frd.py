from typing import NamedTuple, cast
import os
import datetime as dt
import zipfile

import contextlib
from frozendict import frozendict

import io
from io import BytesIO

from functools import wraps, lru_cache

import polars
import polars as pl

import pathlib
from pathlib import Path

from .. import s3

from .. import logs

logger, log_kw = logs.prepare_logger(__file__)

# TZ = US Eastern

# workflow: check last update
# compare to snap
# request the required timeframe of file to fill the gap (if any)
# update value, update snap

# Url EndPoint : https://firstratedata.com/api/data_file

# Parameter : type
# Accepted Values : stock , etf, futures , crypto , index , fx , options

# Description : Specifies the type of instrument that is being requested.

# Example :
# https://firstratedata.com/api/data_file?type=futures&period=week&timeframe=1day&adjustment=contin_UNadj&userid=Gp66_esi0E-jz_-_mfmW0Q

# Parameter : period
# Accepted Values : full , month , week , day

# Description : Specifies the period to request data for. 'full' requests the entire historical archive, 'month' requests the last 30 days, 'week' requests the current trading week (starting on Monday), 'day' requests the last trading day.

# Example :
# https://firstratedata.com/api/data_file?type=futures&period=full&adjustment=contin_UNadj&timeframe=1hour&userid=Gp66_esi0E-jz_-_mfmW0Q

# Parameter : timeframe
# Accepted Values : 1min , 5min , 30min , 1hour , 1day

# Description : Specifies the period the timeframe of the data. '1min' will request 1-minute intraday bars, '5min' requests 5-minute bars etc.
# Note : bars with zero volumes are not included
# Note: 1day data also includes open - interest in the final file. (therefore the data format for 1day futures data is { DateTime, Open, High, Low, Close, Volume, Open Interest })

# Example :
# https://firstratedata.com/api/data_file?type=futures&period=week&adjustment=contin_UNadj&timeframe=1min&userid=Gp66_esi0E-jz_-_mfmW0Q

# Parameter : adjustment
# Accepted Values : contin_UNadj , contin_adj_ratio , contin_adj_absolute

# Description : Specifies the type of adjustment applied to the continuous data. (note the continuous data is created by stiching together front-month contracts into a single continuous series - more details on this and the adjustments can be found on our Futures Adjustment Info Page

# 'contin_UNadj'is the raw unadjusted trade data.
# 'contin_adj_ratio' is the ratio-adjusted data to avoid artifical price jumps on roll dates.
# 'contin_adj_absolute' is the absolute-adjusted data to avoid artifical price jumps on roll dates.

# Example :
# https://firstratedata.com/api/data_file?type=futures&period=week&timeframe=1day&adjustment=contin_UNadj&userid=Gp66_esi0E-jz_-_mfmW0Q


# Url EndPoint : https://firstratedata.com/api/futures_contract

# Parameter : contract_files
# Accepted Values : archive , update

# Description : An 'archive' request returns the pre-2025 data as a single zip archive of all individual futures contracts. 'update' returns a zip of the 2025 + contracts which is updated daily.

# Example :
# https://firstratedata.com/api/futures_contract?contract_files=update&timeframe=1min&userid=Gp66_esi0E-jz_-_mfmW0Q

# Parameter : timeframe
# Accepted Values : 1min , 5min , 30min , 1hour , 1day

# Description : Specifies the period the timeframe of the data. '1min' will request 1-minute intraday bars, '1day' requests 1-day bars etc.
# Note that bars with zero volumes are not included.
# Note : 1day data also includes open-interest in the final file. (therefore the data format for 1day futures data is {DateTime, Open, High, Low, Close, Volume,Open Interest})

# Example :
# https://firstratedata.com/api/futures_contract?contract_files=update&timeframe=1min&userid=Gp66_esi0E-jz_-_mfmW0Q

# Url EndPoint : https://firstratedata.com/api/last_update

# Parameter : type
# Accepted Values : stock , etf, futures , crypto , index , fx

# Description : Specifies the type of instrument that is being requested.

# Example :
# https://firstratedata.com/api/last_update?type=futures&userid=Gp66_esi0E-jz_-_mfmW0Q

# Parameter : is_full_update (optional)
# Accepted Values : false , true (false is the default)

# Description : This parameter is used when requesting the last update of the full historical dataset (otherwise the Last Update function will return the date for the update files for the past day, week or month).

# Example :
# https://firstratedata.com/api/last_update?type=futures&is_full_update=true&userid=Gp66_esi0E-jz_-_mfmW0Q

# Url EndPoint : https://firstratedata.com/api/ticker_listing

# Data Format : {ticker},{name},{startDate},{endDate}

# Parameter : type
# Accepted Values : stock , etf

# Description : Specifies the type of instrument that is being requested.

# Example :
# https://firstratedata.com/api/ticker_listing?type=futures&userid=Gp66_esi0E-jz_-_mfmW0Q

class Folders:
    STOCK = "stock" # NOTE: tbc, stock?
    ETF = "etf"
    FUT = "futures"
    FUTC = "individual_contracts_archive"
    FX = "fx"
    IND = "index"

    # TODO: _update


class SuffixMap(NamedTuple):
    file: frozendict[str, str | None]
    folder: frozendict[str, str]


suffix_maps = cast(
    frozendict[str, SuffixMap],
    frozendict(
        {
            Folders.FUT: SuffixMap(
                file=frozendict(
                    pct="continuous_ratio_adjusted",
                    abs="continuous_absolute_adjusted",
                    raw=None,
                    default="pct",
                ),
                folder=frozendict(
                    pct="contin_adj_ratio",
                    abs="contin_adj_absolute",
                    raw="contin_UNadj",
                    default="pct",
                ),
            ),
            Folders.ETF: SuffixMap(
                file=frozendict(
                    # NOTE: just for type detection
                    empty=""
                ),
                folder=frozendict(
                    pct="adjsplitdiv_",
                    split="adjsplit_",
                    raw="UNADJUSTED",
                    default="pct",
                ),
            ),
            Folders.STOCK: SuffixMap(
                file=frozendict(
                    # NOTE: just for type detection
                    empty=""
                ),
                folder=frozendict(
                    pct="adjsplitdiv_",
                    split="adjsplit_",
                    raw="UNADJUSTED",
                    default="pct",
                ),
            ),
            Folders.IND: SuffixMap(
                file=frozendict(
                    empty=""
                ),
                folder=frozendict(
                    raw="",
                )
            ),
            Folders.FX: SuffixMap(
                file=frozendict(
                    empty=""
                ),
                folder=frozendict(
                    raw="",
                )
            ),
        }
    ),
)


def folder_stem(
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full",  # full
    freq: str = "1day",  # 1day
) -> str:
    suffixes = suffix_maps[folder].folder
    if suffix is None:
        suffix = suffixes.get("default", None)

    suffix_folder = None
    if suffix is not None:
        suffix_folder = suffixes.get(suffix)

    if folder == Folders.ETF or folder == Folders.STOCK:
        folder = f"{folder}_{symbol[0]}"

    stem = f"{folder}_{snap}_{freq}"

    if suffix_folder is not None:
        stem = f"{stem}_{suffix_folder}"

    return stem

def file_stem(
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full",  # full
    freq: str = "1day",  # 1day
):
    suffixes = suffix_maps[folder].file
    if suffix is None:
        suffix = suffixes.get("default", None)

    suffix_file = None

    if suffix is not None:
        suffix_file = suffixes.get(suffix)

    file_stem = f"{symbol}_{snap}_{freq}"
    if suffix_file is not None:
        file_stem = f"{file_stem}_{suffix_file}"
    
    return file_stem

def find_folder(
    dir: str,
    stem: str,
) -> Path:
    dp = pathlib.Path(dir)
    fps = []

    for fp in dp.iterdir():
        if not fp.suffix == ".zip":
            continue
        if fp.name.startswith(stem):
            fps.append(fp)

    assert len(fps) == 1, (fps, stem)
    fp = fps[0]

    return fp

@contextlib.contextmanager
def find_file(dp: Path, stem: str):

    with zipfile.ZipFile(dp) as archive:

        zip_names = archive.namelist()
        match_names = []

        for name in zip_names:
            if name.startswith(stem) or name == stem:
                match_names.append(name)

        assert len(match_names) == 1, dict(
            symbols=[n.split("_")[0] for n in zip_names],
            stem=stem,
        )

        name = match_names[0]
        yield name, archive

@contextlib.contextmanager
def open_file(dp: Path, stem: str):
    with find_file(dp, stem) as (name, archive):
        with archive.open(name, mode="r") as f:
            yield f.read()


def parse_daily_file(
    f: bytes | io.TextIOWrapper,
    start: dt.date | None,
    end: dt.date | None,
) -> polars.DataFrame:
    # TODO: maybe stream filter / binary search the rows in the date range
    df = polars.read_csv(
        f,
        has_header=False,
        schema_overrides=dict(
            # column_1=polars.Date,
            column_2=polars.Float64,
            column_3=polars.Float64,
            column_4=polars.Float64,
            column_5=polars.Float64,
        ),
        # try_parse_dates=True,
        ignore_errors=True,
    )
    d0 = df.get_column("column_1")[0]
    try:
        if isinstance(d0, int):
            df = df.with_columns(
                pl.col("column_1")
                .cast(pl.String)
                .str.to_date("%Y%m%d")
            )
        elif len(d0) == 8:
            df = df.with_columns(
                pl.col("column_1").str.to_date("%Y%m%d")
            )
        elif "/" in d0:
            df = df.with_columns(
                pl.col("column_1").str.to_date("%Y/%m/%d")
            )
        else:
            df = df.with_columns(
                pl.col("column_1").str.to_date()
            )
    except:
        raise ValueError((df, d0))
    df = df.with_columns()
    df = df.rename(
        {
            "column_1": "date",
            "column_2": "open",
            "column_3": "high",
            "column_4": "low",
            "column_5": "close",
        }
    )
    if "column_6" in df:
        df = df.rename({"column_6": "volume"})
    else:
        df = df.with_columns(
            polars.lit(None).cast(polars.Float64).alias("volume")
        )
    if "column_7" in df:
        df = df.rename({"column_7": "open_interest"})
    else:
        df = df.with_columns(
            polars.lit(None)
            .cast(polars.Float64)
            .alias("open_interest")
        )
    assert "date_right" not in df.schema, df.schema
    return df.pipe(filter_start_end, start, end)

SCHEMA = {
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "open_interest": pl.Float64,
}

FIELD_INDICES = {
    k: i - 1 for i, k in enumerate(SCHEMA) if k != "date"
}

class FieldIndices:
    OPEN = FIELD_INDICES["open"]
    HIGH = FIELD_INDICES["high"]
    LOW = FIELD_INDICES["low"]
    CLOSE = FIELD_INDICES["close"]
    VOLUME = FIELD_INDICES["volume"]
    OPEN_INTEREST = FIELD_INDICES["open_interest"]

def filter_start_end(
    df: polars.DataFrame,
    start: dt.date | None = None,
    end: dt.date | None = None,
):
    if start is not None:
        df = df.filter(polars.col("date") >= start)
    else:
        start = df.get_column("date").min()  # type: ignore
    if end is not None:
        df = df.filter(polars.col("date") <= end)
    else:
        end = df.get_column("date").max()  # type: ignore
    assert start is not None, start
    assert end is not None, end
    df = polars.select(
        polars.date_range(start, end).alias("date")
    ).join(
        df,
        how="left",
        on="date",
    )
    assert "date_right" not in df.schema, df.schema
    return df

def read_daily_file(
    fp: str | bytes,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> polars.DataFrame:

    if isinstance(fp, str):
        with pathlib.Path(fp).open("r") as f:
            df = parse_daily_file(f=f, start=start, end=end)
    else:
        df = parse_daily_file(f=fp, start=start, end=end)

    return df

# TODO: streamlit tiemzone offset
def get_bucket():
    tz_offset = os.environ.get("TIMEZONE_OFFSET", 0)
    offset = int(tz_offset)
    if offset < -180 or offset > 180:
        return "data-frd-sg"
    return "data-frd-uk"


def write_file_s3(
    parent: str,
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full",  # full
    freq: str = "1day",  # 1day
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


from ..st import cache


@cache()
def read_file_s3(
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full",  # full
    freq: str = "1day",  # 1day
    #
):
    bucket = get_bucket()
    key = f"{snap}/{freq}/{folder}/{symbol}/{suffix}"
    bs: bytes = s3.get_object(bucket, key)
    return polars.DataFrame.deserialize(BytesIO(bs))

def read_file(
    parent: str,
    folder: str,
    symbol: str,
    suffix: str | None = None,
    snap: str = "full",  # full
    freq: str = "1day",  # 1day
    #
    start: dt.date | None = None,
    end: dt.date | None = None,
    fields: list[str] | None = None,
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
        dp = find_folder(
            parent,
            stem=folder_stem(
                folder=folder,
                symbol=symbol,
                suffix=suffix,
                snap=snap,
                freq=freq,
            )
        )
        with open_file(
            dp,
            stem=file_stem(
                folder=folder,
                symbol=symbol,
                suffix=suffix,
                snap=snap,
                freq=freq,
            )
        ) as f:
            return read_daily_file(f, start=start, end=end)
    else:
        raise ValueError(freq)