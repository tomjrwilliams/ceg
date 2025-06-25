import os
import sys
sys.path.append("./src")

import json
import pathlib

from typing import Any, cast, Callable
from functools import wraps

from frozendict import frozendict

import streamlit as st

os.environ["STREAMLIT"] = "true"
os.environ["TIMEZONE_OFFSET"] = str(st.context.timezone_offset)

if (
    os.environ.get("AWS") == "true"
    and os.environ.get("LOCAL", "true") == "true"
):
    os.environ["DATA_SOURCE_FRD"] = "AWS"

    with pathlib.Path("./__local__/aws.json").open('r') as f:
        creds = json.load(f)

    os.environ["AWS_ACCESS_KEY_ID"] = creds["key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["key_secret"]

import ceg
import ceg.fs as fs
import ceg.data as data
import ceg.app as app

from ceg.app.markets import bars
from ceg.app.markets import vol
from ceg.app.markets import range_spectrum

# TODO:
# alternatively, have a flag that we default to false
# for allow_none
# where if we return none, we dont append
# if we do allow, we cast to nan and append

# so the default is that we allow nan, but not none
# and up to func to return the relevant one

SYMBOLS = ['A6', 'AD', 'ALI', 'B6', 'BFX', 'BR', 'BTC', 'BZ', 'B', 'CB', 'CC', 'CL', 'CNH', 'CSC', 'CT', 'C', 'DC', 'DX', 'E1', 'E6', 'E7', 'EBM', 'ED', 'ER', 'ESG', 'ES', 'EW', 'FBON', 'FBTP', 'FBTS', 'FCE', 'FDAX', 'FDIV', 'FDXM', 'FDXS', 'FESX', 'FEU3', 'FGBL', 'FGBM', 'FGBS', 'FGBX', 'FOAT', 'FSMX', 'FTDX', 'FTI', 'FTUK', 'FVSA', 'FXXP', 'GC', 'GF', 'GSCI', 'G', 'HE', 'HG', 'HH', 'HO', 'HRC', 'J1', 'J7', 'KC', 'KRW', 'LBS', 'LE', 'L', 'M2K', 'MAX', 'MBT', 'MCL', 'MES', 'MET', 'MFC', 'MFS', 'MGC', 'MME', 'MNQ', 'MP', 'MURA', 'N6', 'NG', 'NIY', 'NKD', 'NOK', 'NQ', 'OJ', 'PA', 'PL', 'PRK', 'PSI', 'QG', 'RB', 'RM', 'RP', 'RS', 'RTY', 'SB', 'SEK', 'SIL', 'SIR', 'SI', 'SO3', 'SR1', 'SR3', 'T6', 'TN', 'UB', 'US', 'VXM', 'VX', 'XAE', 'XAF', 'XAI', 'XC', 'YM', 'ZC', 'ZF', 'ZL', 'ZM', 'ZN', 'ZO', 'ZQ', 'ZRPA', 'ZR', 'ZS', 'ZTWA', 'ZT', 'ZW']

shared: app.Shared = cast(app.Shared, frozendict())

pages: frozendict[
    str, tuple[app.nav.Page, ...]
] = frozendict() # type: ignore

if os.environ.get("LOCAL", "true") == "false":
    universe = [
        dict(symbol="ES", product="FUT"),
        dict(symbol="CL", product="FUT"),
    ]
else:
    universe = [
        dict(symbol="ES", product="FUT"),
        dict(symbol="NQ", product="FUT"),
        dict(symbol="RTY", product="FUT"),
        dict(symbol="CL", product="FUT"),
        dict(symbol="NG", product="FUT"),
        dict(symbol="GC", product="FUT"),
        dict(symbol="HG", product="FUT"),
        dict(symbol="C", product="FUT"),
        dict(symbol="G", product="FUT"),
        dict(symbol="US", product="FUT"),
        dict(symbol="TN", product="FUT"),
        dict(symbol="XLF", product="ETF"),
        dict(symbol="XLE", product="ETF"),
    ]

pages = (
    pages.set("bars", tuple([
        bars.lines(**product_symbol)
        for product_symbol in universe
    ])).set("vol", tuple([
        vol.lines(**product_symbol)
        for product_symbol in universe
    ])).set("range-spectrum", tuple([
        range_spectrum.trend.lines(**product_symbol)
        for product_symbol in universe
    ])).set("range-decomp", tuple([
        range_spectrum.decomp.lines(**product_symbol)
        for product_symbol in universe
    ]))
)

# TODO: try plot each element of the vectors on a single time series?
# given how many bar charts it would be
# to see if the loadngs flip
# stabilistaion as a set of samples to vote

# TODO: pnl break, range_spectrum
# FUT-G/
# ETF-XLF/

app.nav.page(pages, page_config=dict(
    page_title="ceg",
    layout="wide"
)).run()