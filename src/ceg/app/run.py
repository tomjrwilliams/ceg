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

import ceg.app.examples as examples

SYMBOLS = ['A6', 'AD', 'ALI', 'B6', 'BFX', 'BR', 'BTC', 'BZ', 'B', 'CB', 'CC', 'CL', 'CNH', 'CSC', 'CT', 'C', 'DC', 'DX', 'E1', 'E6', 'E7', 'EBM', 'ED', 'ER', 'ESG', 'ES', 'EW', 'FBON', 'FBTP', 'FBTS', 'FCE', 'FDAX', 'FDIV', 'FDXM', 'FDXS', 'FESX', 'FEU3', 'FGBL', 'FGBM', 'FGBS', 'FGBX', 'FOAT', 'FSMX', 'FTDX', 'FTI', 'FTUK', 'FVSA', 'FXXP', 'GC', 'GF', 'GSCI', 'G', 'HE', 'HG', 'HH', 'HO', 'HRC', 'J1', 'J7', 'KC', 'KRW', 'LBS', 'LE', 'L', 'M2K', 'MAX', 'MBT', 'MCL', 'MES', 'MET', 'MFC', 'MFS', 'MGC', 'MME', 'MNQ', 'MP', 'MURA', 'N6', 'NG', 'NIY', 'NKD', 'NOK', 'NQ', 'OJ', 'PA', 'PL', 'PRK', 'PSI', 'QG', 'RB', 'RM', 'RP', 'RS', 'RTY', 'SB', 'SEK', 'SIL', 'SIR', 'SI', 'SO3', 'SR1', 'SR3', 'T6', 'TN', 'UB', 'US', 'VXM', 'VX', 'XAE', 'XAF', 'XAI', 'XC', 'YM', 'ZC', 'ZF', 'ZL', 'ZM', 'ZN', 'ZO', 'ZQ', 'ZRPA', 'ZR', 'ZS', 'ZTWA', 'ZT', 'ZW']

shared: app.Shared = cast(app.Shared, frozendict())

pages: frozendict[
    str, tuple[app.nav.Page, ...]
] = frozendict() # type: ignore

pages = (
    pages.set("bars", (
        examples.bars.lines("ES"),
        examples.bars.lines("CL"),
    )).set("vol", (
        examples.vol.lines("ES"),
        examples.vol.lines("CL"),
    ))
    # )).set("minmax", (
    #     examples.minmax.lines("ES", start="2014"),
    #     examples.minmax.lines("NQ", start="2014"),
    #     examples.minmax.lines("RTY", start="2014"),
    #     examples.minmax.lines("CL", start="2014"),
    #     examples.minmax.lines("NG", start="2014"),
    #     examples.minmax.lines("GC", start="2014"),
    #     examples.minmax.lines("HG", start="2014"),
    #     examples.minmax.lines("C", start="2014"),
    #     examples.minmax.lines("G", start="2014"),
    #     examples.minmax.lines("US", start="2014"),
    #     examples.minmax.lines("TN", start="2014"),
    #     examples.minmax.lines("XLF", product="ETF", start="2014"),
    #     examples.minmax.lines("XLE", product="ETF", start="2014"),
    # ))
)

# TODO: pnl break:
# FUT-G
# ETF-XLF

app.nav.page(pages, page_config=dict(
    page_title="ceg",
    layout="wide"
)).run()