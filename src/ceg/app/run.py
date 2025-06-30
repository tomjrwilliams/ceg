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
        dict(symbol="FESX", product="FUT"),
        dict(symbol="NIY", product="FUT"),
        dict(symbol="NKD", product="FUT"),
        dict(symbol="NQ", product="FUT"),
        dict(symbol="RTY", product="FUT"),
        dict(symbol="FTUK", product="FUT"),
        dict(symbol="EW", product="FUT"), # midcap
        dict(symbol="XAE", product="FUT"), # energy eq
        dict(symbol="XAF", product="FUT"), # fins eq
        dict(symbol="XAI", product="FUT"), # energy eq
        # FTI = aex
        dict(symbol="BTC", product="FUT"),
        dict(symbol="CL", product="FUT"),
        dict(symbol="NG", product="FUT"),
        dict(symbol="GC", product="FUT"),
        dict(symbol="SIL", product="FUT"),
        dict(symbol="HG", product="FUT"),
        # dict(symbol="C", product="FUT"),
        dict(symbol="G", product="FUT"),
        dict(symbol="US", product="FUT"),
        dict(symbol="TN", product="FUT"),
        dict(symbol="ZF", product="FUT"),
        dict(symbol="DX", product="FUT"), # dollar?
        dict(symbol="E6", product="FUT"),
        dict(symbol="E7", product="FUT"),
        dict(symbol="J1", product="FUT"),
        dict(symbol="J7", product="FUT"),
        dict(symbol="VX", product="FUT"), # vix
        dict(symbol="VXM", product="FUT"), # vix
        dict(symbol="FVSA", product="FUT"), # vstoxx
        # dict(symbol="XLF", product="ETF"),
        # dict(symbol="XLE", product="ETF"),
        # N6=NZD
        # KRW
        # B6=GBP
        # NOK
        # SEK
        # SO3 = Sonia
        # SO1=1m sofr
        # SO3 = 3m sofr
        # FEU3 3m euribore
        # FOAT = oat (french)
        # FBTP = btp (italy)
    ]

# universe_w_drift = [
#     {**product_symbol, **dict(
#         drifts=DRIFTS.get(product_symbol["symbol"])
#     )}
#     for product_symbol in universe
# ]

pages = (
    pages.set("bars", tuple([
        bars.lines(**product_symbol)
        for product_symbol in universe
    ])).set("vol", tuple([
        vol.lines(**product_symbol)
        for product_symbol in universe
    ])).set("range-spread", tuple([
        range_spectrum.spreads.lines(
            **product_symbol,
            relative=relative,
        )
        for product_symbol in universe
        # for truncate in [None, 0.2]
        for relative in [False, "mid", "mid-inner", "mid-pct"]
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


# TODO: oen way to deal with the near 0 for sx5e is a back adjust series so you add back the delta so each day is as if that was the last future



# TODO: one issue with fitting a factor model from eg. oil stocks to just oil, is that growth will also place a role

# or more tangibly, overall equity market


# you eg. want to either adjust for the overall market, and really regress the market-beta adjusted residual to oil


# but then your hedge factor will be a long short basket against an equity position


# you might even want to start by regressing the market and oil, so you can then regress the residual post market of the equities, against the reisudal post market of oil?




# or, more abstractly, it's as if your outputs are under-parametrised

# you could instead use a two layer model, into a shared factor and back up to the oil and (say) market

# but if all linear that's redundant, so the inner factor has to go through a non linearity (eg. sigmoid, assuming all are vol adjusted first, so you're predicting pre-vol scaling back up)



# interesting idea to do this for a few pairs, overlapping, and then sum back up the individual contributions to get your final predictions
# for eg. oil, equities, ec.

# and working backwards, your combined factor weights? (requiring netting of the others, per the above, or not?)