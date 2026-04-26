
import polars as pl
import datetime as dt
import os
import ceg

for symbol in ["SPY", "VONG", "VYM"]:
    for adjust in [
        "pct", "split", "raw",
    ]:
        df = ceg.apis.data.frd.read_file(
            "./__local__/data/frd",
            folder=ceg.apis.data.frd.Folders.ETF,
            symbol=symbol,
            suffix=adjust,
            snap="full",  # full
            freq="1day",  # 1day
            #
            start=None,
            end=None,
            update=True,
        )

        print(symbol, adjust, df)