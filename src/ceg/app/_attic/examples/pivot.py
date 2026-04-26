import datetime as dt
import polars as pl
# from streamlit.navigation.page import StreamlitPage

import ceg.vis.st.pivot as pivot
from ...vis.st.nav import Page, Shared

ONE_DAY = dt.timedelta(days=1)


def dates_between(d0: dt.date, d1: dt.date):
    res: list[dt.date] = [
        d0 for _ in range((d1 - d0).days + 1)
    ]
    i = 0
    while d0 <= d1:
        res[i] = d0
        d0 += ONE_DAY
        i += 1
    return res


# ideally can then have a generic query api for the initial dtaframe over the series keys

# where the tags, and later relational joins on the keys with that tag, can be used to provide grouping keys on the time series we get back


class VisPage(Page):
    def run(self) -> StreamlitPage | None:
        ds = dates_between(
            dt.date(2000, 1, 1), dt.date(2003, 1, 1)
        )
        df = pl.DataFrame(
            {
                "d": ds,
                "a": ["a" for _ in ds[:50]]
                + ["b" for _ in ds[50:]],
                "b": ["b" for _ in ds[:50]]
                + ["a" for _ in ds[50:]],
                "x": ["a" for _ in ds],
                "v0": range(len(ds)),
                "v1": range(len(ds), len(ds) * 2),
            }
        ).with_columns(pl.col("a").cast(pl.Categorical))
        return pivot.MultiPivot(df).add()


# TODO: rename (multi) pivot
def example(title: str, shared: Shared):
    return VisPage(title, shared)
