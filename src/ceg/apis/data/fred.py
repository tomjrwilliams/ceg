
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Annotated,
)
import time
import datetime as dt
from pathlib import Path

import polars as pl

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from pydantic import (
    TypeAdapter,
    BaseModel,
    Field,
    field_validator
)

import requests

from . import FRED_API

from .. import cal

URL_FRED = "https://api.stlouisfed.org/fred"


def get_text(
    path: str,
    query: dict | None = None,
):
    url = URL_FRED + f"/{path}"
    query = query or {}
    query["file_type"] = "json"
    query["api_key"] = FRED_API.KEY
    return requests.get(
        url,
        params=query
    ).text

def validate_datetime(s: str):
    if len(s) == 22:
        s = s[:-3]
    return dt.datetime.strptime(
        s, "%Y-%m-%d %H:%M:%S"
    )



@dataclass(frozen=True)
class SeriesDataItem:
    id: str
    realtime_start: dt.date
    realtime_end: dt.date
    title: str
    observation_start: dt.date
    observation_end: dt.date
    frequency: str
    frequency_short: str
    units: str
    units_short: str
    seasonal_adjustment: str
    seasonal_adjustment_short: str
    last_updated: dt.datetime
    popularity: int
    notes: str

    @field_validator('last_updated', mode='before')
    @classmethod
    def parse_birthdate(cls, value):
        return validate_datetime(value)

@dataclass(frozen=True)
class SeriesData:
    realtime_start: dt.date
    realtime_end: dt.date
    items: list[SeriesDataItem] = Field(
        validation_alias="seriess"
    )

    @classmethod
    def parse(cls, text: str) -> SeriesData:
        return TypeAdapter(cls).validate_json(text)

@dataclass(frozen=True)
class Series:
    id: str
    data: SeriesData

def get_series(id: str) -> Series:
    # """
    # >>> get_series("BAMLH0A0HYM2")
    # """
    try:
        return Series(
            id,
            data=SeriesData.parse(get_text(
                "series",
                dict(series_id=id),
            ))
        )
    except:
        raise ValueError(id)

@dataclass(frozen=True)
class VintageDatesData:
    realtime_start: dt.date
    realtime_end: dt.date
    order_by: str
    sort_order: str
    count: int
    offset: int
    limit: int
    vintage_dates: list[dt.date]

    @classmethod
    def parse(cls, text: str) -> VintageDatesData:
        return TypeAdapter(cls).validate_json(text)

@dataclass(frozen=True)
class VintageDates:
    id: str
    data: VintageDatesData

# dates when the series values changed
def get_vintage_dates(
    id: str,
    sort_order: str = "desc",
    offset: int=0,
) -> VintageDates:
    # """
    # >>> get_vintage_dates("BAMLH0A0HYM2")
    # """
    return VintageDates(
        id,
        VintageDatesData.parse(get_text(
            "series/vintagedates",
            dict(
                series_id=id,
                sort_order=sort_order,
                offset=offset,
            ),
        ))
    )

@dataclass(frozen=True)
class ObservationDataItem:
    realtime_start: dt.date
    realtime_end: dt.date
    date: dt.date
    value: float | None

    @field_validator('value', mode='before')
    @classmethod
    def parse_value(cls, value):
        if value == ".":
            return None
        return float(value)

@dataclass(frozen=True)
class ObservationData:
    realtime_start: dt.date
    realtime_end: dt.date
    observation_start: dt.date
    observation_end: dt.date
    units: str
    output_type: int
    file_type: str
    order_by: str
    sort_order: str
    count: int
    offset: int
    limit: int
    observations: list[ObservationDataItem]
    
    @classmethod
    def parse(cls, text: str) -> ObservationData:
        try:
            return TypeAdapter(cls).validate_json(text)
        except:
            raise ValueError(
                text[:300] + "..." + text[-300:]
            )

@dataclass(frozen=True)
class Observations:
    id: str
    data: ObservationData

# 1 = Observations by Real-Time Period
# 2 = Observations by Vintage Date, All Observations
# 3 = Observations by Vintage Date, New and Revised Observations Only
# 4 = Observations, Initial Release Only

def get_observations(
    id: str,
    sort_order: str = "desc",
    offset: int=0,
    output_type: int = 1,
    ending: dt.date | None = None,
    limit: int | None = None,
    vintage_dates: list[dt.date] | None = None,
    # TODO: asof range
) -> Observations:
    query = dict(
        series_id=id,
        sort_order=sort_order,
        offset=offset,
        output_type=output_type,
    )
    if ending is not None:
        query["realtime_end"] = ending.strftime("%Y-%m-%d")
    if limit is not None:
        query["limit"] = limit
    if vintage_dates is not None:
        query["vintage_dates"] = [ # type: ignore
            d.strftime("%Y-%m-%d") for d in vintage_dates
        ]
    return Observations(
        id,
        ObservationData.parse(get_text(
            "series/observations",
            query=query
        ))
    )


@dataclass(frozen=True)
class Spec:
    refreshed: dt.datetime
    id: str
    # realtime_start: dt.date
    # realtime_end: dt.date
    title: str
    observation_start: dt.date
    observation_end: dt.date
    frequency: str
    frequency_short: str
    units: str
    units_short: str
    seasonal_adjustment: str
    seasonal_adjustment_short: str
    last_updated: dt.datetime
    popularity: int
    # notes: str

def load_spec(
    id: str,
    cache: str,
    refresh: dt.timedelta | None = dt.timedelta(days=1),
):
    id = ALIASES.get(id, id)

    now = dt.datetime.now()
    ta = TypeAdapter(Spec)
    fp = Path(cache) / f"{id}_spec.json"

    if fp.exists():
        with fp.open('rb') as f:
            spec = ta.validate_json(f.read())
    else:
        spec = None

    if spec is not None:
        if refresh is None:
            return spec
        elif spec.refreshed > now - refresh:
            return spec
    
    s = get_series(id).data.items[0]
    spec = Spec(
        refreshed=now,
        id=s.id,
        title=s.title,
        observation_start=s.observation_start,
        observation_end=s.observation_end,
        frequency=s.frequency,
        frequency_short=s.frequency_short,
        units=s.units,
        units_short=s.units_short,
        seasonal_adjustment=s.seasonal_adjustment,
        seasonal_adjustment_short=s.seasonal_adjustment_short,
        last_updated=s.last_updated,
        popularity=s.popularity,
    )
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open('wb+') as f:
        f.write(ta.dump_json(spec))
    return spec

@dataclass(frozen=True)
class DataItem:
    date: dt.date
    value: float | None
    asof: dt.date | None = None

@dataclass(frozen=True)
class Data:
    refreshed: dt.datetime
    observation_start: dt.date
    observation_end: dt.date
    units: str
    output_type: int
    file_type: str
    order_by: str
    sort_order: str
    count: int
    offset: int
    limit: int

    values: list[DataItem]

def load_data(
    id: str,
    cache: str,
    refresh: dt.timedelta | None = dt.timedelta(days=1),
    asof: str | None = None,
    starting: dt.date = dt.date(1995, 1, 1)
):
    id = ALIASES.get(id, id)

    now = dt.datetime.now()
    ta = TypeAdapter(Data)

    f_name = f"{id}_data"
    if asof is not None:
        f_name = f"{f_name}_{asof}"

    fp = Path(cache) / f"{f_name}.json"
    
    if fp.exists():
        with fp.open('rb') as f:
            data = ta.validate_json(f.read())
    else:
        data = None

    if data is not None:
        if refresh is None:
            return data
        elif data.refreshed > now - refresh:
            return data
    
    if asof is None:
        obs = get_observations(id).data
        data = Data(
            refreshed=now,
            observation_start=obs.observation_start,
            observation_end=obs.observation_end,
            units=obs.units,
            output_type=obs.output_type,
            file_type=obs.file_type,
            order_by=obs.order_by,
            sort_order=obs.sort_order,
            count=obs.count,
            offset=obs.offset,
            limit=obs.limit,
            values=[
                DataItem(i.date, i.value)
                for i in obs.observations
            ],
        )
    else:
        spec = load_spec(
            id, cache=cache, refresh=refresh
        )
        assert starting is not None, (starting, asof)
        ending = dt.date.today()

        asof_dates = pl.select(
            pl.date_range(starting, ending)
            .alias("date")
        ).filter(
            cal.rebalance_mask(asof)
        ).get_column("date").to_list()

        asof_dates_data = [] if data is None else [
            dv.asof
            for dv in data.values
        ]
        asof_dates_req = [
            d for d in asof_dates
            if d not in asof_dates_data
        ]
        assert len(asof_dates_req), asof_dates_req

        values = [] if data is None else data.values

        acc: list[DataItem] = []

        for d in reversed(asof_dates_req):
            time.sleep(1/20)
            try:
                obs = get_observations(
                    id,
                    ending=d,
                    limit=cal.calc_sample_limit(
                        asof, spec.frequency_short.lower()
                    ),
                    # vintage_dates=asof_dates_req
                ).data
            except ValueError as e:
                if "No vintage dates exist" in str(e):
                    continue
                else:
                    raise e
            acc.extend([
                DataItem(
                    i.date,
                    i.value,
                    asof=d,
                )
                for i in reversed(obs.observations)
            ])

        values.extend(reversed(acc))

        data = Data(
            refreshed=now,
            observation_start=obs.observation_start,
            observation_end=obs.observation_end,
            units=obs.units,
            output_type=obs.output_type,
            file_type=obs.file_type,
            order_by=obs.order_by,
            sort_order=obs.sort_order,
            count=obs.count,
            offset=obs.offset,
            limit=obs.limit,
            values=values,
        )
    
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open('wb+') as f:
        f.write(ta.dump_json(data))

    return data

def load_time_series(
    id: str,
    cache: str,
    refresh: dt.timedelta | None = dt.timedelta(days=1),
    dense: bool = False,
    asof: str | None = None,
    lag: dt.timedelta | None = None,
) -> pl.DataFrame:
    """
    >>> load_time_series(
    ...     "TCU",
    ...     cache="./__local__/fred",
    ...     asof="3W",
    ... )
    """

    data = load_data(id, cache, refresh=refresh, asof=asof)
    values = data.values

    if asof is None:
        if lag is None:
            lag = dt.timedelta(days=0)
        d0 = values[-1].date
        d1 = values[0].date
        df = pl.DataFrame({
            "date": [d.date + lag for d in values],
            id: [d.value for d in values],
        }).reverse()
    else:
        df = pl.DataFrame({
            "date": [d.date for d in values],
            # id: [d.value for d in values],
            "asof": [d.asof for d in values],
        }).group_by(
            pl.col("date"),
        ).agg(
            pl.col("asof").first(),
            # pl.col(id).first(),
        )
        df_today = load_time_series(
            id,
            cache,
            refresh=refresh,
            dense=dense,
            lag=None,
            asof=None,
        )
        df = df.join(
            df_today,
            how="left",
            on="date",
            maintain_order="left_right",
            coalesce=True,
        ).group_by(
            pl.col("asof"),
        ).agg(
            pl.col(id).first(),
            pl.col("date").first()
        ).select(pl.col("asof").alias("date"), id)
        
        dates = df.get_column("date")

        if lag is not None:
            df_lag = load_time_series(
                id,
                cache,
                refresh=refresh,
                dense=dense,
                lag=lag,
                asof=None,
            )
            df_lag = df_lag.filter(
                pl.col("date") < dates[0]
            )
            df = pl.concat([df_lag, df]).sort(by="date")

        d0 = df.select(pl.col("date").min()).item()
        d1 = df.select(pl.col("date").max()).item()
    if dense:
        df = pl.select(
            pl.date_range(d0, d1).alias("date")
        ).join(
            df,
            on="date",
            how="left",
            coalesce=True,
        )
    return df

ALIASES = {
    "VIX": "VIXCLS",
    "IV_SPX_3M": "VXVCLS",
    "IV_GOLD": "GVZCLS",
    "IV_CRUDE": "OVXCLS",
    "IV_NASDAQ": "VXNCLS",
    "IV_EM": "VXEEMCLS",
    "IV_R2000": "RVXCLS",
    "IV_DJIA": "VXDCLS",

    # LOANS

    "US_OVERDUE_30D_90D": "QBPLNTLN3089DU", # overdue
    "US_OVERDUE_90D": "QBPLNTLNNACSTAT",
    "US_OVERDUE_NONACCRUAL": "QBPLNTLNNACSTAT",
    "US_OVERDUE_CHARGEOFF": "QBPLNTLNTCGOFF",
    "US_OVERDUE_NONCURRENT": "QBPLNTLNNCUR",

    # chargeoff rate also available
    # delinquency rate also available

    "US_CHAREOFF_TOTL": "NCOALLACB",
    "US_CHARGEOFF_CORP": "NCOALLCIACB",
    "US_CHARGEOFF_CONS": "NCOALLCACB",
    "US_CHARGEOFF_CONS_CC": "NCOALLCCACB",
    "US_CHARGEOFF_RE": "NCOALLSREACB",

    "US_LOANS_TOTL": "QBPBSTASTLN",
    "US_LOANS_CORP": "QBPBSTASCOMLN",
    "US_LOANS_RE": "QBPBSTASLNREAL",
    "uS_LOANS_RE_CONSTR": "QBPBSTASLNREALCONDEV",
    "US_LOANS_CONS": "QBPBSTASLNINDV",

    # US ECO

    # available by sector
    "US_CAP_UTIL": "TCU",
    "US_IP": "INDPRO",
    "US_IP_AUTO": "IPG3361T3S",
    "US_IP_MINE": "IPMINE",
    "US_IP_CONSTR_SUPP": "IPB54100S",
    "US_IP_MATRL": "IPMAT", # materials
    "US_IP_DEFN": "IPB52300S",

    "US_CU_OIL_GAS": "CAPUTLG211S",

    "US_UNEMPL": "UNRATE",
    "US_INFL_CONS": "FPCPITOTLZGUSA",

    "EU_UNEMPL": "LRHUTTTTEUM156S",
    "EU_INFL_TOTL": "CPHPTT01EZM659N", # yoy
    "EU_INFL_CONS": "FPCPITOTLZGEUU",

    "US_LOAN_LOSS_PROV": "QBPQYLNLOSS",

    # EUR yield

    "EUR_INTERBANK_1D": "IRSTCI01EZM156N",
    "EUR_INTERBANK_3M": "IR3TIB01EZM156N",

    "EUR_GOV_10Y": "IRLTLT01EZM156N",
    "EUR_DISCOUNT": "INTDSREZM193N",

    "JPN_GOV_10Y": "IRLTLT01JPM156N",
    "JPN_DISCOUNT": "INTDSRJPM193N",

    "GBP_GOV_10Y": "IRLTLT01GBM156N",
    "GBP_DISCOUNT": "INTDSREZM193N",
    "GBP_INTERBANK_3M": "IR3TIB01GBM156N",

    "GER_GOV_10Y": "IRLTLT01DEM156N",
    "ESP_GOV_10Y": "IRLTLT01ESM156N",
    "NET_GOV_10Y": "IRLTLT01NLM156N",
    "GRE_GOV_10Y": "IRLTLT01GRM156N",
    "ITA_GOV_10Y": "IRLTLT01ITM156N",
    "FRA_GOV_10Y": "IRLTLT01FRM156N",

    # US yields

    "US_FF": "DFF",

    "US_BE_10Y": "T10YIE",
    "US_BE_5Y": "T5YIE",

    "US_INFL_5Y5Y": "T5YIFR",

    "US_GOV_1M_CM": "DGS1MO",
    "US_GOV_3M_CM": "DGS3MO",
    "US_GOV_6M_CM": "DGS6MO",
    "US_GOV_1Y_CM": "DGS1",
    "US_GOV_2Y_CM": "DGS2",
    "US_GOV_3Y_CM": "DGS3",
    "US_GOV_5Y_CM": "DGS5",
    "US_GOV_7Y_CM": "DGS7",
    "US_GOV_10Y_CM": "DGS10",
    "US_GOV_20Y_CM": "DGS20",
    "US_GOV_30Y_CM": "DGS30", # "investmnet basis"
    # ^ also inflation indexed?

    "US_GOV_10Y": "BC_10YEAR", # const maturity
    # ^ M suffix?

    "US_AAA": "DAAA", # mmodys seasoned
    "US_BAA": "DBAA", # mmodys seasoned

    "US_ICE_IG": "BAMLC0A0CMEY",
    "US_ICE_HY": "BAMLH0A0HYM2EY",

    "US_ICE_AAA": "BAMLC0A1CAAAEY",
    "US_ICE_AA": "BAMLC0A2CAAEY",
    "US_ICE_A": "BAMLC0A3CAEY",
    "US_ICE_BBB": "BAMLC0A4CBBBEY",
    "US_ICE_BB": "BAMLH0A1HYBBEY",
    "US_ICE_B": "BAMLH0A2HYBEY",
    "US_ICE_CCC": "BAMLH0A3HYCEY",

    "EUR_HY": "BAMLHE00EHYIEY",

    "EM_HY": "BAMLEMHBHYCRPIEY",
    "ASIA_EM_IG": "BAMLEMRACRPIASIAEY",

    # US yield spreads

    "D_US_AAA": "AAA10Y",
    "D_US_BAA": "BAA10Y",
    "D_US_AAA_GOV_10Y": "AAA10Y",
    "D_US_BAA_GOV_10Y": "BAA10Y",

    "D_US_BAA_FF": "BAAFF", # baa seasoned to fed funds
    "D_US_AAA_FF": "AAAFF", # baa seasoned to fed funds

    "US_HY_OAS": "BAMLH0A0HYM2",
    "US_IG_OAS": "BAMLC0A0CM",

    "US_AAA_OAS": "BAMLC0A1CAAA",
    "US_AA_OAS": "BAMLC0A2CAA",
    "US_A_OAS": "BAMLC0A3CA",
    "US_BBB_OAS": "BAMLH0A2HYB",
    "US_BB_OAS": "BAMLH0A1HYBB",
    "US_B_OAS": "BAMLH0A2HYB",
    "US_CCC_OAS": "BAMLH0A3HYC",

    "EUR_HY_OAS": "BAMLHE00EHYIOAS",

    "EM_IG_OAS": "BAMLEMHBHYCRPIEY",
    "EM_HY_OAS": "BAMLEMHBHYCRPIOAS",

    "ASIA_EM_IG_OAS": "BAMLEMRACRPIASIAOAS",
    "LATAM_EM_IG_OAS": "BAMLEMRLCRPILAOAS",

    # USD INDICES

    "NOML_USD": "TWEXBGSMTH", # nominal broad
    "REAL_USD": "RTWEXBGS", # broad
    "NOML_USD_ADV": "TWEXAFEGSMTH", # nominal advanced economies
    "REAL_USD_ADV": "RTWEXAFEGS",
    "NOML_USD_EMG": "TWEXEMEGSMTH", # nominal EM
    "REAL_USD_EMG": "RTWEXEMEGS",
}