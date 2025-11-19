
import os
import sys
sys.path.append("./src")

import polars as pl
import streamlit as st

import ceg
from ceg.apis.st import cache

os.environ["FRD_DATA"] = "./__local__/frd"

t = ceg.apis.t
objs = ceg.objs
fut = objs.dev.fut
ui = ceg.app.dev.academic.ui
vis = ceg.vis
apis = ceg.apis

def drop_zero(
    df: pl.DataFrame,
    col: str,
    drop_nan: bool = True,
    drop_null: bool = True,
    dp: int | None = 3,
) -> pl.DataFrame:
    expr = pl.col(col)
    if drop_nan:
        expr = expr.fill_nan(0)
    if drop_null:
        expr = expr.fill_null(0)
    if dp is not None:
        expr = expr.round(dp)
    return df.filter(expr != 0)

def collect_obj_domains(
    domains: dict[str, ceg.Domain]
):
    obj_domains: dict[str, str] = {}

    for d in domains.values():
        for _, o in d.objects.items():
            obj_domains[o.path] = d.root.path
    
    return obj_domains

DOMAINS = {
    d.root.path: d
    for d in [
        fut.d.bars.ord.FutDailyOrdinalDomain.new(),
        fut.d.bars.rx.FutDailyRxDomain.new(),
    ]
}
OBJ_DOMAINS = collect_obj_domains(DOMAINS)
    

@cache()
def obj_to_daily_ts(
    **kwargs: str
) -> pl.DataFrame:
    assert "obj" in kwargs, kwargs
    obj = kwargs.pop("obj")
    if obj is None:
        raise ValueError(kwargs)
    domain = OBJ_DOMAINS.get(obj)
    assert domain is not None, obj

    d = DOMAINS.get(domain)
    assert d is not None, (domain, obj)

    o = d.get(obj, t = None)
    d = d.sync()
    val_kv = o.value_key_val(d)
    snap_kv = o.snap_key_val(d)
    v = val_kv.get(
        o.path,
        apis.kv.BODY # type: ignore
        # TODO: get_concat for chunked
    )

    snap = snap_kv.get(o.path)
    assert snap is not None, (domain, obj)
    assert v is not None, (domain, obj)
    assert isinstance(v, ceg.ValuePlSeries), (
        domain, obj, v
    )
    d0, d1 = snap.first, snap.last # type: ignore
    df = pl.DataFrame([
        v.ser.alias("value"),
    ]).with_columns(
        pl.lit(o.path).alias("key"),
        pl.date_range(d0, d1).alias("date")
    )
    
    params = kwargs.pop("params")
    # TODO: if params is not None (eg. transform, etc.)

    for k in ["group"]:
        if k in kwargs:
            df = df.with_columns(pl.lit(kwargs[k]).alias(k))
    return df

if __name__ == "__main__":
    app = vis.apps.App("dev", "./__local__/cache")
    ui.apps.app({
        "home": {
            "pages": app.list_pages(),
        },
        "pages": app.get_pages(
            obj_to_daily_ts
        ),
    }, init_elements=app.init_page).run()
