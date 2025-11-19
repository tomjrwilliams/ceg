from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Callable,
)
import polars as pl
import streamlit as st

from ..engine import *
from ..objs import *
from .. import vis
from .dev.academic import ui

from .. import apis
from ..apis.st import cache

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

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
    domains: dict[str, Domain]
):
    obj_domains: dict[str, str] = {}

    for d in domains.values():
        for _, o in d.objects.items():
            obj_domains[o.path] = d.root.path
    
    return obj_domains

@dataclass(frozen=True)
class AppDomain:
    domains: list[Domain]
    domain_sync: dict[str, bool]

    def obj_domain(self, obj: str):
        for d in self.domains:
            if obj in d.objects:
                if not self.domain_sync.get(d.root.path, False):
                    d = d.sync()
                    self.domain_sync[d.root.path] = True
                return d
        raise ValueError((obj))

def prepare_app(
    name: str,
    root_fp: str,
    domain_funcs: dict[
        str, Callable[[dict], AppDomain]
    ]
):
    DOMAINS: dict[str, AppDomain] = {}
    DOMAIN_SYNC = {}

    @cache()
    def obj_to_daily_ts(
        domain: str | None, **kwargs: str
    ) -> pl.DataFrame | None:
        if domain is None or not len(domain):
            return None

        assert "obj" in kwargs, kwargs
        obj = kwargs.pop("obj")

        if obj is None:
            raise ValueError(kwargs)
        
        if domain in DOMAINS:
            dom = DOMAINS[domain]
        else:
            dom = domain_funcs[domain](DOMAIN_SYNC)
            DOMAINS[domain] = dom

        d = dom.obj_domain(obj)
        assert d is not None, obj

        o = d.get(obj, t = None)
        
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
        assert isinstance(v, ValuePlSeries), (
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
    
    @dataclass(frozen=True)
    class ExploreDailyObjsFromRows(
        vis.ExploreDailyFromRows
    ):

        def gen_domain(self):
            return vis.CachedText(
                "domain",
                f"{self.cache}/domain.json",
                "domain:",
            )

        def pre_render(self):
            domain = self.gen_domain().render()
            d = domain.data.get()
            if d is None or not len(d):
                return
            
            if d in DOMAINS:
                dom = DOMAINS[d]
            else:
                dom = domain_funcs[d](DOMAIN_SYNC)
                DOMAINS[d] = dom

            with st.expander("objects:"):
                objs, types = zip(*set(sum([
                    [
                        (o, type(obj).__name__)
                        for o, obj in domain.objects.items()
                    ]
                    for domain in dom.domains
                ], [])))
                st.table(pl.DataFrame({
                    "type": list(types),
                    "object": list(objs)
                    # types?
                }).sort(by=["type", "object"]))

                # TODO: on click, edit the objects table with the item from the objects list?
                # or at least an add to objects button?

        def row_to_df(self, r: dict) -> pl.DataFrame | None:
            domain = self.gen_domain().data.get()
            return obj_to_daily_ts(domain, **r)

        def row_to_alias(self, r: dict) -> str:
            return (
                r["label"] 
                if r["label"] is not None
                else r["obj"]
            )

    app = vis.apps.App(name, root_fp)
    return ui.apps.app(
        [app.list_pages()]
        + list(
            app.get_pages(
                daily_cls=ExploreDailyObjsFromRows,
                # ...
            ).values()
        ),
        init_elements=app.init_page
    )
