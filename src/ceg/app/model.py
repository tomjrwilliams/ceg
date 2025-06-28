from typing import cast, Iterable, Callable, get_type_hints, get_args, get_origin, Type, NamedTuple, Union, Optional, Any, Annotated
import types
import datetime as dt
import math

from dataclasses import dataclass

import numpy as np
import polars as pl
import streamlit as st


from frozendict import frozendict

from ceg.app.nav import Shared, Page
from ceg.app.page import DataFrame, Transformation, Dynamic

import ceg

#  ------------------

TYPES = {
    "str": str,
    "bool": bool,
    "int": int,
    "float": float,
}

def repr_type(t: Type):
    s = str(t)
    if s.endswith(".args"):
        return "*args"
    elif s.endswith(".kwargs"):
        return "**kwargs"
    try:
        s = s.split("<class '")[1].split("'>")[0]
    except:
        return s
    s = s.replace("datetime.", "")
    s = (
        s.replace("_D0_", "_")
        .replace("_D1_", "_Vec_")
        .replace("_D2_", "_Mtx_")
    )
    return s.split(".")[-1].replace("Ref_", "").strip()

class Annot(NamedTuple):
    t: Type | tuple[Type, ...]
    optional: bool
    union: bool
    annots: tuple[ceg.define.Annotation]

    def repr(self):
        if self.union:
            s = "|".join([repr_type(t) for t in self.t])
        else:
            s = repr_type(self.t)
        if self.optional:
            return f"{s}|N"
        return s

class Signature(NamedTuple):
    annots: frozendict[str, Annot]
    depth: int
    f: Callable

    # TODO: allow explicitly pass
    # and in construction allow masking certain kwargs
    # eg. field for daily_close, bar etc.

Universe = frozendict[str, ceg.Node.Any | Callable]
Signatures = frozendict[str, Signature]

def signatures(
    universe: Universe
) -> Signatures:
    # TODO: drop keep? that's purely representational
    res = cast(Signatures, frozendict())
    for key, f in universe.items():
        depth = 0
        sig = {}
        if isinstance(f, tuple):
            f, r = f
            hints = get_type_hints(r)
            hints_extra = get_type_hints(r, include_extras=True)
        else:
            hints = get_type_hints(f)
            hints_extra = get_type_hints(f, include_extras=True)
        returns = hints.pop("return", None)
        keep = hints.pop("keep", None)
        annots = cast(tuple[ceg.define.Annotation], ())
        for k, h in hints.items():
            extra = hints_extra[k]
            origin = get_origin(h)
            if get_origin(extra) is Annotated:
                args = get_args(extra)
                origin = args[0]
                annots = args[1:]
            if (
                origin is types.UnionType
                or origin is Union
            ):
                args = get_args(h)
                if type(None) in args:
                    args = tuple((
                        a for a in args if a is not type(None)
                    ))
                    if len(args) == 1:
                        sig[k] = Annot(
                            args[0],
                            optional=True,
                            union=False,
                            annots=annots
                        )
                    else:
                        sig[k] = Annot(
                            args,
                            optional=True,
                            union=True,
                            annots=annots
                        )
                else:
                    sig[k] = Annot(
                        args,
                        optional=False,
                        union=True,
                        annots=annots
                    )
            else:
                sig[k] = Annot(h, False, False, annots)
        res = res.set(key, Signature(
            cast(frozendict[str, Annot], frozendict(sig)),
            depth,
            f
        ))
    return cast(Signatures, res)

def signatures_df(sigs: Signatures) -> pl.DataFrame:
    return pl.DataFrame([
        {
            **{"func": k},
            **{
                f"kw-{i}": f"{k}:{annot.repr()}"
                for i, (k, annot)
                in enumerate(sig.annots.items())
                if not any([
                    ann.type == "internal"
                    for ann in annot.annots
                ])
            }
        }
        for k, sig in sigs.items()
    ])

def empty_cell(cell):
    if cell is None:
        return True
    if isinstance(cell, str):
        return len(cell) == 0
    return False

def empty_row(
    row: dict, 
    *keys: str, 
    method: Callable[[Iterable[bool]], bool] = any
):
    return method([empty_cell(row[k]) for k in keys])

def label_padding(df: pl.DataFrame):
    return (
        0 if not len(df) else 2 + int(math.log(len(df), 10))
    )

def parse_date(v: str | int):
    if isinstance(v, int):
        if v < 10000:
            return dt.date(v, 1, 1)
        else:
            assert v > 1000000, v
            v = str(v)
    if v.isnumeric() and len(v) == 4:
        y = int(v)
        return dt.date(y, 1, 1)
    elif v.isnumeric() and len(v) == 8:
        y = int(v[:4])
        m = int(v[4:6])
        d = int(v[6:])
        return dt.date(y, m, d)
    vs = v.split("-")
    if len(vs) == 2:
        y, m = v
        return dt.date(int(y), int(m), 1)
    elif len(vs) == 3:
        y, m, d = v
        return dt.date(int(y), int(m), int(d))
    else:
        raise ValueError(v)

def parse_kwarg(
    i: int,
    kw: str, 
    refs: dict[str, ceg.Ref.Any],
    annots: frozendict[str, Annot],
    ks: list[str]
):
    if "=" in kw:
        k, v = kw.split("=")
    else:
        k = ks[i]
        v = kw
    if ":" in k:
        k, t = k.split(":")
        if t == "ref":
            v = refs[v]
        elif t == "date":
            v = parse_date(v)
        else:
            v = TYPES[t](v)
    elif k in annots:
        t = annots[k]
        if t.union:
            for tt in t.t:
                try:
                    if tt is dt.date:
                        v = parse_date(v)
                    else:
                        v = tt(v)
                    return k, v
                except:
                    pass
        else:
            assert isinstance(t.t, type), t
            try:
                if t.t is dt.date:
                    v = parse_date(v)
                else:
                    v = t.t(v)
            except:
                raise ValueError(v)
    elif v.isnumeric():
        v = float(v)
    return k, v

def kwargs_ready(
    vs: list[str | None]
) -> tuple[bool, list[str]]:
    try:
        end = vs.index(".")
        return True, cast(list[str], [
            v for v in vs[:end] if v is not None
        ])
    except:
        if all([v is not None for v in vs]):
            return True, cast(list[str], vs)
        return False, []

def parse_kwargs(
    row: dict[str, str | None],
    refs: dict[str, ceg.Ref.Any],
    sig: Signature
):
    # TODO: pass in the type so we can infer from sig
    vs = [v for k, v in row.items() if k.startswith("kw-")]
    ready, vs = kwargs_ready(vs)
    if not ready:
        return {}
    ks = list(sig.annots.keys())
    return dict((
        parse_kwarg(1+i, v, refs, sig.annots, ks)
        for i, v in enumerate(vs)
    ))

#  ------------------

def rows_to_refs(
    g: ceg.Graph,
    df: pl.DataFrame,
    refs: dict[str, ceg.Ref.Any],
    universe: Universe,
    sigs: Signatures,
    keep: dict[str, int] = {}
):
    """
    >>> steps = 100
    >>> fs = ceg.fs
    >>> data = ceg.data
    >>> app = ceg.app
    >>> universe = cast(app.model.Universe, frozendict({
    ...     "days": fs.dates.daily.loop,
    ...     "close": data.bars.daily_close.bind
    ... }))
    >>> sigs = signatures(universe)
    >>> df = pl.DataFrame({
    ...     "label": ["date", "ES"],
    ...     "I": [True, False],
    ...     "func": ["days", "close"],
    ...     "kw-0": ["start=2024", "d:ref=date"],
    ...     "kw-1": ["end=2025", "product=FUT"],
    ...     "kw-2": [".", "symbol=ES"],
    ...     "kw=3": [None, "."]
    ... })
    >>> g, refs, es = ceg.Graph.new().pipe(
    ...     rows_to_refs, 
    ...     df, 
    ...     {}, 
    ...     universe, 
    ...     sigs, 
    ...     keep = {"date": 100, "ES": 100}
    ... )
    >>> g, es, ts = ceg.batches(
    ...     g, *es, n = steps, g = len(refs)
    ... )
    >>> res = pl.DataFrame({
    ...     label: refs[label].history(g)
    ...     .last_n_before(steps, ts[-1])
    ...     for label in ["date", "ES"]
    ... })
    """
    es = []
    for i, row in enumerate(df.iter_rows(named=True)):
        if empty_row(row, "label", "func"):
            continue

        init = row.pop("I")

        func_name = row["func"]
        if func_name not in universe:
            raise ValueError(f"Not found: {func_name}")

        label = row["label"]

        sig = sigs[func_name]
        func = sig.f

        kwargs = parse_kwargs(row, refs, sig)

        if not len(kwargs):
            continue

        if label in keep:
            kwargs["keep"] = keep[label]
        
        if sig.depth:
            res = func(g)(**kwargs) # type: ignore
        else:
            res = func(g, **kwargs) # type: ignore

        g, r = cast(tuple[ceg.Graph, ceg.Ref.Any], res)

        # i += offset
        # label = f"{str(i).rjust(pad, '0')}-{row['label']}"
        refs[label] = r

        if init:
            es.append(ceg.Event.zero(r))

    return g, refs, es

import plotly
import plotly.express

def nan_map(e: pl.Expr, v: pl.Expr):
    return pl.when(e.is_null()).then(None).otherwise(v)

def cumsum(e, trim: int| str | None = None, LEN: int | None = None):
    if trim:
        e = pl.when(pl.int_range(0, LEN) < int(trim)).then(None).otherwise(e)
    return e.fill_nan(None).pipe(
        nan_map, e.fill_nan(0).fill_null(0).cum_sum()
    )

EXPR_MAP: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    "cumsum": cumsum
}
def expr_params(e):
    if ":" not in e:
        return {}
    return dict((kv.split("=") for kv in e.split(":")[1].split(",")))

def df_to_line_plot(
    df: pl.DataFrame,
    g: ceg.Graph,
    refs: dict[str, ceg.Ref.Any],
    t: float,
    id: str | None = None,
):
    steps = None
    for label in df.get_column("label"):
        ref = refs[label]
        steps = ref.history(g).mut.occupied

    assert steps is not None, steps
    
    data = pl.DataFrame({
        label: (
            refs[label].history(g)
            # , **(
            #     {} if slot is None else dict(slot=slot)
            # ))
            .last_n_before(steps, t)
        )
        for label, slot in zip(
            df.get_column("label"),
            df.get_column("slot")
        )
    })

    df_exprs = df.filter(
        pl.col("expr").is_not_null()
    )

    exprs = {
        label: EXPR_MAP[
            e.split(":")[0]
        ](pl.col(label), **expr_params(e), LEN=len(data)) for label, e in zip(
            df_exprs.get_column("label"),
            df_exprs.get_column("expr"),
        )
    }

    # TODO: optionally take transform col 
    # simple one col polars expr, or can be easily cast as such eg. log(2)

    x_label = df.filter(pl.col("x")).get_column("label")

    if len(x_label) == 1:
        x = x_label[0]

        x_min = data.select(pl.col(x).min()).item()
        x_max = data.select(pl.col(x).max()).item()
        
        slider_key = f"date-slider"

        if slider_key not in st.session_state:
            x_l, x_r = st.slider(
                f"{x}:",
                min_value=x_min,
                max_value=x_max,
                value=(x_min, x_max),
                key=slider_key
            )
        else:
            x_l, x_r = st.session_state[slider_key]

        plot = plotly.express.line(
            data.filter(
                (pl.col(x) >= x_l) & (pl.col(x) <= x_r)
            ).with_columns(**exprs), x=x, y = [
                k for k in data.schema.keys() if k not in x_label
            ]
        )
        st.plotly_chart(plot, key = id)
    elif not len(x_label):
        plot = plotly.express.line(data, y = list(data.schema.keys()))
        st.plotly_chart(plot, key = id)
    else:
        raise ValueError(df)

#  ------------------

class ModelKW(NamedTuple):
    name: str
    shared: Shared
    dfs: tuple[DataFrame, ...] = ()
    tfs: tuple[Transformation, ...] = ()

    dfs_data: dict[str, pl.DataFrame] = {}

    universe: Universe = cast(Universe, frozendict())
    signatures: Signatures = cast(Signatures, frozendict())
    
    def with_universe(self, universe: pl.DataFrame):
        return self._replace(
            dfs=self.dfs + (
                DataFrame.new(
                    self.name, 
                    "universe", 
                    data=universe, 
                    label="available universe"
                ),
            )
        )

    def with_functions(self, universe: Universe):
        sigs = signatures(universe)
        sig_df = signatures_df(sigs)
        return self._replace(
            universe=universe,
            signatures=sigs,
            dfs=self.dfs + (
                DataFrame.new(
                    self.name, 
                    "sigs", 
                    data=sig_df, 
                    label="available functions"
                ),
            )
        )

    def with_model(
        self, init: pl.DataFrame | list[dict] | None=None,
    ):
        i_schema = None
        for i, df in enumerate(self.dfs):
            if df.name == "sigs":
                i_schema = i
        if i_schema is None:
            raise ValueError(self)

        # TODO: make all the meta keys upper case so no conflicts

        empty = pl.DataFrame(schema={
            "label": pl.String,
            "I": pl.Boolean,
            **self.dfs[i_schema].schema,
            # TODO: or signatures explicitly?
        })
        k_last = list(empty.schema.keys())[-1]
        c_last = pl.col(k_last)
        if init is None:
            init = empty
        elif isinstance(init, pl.DataFrame):
            assert all([
                k in init for k in ("label", "func")
            ]), init.schema
            init = init.select(
                pl.col(k) 
                if k in init.schema 
                else pl.lit(None).alias(k).cast(empty.schema[k])
                for k in empty.schema.keys()
            )
        elif isinstance(init, list):
            init = pl.DataFrame([
                {
                    "label": r["label"],
                    "I": r.get("I", None),
                    "func": r["func"],
                    **{
                        f"kw-{i}": v
                        for i, v in enumerate(list(r.values())[(
                            2 if "I" not in r else 3
                        ):])
                    }
                }
                for r in init
            ], schema=empty.schema)
        else:
            raise ValueError(init)

        init = init.with_columns(
            pl.when(c_last.is_null())
            .then(pl.lit("."))
            .otherwise(c_last)
            .alias(k_last)
        )

        df = DataFrame.new(
            self.name, "model", data=init, editable=True, label="model"
        )
        return self._replace(
            dfs=self.dfs + (df,),
            tfs = self.tfs + (RunGraph(),)
            # TODO: callback
        )

    def with_plot(self, init: pl.DataFrame | list[dict] | None=None, name: str = "plot"):
        empty = pl.DataFrame(schema={
            "label": pl.String,
            "x": pl.Boolean,
            "y": pl.Boolean,
            "y2": pl.Boolean,
            "align": pl.String,
            "expr": pl.String,
            "slot": pl.Int32,
        })
        if init is None:
            init = empty
        elif isinstance(init, pl.DataFrame):
            pass
        elif isinstance(init, list):
            init = pl.DataFrame(init, schema=empty.schema)
        df = DataFrame.new(
            self.name, name, data=init, editable=True, label=name
        )
        return self._replace(
            dfs=self.dfs + (df,),
            tfs=self.tfs + (AddPlot(name=name),)
        )

class Model(ModelKW, Dynamic):
    pass

# TODO: add param objects

# and then allow kwargs passed as d:param=date
# as well as ref

# where the loading is up to you, after dfs

# eg. so you can toggle them right next to graph

#  ------------------

class RunGraph(Transformation):
    
    def apply(
        self,
        page: Model,
        g: ceg.Graph,
        refs: dict[str, ceg.Ref.Any],
        es: list[ceg.Event],
        dfs: dict[str, pl.DataFrame],
        shared: frozendict[str, Any],
    ):
        plots = [tf for tf in page.tfs if isinstance(tf, AddPlot)]
        df_plot = pl.concat([
            dfs[tf.name] for tf in plots
        ])

        align_label = (pl.col("align") == pl.lit("")) | pl.col("align").is_null()

        df_align = df_plot.filter(align_label.not_())
        df_keep = df_plot.filter(align_label)

        keep_labels = df_keep.get_column("label")
        align_labels = df_align.get_column("label")

        g, refs, es = g.pipe(
            rows_to_refs, 
            dfs["model"],
            refs=refs,
            universe=page.universe,
            sigs=page.signatures,
            keep={
                **{label: True for label in keep_labels},
                **{label: 4 for label in align_labels},
            }
        )

        aligned = {}

        for label, align, slot in zip(
            df_align.get_column("label"),
            df_align.get_column("align"),
            df_align.get_column("slot"),
        ):
            ref = cast(ceg.Ref.Scalar_F64, refs[label])
            ref_align = refs[align]

            if slot is not None:
                ref = ref._replace(slot=slot)

            g, ref = g.bind(
                ceg.fs.align.scalar_f64.new(ref, ref_align),
                keep=True,
                when=ceg.Ready.ref(ref_align)
            )
            
            aligned[label] = ref

        refs = {**refs, **aligned}

        # TODO: for keep, iter tfs for add_plot, get name
        # combine all labels in plot dfs
        # according to relevant plot keep requirements

        if not len(refs) or not len(es) or len(df_plot) < 2:
            return g, refs, es, shared

        e = es[-1]
        for g, e, t in ceg.steps(
            g, *es, n = int(10e6), iter=True
        )():
            continue

        return g, refs, [e], shared

@dataclass(frozen=True)
class AddPlot(Transformation):
    name: str

    # TODO: take plot name and type (or type as a user selection?)

    def apply(
        self,
        page: Model,
        g: ceg.Graph,
        refs: dict[str, ceg.Ref.Any],
        es: list[ceg.Event],
        dfs: dict[str, pl.DataFrame],
        shared: frozendict[str, Any],
    ):

        df_plot = dfs[self.name]
        df_plot.filter(
            pl.col("label").is_not_null()
        )

        if not len(refs) or not len(es) or len(df_plot) < 2:
            return g, refs, es, shared
        
        df_plot.pipe(
            df_to_line_plot,
            g,
            refs,
            t=es[-1].t,
            id=f"{page.name}.{self.name}"
        )
        return g, refs, es, shared

#  ------------------
