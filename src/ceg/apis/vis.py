from typing import Callable, Any
import inspect
import json
from pathlib import Path
from contextlib import contextmanager
from datetime import date
import plotly
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pandas as pd
import ceg

from frozendict import frozendict
from functools import partial

# utils

# 

@contextmanager
def empty_context():
    yield

TYPE_NAMES = {
    t.__name__: t
    for t in [
        bool,
        int,
        float,
        str,
        date,
    ]
}

def render_param(
    acc: frozendict,
    name: str,
    k: str,
    v: str,
    default: Any | None = None,
):
    st = ceg.apis.st.library()
    if v.startswith("param:"):
        try:
            return acc, acc[v]
        except Exception as e:
            raise ValueError(acc, v)
    if TYPE_NAMES[v] is bool:
        param_id = f"param:{name}.{k}"
        p = st.checkbox(k, value=default, key=param_id)
        return acc.set(param_id, p), p
    elif TYPE_NAMES[v] is int:
        param_id = f"param:{name}.{k}"
        p = st.number_input(k, value=default, key=param_id)
        return acc.set(param_id, p), p
    elif TYPE_NAMES[v] is float:
        param_id = f"param:{name}.{k}"
        p = st.number_input(k, value=default, key=param_id)
        return acc.set(param_id, p), p
    elif TYPE_NAMES[v] is str:
        param_id = f"param:{name}.{k}"
        p = st.text_input(k, value=default, key=param_id)
        return acc.set(param_id, p), p
    elif TYPE_NAMES[v] is date:
        param_id = f"param:{name}.{k}"
        p = st.slider(
            k, value=(
                default if default is not None
                else date(2000, 1, 1)
                if k == "d0"
                else date.today()
                if k == "d1"
                else None
            ), key=param_id,
            min_value= date(2000, 1, 1),
            max_value = date.today(),
        )
        return acc.set(param_id, p), p
    else:
        raise ValueError(k, v)

def apply_pipe(
    df: pl.DataFrame,
    acc: frozendict,
    name: str,
    pipe: str | None = None,
    params: dict | None = None,
    kwargs: dict | None = None,
):
    st = ceg.apis.st.library()
    if pipe is None:
        return acc, df
    
    f = ceg
    for i, seg in enumerate(pipe.split(".")):
        if i == 0:
            assert seg == "ceg", pipe
        else:
            f = getattr(f, seg)

    if kwargs is None:
        kwargs = {}
    
    if params is not None:
        rendered = any([
            isinstance(t, type) for k, t in params.items()
        ])
        with (
            st.container(horizontal=True)
            if rendered
            else empty_context()
        ):
            for k, t in params.items():
                acc, kwargs[k] = render_param(
                    acc, name, k, t, default = kwargs.get(k, None)
                )

    return acc, f(df, **kwargs)


def render_graph(
    config: dict,
    acc: frozendict,
    f: str,
    v: bool = False,
    pipe: str | None = None,
    params: dict | None = None,
    kwargs: dict | None = None,
):
    st = ceg.apis.st.library()
    fp: str = config["fp"]

    group = config.get("group", "group")
    
    x_col = config.get("x", "date")

    # TODO: eneds fixing if you want float vs float scatter
    # eg. proper .get defualts for color, group, etc.
    not_float = ["date", group, "variable", x_col]

    if fp.startswith("data:"):
        df = acc[fp]
    else:
        df = pl.read_parquet(fp)

    acc, df = apply_pipe(df, acc, config.get("name"), pipe, params, kwargs)
    assert isinstance(df, pl.DataFrame), config

    df = df.with_columns(
        pl.all()
        .exclude(*not_float)
        .cast(pl.Float64)
    )
    if config.get("cum"):
        df = pl.read_parquet(fp).with_columns(
            pl.all()
            .exclude(*not_float)
            .cast(pl.Float64)
            .cum_sum()
        )
    x, y = "x", "y"
    if v and f != "bar":
        x, y = y, x
    elif not v and f == "bar":
        x, y = y, x
        
    if config.get("tall"):
        if group in df:
            kw = {
                x: x_col,
                y: "value", 
                "color": group,
            }
            if f != "bar":
                kw["line_group"]= "variable"
            else:
                kw["barmode"] = "group"
        else:
            kw = {x: x_col, y: "value", "color": "variable"}
    else:
        kw = {
            x: x_col,
            y: [
                k for k in df.columns if k != x_col
            ]
        }
    for k in [
        "title",
        "groupnorm",
    ]:
        if k in config:
            kw[k] = config[k]
    fig = getattr(px, f)(df, **kw)
    if v and f != "bar":
        fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, height=config.get(
        "height", 
        750 if not v else (
            4000 if f != "bar" else 750
        )
    ))
    return acc


def render_scatter(config: dict, acc,v: bool = False):
    return render_graph(config, acc,"scatter", v=v)

def render_line(config: dict, acc,v: bool = False):
    return render_graph(config, acc,"line", v=v)

def render_area(config: dict, acc,v: bool = False):
    return render_graph(config,acc, "area", v=v)

def render_bar(config: dict, acc,v: bool = False):
    return render_graph(config,acc, "bar", v=v)


def render_obj(config: dict, acc):
    type = config["type"]
    try:
        if type == "df":
            acc = load_df(config, acc)
        elif type == "tf":
            acc = transform_df(config, acc)
        elif type == "transition_matrix":
            acc = render_transition_matrix(config, acc)
        elif type == "transition_corr":
            acc = render_transition_corr(config, acc)
        elif type == "transition_tree":
            acc = render_transition_tree(config, acc)
        elif type == "state_codes_w_y":
            acc = render_state_codes_w_y(config, acc)
        elif type == "conditional_dist_w_fwd":
            acc = render_conditional_dist_w_fwd(config, acc)
        elif type == "conditional_beta_w_vol":
            acc = render_conditional_beta_w_vol(config, acc)
        elif type == "scatter":
            acc = render_scatter(config, acc)
        elif type == "vscatter":
            acc = render_scatter(config, acc, v = True)
        elif type == "line":
            acc = render_line(config, acc)
        elif type == "vline":
            acc = render_line(config, acc, v = True)
        elif type == "area":
            acc = render_area(config, acc)
        elif type == "varea":
            acc = render_area(config, acc, v = True)
        elif type == "bar":
            acc = render_bar(config, acc, v = True)
        elif type == "hbar":
            acc = render_bar(config, acc, v = False)
        else:
            raise ValueError(config)
    except Exception as e:
        raise ValueError(config)
    return acc
    


import json
from pathlib import Path

def next_config(app: Path | frozendict, i: int | None = None, name: str | None = None):
    if isinstance(app, dict):
        i = app["i"]
        if name is None:
            raise ValueError(app)
        if app["path"].exists():
            n = 0
            for subdir in app["path"].iterdir():
                if not subdir.is_dir():
                    continue
                elif subdir.name.startswith("_"):
                    continue
                elif subdir.name.split("-", maxsplit=1)[1] == name:
                    n = int(subdir.name.split("-")[0])
                    break
                else:
                    n += 1
        else:
            n = 0
        app = app["path"] / f"{n}-{name}"
    if not app.exists():
        app.mkdir(exist_ok=True, parents=True)
    if i is None:
        i = 0
        config_fp = app / f"config_{i}.json"
        while config_fp.exists():
            i += 1
            config_fp = app / f"config_{i}.json"
    config_fp = app / f"config_{i}.json"
    return i, config_fp

def write_config(
    config: dict,
    config_fp: Path
):
    with config_fp.open("w+") as f:
        json.dump(config, f)

def encode_pipe_params(
    d: dict,
    name: str,
):
    kvs= {
        k: (
            v.__name__ if isinstance(v, type)
            else v
            if isinstance(v, str)
            and v.startswith("param:")
            else None
        ) for k, v in d.items() 
    }
    return {k: v for k, v in kvs.items() if v is not None}

def encode_pipe_kwargs(
    d: dict
):
    return {
        k: v for k, v in d.items() if not isinstance(v, type)
        and (
            not isinstance(v, str)
            or (isinstance(v, str) and not v.startswith("param:"))
        )
    }

def encode_pipe(
    pipe: Callable | str | None = None,
    params: dict[str, Any | type] | None = None,
    name: str | None = None,
    **kwargs,
):
    if pipe is None:
        return {}
    elif isinstance(pipe, str):
        return {
            "pipe": pipe,
            "params": encode_pipe_params(params, name),
            "kwargs": encode_pipe_kwargs(params | kwargs),
        }
    else:
        path = inspect.getabsfile(pipe).split("/")
        for i, s in enumerate(path):
            if s == "ceg":
                pipe = ".".join(path[i:])
                break
        assert isinstance(pipe, str), (pipe, path)

        return {
            "pipe": pipe,
            "params": encode_pipe_params(params, name),
            "kwargs": encode_pipe_kwargs(params | kwargs),
        }

def render_transition_matrix(
    config: dict,
    acc: frozendict,
):
    st = ceg.apis.st.library()
    ROW_HEIGHT = (500 / 8)

    dfs: list[pl.DataFrame] = []

    objs: list[str] = config["objs"]

    symbols = [ceg.apis.states.extract_symbol(k) for k in objs]
    _, sign_codes = ceg.apis.states.sign_code_labels(objs)

    data = ceg.assert_is(acc[config["data"]], pl.DataFrame)
    
    for smoothed in [False, True]:
        df = data.filter(
            pl.col("smoothed") == smoothed
        ).pivot(on="sign_code", values="state", index="date").filter(
            pl.any_horizontal(*(
                pl.col(code).is_not_null()
                for code in sign_codes
            ))
        )

        df_ts = df.select(
            "date",
            *(
                pl.when(pl.col(sign_code))
                .then(pl.lit(i))
                .otherwise(None)
                .alias(sign_code)
                for i, sign_code in enumerate(sign_codes)
            )
        ).unpivot(
            index="date",
            variable_name="state",
            value_name="active"
        )
        fig = px.line(
            df_ts,
            x="date",
            y="active",
            color="state"
        )
        fig.update_layout(margin=dict(t=25, b=0))
        st.plotly_chart(fig, height=config.get(
            "height", 
            250, 
        ), width="stretch",
        key =f"{smoothed} state ts")


        counts = df.select(*(
            pl.col(code_i)
            .eq(True)
            .and_(pl.col(code_o).shift(-1).eq(True))
            .sum()
            .alias(f"{code_i}|{code_o}")
            for code_i in sign_codes
            for code_o in sign_codes
        )).unpivot(
            variable_name="code_pair",
            value_name="count"
        ).with_columns(
            pl.col("code_pair")
            .str.split(by=pl.lit("|"))
            .list.get(0)
            .alias("code_i"),

            pl.col("code_pair")
            .str.split(by=pl.lit("|"))
            .list.get(1)
            .alias("code_o"),
        )

        count_totals = counts.with_columns(pl.lit(True).alias("g")).pivot(
            on="code_i", index="g", values="count", aggregate_function="sum"
        ).drop("g").row(named=True)

        counts = counts.with_columns(
            (
                pl.col("count")
                / pl.col("code_i")
                .replace(count_totals, return_dtype=pl.Float64)
                .round(3)
            ).alias("prob")
        ).pivot(
            on="code_o",
            index="code_i",
            values="prob",
            sort_columns=True,
            aggregate_function="sum"
        ).sort(by="code_i")

        counts = counts.with_columns(
            pl.concat_str(
                pl.col("code_i"),
                pl.col("code_i")
                .replace(count_totals, return_dtype=pl.Float64)
                .mul(1 / len(df))
                .round(3)
                .cast(pl.String),
                separator=": "
            )
            .alias("code_i")
        )

        df_pd = counts.sort(
            pl.col("code_i")
            .replace({s: i for i, s  in enumerate(sign_codes)})
        ).select(
            "code_i",
            *sign_codes
        )

        dfs.append(df_pd)

    labels_truncated = ceg.apis.states.sign_code_labels_truncated(
        objs
    )

    if config.get("bar", True):
        for i, df_pd in enumerate(dfs):
            with st.container(horizontal=True):
                for r in df_pd.reverse().iter_rows(named=True):
                    ix = r.pop("code_i")
                    ks = []
                    vs = []
                    for label, sign_code in labels_truncated:
                        ks.append("-".join([
                            "X" if vv is None else vv
                            for vv in label
                        ]))
                        vs.append(r.get(sign_code, 0.))
                        # or none?
                
                    df_r = pl.DataFrame({
                        "state": [
                            f"{str(i).rjust(2, '0')}: {k}"
                            for i, k in enumerate(ks)
                        ],
                        "prob": vs,
                    }).sort(by="state", descending=True)
                        
                    fig = px.bar(
                        df_r,
                        y="state",
                        x="prob",
                        # color="direction",
                        # barmode="group",
                        title=ix,
                        range_x = [0, 1],
                        orientation="h"
                    )
                    fig.update_layout(margin=dict(t=25, b=0))

                    st.plotly_chart(fig, height=config.get(
                        "height", 
                        int((200 / 8) * len(df_r)), 
                    ), width="stretch", key =f"{ix} - t - {i}")
    else:
        with st.container(horizontal=True):
            for df_pd in dfs:
                df_pd = df_pd.reverse().to_pandas().set_index("code_i")
                fig = px.imshow(
                    df_pd,
                    title=config["name"],
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0.5
                )
                fig.update_layout(margin=dict(t=25, b=0))
                st.plotly_chart(fig, height=config.get(
                    "height", 
                    int(ROW_HEIGHT * len(df_pd)), 
                ), width="stretch")

    return acc

def render_transition_corr(
    config: dict,
    acc: frozendict,
):
    st = ceg.apis.st.library()
    ROW_HEIGHT = (500 / 8)

    objs: list[str] = config["objs"]

    symbols = [ceg.apis.states.extract_symbol(k) for k in objs]

    sign_tuples, sign_codes = ceg.apis.states.sign_code_labels(objs)
    
    data = ceg.assert_is(acc[config["data"]], pl.DataFrame)

    df = data.filter(
        pl.col("active")
    ).pivot(on="sign_code", values="state", index="date").filter(
        pl.any_horizontal(*(
            pl.col(code).is_not_null()
            for code in sign_codes
        ))
    )

    df_corr = df.drop("date").select(
        pl.corr(
            pl.col(code_i)
            .fill_null(0)
            .ewm_mean(span=16),
            pl.col(code_o)
            .reverse()
            .fill_null(0)
            .ewm_mean(span=16)
            .reverse(),
        )
        .alias(f"{code_i}|{code_o}")
        for code_i in sign_codes
        for code_o in sign_codes
        # if code_i != code_o
    ).pipe(
        ceg.apis.pl.unpivot_concatenated_keys,
        [
            "code_i",
            "code_o",
        ],
        "corr"
    )
    df_corr_r = df.select(
        "date",
        *(
            pl.rolling_corr(
                pl.col(code_i)
                .fill_null(0)
                .ewm_mean(span=16),
                pl.col(code_o)
                .reverse()
                .fill_null(0)
                .ewm_mean(span=16)
                .reverse(),
                window_size=252,
                min_samples=8,
            )
            .alias(f"{code_i}|{code_o}")
            for code_i in sign_codes
            for code_o in sign_codes
            # if code_i != code_o
        )
    ).pipe(
        ceg.apis.pl.unpivot_concatenated_keys,
        [
            "code_i",
            "code_o",
        ],
        "corr",
        index="date",
    )

    max_p = max(df_corr.get_column("corr").abs())
    df_corr = df_corr.pivot(
        index="code_i",
        on="code_o",
        values="corr",
    )

    labels_truncated = ceg.apis.states.sign_code_labels_truncated(
        objs
    )

    with st.container(horizontal=True):
        for sign_code in sign_codes:
            df_r = df_corr.filter(pl.col("code_i") == sign_code)
            ks = []
            vs = []
            r = df_r.drop("code_i").row(0, named=True)
            for label, sc in labels_truncated:
                ks.append("-".join([
                    "X" if vv is None else vv
                    for vv in label
                ]))
                vs.append(r.get(sc, 0.))
                # or none?
        
            df_r = pl.DataFrame({
                "state": [
                    f"{str(i).rjust(2, '0')}: {k}"
                    for i, k in enumerate(ks)
                ],
                "corr": vs,
            }).sort(by="state", descending=True)
                
            fig = px.bar(
                df_r,
                y="state",
                x="corr",
                # color="direction",
                # barmode="group",
                title=sign_code,
                range_x = [-max_p, max_p],
                orientation="h"
            )
            fig.update_layout(margin=dict(t=25, b=0))

            st.plotly_chart(fig, height=config.get(
                "height", 
                int((200 / 8) * len(df_r)), 
            ), width="stretch", key =f"{sign_code} - t corr")

    # for sign_code in sign_codes:
    #     df_r = df_corr_r.filter(pl.col("code_i") == sign_code).rename({
    #         "code_o": sign_code
    #     })

    #     fig = px.line(
    #         df_r,
    #         x="date",
    #         y="corr",
    #         color=sign_code,
    #         title=sign_code,
    #         range_y=[-1, 1.]
    #     )
    #     fig.update_layout(margin=dict(t=25, b=0))
    #     st.plotly_chart(fig, height=config.get(
    #         "height", 
    #         250, 
    #     ), width="stretch",
    #     key =f"{sign_code} - t corr ts")
    return acc
    

def plot_transition_corr(
    objs: list[str],
    acc: frozendict,
    name: str,

    **kwargs,
):
    
    i, config_fp = next_config(acc, name=name)

    config = {
        "type": "transition_corr",
        "name": name,
        "objs": objs,
        # **encode_pipe(pipe, params, name),
        **kwargs,
    }

    write_config(config, config_fp)

    return acc

def plot_transition_matrix(
    objs: list[str],
    acc: frozendict,
    name: str,

    **kwargs,
):
    
    i, config_fp = next_config(acc, name=name)

    config = {
        "type": "transition_matrix",
        "name": name,
        "objs": objs,
        # **encode_pipe(pipe, params, name),
        **kwargs,
    }

    write_config(config, config_fp)

    return acc

def render_transition_tree(
    config: dict,
    acc: frozendict,
):
    st = ceg.apis.st.library()
    ROW_HEIGHT = (500 / 8)

    objs: list[str] = config["objs"]

    symbols = [ceg.apis.states.extract_symbol(k) for k in objs]

    sign_tuples, sign_codes = ceg.apis.states.sign_code_labels(objs)
    
    data = ceg.assert_is(acc[config["data"]], pl.DataFrame)

    dfs: dict[int, dict[str, pl.DataFrame]]= {}

    p_max = 0

    for n in [1, 2]:
        dfs[n] = {}
        df = data.filter(
            pl.col("active")
        ).pivot(on="sign_code", values="state", index="date").filter(
            pl.any_horizontal(*(
                pl.col(code).is_not_null()
                for code in sign_codes
            ))
        )

        df_t = ceg.apis.states.transition_tree(
            df,
            symbols,
            sign_tuples,
            sign_codes,
            n=n,
        ).pipe(
            ceg.apis.pl.unpivot_concatenated_keys,
            [
                "sign_code",
                "asset",
                "direction"

            ],
            "count"
        )

        for sign_code in sign_codes:
            n_s = len(df.filter(pl.col(sign_code)))
            df_s = df_t.filter(
                pl.col("sign_code") == sign_code
            ).with_columns(
                pl.col("count")
                .mul(1 / n_s)
                .alias("prob")
            )
            dfs[n][sign_code] = df_s

            p_max = max([
                p_max, df_s.get_column("prob").max()
            ])

    for n, sign_code_dfs in dfs.items():
        with st.container(horizontal=True):
            for sign_code, df_s in sign_code_dfs.items():
                
                df_s = df_s.sort(by = pl.col("asset").replace({
                    asset: i for i, asset in enumerate(symbols)
                })).to_pandas()

                fig = px.bar(
                    df_s,
                    x="asset",
                    y="prob",
                    color="direction",
                    barmode="group",
                    title=sign_code,
                    range_y = [0, p_max]
                )
                fig.update_layout(margin=dict(t=25, b=0))

                st.plotly_chart(fig, height=config.get(
                    "height", 
                    200, 
                ), width="stretch", key =f"{sign_code} - t tree {n}")

    return acc
def plot_transition_tree(
    objs: list[str],
    acc: frozendict,
    name: str,

    **kwargs,
):
    
    i, config_fp = next_config(acc, name=name)

    config = {
        "type": "transition_tree",
        "name": name,
        "objs": objs,
        # **encode_pipe(pipe, params, name),
        **kwargs,
    }

    write_config(config, config_fp)

    return acc
def render_state_codes_w_y(
    config: dict,
    acc: frozendict,
):
    st = ceg.apis.st.library()
    N_COLS = 50
    ROW_HEIGHT = 750 / 16

    data_s = ceg.assert_is(acc[config["data_s"]], pl.DataFrame).filter(
        pl.col("active")
    ).pivot(on="sign_code", values="state", index="date")
    data_y = ceg.assert_is(acc[config["data_y"]], pl.DataFrame)
    objs_s: list[str] = config["objs_s"]
    objs_y: list[str] = config["objs_y"]

    symbols_s = [ceg.apis.states.extract_symbol(k) for k in objs_s]
    symbols_y = [ceg.apis.states.extract_symbol(k) for k in objs_y]

    _, sign_codes = ceg.apis.states.sign_code_labels(objs_s)

    data = data_s.join(
        data_y.rename({
            obj: symbol for obj, symbol in zip(objs_y, symbols_y)
        }), on ="date", how="full", coalesce=True
    ).sort(by="date")

    data = data.filter(
        pl.any_horizontal(*(
            pl.col(code).is_not_null()
            for code in sign_codes
        ))
    )

    n_days = len(data)

    f_prob = lambda e: e.sum() / e.count()

    variable_order = {s: i for i, s in enumerate(symbols_s)}
    variable_order = {
        **variable_order,
        **{s:i + len(variable_order) for i, s in enumerate(symbols_y)}
    }

    data = data.with_columns(
        (pl.col("date") - pl.col("date").min())
        .dt.total_days()
        .floordiv(n_days / N_COLS)
        .alias("g")
    ).group_by("g").agg(
        pl.col("date").last(),
        *(
            pl.col(code)
            .cast(pl.Int32)
            .fill_null(0)
            .pipe(f_prob)
            .alias(code)
            for code in sign_codes
        ),
        *(
            pl.col(rx)
            .sum()
            for rx in symbols_y
        )
    ).drop("g").unpivot(
        index="date",
        variable_name="variable",
        value_name="value"
    )

    df_y = data.filter(
        pl.col("variable").is_in(symbols_y)
    )
    
    min_y, max_y = df_y.select(
        pl.col("value").min().alias("min"),
        pl.col("value").max()
    ).row(0)

    rng_y = max([abs(max_y), abs(min_y)])

    data = data.with_columns(
        pl.when(pl.col("variable").is_in(sign_codes))
        .then((pl.col("value") - 0.5) * rng_y)
        .otherwise(pl.col("value"))
        .alias("value")
    ).pivot(
        on="date",
        values="value",
        sort_columns=True,
    ).sort(by=pl.col("variable").replace(variable_order)).reverse()

    df_pd = data.to_pandas().set_index("variable")

    fig = px.imshow(
        df_pd,
        title=config["name"],
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0
    )
    fig.update_layout(margin=dict(t=25, b=0))
    st.plotly_chart(fig, height=config.get(
        "height", 
        int(ROW_HEIGHT * (len(sign_codes) + len(objs_y))), 
    ), width="stretch")

    return acc

def plot_state_codes_w_y(
    acc: frozendict,
    name: str,
    **kwargs,
):
    # parametrise orientation - v vs h?
    # states color by intensity, one row per state
    # otherwise, sum rx in each cell (parametrise max cells per row?)
    # by just summing (to start - parmetrise compound?)
    
    i, config_fp = next_config(acc, name=name)

    config = {
        "type": "state_codes_w_y",
        "name": name,
        # **encode_pipe(pipe, params, name),
        **kwargs,
    }

    write_config(config, config_fp)

    return acc

import itertools
def render_conditional_dist_w_fwd(
    config: dict,
    acc: frozendict,
):
    st = ceg.apis.st.library()

    data_s = ceg.assert_is(acc[config["data_s"]], pl.DataFrame).filter(
        pl.col("active")
    ).pivot(on="sign_code", values="state", index="date")
    data_y = ceg.assert_is(acc[config["data_y"]], pl.DataFrame)
    objs_s: list[str] = config["objs_s"]
    objs_y: list[str] = config["objs_y"]

    symbols_s = [ceg.apis.states.extract_symbol(k) for k in objs_s]
    symbols_y = [ceg.apis.states.extract_symbol(k) for k in objs_y]
    
    _, sign_codes = ceg.apis.states.sign_code_labels(objs_s)

    data = data_s.join(
        data_y.rename({
            obj: symbol for obj, symbol in zip(objs_y, symbols_y)
        }),
        on ="date", how="full", coalesce=True
    ).sort(by="date")

    shifts = [1, 2]

    data = data.filter(
        pl.any_horizontal(*(
            pl.col(code).is_not_null()
            for code in sign_codes
        ))
    ).with_columns(
        pl.col(s).shift(-i).alias(f"{s}_{i}").fill_null(0)
        for s in symbols_y
        for i in shifts
        # TODO: parametrise fwd - should be -2?
    )

    sign_rows = {}
    sign_dfs = {}
    sign_probs = {}

    max_v = {}
    max_p = 0

    for sign_code, shift in itertools.product(
        sign_codes,
        ["", "_N"] + [f"_{i}" for i in shifts]
    ):
        if shift == "_N":
            data_s = data.filter(
                pl.col(sign_code),
                pl.col(sign_code) == pl.col(sign_code).shift(1)
            )
        else:
            data_s = data.filter(pl.col(sign_code))

        sign_key = (sign_code, shift)

        sign_probs[sign_code] = round(len(data_s) / len(data), 3)

        sign_dfs[sign_code] = data.select("date", sign_code).with_columns(
            pl.col(sign_code)
            .fill_null(0)
            .ewm_mean(span=90)
            .alias("mu")
        ).with_columns(
            pl.when(pl.col(sign_code).not_())
            .then(None)
            .otherwise(pl.col("mu"))
            .alias(sign_code)
        )

        max_p = max([
            max_p,
            sign_dfs[sign_code].select(pl.col(sign_code).max()).item()
        ])


        suff = shift if shift != "_N" else ""

        data_s = data_s.select(*[
            pl.col(f"{s}{suff}").alias(s) for s in 
            symbols_y
        ]).unpivot(
            variable_name="symbol",
            value_name="rx"
        ).group_by(
            "symbol"
        ).agg(
            pl.col("rx").mean().alias("mean").round(3) * 252,
            pl.col("rx").median().alias("median").round(3) * 252,
            pl.col("rx").quantile(0.25).alias("q_25").round(3) * 252,
            pl.col("rx").quantile(0.75).alias("q_75").round(3) * 252,
            pl.col("rx").pow(2).mean().sqrt().alias("std").round(3) * 16,
        ).with_columns(
            (pl.col("q_25") - pl.col("mean")).mul(-1.).alias("q_25"),
            (pl.col("q_75") - pl.col("mean")).alias("q_75"),
        )

        max_mu, max_med, q25_max, q75_max = data_s.select(
            pl.col("mean").abs().max(),
            pl.col("median").abs().max(),
            pl.col("q_25").abs().max() / 2,
            pl.col("q_75").abs().max() / 2,
        ).row(0)

        shift = shift if shift != "_N" else ""

        max_v[shift] = max([max_v.get(shift, 0), max_mu + q25_max, max_mu + q75_max])

        rows = {
            k: r.row(0, named=True)
            for (k,), r in data_s.partition_by("symbol", as_dict=True).items()
        }
        sign_rows[sign_key] = rows

    containers = {
        sign_code: st.container(horizontal=True)
        for sign_code in sign_codes
    }

    for sign_code, rows in sign_rows.items():
        sign_code, shift = sign_code
        suff = shift if shift != "_N" else ""

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="symbol",
            x=[
                f"{s}: {rows.get(s, {}).get('std', None)}"
                for s in symbols_y
            ],
            y=[
                rows.get(s, {}).get("mean", None)
                for s in symbols_y
            ],
            error_y=dict(
                type='data',
                symmetric=False,
                arrayminus=[
                    rows.get(s, {}).get("q_25", None)
                    for s in symbols_y
                ],
                array=[
                    rows.get(s, {}).get("q_75", None)
                    for s in symbols_y
                ]
            ),
        ))
        fig.update_layout(
            title=sign_code + f"{shift}: " + str(sign_probs[sign_code]),
            margin=dict(t=25, b=0)
        )
        fig.update_yaxes(range=[-max_v[suff], max_v[suff]])

        with containers[sign_code]:
            st.plotly_chart(fig, height=config.get(
                "height", 
                200, 
            ), width="stretch", key =f"{sign_code}{shift} - dist")

            # df_pd = pl.DataFrame(dict(
            #     x=[
            #         s
            #         for s in symbols_y
            #     ],
            #     y=[
            #         rows.get(s, {}).get("mean", None)
            #         for s in symbols_y
            #     ],
            # )).with_columns(pl.lit(sign_code).alias("code")).pivot(index="code", on="x", values="y").to_pandas().set_index("code")

            # fig = px.imshow(
            #     df_pd,
            #     title=sign_code + f"{shift}: " + str(sign_probs[sign_code]),
            #     color_continuous_scale="RdBu_r",
            #     color_continuous_midpoint=0,
            #     zmin=-max_v,
            #     zmax=max_v,
            # )
            # fig.update_layout(margin=dict(t=25, b=0))
            # st.plotly_chart(fig, height=config.get(
            #     "height", 
            #     int(200), 
            # ), width="stretch")
            # fig = px.scatter(
            #     sign_dfs[sign_code],
            #     x="date",
            #     y=sign_code,
            #     title=sign_code + ": " + str(sign_probs[sign_code]),
            # )
            # fig.update_layout(margin=dict(t=25, b=0))
            # fig.update_yaxes(range=[0, max_p])
            # st.plotly_chart(fig, height=config.get(
            #     "height", 
            #     200, 
            # ), width="stretch", key =f"{sign_code} - active")


    return acc

def plot_conditional_dist_w_fwd(
    acc: frozendict,
    name: str,
    **kwargs,
):
    # stack per state

    # conditional bwd and fwd retrusn, for range of assets
    # toggle group by bwd / fwd or group by asset
    # also window for fwd, none=1 day
    
    i, config_fp = next_config(acc, name=name)

    config = {
        "type": "conditional_dist_w_fwd",
        "name": name,
        # **encode_pipe(pipe, params, name),
        **kwargs,
    }

    write_config(config, config_fp)

    return acc

def render_conditional_beta_w_vol(
    config: dict,
    acc: frozendict,
):
    st = ceg.apis.st.library()
    ROW_HEIGHT = (500 / 32)

    data_s = ceg.assert_is(acc[config["data_s"]], pl.DataFrame).filter(
        pl.col("active")
    ).pivot(on="sign_code", values="state", index="date")
    data_y = ceg.assert_is(acc[config["data_y"]], pl.DataFrame)
    data_b = ceg.assert_is(acc[config["data_b"]], pl.DataFrame)
    objs_s: list[str] = config["objs_s"]
    objs_y: list[str] = config["objs_y"]
    objs_b: list[str] = config["objs_b"]

    symbols_s = [ceg.apis.states.extract_symbol(k) for k in objs_s]
    symbols_y = [ceg.apis.states.extract_symbol(k) for k in objs_y]
    symbols_b = [ceg.apis.states.extract_symbol(k) for k in objs_b]
    
    _, sign_codes = ceg.apis.states.sign_code_labels(objs_s)

    data = data_s.join(
        data_y.rename({
            obj: symbol for obj, symbol in zip(objs_y, symbols_y)
        }),
        on ="date", how="full", coalesce=True
    ).join(data_b.rename({
        obj: symbol for obj, symbol in zip(objs_b, symbols_b)
    }).select(
        "date",
        *[
            k for k in data_b.schema.keys()
            if k not in data_y.schema
        ]
    ), on ="date", how="full", coalesce=True).sort(by="date")

    data = data.filter(
        pl.any_horizontal(*(
            pl.col(code).is_not_null()
            for code in sign_codes
        ))
    )
    sign_probs = {}

    for sign_code in sign_codes:
        data_s = data.filter(pl.col(sign_code))
        sign_probs[sign_code] = round(len(data_s) / len(data), 3)

        vol = data_s.select(*(
            pl.col(s).pow(2).mean().sqrt().mul(16).alias(s)
            for s in symbols_b + [
                s for s in symbols_y if s not in symbols_b
            ]
        ))

        corr_b = data_s.select(*(
            pl.corr(pl.col(y), pl.col(b)).alias(f"{y}|{b}").round(3)
            for y in symbols_y
            for b in symbols_b
        )).unpivot(
            variable_name="pair",
            value_name="corr"
        ).with_columns(
            pl.col("pair").str.split(pl.lit("|")).list.get(0).alias("y"),
            pl.col("pair").str.split(pl.lit("|")).list.get(1).alias("b"),
        ).sort(by=pl.col("b").replace({
            s: i for i, s in enumerate(symbols_b)
        })).pivot(
            values="corr",
            index="b",
            on="y"
        ).select("b", *symbols_y)

        corr_y = data_s.select(*(
            pl.corr(pl.col(y1), pl.col(y2)).alias(f"{y1}|{y2}").round(3)
            for y1 in symbols_y
            for y2 in symbols_y
            if y1 not in symbols_b
            and y2 not in symbols_b
        )).unpivot(
            variable_name="pair",
            value_name="corr"
        ).with_columns(
            pl.col("pair").str.split(pl.lit("|")).list.get(0).alias("y1"),
            pl.col("pair").str.split(pl.lit("|")).list.get(1).alias("y2"),
        ).sort(by=pl.col("y1").replace({
            s: i for i, s in enumerate(symbols_y)
        })).pivot(
            values="corr",
            index="y1",
            on="y2"
        ).select("y1", *[s for s in symbols_y if s not in symbols_b])

        df_b = corr_b.to_pandas().set_index("b")
        df_y = corr_y.to_pandas().set_index("y1")

        with st.container(horizontal=True):
            fig = px.imshow(
                df_b,
                title=sign_code+ ": " + str(sign_probs[sign_code]),
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0
            )
            fig.update_layout(margin=dict(t=25, b=0))
            fig.update_yaxes(range=[-1., 1.])
            st.plotly_chart(fig, height=config.get(
                "height", 
                int(ROW_HEIGHT * (len(sign_codes) + len(objs_y))), 
            ), width="stretch")
            if len(df_y):
                fig = px.imshow(
                    df_y,
                    title=sign_code+ ": " + str(sign_probs[sign_code]),
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0
                )
                fig.update_layout(margin=dict(t=25, b=0))
                fig.update_yaxes(range=[-1., 1.])
                st.plotly_chart(fig, height=config.get(
                    "height", 
                    int(ROW_HEIGHT * (len(sign_codes) + len(objs_y))), 
                ), width="stretch")


            fig = px.bar(vol.unpivot(
                variable_name="symbol",
                value_name="vol"
            ), x = "symbol", y="vol", range_y = [0.75, 1.75])
            fig.update_layout(margin=dict(t=25, b=0))

            st.plotly_chart(fig, height=config.get(
                "height", 
                int(ROW_HEIGHT * (len(sign_codes) + len(objs_y))), 
            ), width="stretch")
        # TODO: add bar ploto of conditional vol across panel (y + b)

    return acc

def plot_conditional_beta_w_vol(
    acc: frozendict,
    name: str,

    **kwargs,
):
    # per state
    # parametrise group by beta ref, or by asset
    
    i, config_fp = next_config(acc, name=name)

    config = {
        "type": "conditional_beta_w_vol",
        "name": name,
        # **encode_pipe(pipe, params, name),
        **kwargs,
    }

    write_config(config, config_fp)

    return acc

def transform_df(
    config: dict,
    acc: frozendict,
):

    pipe: str | None = config.get("pipe")
    params: dict | None = config.get("params")
    kwargs: dict | None = config.get("kwargs")

    df = acc[config["data"]]

    assert isinstance(df, pl.DataFrame), (config, acc)

    if kwargs is None:
        kwargs = {}
    kwargs["objs"] = config["objs"]

    acc, df = apply_pipe(df, acc, config.get("name"), pipe, params, kwargs)

    return acc.set(f"data:{config['name']}", df)

def write_transform(
    objs: list[str],
    acc: frozendict,

    name: str,
    pipe: Callable | str | None = None,
    params: dict[str, Any | type] | None = None,

    data: str | None = None,
    **kwargs,
):
    # used to bind a transformation
    # given eg. col names, and a previous data output (by name?)
    
    i, config_fp = next_config(acc, name=name)

    # if "data" not in params:
    #     raise ValueError(params)

    config = {
        "type": "tf",
        "objs": objs,
        "name": name,
        "data": data,
        **encode_pipe(pipe, params, name, **kwargs),
        **kwargs,
    }

    write_config(config, config_fp)

    return acc

def load_df(
    config: dict,
    acc: frozendict,
):

    fp: str = config["fp"]

    pipe: str | None = config.get("pipe")
    params: dict | None = config.get("params")
    kwargs: dict | None = config.get("kwargs")

    df = pl.read_parquet(fp)

    acc, df = apply_pipe(df, acc, config.get("name"), pipe, params, kwargs)

    return acc.set(f"data:{config['name']}", df)

def write_data(
    df: pl.DataFrame,
    acc: frozendict,
    # t? = df
    name: str,
    pipe: Callable | str | None = None,
    params: dict[str, Any | type] | None = None,
    **kwargs,
):
    i, config_fp = next_config(acc, name=name)

    df_fp = config_fp.parent / f"df_{i}.parquet"
    df.write_parquet(df_fp)
    config = {
        "type": "df",
        "fp": str(df_fp),
        "name": name,
        **encode_pipe(pipe, params, name),
        **kwargs,
    }

    write_config(config, config_fp)

    return acc


def write_graph(
    df: str | pl.DataFrame,
    acc: frozendict,
    t: str,
    name: str,
    pipe: Callable | str | None = None,
    params: dict[str, Any | type] | None = None,
    **kwargs,
):
    i, config_fp = next_config(acc, name=name)

    if isinstance(df, str):
        df_fp = f"data:{df}"
    else:
        df_fp = config_fp.parent / f"df_{i}.parquet"
        df.write_parquet(df_fp)

    config = {
        "type": t,
        "fp": str(df_fp),
        "name": name,
        **encode_pipe(pipe, params, name),
        **kwargs,
    }
    write_config(config, config_fp)

    return acc



plot_line = partial(write_graph, t = "line")
plot_vline = partial(write_graph, t = "vline")
plot_scatter = partial(write_graph, t = "scatter")
plot_vscatter = partial(write_graph, t = "vscatter")
plot_area = partial(write_graph, t = "area")
plot_varea = partial(write_graph, t = "varea")
plot_bar = partial(write_graph, t = "bar")
plot_hbar = partial(write_graph, t = "hbar")
