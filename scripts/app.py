import json
from pathlib import Path
import plotly
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import ceg

def render_px(
    config: dict,
    f: str,
    v: bool = False
):
    fp = config["fp"]

    group = config.get("group", "group")
    
    x_col = config.get("x", "date")

    # TODO: eneds fixing if you want float vs float scatter
    # eg. proper .get defualts for color, group, etc.
    not_float = ["date", group, "variable", x_col]

    df = pl.read_parquet(fp).with_columns(
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


def render_scatter(config: dict, v: bool = False):
    return render_px(config, "scatter", v=v)

def render_line(config: dict, v: bool = False):
    return render_px(config, "line", v=v)

def render_area(config: dict, v: bool = False):
    return render_px(config, "area", v=v)

def render_bar(config: dict, v: bool = False):
    return render_px(config, "bar", v=v)


def render_obj(config: dict):
    type = config["type"]
    if type == "scatter":
        render_scatter(config)
    elif type == "vscatter":
        render_scatter(config, v = True)
    elif type == "line":
        render_line(config)
    elif type == "vline":
        render_line(config, v = True)
    elif type == "area":
        render_area(config)
    elif type == "varea":
        render_area(config, v = True)
    elif type == "bar":
        render_bar(config, v = True)
    elif type == "hbar":
        render_bar(config, v = False)
    else:
        raise ValueError(config)

def page(name: str, k: str):
    def run():
        data=DIR_ST / name / k
        i = 0
        config_fp = data / f"config_{i}.json"
        while config_fp.exists():
            with config_fp.open('r') as f:
                config = json.load(f)
            render_obj(config)
            i += 1
            config_fp = data / f"config_{i}.json"
    run.__name__ = k
    return run

DIR_ST = Path("./__local__/st")

if __name__ == "__main__":
    st = ceg.apis.st.library()
    
    st.set_page_config(**{"layout": "wide"})

    pages = {}
    if DIR_ST.exists():
        for dp in DIR_ST.iterdir():
            pages[dp.name] = []
            for sub_dp in dp.iterdir():
                pages[dp.name].append(sub_dp.name)

    st.navigation({
        **{
            "home": [st.Page(lambda: None, title="home")],
        },
        **{
            name: [
                st.Page(page(name, k), title=k)
                for k in sub_pages
            ]
            for name, sub_pages in pages.items()
        }
    }).run()