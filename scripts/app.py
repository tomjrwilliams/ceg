import json
from pathlib import Path
import plotly
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import ceg
from frozendict import frozendict
def render_config_dir(dp: Path, acc: frozendict):

    i = 0
    config_fp = dp / f"config_{i}.json"
    while config_fp.exists():
        with config_fp.open('r') as f:
            config = json.load(f)
        acc = ceg.apis.vis.render_obj(config, acc)
        i += 1
        config_fp = dp / f"config_{i}.json"
    return acc

def page(name: str, k: str):
    def run():
        acc = frozendict()
        data=DIR_ST / name / k
        acc = render_config_dir(data, acc)
        for sub_dir in sorted(data.iterdir()):
            if not sub_dir.is_dir():
                continue
            elif sub_dir.name.startswith("_"):
                continue
            else:
                acc = render_config_dir(sub_dir, acc)
    run.__name__ = k
    return run

DIR_ST = Path("./__local__/st")

from ceg.apis import vis
from ceg.apis import states

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