import os
import sys

sys.path.append("./src")

import polars as pl
from ...apis.st import cache, library

st = library()

import ceg

core = ceg.app.academic.core
ui = ceg.vis.legacy


def init_app():
    with st.sidebar:
        sim = ui.components.session_state_text_input(
            "sim",
            variable="SIM",
            default=os.environ.get("SIM"),
        )
        ui.components.reset_cache_button()


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


if __name__ == "__main__":
    ui.apps.app(
        {
            "constants": {
                "technologies": ui.components.single_table(
                    "technologies",
                    core.technologies.df_wide,
                    f_data=lambda v: v[0],
                ),
                "rels": ui.components.single_table(
                    "rels",
                    core.rels.df_wide,
                    f_data=lambda v: v[0],
                ),
                "functions": ui.components.single_table(
                    "functions",
                    core.functions.df_wide,
                    f_data=lambda v: v[0],
                ),
            },
            "variables": {
                "deliveries": ui.components.single_table(
                    "deliveries",
                    core.deliveries.df_tall,
                    f_df=lambda df: df.pipe(
                        drop_zero, "delivery"
                    ),
                ),
                "production": ui.components.single_table(
                    "production",
                    core.production.df_tall,
                    f_df=lambda df: df.pipe(
                        drop_zero, "production"
                    ),
                ),
                "inventory": ui.components.single_table(
                    "inventory",
                    core.inventory.df_tall,
                    f_df=lambda df: df.pipe(
                        drop_zero, "inventory"
                    ),
                ),
                "cash": ui.components.single_table(
                    "cash",
                    core.cash.df_tall,
                    f_df=lambda df: df.pipe(
                        drop_zero, "cash"
                    ),
                ),
                "prices": ui.components.single_table(
                    "prices",
                    core.prices.df_tall,
                    f_df=lambda df: df.pipe(
                        drop_zero, "price"
                    ),
                ),
                "orders": ui.components.single_table(
                    "orders",
                    core.orders.df_tall,
                    f_df=lambda df: df.pipe(
                        drop_zero, "order"
                    ),
                ),
            },
        },
        init_elements=init_app,
    ).run()

# simple streamlit page with charts

# possibly, share state on a file path

# and then different pages (all that file path) have different views onto the data


# eg. one that's the static config: the product io, agent io, rels


# eg. others for production, orders, deliveries

# prices, cash balances


# essentially just a page for each variable?
# can presumably open two pages and compare side by side if interested in more than one data point at a time


# possibly, you want side split, at the top, you pick the variable to explore and then given that it shows charts
# even take as param side split

# so then its only one web window

# pick columns, then given that, for each col, pick vertical

# nothing fancy on the vertical, just takes up the space it takes, use container with pixel height to force scroll


# if we can detect the viewport pixel size, can then set grid of fixed pixel height and scroll within


# arguably just a grid table would work, put heights in what you want to fill

# none are removed, per col (if whole col, dropped)


# possibly the cell is variable and height? (optional height)
# that means text input for variables


# but then you want a way of saving a config or easy loading to a particular view?


# then just call python streamlit ../ui.py


# reload button for after file change
