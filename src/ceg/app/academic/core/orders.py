from typing import Callable
import numpy as np
import pandas as pd
import polars as pl
from .types import *
from .states import State


def df_tall(
    v: F64_4D,
    agents: list[str] | None = None,
    products: list[str] | None = None,
    agent: str = "a",
    product: str = "p",
) -> pl.DataFrame:
    t = v.shape[0]
    p = v.shape[1]
    a = v.shape[2]
    assert a == v.shape[3]
    if products is not None:
        assert len(products) == v.shape[1]
    else:
        products = enumerated_labels(product, p, justify=3)
    if agents is not None:
        assert len(agents) == v.shape[1]
    else:
        agents = enumerated_labels(agent, a, justify=3)
    t_agents = pl.Enum(agents)
    t_products = pl.Enum(products)
    return f64_4d_to_df(
        v,
        "order",
        {
            "time": pl.Int32,
            "product": t_products,
            "agent-o": t_agents,
            "agent-i": t_agents,
        },
    )


def cum_net_chg_inventory(
    state: State,
    # TODO: name proportional_to_rel
):

    prices = state.prices
    funcs = state.functions

    is_input = funcs < 0
    is_ouput = funcs > 0

    inputs = -1 * np.where(is_input, funcs, 0)

    order_costs: F64_1D = np.sum(
        inputs * prices,
        axis=1,
        # NOTE: only works for a single shared price
        # as then the agent price is given for all outputs (not just their production), by the global price
    )

    order_max: F64_2D = (
        np.expand_dims(state.cash / order_costs, 0)
        # * np.where(is_ouput, funcs, 0)
    )
    # print(order_costs, state.cash, order_max)

    delivered: F64_2D = np.sum(state.deliveries, axis=2)
    delivered_units: F64_2D = (
        delivered / np.where(is_ouput, funcs, np.nan)
        # no need to mask as assume only deliver output
    )

    max_deliverable: F64_1D = np.nanmax(
        np.clip(
            delivered_units, a_min=None, a_max=order_max
        ),
        axis=1,
    )

    order_totals = (
        np.expand_dims(max_deliverable, axis=1) * inputs
    )

    rels_io = state.rels * np.where(is_input, 1, 0)

    rels_prop = (
        rels_io / np.expand_dims(np.sum(rels_io, axis=1), 0)
        # TODO: right axis to sum out to norm?
    )

    res = np.zeros(state.orders.shape)

    for a in range(state.n_agents):
        a_order = order_totals[a]
        a_rels = rels_prop[a]
        # print(a, a_order)
        for p, v in enumerate(a_order):
            res[p][p][a] = v * a_rels[p]

    # print(res)

    return res
