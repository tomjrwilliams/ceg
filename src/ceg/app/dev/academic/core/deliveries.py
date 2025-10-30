
import numpy as np

from .types import *
from . import states, State


def df_tall(
    v: F64_4D,
    agents: list[str] | None = None,
    products: list[str] | None=None,
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
        "delivery",
        {
            "time": pl.Int32,
            "product": t_products,
            "agent-o": t_agents,
            "agent-i": t_agents,
        }
    )
    
def order_by_rel(state: State) -> F64_3D:
    
    is_output = state.functions > 0

    inventory: F64_1D = np.where(
        is_output,
        state.inventory,
        0,
    )

    rels = state.rels
    # TODO: jitter for random ordering if equal

    res = np.zeros(state.orders.shape)

    # TODO: only works if one good output per agent
    for a in range(state.n_agents):
        for p in range(state.n_products):

            a_orders: F64_1D = state.orders[p, a, :]
            a_inventory: F64_1D = inventory[a][p]
            a_rels: F64_1D = rels[a]

            a_order = a_rels.argsort()

            a_ordered = a_orders[a_order]
            a_ordered_cum = a_ordered.cumsum()
            a_ordered_cum_clip = np.clip(
                a_ordered_cum, 
                a_min=None,
                a_max=a_inventory
            )

            a_ordered_unsatisfied = -1 * (
                a_ordered_cum - a_ordered_cum_clip
            )
            a_partial_satisfy = np.cumsum(
                a_ordered_unsatisfied > 0
            ) == 1
            
            a_ordered_satisfied = np.where(
                a_ordered_unsatisfied == 0,
                a_ordered,
                np.where(
                    a_partial_satisfy,
                    a_ordered - a_ordered_unsatisfied,
                    0,
                )
            )

            a_del = np.empty(a_orders.shape)
            a_del[a_order] = a_ordered_satisfied

            res[p][a] = a_del

    return res

# def proportional_to_rel(state: State) -> F64_3D:

#     orders = state.orders
#     rels = state.rels

#     orders_total: F64_2D = np.sum(state.orders, axis=2)

#     rels_prop = rels / np.expand_dims(
#         np.sum(rels, axis=1), 1
#     )


#     # f_deliver(product_orders, agent_inventory, agent_rels)

#     return