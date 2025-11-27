from typing import Callable
import numpy as np
import pandas as pd
from .types import *
from .states import State

def df_tall(
    v: F64_3D,
    agents: list[str] | None = None,
    products: list[str] | None=None,
    agent: str = "a",
    product: str = "p",
) -> pl.DataFrame:
    t = v.shape[0]
    a = v.shape[1]
    p = v.shape[2]
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
    return f64_3d_to_df(
        v,
        "price",
        {
            "time": pl.Int32,
            "agent": t_agents,
            "product": t_products,
        }
    )

def df_wide(
    v: F64_2D,
    agents: list[str] | None = None,
    products: list[str] | None=None,
    agent: str = "a",
    product: str = "p",
) -> pd.DataFrame:
    a = v.shape[0]
    p = v.shape[1]
    if products is not None:
        assert len(products) == v.shape[1]
    else:
        products = enumerated_labels(product, p, justify=3)
    if agents is not None:
        assert len(agents) == v.shape[1]
    else:
        agents = enumerated_labels(agent, a, justify=3)
    return pd.DataFrame(
        data={
            name: v[:, i]
            for i, name in enumerate(products)
        },
        index=agents
    )

def init_constant(
    n_agents: int,
    n_products: int,
    v: F64_2D,
) -> Callable[[State], F64_2D]:
    assert n_agents == v.shape[0]
    assert n_products == v.shape[1]
    def f(state: State) -> F64_2D:
        assert state.n_agents == n_agents
        assert state.n_products == n_products
        return v
    return f

# def init_product_prices(
#     n_agents: int,
#     n_products: int,
#     # take product io, agent io?
#     f_sample: Callable[[tuple[int, ...]], np.ndarray]
# ) -> np.ndarray:
#     shape = (n_agents, n_products)
#     # for now, all agents share a price
#     return f_sample(shape)


# TODO: * constant per product / agent
def cum_sqrt_imbalance(
    state: State,
):
    prices = state.prices

    orders: F64_1D = np.sum(
        state.orders, axis=(1, 2)
    )
    deliveries: F64_1D = np.sum(
        state.deliveries, axis=(1, 2)
    )

    assert orders.shape == deliveries.shape

    imbalance = orders - deliveries

    d_price= np.sign(imbalance) * np.sqrt(np.abs(imbalance))

    return prices + np.expand_dims(d_price, 0)
    