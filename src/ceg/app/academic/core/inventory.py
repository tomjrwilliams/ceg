from typing import Callable
import numpy as np
import pandas as pd

from .types import *
from .states import State


def df_tall(
    v: F64_3D,
    agents: list[str] | None = None,
    products: list[str] | None = None,
    agent: str = "a",
    product: str = "p",
) -> pl.DataFrame:
    t = v.shape[0]
    a = v.shape[1]
    p = v.shape[2]
    assert a == v.shape[2]
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
        "inventory",
        {
            "time": pl.Int32,
            "agent": t_agents,
            "product": t_products,
        },
    )


def df_wide(
    v: F64_2D,
    agents: list[str] | None = None,
    products: list[str] | None = None,
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
            name: v[:, i] for i, name in enumerate(products)
        },
        index=agents,
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


# def init_agent_inventory(
#     n_agents: int,
#     n_products: int,
#     agent_io: np.ndarray, # product io?
#     f_sample: Callable[[tuple[int, ...], np.ndarray], np.ndarray]
# ) -> np.ndarray:
#     shape = (n_agents, n_products)
#     assert agent_io.shape == shape, (agent_io.shape, shape)
#     return f_sample(shape, agent_io)

# # target inventory?


# ------------------------------ old

# arrival process for orders
# sometimes state change on process parameter (s)

# initially fixed:

# profit per sale
# cost per inventory held

# later can allow cost of production to vary
# and price of sale
# and inventory holding

# sale starts off as 100% probability given order, if inventory
# later can make probabilistic

# agent chooses production
# starts off immediate arrival (?)
# later add lag (can then make probabilistic)

# each time:

# agent finds out sales that period
# calculates profit in that period

# then two options
# rl or parametric optimisation

# parametric, likely two steps

# eg. fit agent estimate of model parameters (eg. arrival process rate)
# and then given arrival estimate
# and current inventory, unit profit, holding cost
# pick production so as to maximise future discounted reward from here


# rl you need a single parametric function from current state and new data to production
# there its also possibly easier to separate process param estimation
# and production given arrival param

# ie. start with just a linear model over the unit costs, inventory, and arrival rate
# can later make more complicated as intuition develops

# that you then fit RL style based on the observed reward each time
# but then have to fit the params of the linear model by running many times


# vs the parametric approach where the maximisation is determined each stype by the functional form, so there's not the same fitting
# rather finding the right functional form for your future expected reward that you can optimise it directly (given arrival estimate)


# what are the outputs?

# arrival process param
# orders time series

# agents arrival process param estimate

# agent inventory time series (including newly received production)
# agent sales times series
# agent profit time series


# agent new production decision time series
# agent maximised estimated total future profit from here, time series


# exactly the same graphs would also describe the model with lag between production decision and receipt
# and with varying profit per sale / cost per unit / inventory cost


# just the complexity of the params might change, eg. if we move to a markov model with state estimates

# even a two variable model where agents optimise over both price and production
# where there you just need another model for agents that gives a sale probability given an arrival and a price
