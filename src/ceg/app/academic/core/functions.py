from typing import Callable
import numpy as np
import pandas as pd
from .types import *
from .states import State

# TODO: tall
# TODO: for now, we assume constant!


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


# TODO:

# def init_agent_io(
#     n_agents: int,
#     n_products: int,
#     product_io: np.ndarray,
#     f_sample: Callable[[tuple[int, ...], np.ndarray], np.ndarray]
# ) -> np.ndarray:
#     shape = (n_agents, n_products)
#     # one product output per agent, so one vector per agent (summative, not matmul)
#     # more generally, one production process - even if multiple outputs
#     # which i guess is summing over two individual product processes?
#     # again, can possibly mutate the i and o components separately and re-combine
#     assert product_io.shape == (n_products, n_products), (product_io.shape, (n_products, n_products))
#     return f_sample(shape, product_io)
