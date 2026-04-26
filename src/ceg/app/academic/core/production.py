import pandas as pd

from .types import *
from . import states, State


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
        "production",
        {
            "time": pl.Int32,
            "agent": t_agents,
            "product": t_products,
        },
    )


# assume single t
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


def offset_deliveries(state: State) -> F64_2D:

    deliveries: F64_2D = np.sum(state.deliveries, axis=2)

    is_input = state.functions < 0
    is_ouput = state.functions > 0

    units_in_inventory: F64_2D = np.where(
        is_input, state.inventory, np.nan
    ) / np.where(
        is_input,
        -1 * state.functions,
        1,
    )

    units_available: F64_1D = np.nanmin(
        units_in_inventory, axis=1
    )

    units_delivered: F64_2D = np.where(
        is_ouput, deliveries, np.nan
    ) / np.where(is_ouput, state.functions, 1)

    # max?
    units_required: F64_1D = np.nanmax(
        units_delivered, axis=1
    )

    prod: F64_1D = np.clip(
        units_required, a_min=None, a_max=units_available
    )

    prod_io: F64_2D = prod * state.functions

    return prod_io
