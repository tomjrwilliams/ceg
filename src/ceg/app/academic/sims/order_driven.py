
from pathlib import Path
import numpy as np

from .. import core
from ..core.types import *

def example(
    fp: Path,
    agents: int = 3,
    products: int = 3,
    length: int = 100,
    batch: int | None = None,
    # TODO: other args - eg. n agents, etc.
) -> Path:
    n_agents=agents
    n_products=products

    results = core.results.Results.init(
        str(fp),
        n_agents=n_agents,
        n_products=n_products,
        length=length,
        batch=batch,
    )
    
    assert n_products == n_agents
    functions = technology = np.eye(n_products) - np.roll(
        np.eye(n_products), 1, 1
    )
    inventory = np.eye(n_products)

    state = core.states.State.init(
        n_agents=n_agents,
        n_products=n_products,
        #
        f_technology = core.technologies.init_constant(
            n_products,
            technology,
        ),
        f_rels = core.rels.init_constant(
            n_agents,
            np.ones((n_agents, n_agents))
        ),
        #
        f_functions = core.functions.init_constant(
            n_agents,
            n_products,
            functions,
        ),
        #
        f_prices = core.prices.init_constant(
            n_agents,
            n_products,
            np.ones((n_agents, n_products))
            # only the products for sale count
        ),
        f_inventory = core.inventory.init_constant(
            n_agents,
            n_products,
            inventory,
        ),
        f_cash = core.cash.init_constant(
            n_agents,
            np.ones((n_products,))
        ),
    )

    init_orders = np.zeros(state.orders.shape, float)
    for p in range(state.n_products):
        init_orders[p][p][(p - 1) % n_products] = 1

    state = replace(
        state, orders = init_orders
    )
    sim = core.sims.OrderDriven(state, results)

    for _ in range(length):
        sim = sim.step(
            f_order=core.orders.cum_net_chg_inventory,
            f_deliver=core.deliveries.order_by_rel,
            f_produce=core.production.offset_deliveries,
            f_price=core.prices.cum_sqrt_imbalance,
        )

    results = sim.results.save()

    return fp

def run(
    fp: Path,
    variant: str | None = None,

    # TODO: other args
    # force same types if shared across variant
    # but all should be optional
    # then below pass explicitly per variant, no **kwargs

    # TODO: type cast args

) -> Path:
    if variant is None or variant == "example":
        return example(fp)
    raise ValueError(variant)