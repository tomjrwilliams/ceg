from __future__ import annotations
from abc import abstractmethod
from typing import overload, Callable, Protocol
from .types import *


@dataclass(frozen=True)
class N:
    agents: int
    products: int


class Results(Protocol):
    @classmethod
    def load(cls, fp: str) -> Results: ...


@dataclass(frozen=True)
class State:
    n_agents: int
    n_products: int
    technologies: F64_2D  # products, products
    rels: F64_2D  # agents, agents
    functions: F64_2D  # agents, products
    prices: F64_2D  # agents, products
    inventory: F64_2D  # agents, products
    inflows: F64_2D  # agents, products
    outflows: F64_2D  # agents, products
    production: F64_2D  # agents, products
    orders: F64_3D  # products, agents, agents
    deliveries: F64_3D  # products, agents, agents
    cash: F64_1D  # agents

    t: int = 0

    @property
    def n(self) -> N:
        return N(self.n_agents, self.n_products)

    @overload
    def update(self, *, technologies: F64_2D) -> State: ...

    @overload
    def update(self, *, rels: F64_2D) -> State: ...

    @overload
    def update(self, *, functions: F64_2D) -> State: ...

    @overload
    def update(self, *, prices: F64_2D) -> State: ...

    @overload
    def update(self, *, inventory: F64_2D) -> State: ...

    @overload
    def update(self, *, inflows: F64_2D) -> State: ...

    @overload
    def update(self, *, outflows: F64_2D) -> State: ...

    @overload
    def update(self, *, production: F64_2D) -> State: ...

    @overload
    def update(self, *, orders: F64_3D) -> State: ...

    @overload
    def update(self, *, deliveries: F64_3D) -> State: ...

    @overload
    def update(self, *, cash: F64_1D) -> State: ...

    def update(self, **kwargs) -> State:
        return replace(self, **kwargs)

    # init

    @classmethod
    def init(
        cls,
        n_agents: int,
        n_products: int,
        #
        f_technology: Callable[[int], F64_2D],
        f_rels: Callable[[int], F64_2D],
        #
        f_functions: Callable[[State], F64_2D],
        #
        f_prices: Callable[[State], F64_2D],
        f_inventory: Callable[[State], F64_2D],
        f_cash: Callable[[State], F64_1D],
    ) -> State:
        A = (n_agents,)
        A_x_P = (n_agents, n_products)
        P_x_A_x_A = (n_products, n_agents, n_agents)

        state = State(
            n_agents=n_agents,
            n_products=n_products,
            technologies=f_technology(n_products),
            rels=f_rels(n_agents),
            functions=np.ones(A_x_P, float),
            prices=np.ones(A_x_P, float),
            inventory=np.zeros(A_x_P, float),
            inflows=np.zeros(A_x_P, float),
            outflows=np.zeros(A_x_P, float),
            production=np.zeros(A_x_P, float),
            orders=np.zeros(P_x_A_x_A, float),
            deliveries=np.zeros(P_x_A_x_A, float),
            cash=np.zeros(A, float),
        )

        state = state.update(functions=f_functions(state))
        state = state.update(
            prices=f_prices(state),
        ).update(
            inventory=f_inventory(state),
        )
        state = state.update(cash=f_cash(state))

        return state


__all__ = ["State"]
