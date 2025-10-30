from __future__ import annotations
from typing import Callable, Protocol
from .types import *
from .states import State
from .results import Results

class Sim(Protocol):

    @property
    def state(self) -> State: ...

    @property
    def results(self) -> Results: ...

    def step(self, *args, done: bool = False) -> Sim: ...

@dataclass(frozen=True)
class OrderDriven:
    state: State
    results: Results
    
    def step(
        self,
        f_order: Callable[[State], F64_3D],
        f_deliver: Callable[[State], F64_3D],
        f_produce: Callable[[State], F64_2D],
        f_price: Callable[[State], F64_2D],
    ) -> OrderDriven:
        state = self.state

        # make deliveries given last round of orders
        state = state.update(
            deliveries=f_deliver(state)
        )

        # TODO: does this double count?

        state = state.update(
            outflows=-np.sum(state.deliveries, axis=(2,))
        ).update(
            inflows=np.sum(state.deliveries, axis=(1,)).T
        )

        state = state.update(
            inventory=(
                state.inventory
                + state.inflows
                + state.outflows
            )
        )

        state = state.update(
            production=f_produce(state)
        )
        state = state.update(
            inventory = state.inventory + state.production
        )
        
        state = state.update(
            cash=state.cash + np.sum(
                (
                    state.outflows
                    * state.prices
                ) + (
                    state.inflows
                    * state.prices
                ),
                axis=(1,),
            )
        )

        # we change prices here so they're known at the point orders are made
        # only the inventory holder changes prices here (based on total orders to them)
        # in a shared price model, it's rather total orders vs total available to offer
        state = state.update(
            prices=f_price(state)
        )

        state = state.update(
            orders=f_order(state)
        )

        results = self.results.acc(state)

        state = replace(state, t = state.t + 1)

        return OrderDriven(state, results)

__all__ = ["Sim", "OrderDriven"]