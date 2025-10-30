
from __future__ import annotations
from typing import overload
import numpy as np
from .types import *
from . import State

def concat_empty(incr: int, v: np.ndarray, axis: int = 0):
    shape = tuple([
        d if i != axis else incr
        for i, d in enumerate(v.shape)
    ])
    return np.concat(
        (v, np.zeros(shape, v.dtype) * np.nan), axis=axis
    )

@dataclass(frozen=True)
class Results:
    fp: str
    n_agents: int
    n_products: int
    technologies: F64_3D # products, products
    rels: F64_3D # agents, agents
    functions: F64_3D # agents, products
    prices: F64_3D # agents, products
    inventory: F64_3D # agents, products
    inflows: F64_3D # agents, products
    outflows: F64_3D # agents, products
    production: F64_3D # agents, products
    orders: F64_4D # products, agents, agents
    deliveries: F64_4D # products, agents, agents
    cash: F64_2D # agents

    t: int = 0

    batch: int | None = None
    length: int | None = None

    # TODO: ideally can reload an existing sim and continue
    @classmethod
    def init(
        cls,
        fp: str,
        n_agents: int,
        n_products: int,
        batch: int | None = None,
        length: int | None = None,
        fallback: int = 128, # TODO: rename better
    ) -> Results:
        A = n_agents
        P = n_products

        if length is not None:
            T = length
        elif batch is not None:
            T = batch
        else:
            T = fallback
            
        return Results(
            # TODO: given state, just expand_dims
            fp=fp,
            n_agents=n_agents,
            n_products=n_products,
            #
            technologies=np.zeros((T, P, P), float)* np.nan,
            rels=np.zeros((T, A, A), float)* np.nan,
            functions=np.zeros((T, A, P), float)* np.nan,
            prices=np.zeros((T, A, P), float)* np.nan,
            inventory=np.zeros((T, A, P), float)* np.nan,
            inflows=np.zeros((T, A, P), float)* np.nan,
            outflows=np.zeros((T, A, P), float)* np.nan,
            production=np.zeros((T, A, P), float)* np.nan,
            orders=np.zeros((T, P, A, A), float)* np.nan,
            deliveries=np.zeros((T, P, A, A), float)* np.nan,
            cash=np.zeros((T, A), float)* np.nan,
            #
            batch=batch,
            length=length,
        )

    def save(self, compressed=False) -> Results: 
        f = np.savez_compressed if compressed else np.savez
        f(
            self.fp,
            shape=np.array([
                self.n_agents,
                self.n_products,
                self.t,
            ], int),
            technologies=self.technologies,
            rels=self.rels,
            functions=self.functions,
            prices=self.prices,
            inventory=self.inventory,
            inflows=self.inflows,
            outflows=self.outflows,
            production=self.production,
            orders=self.orders,
            deliveries=self.deliveries,
            cash=self.cash,
        )
        return self

    def acc(self, state: State) -> Results:
        row = state.t

        # TODO: incremental save if batch
        # if self.batch is not None:
            # row = (state.t + 1) % self.batch
            # if row == 0:
            #     self = self.save(state)
            # else:
            #     row -= 1
            
        if row + 1 > self.cash.shape[0]:
            self = self.resize()

        self.technologies[row] = state.technologies
        self.rels[row] = state.rels
        self.functions[row] = state.functions
        self.prices[row] = state.prices
        self.inventory[row] = state.inventory
        self.inflows[row] = state.inflows
        self.outflows[row] = state.outflows
        self.production[row] = state.production
        self.orders[row] = state.orders
        self.deliveries[row] = state.deliveries
        self.cash[row] = state.cash

        return replace(self, t = state.t)

    def resize(self) -> Results:
        if self.batch is None:
            incr = self.technologies.shape[0]
        else:
            incr = self.batch
        return replace(
            self,
            technologies=concat_empty(
                incr, self.technologies, axis=0
            ),
            rels=concat_empty(
                incr, self.rels, axis=0
            ),
            functions=concat_empty(
                incr, self.functions, axis=0
            ),
            prices=concat_empty(
                incr, self.prices, axis=0
            ),
            inventory=concat_empty(
                incr, self.inventory, axis=0
            ),
            inflows=concat_empty(
                incr, self.inflows, axis=0
            ),
            outflows=concat_empty(
                incr, self.outflows, axis=0
            ),
            production=concat_empty(
                incr, self.production, axis=0
            ),
            orders=concat_empty(
                incr, self.orders, axis=0
            ),
            deliveries=concat_empty(
                incr, self.deliveries, axis=0
            ),
            cash=concat_empty(
                incr, self.cash, axis=0
            ),
        )

    @overload
    @classmethod
    def load(cls, fp: str, *, technologies: bool) -> F64_3D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, rels: bool) -> F64_3D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, functions: bool) -> F64_3D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, prices: bool) -> F64_3D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, inventory: bool) -> F64_3D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, inflows: bool) -> F64_3D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, outflows: bool) -> F64_3D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, production: bool) -> F64_3D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, orders: bool) -> F64_4D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, deliveries: bool) -> F64_4D: ...

    @overload
    @classmethod
    def load(cls, fp: str, *, cash: bool) -> F64_2D: ...

    @overload
    @classmethod
    def load(cls, fp: str) -> Results: ...

    @classmethod
    def load(cls, fp: str, **kwargs):
        with np.load(fp) as blob:
            if len(kwargs):
                key, *keys = tuple(kwargs.keys())
                assert len(keys) == 0, kwargs
                assert kwargs[key], kwargs
                return blob[key]

            shape = blob["shape"]
            
            return Results(
                fp=fp,
                n_agents = int(shape[0]),
                n_products = int(shape[1]),
                technologies=blob["technologies"],
                rels=blob["rels"],
                functions=blob["functions"],
                prices=blob["prices"],
                inventory=blob["inventory"],
                inflows=blob["inflows"],
                outflows=blob["outflows"],
                production=blob["production"],
                orders=blob["orders"],
                deliveries=blob["deliveries"],
                cash=blob["cash"],
                t=int(shape[2])
            )

    def update(
        self,
        fp: str | None = None,
        batch: int | None = -1,
        length: int | None = -1,
    ):
        return replace(
            self,
            fp=self.fp if fp is None else fp,
            batch=self.batch if batch == -1 else batch,
            length=self.length if length == -1 else length,
        )

# TODO:
# utils to reshape from a single row to the original tensor
# and then to a df with <name><i><...> col naming
