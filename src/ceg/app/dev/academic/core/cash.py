from typing import Callable
import numpy as np
import pandas as pd

from .types import *
from .states import State

def df_tall(
    v: F64_2D,
    agents: list[str] | None = None,
    agent: str = "a",
) -> pl.DataFrame:
    t = v.shape[0]
    a = v.shape[1]
    if agents is not None:
        assert len(agents) == v.shape[1]
    else:
        agents = enumerated_labels(agent, a, justify=3)
    t_agents = pl.Enum(agents)
    return f64_2d_to_df(
        v,
        "cash",
        {
            "time": pl.Int32,
            "agent": t_agents,
        }
    )

def df_wide(
    v: F64_2D,
    agents: list[str] | None = None,
    agent: str = "a",
) -> pd.DataFrame:
    a = v.shape[1]
    if agents is not None:
        assert len(agents) == v.shape[1]
    else:
        agents = enumerated_labels(agent, a, justify=3)
    return pd.DataFrame(
        data={
            name: v[:, i]
            for i, name in enumerate(agents)
        },
        index=list(range(v.shape[0]))
    )


def init_constant(
    n_agents: int,
    v: F64_1D,
) -> Callable[[State], F64_1D]:
    assert n_agents == v.shape[0]
    def f(state: State) -> F64_1D:
        assert state.n_agents==n_agents
        return v
    return f


# def init_agent_cash(
#     n_agents: int,
#     f_sample: Callable[[tuple[int, ...]], np.ndarray]
# ) -> np.ndarray:
#     shape = (n_agents,)
#     # will this later include other cash types? debt?
#     return f_sample(shape)

