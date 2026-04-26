from typing import Callable
import numpy as np
import pandas as pd
from .types import *

# TODO: tall
# TODO: for now, we assume constant!


def df_wide(
    v: F64_2D,
    agents: list[str] | None = None,
    agent: str = "p",
) -> pd.DataFrame:
    n = v.shape[0]
    assert v.shape[1] == n
    if agents is not None:
        assert len(agents) == n
        assert len(agents) == n
    else:
        agents = enumerated_labels(agent, n, justify=3)
    return pd.DataFrame(
        data={
            name: v[:, i] for i, name in enumerate(agents)
        },
        index=agents,
    )


def init_constant(
    n_agents: int,
    v: F64_2D,
) -> Callable[[int], F64_2D]:
    assert n_agents == v.shape[0]
    assert n_agents == v.shape[1]

    def f(n: int) -> F64_2D:
        assert n == n_agents, (n, n_agents)
        return v

    return f


# def init_agent_rels(
#     n_agents: int,
#     f_sample: Callable[[tuple[int, ...]], np.ndarray]
# ) -> np.ndarray:
#     # note: directed graph (rel is not bi-directional)
#     # eg. first generate random noise
#     # then perhaps unit norm?
#     # then boolean mask
#     shape = (n_agents, n_agents)
#     # TODO: in future, if agents produce multiple outputs, also over outputs
#     return f_sample(shape)
