import datetime as dt

from .assets import *
from .calcs import *
from .params import *
from .domain import *
from .project import *
from . import baskets

from ....engine import *

def f_cv(
    container: str | Strategies, *args, **kwargs
):
    s = cv(*args, **kwargs)
    if isinstance(container, str):
        container = context.get().get(container, Strategies)
    return s.bind(container, t=Strategies)

def f_strategy(
    container_id: str, *args, **kwargs    
):
    s = Strategy(*args, **kwargs)
    container = context.get().get(container_id, Strategies)
    container, s = s.bind(container)
    return container, s

def f_basket(
    container_id: str, *args, **kwargs
):
    basket = Basket(*args, **kwargs)
    container = context.get().get(container_id, Strategies)
    container, basket = basket.bind(container, t=Strategies)
    # TODO: wrap into some kind of shared object?
    return container, basket



def bind_basket(
    name: str,
    container: str,
    asset: tuple[Strategy],
    params: StrategyParams,
):
    return Bound(
        f_basket,
        name,
        container=bound(container),
        params=fdict(
            asset=asset,
            params=params,
            name=name,
        )
    ).unpack("asset", "params")
    
def bind_strategy(
    name: str,
    container: str,
    asset: TAsset,
    params: StrategyParams,
):
    return Bound(
        f_basket,
        name,
        container=bound(container),
        params=fdict(
            asset=asset,
            params=params,
            # name?
        )
    ).unpack("asset", "params")
    


def bind_cv(
    symbol: str,
    container: str | Strategies,
    # start: dt.date,
    # end: dt.date,
    signal: TSignal | None = None,
    notional_cap: float | None = 10.,
    span: int = 48,
):
    ac, sym = symbol.split("-", maxsplit=1)
    s = cv(
        (ac, sym),
        span=span,
        cap=None if notional_cap is None else CapNotional(notional_cap),
        signal=signal,
    )
    # res = res.unpack("signal")
    if isinstance(container, str):
        container = context.get().get(container, Strategies)
    return s.bind(container, t=Strategies)