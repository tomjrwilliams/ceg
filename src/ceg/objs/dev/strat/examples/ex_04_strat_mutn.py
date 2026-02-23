
import polars as pl

from ..... import apis
from .....apis import t
from .....apis import kv
from .....engine import *

from ...agg import *
from ...pfo import *

from ..assets import *
from ..calcs import *
from ..signals import *
from ..params import *
from ..domain import *



def ex_04_strat_ls():
    """
    >>> spec = Strategy(
    ...     asset=AssetConcrete("FUT", "ES"),
    ...     signal=SignalOsc(32, 96, 48),
    ...     scale=ScaleVol(span=48, ann=False)
    ... )
    >>> dom = StratDomain.new()
    >>> dom, univ = dom.with_univ("strat")
    >>> dom, univ, (res,) = dom.with_bt(univ, spec, [])
    >>> dom, (res,) = dom.with_perf(
    ...     [res],
    ...     mutation=[
    ...         mutate_signal("levl", "sign"),
    ...     ]
    ... )
    >>> _ = dom.sync()
    >>> res.get_perf(dom).display(dom)
    key strat-0-bt-uv-000-rx
    mu 4.2188
    sd 11.2066
    rms 11.2085
    skew -1.3384
    shrp 0.3794
    sort 0.4375
    dd 26.6185
    calm 0.1585
    >>> res.get_perf(dom, 0).display(dom, row=1)
    key strat-0-bt-uv-000-rx
    signal_lower 0.0
    signal_upper None
    mu 5.1832
    sd 10.5798
    rms 10.5837
    skew -1.5898
    shrp 0.4938
    sort 0.5643
    dd 23.2205
    calm 0.2232
    >>> res.get_perf(dom, 0).display(dom, row=0)
    key strat-0-bt-uv-000-rx
    signal_lower None
    signal_upper 0.0
    mu -0.9647
    sd 3.6903
    rms 3.6904
    skew -0.1631
    shrp -0.2635
    sort -0.3551
    dd 32.9803
    calm -0.0292
    """
    return


def ex_04_strat_noise():
    """
    >>> spec = Strategy(
    ...     asset=AssetConcrete("FUT", "ES"),
    ...     signal=SignalOsc(32, 96, 48),
    ...     scale=ScaleVol(span=48, ann=False)
    ... )
    >>> dom = StratDomain.new()
    >>> dom, univ = dom.with_univ("strat")
    >>> dom, univ, (res,) = dom.with_bt(univ, spec, [])
    >>> dom, (res,) = dom.with_perf(
    ...     [res],
    ...     mutation=[
    ...         mutate_signal("rel", 0.1),
    ...         mutate_signal("rel", 0.2),
    ...         mutate_signal("rel", 0.3),
    ...         mutate_signal("abs", 0.1),
    ...         mutate_signal("abs", 0.2),
    ...         mutate_signal("abs", 0.3),
    ...     ]
    ... )
    >>> _ = dom.sync()
    >>> agg = dict(
    ...     mu=lambda k: pl.col(k).mean(),
    ...     sd=lambda k: pl.col(k).std(),
    ... )
    >>> res.get_perf(dom).display(dom)
    key strat-0-bt-uv-000-rx
    mu 4.2188
    sd 11.2066
    rms 11.2085
    skew -1.3384
    shrp 0.3794
    sort 0.4375
    dd 26.6185
    calm 0.1585
    """
    # >>> res.get_perf(dom, 0).display(dom, **agg)
    return