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


def ex_00_strat():
    """
    >>> spec = Strategy(
    ...     asset=AssetConcrete("FUT", "ES"),
    ...     signal=SignalOsc(32, 96, 48),
    ...     scale=ScaleVol(span=48, ann=False)
    ... )
    >>> dom = StratDomain.new()
    >>> dom, univ = dom.with_univ("strat")
    >>> dom, univ, bts = dom.with_strats(univ, spec)
    >>> bt = bts[0].get_bt(dom, univ)
    >>> pf = DailyPfolio.new(bt)
    >>> dom = dom.with_obj(pf).sync()
    >>> pos = bts[0].get_pos(dom)
    >>> rx = bts[0].get_rx(dom)
    >>> perf = bts[0].get_perf(dom)
    >>> pf_pos = DailyPfolioPos.new(pf, "ES")
    >>> pf_rx = DailyPfolioRx.new(pf, "ES")
    >>> pf_agg = DailyPfolioRx.new(pf, None)
    >>> pos.value_key_val(dom).get(pos.path).tail(8)
    shape: (8,)
    Series: 'strat-0-bt-uv-000-pos' [f64]
    [
        -0.006311
        null
        null
        -0.006753
        -0.006675
        -0.006903
        -0.007089
        -0.007451
    ]
    >>> pf_pos.value_key_val(dom).get(pf_pos.path).tail(8)
    shape: (8,)
    Series: 'strat-0-bt-uv-000-pf-pos-ES' [f64]
    [
        -0.006311
        null
        null
        -0.006753
        -0.006675
        -0.006903
        -0.007089
        -0.007451
    ]
    >>> rx.value_key_val(dom).get(rx.id).tail(8)
    shape: (8,)
    Series: 'strat-0-bt-uv-000-rx' [f64]
    [
        -0.032185
        null
        null
        -0.613726
        -0.074286
        0.447206
        0.139791
        0.824148
    ]
    >>> pf_rx.value_key_val(dom).get(pf_rx.id).tail(8)
    shape: (8,)
    Series: 'strat-0-bt-uv-000-pf-rx-ES' [f64]
    [
        -0.032185
        null
        null
        -0.613726
        -0.074286
        0.447206
        0.139791
        0.824148
    ]
    >>> pf_agg.value_key_val(dom).get(pf_agg.id).tail(8)
    shape: (8,)
    Series: 'strat-0-bt-uv-000-pf-rx' [f64]
    [
        -0.032185
        null
        null
        -0.613726
        -0.074286
        0.447206
        0.139791
        0.824148
    ]
    >>> v_perf = perf.value_key_val(dom).get(perf.id)
    >>> v_perf.row(0, round=4, repr=True)
    key strat-0-bt-uv-000-rx-bench
    mu 4.2188
    sd 11.2066
    rms 11.2085
    skew -1.3384
    shrp 0.3794
    sort 0.4375
    dd 26.6185
    calm 0.1585
    """
    return
