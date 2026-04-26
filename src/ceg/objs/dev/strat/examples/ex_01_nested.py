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


def ex_01_nested():
    """
    >>> spec = Strategy(
    ...     asset=AssetConcrete("FUT", "ES"),
    ...     signal=SignalOsc(32, 96, 48),
    ...     scale=ScaleVol(span=48, ann=True)
    ... )
    >>> dom = StratDomain.new()
    >>> dom, univ = dom.with_univ("nested", depth=0)
    >>> dom, univ, bts = dom.with_strats(univ, spec)
    >>> dom, univ = dom.with_univ("nested", depth=1)
    >>> spec = Strategy(
    ...     asset=bts[0],
    ...     scale=ScaleVol(span=256, ann=True)
    ... )
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
    Series: 'nested-1-bt-uv-000-pos' [f64]
    [
        1.352015
        null
        null
        1.357296
        1.358916
        1.36418
        1.367525
        1.372681
    ]
    >>> pf_pos.value_key_val(dom).get(pf_pos.path).tail(8)
    shape: (8,)
    Series: 'nested-1-bt-uv-000-pf-pos-ES' [f64]
    [
        -0.000533
        null
        null
        -0.000573
        -0.000567
        -0.000589
        -0.000606
        -0.000639
    ]
    >>> rx.value_key_val(dom).get(rx.id).tail(8)
    shape: (8,)
    Series: 'nested-1-bt-uv-000-rx' [f64]
    [
        -0.002709
        null
        null
        -0.05186
        -0.006302
        0.037982
        0.011919
        0.07044
    ]
    >>> pf_rx.value_key_val(dom).get(pf_rx.id).tail(8)
    shape: (8,)
    Series: 'nested-1-bt-uv-000-pf-rx-ES' [f64]
    [
        -0.002709
        null
        null
        -0.05186
        -0.006302
        0.037982
        0.011919
        0.07044
    ]
    >>> pf_agg.value_key_val(dom).get(pf_agg.id).tail(8)
    shape: (8,)
    Series: 'nested-1-bt-uv-000-pf-rx' [f64]
    [
        -0.002709
        null
        null
        -0.05186
        -0.006302
        0.037982
        0.011919
        0.07044
    ]
    >>> v_perf = perf.value_key_val(dom).get(perf.id)
    >>> v_perf.row(0, round=4, repr=True)
    key nested-1-bt-uv-000-rx-bench
    mu 0.4026
    sd 1.0574
    rms 1.0576
    skew -1.1026
    shrp 0.3838
    sort 0.4495
    dd 2.5966
    calm 0.155
    """
    return
