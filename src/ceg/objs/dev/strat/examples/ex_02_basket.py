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


def ex_02_basket():
    """
    >>> spec_es = Strategy(
    ...     asset=AssetConcrete("FUT", "ES"),
    ...     signal=SignalOsc(32, 96, 48),
    ...     scale=ScaleVol(span=48, ann=True),
    ...     rescale=ScaleVol(span=256, ann=True)
    ... )
    >>> spec_tn = Strategy(
    ...     asset=AssetConcrete("FUT", "TN"),
    ...     signal=SignalOsc(32, 96, 48),
    ...     scale=ScaleVol(span=48, ann=True),
    ...     rescale=ScaleVol(span=256, ann=True)
    ... )
    >>> dom = StratDomain.new()
    >>> dom, univ = dom.with_univ("basket", depth=0)
    >>> dom, univ, bts = dom.with_strats(univ, spec_es, spec_tn)
    >>> dom, _, agg = dom.with_rx_agg(
    ...     Container.new(univ.get_parent(dom), "rx"), bts
    ... )
    >>> spec_b = Basket(
    ...     asset=tuple([
    ...         Strategy(asset=bt)
    ...         for bt in bts
    ...     ]),
    ...     name="ES-TN",
    ...     scale=ScaleVol(span=256, rx = agg.id)
    ... )
    >>> dom, univ = dom.with_univ("basket", depth=1)
    >>> dom, univ, bts = dom.with_strats(univ, spec_b)
    >>> bt = bts[0].get_bt(dom, univ)
    >>> pf = DailyPfolio.new(bt)
    >>> dom = dom.with_obj(pf).sync()
    >>> pos = bts[0].get_pos(dom, 0)
    >>> rx = bts[0].get_rx(dom, 0)
    >>> rx_agg = bts[0].get_rx(dom)
    >>> perf = bts[0].get_perf(dom)
    >>> pf_pos = DailyPfolioPos.new(pf, "ES")
    >>> pf_rx = DailyPfolioRx.new(pf, "ES")
    >>> pf_agg = DailyPfolioRx.new(pf, None)
    >>> pos.value_key_val(dom).get(pos.path).tail(8)
    shape: (8,)
    Series: 'basket-1-bt-mv-000-pos-0' [f64]
    [
        1.197239
        null
        null
        1.201915
        1.203492
        1.20819
        1.211779
        1.216455
    ]
    >>> pf_pos.value_key_val(dom).get(pf_pos.path).tail(8)
    shape: (8,)
    Series: 'basket-1-bt-mv-000-pf-pos-ES' [f64]
    [
        -0.000472
        null
        null
        -0.000507
        -0.000502
        -0.000521
        -0.000537
        -0.000566
    ]
    >>> rx.value_key_val(dom).get(rx.id).tail(8)
    shape: (8,)
    Series: 'basket-1-bt-mv-000-rx-0' [f64]
    [
        -0.002399
        null
        null
        -0.045924
        -0.00558
        0.033638
        0.010556
        0.062418
    ]
    >>> pf_rx.value_key_val(dom).get(pf_rx.id).tail(8)
    shape: (8,)
    Series: 'basket-1-bt-mv-000-pf-rx-ES' [f64]
    [
        -0.002399
        null
        null
        -0.045924
        -0.00558
        0.033638
        0.010556
        0.062418
    ]
    >>> rx_agg.value_key_val(dom).get(rx.id).tail(8)
    shape: (8,)
    Series: 'basket-1-bt-mv-000-rx-0' [f64]
    [
        -0.002399
        null
        null
        -0.045924
        -0.00558
        0.033638
        0.010556
        0.062418
    ]
    >>> pf_agg.value_key_val(dom).get(pf_agg.id).tail(8)
    shape: (8,)
    Series: 'basket-1-bt-mv-000-pf-rx' [f64]
    [
        -0.002758
        null
        null
        -0.05071
        -0.003254
        0.030501
        0.007361
        0.074568
    ]
    >>> v_perf = perf.value_key_val(dom).get(perf.id)
    >>> v_perf.row(0, round=4, repr=True)
    key basket-1-bt-mv-000-rx-bench
    mu 0.5207
    sd 1.0542
    rms 1.0546
    skew -0.7435
    shrp 0.4979
    sort 0.623
    dd 2.3805
    calm 0.2188
    """
    return
