from .root import DOMAINS
from ... import explorer

from ....objs.dev import fut

DOMAINS["es"] = lambda sync: explorer.AppDomain([
    fut.d.bars.ord.FutDailyOrdinalDomain.new(),
    fut.d.bars.rx.FutDailyRxDomain.new(),
], sync)
DOMAINS["tn"] = lambda sync: explorer.AppDomain([
    fut.d.bars.ord.FutDailyOrdinalDomain.new(symbol="TN"),
    fut.d.bars.rx.FutDailyRxDomain.new(symbol="TN"),
], sync)