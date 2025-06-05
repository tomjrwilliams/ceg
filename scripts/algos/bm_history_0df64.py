import sys
import time

import numpy as np
import numba as nb

sys.path.append("./src")
from ceg.core.histories import History

variants = {
    "py": History.D0_F64,
    # "jit": histories.History_0D_F64_jit,
    # NOTE: jit is materially slower?
}

expon_len = list(range(3, 6))
expon_req = list(range(5, 10))
expon_lim = list(range(2, 7))

RUNS = 50

for exp_len in expon_len:
    for exp_req in expon_req:
        req = 2 ** exp_req
        vs = np.linspace(0, 10 ** exp_len, 10 ** exp_len)
        res = {}
        for variant, cls in variants.items():
            for lim in expon_lim:
                cls = History.D0_F64
                acc = 0
                for _ in range(RUNS):
                    obj = cls.new(
                        vs[0], req, limit = int(2 ** lim)
                    )
                    t = time.perf_counter_ns()
                    for i, v in enumerate(vs):
                        obj.append(v, i)
                    acc += (time.perf_counter_ns()-t) / 1_000_000
                res[variant + f"_{lim}"] = round(acc / RUNS, 4)
        r = dict(
            length = 10 ** exp_len,
            req = 2 ** exp_req,
            **res,
        )
        print(r)

# NOTE: seems the difference is marginal at most