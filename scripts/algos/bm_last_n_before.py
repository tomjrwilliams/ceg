
import sys
import time

import numpy as np
import numba as nb

sys.path.append("./src")
from ceg.core import algos

variants = {
    "bin": algos.last_n_before,
    "loop": algos.last_n_before_naive,
    "np": algos.last_n_before_np, 
}
vs = {
    p: np.linspace(0, 2 ** p, 2 ** p)
    for p in range(3, 18, 3)
}
results = []

print({p: v.shape for p, v in vs.items()})

runs = 30

for p, v in vs.items():
    n = 2 ** p
    offsets= [
        int(n/30), int(n / 10), int(n/3), int(n * .8)
    ]
    for offset in offsets:
        for take in offsets:
            if offset == 0 or take == 0:
                continue
            for i in range(n):
                v0 = algos.last_n_before_np(v, v, take, i, offset, p)
                v1 = algos.last_n_before(v, v, take, i, offset, p)
                assert np.all(v0 == v1), dict(v0=v0, v1=v1, i=i, offset=offset, n=n, p=p)
            res = {}
            for variant, f in variants.items():
                acc = 0
                f(v, v, take, 0, 1, p)
                for _ in range(runs):
                    t = time.perf_counter_ns()
                    for i in range(n):
                        _ = f(v, v, take, i, offset, p)
                    acc += (time.perf_counter_ns()-t) / 1_000_000
                res[variant] = round(acc / runs, 4)
            r = dict(
                n=n,
                offset=offset,
                take=take,
                **res,
            )
            print(r)
            results.append(r)

