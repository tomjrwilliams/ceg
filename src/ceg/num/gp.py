from typing import Callable
from functools import partial, lru_cache

from dataclasses import dataclass

from llvmlite.ir import Value
from numba import vectorize, float64, int32
import numpy as np

# cov_k is column vector, .T is row

def cov_next(
    cov_C, # square mat
    cov_k, # vec of cov from new datapoint to prev (via cov kernel funcs)
    k # scalar (cov of point to itself via kernel func)
):
    return np.hstack((
        np.vstack((
            cov_C, cov_k.T
        )),
        np.vstack((
            cov_k, k
        ))
    ))

def cov_inv_next(
    cov_C_inv, # as above
    cov_k, # as above
    k, # as above
):
    m = (k - np.matmul(np.matmul(cov_k.t, cov_C_inv), cov_k)) ** -1
    m_vec = -m * (np.matmul(cov_C_inv, cov_k))
    M = cov_C_inv + (1 / m) * (np.matmul(m_vec, m_vec.T) / m)
    return np.hstack((
        np.vstack((M, m.T)),
        np.vstack((m, m))
    ))

def mean_next(
    cov_C_inv, # square mat
    cov_k, # vec of cov from new datapoint to prev (via cov kernel funcs)
    data # of size cov_C
):
    # data of size cov_C_inv
    # cov_k doesnt include self to self (??)
    return np.matmul(cov_k.T, np.matmul(cov_C_inv, data))

def variance_next(
    cov_C_inv, # square mat
    cov_k, # vec of cov from new datapoint to prev (via cov kernel funcs)
    k, # scalar (cov of point to itself via kernel func)
    # data, # of size cov_C
):
    return k - np.sum(
        np.multiply(np.matmul(cov_k.T, cov_C_inv), cov_k.T), axis=1
    )


@lru_cache(maxsize=32, typed=True)
def kernel_diag_const(sig):

    sig_sq = np.square(sig)
    
    @vectorize([float64(int32, int32)])
    def f(x0, x1):
        if x0 == x1:
            return sig_sq
        return 0

    return f

@lru_cache(maxsize=64, typed=True)
def kernel_sq_exp(l, sig):
    
    sig_sq = np.square(sig)
    l_sq = np.square(l)

    @vectorize([float64(float64, float64)])
    def f(x0, x1):
        return sig_sq * np.exp(-1 * np.square(x0 - x1) / (2 * l_sq))

    return f

@dataclass(frozen=True)
class Kernel:
    f: Callable
    kws: dict
    kind: str = "value"

    def elementwise(self, x0, x1):
        x = np.vstack((x0, x1))
        return self.f(**self.kws).reduce(x, axis=0)

    def outer(self, x0, x1):
        return self.f(**self.kws).outer(x0, x1)

def kernel_cov(x, *kernels: Kernel) -> np.ndarray:
    res = None
    i = np.linspace(0, x.shape[0] - 1, num = x.shape[0]).astype(np.int32)
    for kernel in kernels:
        if kernel.kind == "value":
            v = kernel.outer(x, x)
        elif kernel.kind == "index":
            v = kernel.outer(i, i)
        else:
            raise ValueError(kernel)
        if res is None:
            res = v
        else:
            res += v
    assert res is not None, res
    return res

import matplotlib.pyplot as plt

import sys

if __name__ == "__main__":
    rng = np.random.default_rng(seed=69)

    x = 1 + rng.normal(0., 1., (100,))
    y = (0.5 * x) + rng.normal(2., .1, (100,))

    y_mu = np.mean(y)
    y_cent = y - y_mu

    x_sig = np.std(x)

    l = float(sys.argv[1])
    sig = float(sys.argv[2])

    kernels = (
        Kernel(kernel_sq_exp, dict(l=l, sig=sig)),
        Kernel(kernel_diag_const, dict(sig = x_sig), kind="index"),
    )

    cov_C = kernel_cov(x, *kernels)

    cov_C_inv = np.linalg.inv(cov_C)

    x_new = np.array(sys.argv[3].split(",")).astype(np.float64)

    kernel = kernels[0]

    cov_k = kernel.outer(x, x_new)
    k = kernel.elementwise(x_new, x_new)

    y_prd = mean_next(cov_C_inv, cov_k, y_cent) + y_mu
    y_var = variance_next(cov_C_inv, cov_k, k)

    print(y_var.shape)

    y_sig = np.sqrt(y_var)

    x_res = x_new
    y_res = [-y_sig + y_prd, y_prd, y_sig + y_prd]

    plt.scatter(x, y, color="blue")
    for i, y_r in enumerate(y_res):
        if i == 1:
            plt.scatter(x_res, y_r, color="red")
        plt.plot(x_res, y_r, color="red")

    plt.show()