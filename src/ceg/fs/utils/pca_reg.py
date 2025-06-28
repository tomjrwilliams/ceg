
import numpy as np
from scipy.linalg import qr

from numpy import array, zeros
from numpy.linalg import solve

import scipy.optimize
from scipy.optimize import NonlinearConstraint

rng = np.random.default_rng(69)

def unit_norm_beta(vs, ws: np.ndarray | None = None, strict: bool = True):
    
    con = lambda x: np.dot(x, x.T)
    nlc = NonlinearConstraint(con, 0.99, 1.01)

    # TODO: take optional obs weights, just multiply into second term?

    f_norm = lambda x: np.square((np.dot(x, x.T) - 1))

    if ws is None:
        f_loss = lambda x: np.mean(np.square(np.dot(vs, x)))
    else:
        f_loss = lambda x: np.sum(np.square(np.dot(vs, x)) * ws)
    
    f = lambda x: f_norm(x) - f_loss(x)

    w = rng.normal(0, 1, (vs.shape[1],))
    opt = scipy.optimize.minimize(f, w, constraints=nlc)
    w = opt.x

    if strict:
        assert opt.success

    return w

def exp_weights(n: int, span: float):
    a = 2 / (span + 1)
    decay = 1 - a
    weights = 1 * (decay ** np.arange(n))
    return np.divide(weights, sum(weights))

def compare_pc(vs, factors):
    # n_obs, n_features

    U, e, _ = np.linalg.svd(vs.T, full_matrices=False)
    U = np.round(U, 3)
    e = np.round(e, 3)

    # vs_w_intercept = np.hstack([vs, np.expand_dims(ones, 1)])

    print("eig", e)
    for i in range(vs.shape[1]):
        pc = U[:, i]
        print("pc", i, pc)
        print("pc norm", i, np.dot(pc, pc))

    unit_beta = unit_norm_beta(vs, exp_weights(vs.shape[0], 16))
    print("unit beta", unit_beta)
    print("unit norm", np.round(np.dot(unit_beta, unit_beta.T), 2))

    for f in range(factors.shape[1]):
        beta = np.round(
            np.linalg.lstsq(vs, factors[:, f], rcond=None)[0], 3
        )
        print("beta", f, beta)
        # vs = vs - np.matmul()


factors = np.hstack([
    rng.normal(0., 3., size=(100, 1)),
    rng.normal(0., 1, size=(100, 1)),
    rng.normal(0., 0.01, size=(100, 1)),
])

weights = qr(rng.normal(0, 1., (3, 3)))[0]
weights = np.round(weights, 3)

print("weight norm")
print(np.round(np.dot(weights, weights.T), 3))

v = np.matmul(factors, weights)

for w in range(weights.shape[0]):
    print("w", w, weights[w])

compare_pc(v, factors)
