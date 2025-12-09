
import numpy as np

def iqr(v, width: float):
    up = np.quantile(v, 0.5 + width)
    dn = np.quantile(v, 0.5 - width)
    return (up - dn) / 2

def iqr_ratio(width: float=0.25, n: int = 30000):
    v = np.random.normal(size=n)
    return np.std(v) / iqr(v, width=width)
