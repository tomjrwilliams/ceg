import numpy as np


def iqr(v, width: float):
    up = np.quantile(v, 0.5 + width)
    dn = np.quantile(v, 0.5 - width)
    return (up - dn) / 2


def iqr_ratio(width: float = 0.25, n: int = 30000):
    v = np.random.normal(size=n)
    return np.std(v) / iqr(v, width=width)


def sdev_increments(
    sdev_width: float | None, sdevs: int, mu=0
):
    assert sdevs > 1, sdevs
    if sdev_width is None:
        sdev_width = 1
    l = r = mu
    if sdevs % 2 == 1:
        n = int((sdevs - 1) / 2)
        l -= sdev_width / 2
        r += sdev_width / 2
        mid = [(l, r)]
    else:
        n = int(sdevs / 2)
        mid = []
    return (
        [
            (
                None
                if i == (n - 1)
                else l - ((1 + i) * sdev_width),
                l - (i * sdev_width),
            )
            for i in reversed(range(n))
        ]
        + mid
        + [
            (
                r + (i * sdev_width),
                None
                if i == (n - 1)
                else r + ((1 + i) * sdev_width),
            )
            for i in range(n)
        ]
    )


def increments(_start: float, _stop: float, _num: int):

    sep = np.linspace(
        float(_start),
        float(_stop),
        num=int(_num),
    ).tolist()
    acc = []
    l = None
    r = None
    for v in sep:
        acc.append((l, v))
        l = v
    acc.append((v, None))

    return acc
