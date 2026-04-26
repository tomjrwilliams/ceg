from typing import Type
from dataclasses import dataclass, replace
import numpy as np
import polars as pl

# TODO: rename shapes (or something)

# TODO: does pydantic check numpy shapes??
# then replace the dataclasses import with pydantic equiv

F64_1D = np.ndarray[tuple[int], np.dtype[np.float64]]
F64_2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
F64_3D = np.ndarray[
    tuple[int, int, int], np.dtype[np.float64]
]
F64_4D = np.ndarray[
    tuple[int, int, int, int], np.dtype[np.float64]
]


def enumerated_labels(
    prefix: str,
    n: int,
    justify: int = 3,
):
    return [
        f"{prefix}-{str(i).rjust(justify, '0')}"
        for i in range(n)
    ]


# TODO: generalise the flatten to df
# eg. overload on enum count (?)


def f64_2d_to_df(
    v: F64_2D,
    value: str,
    enums: dict[str, Type[pl.DataType] | pl.Enum],
):
    a, b = reversed(tuple(v.shape))
    assert len(enums) == 2
    # we pass in dim order, we unpack reversed
    enums = dict(reversed(enums.items()))
    k_a, k_b = enums.keys()
    t_a, t_b = enums.values()
    return pl.DataFrame({value: v.flatten()}).with_columns(
        pl.int_range(v.size).mod(a).cast(t_a).alias(k_a),
        pl.int_range(v.size)
        .floordiv(a)
        .mod(b)
        .cast(t_b)
        .alias(k_b),
    )


def f64_3d_to_df(
    v: F64_3D,
    value: str,
    enums: dict[str, Type[pl.DataType] | pl.Enum],
):
    a, b, c = reversed(tuple(v.shape))
    assert len(enums) == 3
    # we pass in dim order, we unpack reversed
    enums = dict(reversed(enums.items()))
    k_a, k_b, k_c = enums.keys()
    t_a, t_b, t_c = enums.values()
    return pl.DataFrame({value: v.flatten()}).with_columns(
        pl.int_range(v.size).mod(a).cast(t_a).alias(k_a),
        pl.int_range(v.size)
        .floordiv(a)
        .mod(b)
        .cast(t_b)
        .alias(k_b),
        pl.int_range(v.size)
        .floordiv(a * b)
        .mod(c)
        .cast(t_c)
        .alias(k_c),
    )


def f64_4d_to_df(
    v: F64_4D,
    value: str,
    enums: dict[str, Type[pl.DataType] | pl.Enum],
):
    a, b, c, d = reversed(tuple(v.shape))
    assert len(enums) == 4
    # we pass in dim order, we unpack reversed
    enums = dict(reversed(enums.items()))
    k_a, k_b, k_c, k_d = enums.keys()
    t_a, t_b, t_c, t_d = enums.values()
    return pl.DataFrame({value: v.flatten()}).with_columns(
        pl.int_range(v.size).mod(a).cast(t_a).alias(k_a),
        pl.int_range(v.size)
        .floordiv(a)
        .mod(b)
        .cast(t_b)
        .alias(k_b),
        pl.int_range(v.size)
        .floordiv(a * b)
        .mod(c)
        .cast(t_c)
        .alias(k_c),
        pl.int_range(v.size)
        .floordiv(a * b * c)
        .mod(d)
        .cast(t_d)
        .alias(k_d),
    )
