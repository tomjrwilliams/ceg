from datetime import date
import polars as pl

import itertools

from functools import lru_cache

from .pl import filter_dates_between, null_mask

from .common import norm_expr

ASSET_CLASSES = ["ETF", "FX", "EQ", "FUT", "ECO"]

DIRECTIONS = ["UP", "DN"]

@lru_cache(maxsize=128)
def extract_symbol(name: str):
    for ac in ASSET_CLASSES:
        if ac in name:
            return name.split(f"{ac}-")[1].split("-")[0].split(".")[0]
    raise ValueError(name)

@lru_cache(maxsize=128)
def _sign_code_labels(objs: tuple[str, ...]):
    # TODO: parametrise cutoff
    symbols = [extract_symbol(obj) for obj in objs]
    return list(itertools.product(*[
        [True, False]
        for symbol in symbols
    ])), [
        "-".join(vs) for vs in itertools.product(*[
            [
                f"{symbol}.{sign}" for sign in ["UP", "DN"]
            ]
            for symbol in symbols
        ])
    ]
    
def sign_code_labels(objs: list[str]):
    return _sign_code_labels(tuple(objs))

@lru_cache(maxsize=128)
def _sign_code_labels_truncated(objs: tuple[str, ...]):
    # TODO: parametrise cutoff
    symbols = [extract_symbol(obj) for obj in objs]
    sign_tups, sign_codes = _sign_code_labels(objs)
    res = []
    prev = [None for _ in objs]
    n = 2 ** len(objs)
    for i, sign_code in enumerate(sign_codes):
        codes = sign_code.split("-")
        match = False
        if i == 0 or i == int(n / 2):
            res.append(([codes[0]] + [None for _ in objs[1:]], None))
            prev = res[-1][0]
        for j, (c, p) in enumerate(zip(codes, prev)):
            if c == p:
                continue
            res.append(([
                None for _ in range(j)
            ] + [c] + [
                None for _ in range(j + 1, len(objs))
            ], sign_code))
            match = True
            break
        assert match, sign_code
        prev = codes
    return res


def sign_code_labels_truncated(objs: list[str]):
    return _sign_code_labels_truncated(tuple(objs))


def sign_expr(k: str, n: int, span: float | None = None):
    if span is None:
        v_smooth = pl.col(k)
    else:
        v_smooth = pl.col(k).pipe(
            norm_expr,
            norm=f"ewmean-{int(span)}-252",
            n=n,
        )
    return pl.when(pl.col(k).is_null()).then(None).otherwise(v_smooth.gt(0))

# NOTE:

# normalised transtion:
# - lead (?) asset reversion (ie. SPY UP, TLT UP, GLD UP -> SPY DN, '')
# ^ actually chacterises lead asset?

def sign_code_product(
    df: pl.DataFrame,
    objs: list[str],
    smoothed: bool = False,
    span: float | None = None,
    smooth: float | None = None,
    d0: date | None = None,
    d1: date | None = None,
):
    df = df.pipe(filter_dates_between, d0=d0, d1=d1)
    sign_tuples, sign_codes = sign_code_labels(objs)
    df = df.with_columns(*(
        sign_expr(k, len(df), span).alias(k)
        for k in objs
    ))

    df = df.select("date", *(
        pl.when(
            pl.all_horizontal(*(
                pl.col(k).is_null()
                for k in objs
            ))
        )
        .then(None)
        .otherwise(pl.all_horizontal(*(
            pl.col(k).eq(v)
            for k, v in zip(objs, sign_tuple)
        )))
        .alias(sign_code)
        for sign_code, sign_tuple in zip(sign_codes, sign_tuples)
    ))

    df_sm = df.with_columns("date", *(
        pl.col(sign_code)
        .fill_null(0)
        .ewm_mean(span=smooth)
        .round(4)
        .alias(f"{sign_code}-{smooth}")
        for sign_code in sign_codes
    )).with_columns(
        pl.concat_list(*(
            f"{sign_code}-{smooth}" for sign_code in sign_codes
        ))
        .list.arg_max()
        .alias("i_max")
    ).select("date", *(
        pl.lit(i)
        .eq(pl.col("i_max"))
        .pipe(lambda e: (
            pl.when(pl.col(sign_code).is_null())
            .then(None)
            .otherwise(e)
        ))
        .alias(sign_code)
        for i, sign_code in enumerate(sign_codes)
    )).unpivot(index="date", variable_name="sign_code", value_name="state").with_columns(
        pl.lit(smoothed).alias("active"),
        pl.lit(True).alias("smoothed"),
    )

    return df.unpivot(index="date", variable_name="sign_code", value_name="state").with_columns(
        pl.lit(not smoothed).alias("active"),
        pl.lit(False).alias("smoothed"),
    ).vstack(df_sm)

def transition_tree(
    df: pl.DataFrame,
    assets: list[str],
    sign_tuples: list[tuple[bool]],
    sign_codes: list[str],
    n: int,
):
    STATE = {True: "UP", False: "DN"}
    
    sign_code_agg: dict[str, dict[str, dict[str, list[str]]]] = {}
    sign_code_other = {}

    for sign_code, sign_tup in zip(sign_codes, sign_tuples):
        # NOTE: hold states other than asset=i constant
        sign_code_agg[sign_code] = {}
        for i, asset in enumerate(assets):
            sign_code_agg[sign_code][asset] = {}
            for s in [True, False]:
                sign_code_agg[sign_code][asset][s] = [
                    sc for sc, st in zip(sign_codes, sign_tuples)
                    # if all([
                    #     st[ii] == sign_tup[ii]
                    #     for ii, a in enumerate(assets)
                    #     if ii != i
                    # ])
                    if st[i] == s
                ]

        sign_code_other[sign_code] = [
            sc for sc in sign_codes if sc not in sum(sum(
                map(
                    lambda d: list(d.values()),
                    sign_code_agg[sign_code].values()
                ), []
            ), [])
        ]

    try:
        return df.select(*(
            pl.col(code_i)
            .eq(True)
            .and_(pl.any_horizontal(*(
                pl.col(code_o).shift(-n).eq(True)
                for code_o in sign_code_agg[sign_code][asset][s]
            )))
            .and_(pl.col(code_i).shift(-n).eq(False))
            .sum()
            .alias(f"{code_i}|{asset}|{STATE[s]}")
            for code_i in sign_codes
            for asset in assets
            for s in [True, False]
        ), *(
            pl.col(code_i)
            .eq(True)
            .and_(
                pl.col(code_i).shift(-n).eq(True)
            )
            .sum()
            .alias(f"{code_i}|SELF|SM")
            for code_i in sign_codes
        ), *(
            # pl.col(code_i)
            # .eq(True)
            # .and_(pl.any_horizontal(*(
            #     pl.col(code_o).shift(-n).eq(True)
            #     for code_o in sign_code_other[sign_code]
            # )))
            # .sum()
            # .alias(f"{code_i}|other|ANY")
            # for code_i in sign_codes
        ))
    except Exception as e:
        raise ValueError(df, sign_codes, sign_code_agg, sign_code_other) from e