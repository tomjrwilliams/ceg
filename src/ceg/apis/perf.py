from typing import overload, Literal
import numpy as np
import polars as pl

# NOTE: threshold for sortino (== target return, eg. 0)

def expr(
    e: bool | int | float | str | pl.Expr | None = None,
    default: bool | int | float | None = None,
) -> pl.Expr | None:
    if e is None:
        assert default is not None, dict(e=e)
        return pl.lit(default)
    return (
        e if isinstance(e, pl.Expr) 
        else pl.lit(e)
        if isinstance(e, (bool, int, float))
        else pl.col(e)
    )

def null_mask(e: pl.Series | pl.Expr, cal: str):
    return pl.when(
        pl.col(cal)
    ).then(e).otherwise(None)

def backtest_rx_cum(
    rx: np.ndarray,
    calendar: np.ndarray,
    signal: np.ndarray, # not lagged yet
    scale: np.ndarray, # not lagged yet
    rebal: np.ndarray,
    flows: np.ndarray, # not lagged yet
):
    res_cap = np.zeros(rx.shape, dtype=rx.dtype) * np.nan
    res_pos = np.zeros(rx.shape, dtype=rx.dtype) * np.nan
    res_pnl = np.zeros(rx.shape, dtype=rx.dtype) * np.nan
    res_rx = np.zeros(rx.shape, dtype=rx.dtype) * np.nan
    live: bool = False
    reb: bool = False
    pos: float = 0
    cap: float = 0
    for t in range(len(rx)):
        if rebal[t]:
            reb = True
        # TODO: if rebal[t] = False, reset? (latest limit)
        # then have to disambiguate none from false
        if not calendar[t]:
            # cant rebal on non trading, no signal
            continue
        if np.isnan(rx[t]):
            continue
        sig = signal[t]
        sca = scale[t]
        flow = flows[t]
        pnl = rx[t] * pos
        res_rx[t] = pnl / cap
        res_pnl[t] = pnl
        cap = cap + pnl + flow
        if not live:
            live = True
        if reb and sca != 0:
            pos = (cap * sig) / sca
            reb = False
        res_pos[t] = pos
        res_cap[t] = cap
    return res_pos, res_rx, res_pnl, res_cap

def backtest_px_cum(
    px: np.ndarray, # always abs
    calendar: np.ndarray,
    signal: np.ndarray,
    scale: np.ndarray,
    rebal: np.ndarray,
    flows: np.ndarray,
):
    res_rx = np.zeros(px.shape, dtype=px.dtype) * np.nan
    live: bool = False
    v: float = 0.
    for t in range(len(px)):
        if not calendar[t]:
            # cant rebal on non trading, no signal
            continue
        if np.isnan(px[t]):
            continue
        if not live:
            live = True
            res_rx[t] = 0
        else:
            res_rx[t] = px[t] - v
        v = px[t]
    return backtest_rx_cum(
        rx=res_rx,
        calendar=calendar,
        signal=signal,
        scale=scale,
        rebal=rebal,
        flows=flows,
    )

# TODO: numba jit

def backtest_rx(
    rx: np.ndarray,
    calendar: np.ndarray,
    signal: np.ndarray, # not lagged yet
    scale: np.ndarray, # not lagged yet
    rebal: np.ndarray,
    unit: float,
):
    res_pos = np.zeros(rx.shape, dtype=rx.dtype) * np.nan
    res_rx = np.zeros(rx.shape, dtype=rx.dtype) * np.nan
    live: bool = False
    reb: bool = False
    pos: float = 0
    for t in range(len(rx)):
        if rebal[t]:
            reb = True

        # TODO: distinguish rebal first after including, versus rebal exactly
        # TODO: if rebal[t] = False, reset? (latest limit)
        # then have to disambiguate none from false

        if not calendar[t]:
            # cant rebal on non trading, no signal
            # do we have returns on non cal days?
            continue
        if np.isnan(rx[t]):
            continue
        
        sig = signal[t]
        sca = scale[t]

        if np.isnan(sig) or not np.isfinite(sig):
            continue
        if np.isnan(sca) or not np.isfinite(sca):
            continue

        res_rx[t] = rx[t] * pos
        if not live:
            live = True
        if reb and sca != 0:
            pos = (unit * sig) / sca
            reb = False
        res_pos[t] = pos
        # if np.isnan(pos):
        #     print(sig, sca, rx[t])
    return res_pos, res_rx

def backtest_px(
    px: np.ndarray, # always abs
    calendar: np.ndarray,
    signal: np.ndarray, # not lagged yet
    scale: np.ndarray, # not lagged yet
    rebal: np.ndarray,
    # boolean mask for first day gt
    # poss false for end cutoff
    unit: float,
):
    res_rx = np.zeros(px.shape, dtype=px.dtype) * np.nan
    live: bool = False
    v: float = 0.
    for t in range(len(px)):
        if not calendar[t]:
            # cant rebal on non trading, no signal
            continue
        if np.isnan(px[t]):
            continue
        if not live:
            live = True
            res_rx[t] = 0
        else:
            res_rx[t] = px[t] - v
        v = px[t]
    return backtest_rx(
        rx=res_rx,
        calendar=calendar,
        signal=signal,
        scale=scale,
        rebal=rebal,
        unit=unit,
    )

def calendar_expr(
    calendar: str | pl.Expr | None = None,
    px: float | str | pl.Expr | None = None,
    rx: float | str | pl.Expr | None = None,
):
    if calendar is not None:
        return expr(calendar)
    elif px is not None:
        e = expr(px)
        assert e is not None, e
        return e.is_not_null()
    elif rx is not None:
        e = expr(rx)
        assert e is not None, e
        return e.is_not_null()
    else:
        return pl.lit(True)

def backtest(
    df: pl.DataFrame,
    #
    px: float | str | pl.Expr | None = None,
    rx: float | str | pl.Expr | None = None,
    #
    calendar: str | pl.Expr | None = None,
    signal: float | str | pl.Expr | None = None,
    scale: float | str | pl.Expr | None = None,
    rebal: float | str | pl.Expr | None = None,
    flows: float | str | pl.Expr | None = None,
    #
    unit: float | None = None,
    cum: bool = False,
):
    calendar = calendar_expr(calendar, px, rx)
    signal = expr(signal, 1.)
    scale = expr(scale, 1.)
    rebal = expr(rebal, True)
    
    df_vs = df.select(
        calendar=calendar,
        signal=signal,
        scale=scale,
        rebal=rebal,
    )

    v_calendar = df_vs.get_column("calendar").to_numpy()
    v_signal = df_vs.get_column("signal").to_numpy()
    v_scale = df_vs.get_column("scale").to_numpy()
    v_rebal = df_vs.get_column("rebal").to_numpy()

    if px is None:
        assert rx is not None, dict(px=px, rx=rx)
        v_rx = df.select(rx=rx).get_column("rx").to_numpy()
        if cum:
            v_flows = (
                df.select(flows=flows)
                .get_column("flows")
                .to_numpy()
            )
            return backtest_rx_cum(
                v_rx,
                v_calendar,
                v_signal,
                v_scale,
                v_rebal,
                v_flows,
            )
        assert unit is not None, dict(cum=cum, unit=unit)
        return backtest_rx(
            v_rx,
            v_calendar,
            v_signal,
            v_scale,
            v_rebal,
            unit,
        )
    elif rx is None:
        assert px is not None, dict(px=px, rx=rx)
        v_px = df.select(px=px).get_column("px").to_numpy()
        if cum:
            v_flows = (
                df.select(flows=flows)
                .get_column("flows")
                .to_numpy()
            )
            return backtest_px_cum(
                v_px,
                v_calendar,
                v_signal,
                v_scale,
                v_rebal,
                v_flows,
            )
        assert unit is not None, dict(cum=cum, unit=unit)
        return backtest_px(
            v_px,
            v_calendar,
            v_signal,
            v_scale,
            v_rebal,
            unit,
        )
    else:
        raise ValueError(dict(px=px, rx=rx))

@overload
def with_backtest(
    df: pl.DataFrame,
    alias: tuple[str, str],
    cum: Literal[False],
    #
    px: float | str | pl.Expr | None = None,
    rx: float | str | pl.Expr | None = None,
    #
    calendar: str | pl.Expr | None = None,
    signal: float | str | pl.Expr | None = None,
    scale: float | str | pl.Expr | None = None,
    rebal: float | str | pl.Expr | None = None,
    flows: float | str | pl.Expr | None = None,
    #
    unit: float | None = None,
) -> pl.DataFrame: ...

@overload
def with_backtest(
    df: pl.DataFrame,
    alias: tuple[str, str, str, str],
    cum: Literal[True],
    #
    px: float | str | pl.Expr | None = None,
    rx: float | str | pl.Expr | None = None,
    #
    calendar: str | pl.Expr | None = None,
    signal: float | str | pl.Expr | None = None,
    scale: float | str | pl.Expr | None = None,
    rebal: float | str | pl.Expr | None = None,
    flows: float | str | pl.Expr | None = None,
    #
    unit: float | None = None,
) -> pl.DataFrame: ...

def with_backtest(
    df: pl.DataFrame,
    alias: tuple[str, str] | tuple[str, str, str, str],
    cum: bool = False,
    #
    px: float | str | pl.Expr | None = None,
    rx: float | str | pl.Expr | None = None,
    #
    calendar: str | pl.Expr | None = None,
    signal: float | str | pl.Expr | None = None,
    scale: float | str | pl.Expr | None = None,
    rebal: float | str | pl.Expr | None = None,
    flows: float | str | pl.Expr | None = None,
    #
    unit: float | None = None,
):
    v_res = backtest(
        df=df,
        px=px,
        rx=rx,
        calendar=calendar,
        signal=signal,
        scale=scale,
        rebal=rebal,
        flows=flows,
        unit=unit,
        cum=cum,
    )
    cal = calendar_expr(calendar, px, rx)
    assert cal is not None, calendar
    cal = cal.alias("CAL")
    if cum:
        assert len(alias) == 4, alias
        assert len(v_res) == 4, len(v_res)
        a_pos, a_pct, a_pnl, a_cap = alias
        v_pos, v_pct, v_pnl, v_cap = v_res
        return df.with_columns(cal).with_columns(
            null_mask(
                pl.Series(v_pos), "CAL"
            ).alias(a_pos),
            null_mask(
                pl.Series(v_pct), "CAL"
            ).alias(a_pct),
            null_mask(
                pl.Series(v_pnl), "CAL"
            ).alias(a_pnl),
            null_mask(
                pl.Series(v_cap), "CAL"
            ).alias(a_cap),
        )
    else:
        assert len(alias) == 2, alias
        assert len(v_res) == 2, len(v_res)
        a_pos, a_rx = alias
        v_pos, v_rx = v_res
        return df.with_columns(cal).with_columns(
            null_mask(
                pl.Series(v_pos), "CAL"
            ).alias(a_pos),
            null_mask(
                pl.Series(v_rx), "CAL"
            ).alias(a_rx),
        )

# ---

def rx_mu(
    rx: str | pl.Expr,
    #
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    #
    min_samples: int = 2,
    threshold: float | None = None,
):
    
    params = (window, alpha, span)
    rx = rx if isinstance(rx, pl.Expr) else pl.col(rx)
    if threshold is not None:
        rx = rx - threshold
    if window is not None:
        assert span is None and alpha is None, params
        return rx.rolling_mean(
            window, min_samples=min_samples
        )
    elif span is not None:
        assert alpha is None, params
        return rx.ewm_mean(
            span=span, min_samples=min_samples
        )
    elif alpha is not None:
        assert span is None, params
        return rx.ewm_mean(
            alpha=alpha, min_samples=min_samples
        )
    else:
        return rx.mean()
    
def rx_rms(
    rx: str | pl.Expr,
    #
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    threshold: float | None = None,
    #
    min_samples: int = 2,
):
    rx = rx if isinstance(rx, pl.Expr) else pl.col(rx)
    if threshold is not None:
        rx = pl.when(
            rx > threshold
        ).then(None).otherwise(rx - threshold)
    rx_2 = rx.pow(2)
    return rx_mu(
        rx_2, 
        window=window,
        alpha=alpha,
        span=span,
        min_samples=min_samples,
    ).sqrt()
    
def rx_vol(
    rx: str | pl.Expr,
    #
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    threshold: float | None = None,
    #
    min_samples: int = 2,
):
    params = (window, alpha, span)
    rx = rx if isinstance(rx, pl.Expr) else pl.col(rx)
    if threshold is not None:
        rx = pl.when(
            rx > threshold
        ).then(None).otherwise(rx - threshold)
    if window is not None:
        assert span is None and alpha is None, params
        return rx.rolling_std(
            window, min_samples=min_samples
        )
    elif span is not None:
        assert alpha is None, params
        return rx.ewm_std(
            span=span, min_samples=min_samples
        )
    elif alpha is not None:
        assert span is None, params
        return rx.ewm_std(
            alpha=alpha, min_samples=min_samples
        )
    else:
        return rx.std()
    
def rx_sharpe(
    rx: str | pl.Expr,
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    ann: int = 252,
    min_samples: int = 2,
    threshold: float | None = None, # for sortino
    rms: bool = False,
) -> pl.Expr:
    ann_vol = pl.lit(ann).sqrt()
    mu = rx_mu(
        rx,
        window=window,
        alpha=alpha,
        span=span,
        min_samples=min_samples,
        threshold=threshold,
    )
    f_vol = rx_vol if not rms else rx_rms
    vol = f_vol(
        rx,
        window=window,
        alpha=alpha,
        span=span,
        min_samples=min_samples,
        threshold=threshold,
    )
    return (mu * ann) / (vol * ann_vol)


def rx_hwm(
    rx: str | pl.Expr,
    #
    window: int | None = None,

    min_samples: int = 2,
    pct: bool = False,
):
    rx = rx if isinstance(rx, pl.Expr) else pl.col(rx)
    if pct and window is not None:
        raise ValueError("NOT IMPLEMENTED")
        # e = (1 + rx).cum_prod().rolling_max(
        #     window,
        #     min_samples=min_samples,
        # )
    elif window is not None:
        raise ValueError("NOT IMPLEMENTED")
        # e = rx.cum_sum().rolling_max(
        #     window,
        #     min_samples=min_samples,
        # )
    elif pct:
        e = (1 + rx).cum_prod()
        return e.cum_max()
    else:
        e = rx.cum_sum()
        return e.cum_max()

def rx_dd(
    rx: str | pl.Expr,
    #
    window: int | None = None,

    min_samples: int = 2,
    pct: bool = False,
):
    rx = rx if isinstance(rx, pl.Expr) else pl.col(rx)
    hwm = rx_hwm(
        rx=rx,
        window=window,
        min_samples=min_samples,
        pct=pct,
    )
    if pct:
        return (((1 + rx).cum_prod() / hwm) - 1) * -1
    else:
        return (rx.cum_sum() - hwm) * -1

def rx_dd_max(
    rx: str | pl.Expr,
    #
    window: int | None = None,

    min_samples: int = 2,
    pct: bool = False,
):
    # TODO: correct rolling?
    return rx_dd(
        rx,
        window=window,
        min_samples=min_samples,
        pct=pct,
    ).max()

def rx_calmar(
    rx: str | pl.Expr,
    #
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    threshold: float | None = None,
    #
    min_samples: int = 2,
    pct: bool = False,
    ann: int = 252,
):
    mu = rx_mu(
        rx,
        window=window,
        alpha=alpha,
        span=span,
        min_samples=min_samples,
        threshold=threshold,
    )
    dd = rx_dd_max(
        rx=rx,
        window=window,
        min_samples=min_samples,
        pct=pct,
    )
    return (mu * ann) / dd

#  ---

def with_rx_mu(
    df: pl.DataFrame,
    rx: str | pl.Expr,
    alias: str,
    #
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    threshold: float | None = None,
    #
    min_samples: int = 2,
):
    e = rx_mu(
        rx=rx,
        window=window,
        alpha=alpha,
        span=span,
        min_samples=min_samples,
        threshold=threshold,
    )
    return df.with_columns(e.alias(alias))

def with_rx_rms(
    df: pl.DataFrame,
    rx: str | pl.Expr,
    alias: str,
    #
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    threshold: float | None = None,
    #
    min_samples: int = 2,
):
    e = rx_rms(
        rx=rx,
        window=window,
        alpha=alpha,
        span=span,
        min_samples=min_samples,
        threshold=threshold,
    )
    return df.with_columns(e.alias(alias))

def with_rx_vol(
    df: pl.DataFrame,
    rx: str | pl.Expr,
    alias: str,
    #
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    threshold: float | None = None,
    #
    min_samples: int = 2,
):
    e = rx_vol(
        rx=rx,
        window=window,
        alpha=alpha,
        span=span,
        min_samples=min_samples,
        threshold=threshold,
    )
    return df.with_columns(e.alias(alias))

def with_rx_sharpe(
    df: pl.DataFrame,
    rx: str | pl.Expr,
    alias: str,
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    ann: int = 252,
    min_samples: int = 2,
    threshold: float | None = None,
    rms: bool = False,
):
    e = rx_sharpe(
        rx=rx,
        window=window,
        alpha=alpha,
        span=span,
        ann=ann,
        min_samples=min_samples,
        threshold=threshold,
        rms=rms,
    )
    return df.with_columns(e.alias(alias))

def with_rx_hwm(
    df: pl.DataFrame,
    rx: str | pl.Expr,
    alias: str,
    window: int | None = None,
    min_samples: int = 2,
):
    e = rx_hwm(
        rx=rx,
        window=window,
        min_samples=min_samples,
    )
    return df.with_columns(e.alias(alias))

def with_rx_dd(
    df: pl.DataFrame,
    rx: str | pl.Expr,
    alias: str,
    window: int | None = None,
    min_samples: int = 2,
):
    e = rx_dd(
        rx=rx,
        window=window,
        min_samples=min_samples,
    )
    return df.with_columns(e.alias(alias))

def with_rx_calmar(
    df: pl.DataFrame,
    rx: str | pl.Expr,
    alias: str,
    window: int | None = None,
    alpha: float | None = None,
    span: float | None = None,
    ann: int = 252,
    min_samples: int = 2,
    threshold: float | None = None,
):
    e = rx_calmar(
        rx=rx,
        window=window,
        alpha=alpha,
        span=span,
        ann=ann,
        min_samples=min_samples,
        threshold=threshold,
    )
    return df.with_columns(e.alias(alias))

# TODO: specific window drawdowns
# ie. none mask outside the window, then cum prod / sum

f_round = round

def perf_row(
    df: pl.DataFrame,
    k: str | None = None,
    pos: str | None = None,
    rx: str | None = None,
    round: int | None = None,
    repr: int | bool = False,
    **kwargs
):
    # if isinstance(repr, int) and round is None:
    #     round = repr
    #     repr = True

    if k is None and pos is None and rx is None:
            pos = "pos"
            rx = "rx"
    elif k is None:
        assert (
            pos is not None and rx is not None
        ), dict(
            k=k,
            pos=pos,
            rx=rx,
        )
    else:
        pos = pos or f"{k}_pos"
        rx = rx or f"{k}_rx"

    # assert pos in df.schema, df.schema
    assert rx in df.schema, df.schema

    df = df.with_columns(
        pl.col(rx).fill_nan(None)
    )
    if pos in df.schema:
        df = df.with_columns(
            pl.col(pos).fill_nan(None),
        )

    (
        r_mu,
        r_sd,
        r_rms,
        r_skew,
        r_shrp,
        r_sort,
        r_dd,
        r_calm,
    ) = df.select(
        rx_mu(rx=rx)
        .mul(252)
        .alias("mu")
        .item(),
        rx_vol(rx=rx)
        .mul(16)
        .alias("sd")
        .item(),
        rx_rms(rx=rx)
        .mul(16)
        .alias("rms")
        .item(),
        pl.col(rx)
        .skew()
        .alias("skew")
        .item(),
        rx_sharpe(rx=rx)
        .alias("shrp")
        .item(),
        rx_sharpe(rx=rx, threshold=0)
        .alias("sort")
        .item(),
        rx_dd_max(rx=rx)
        .alias("dd")
        .item(),
        rx_calmar(rx=rx)
        .alias("calm")
        .item(),
        # *([
    ).row(0, named=False)

    if pos in df.schema:
        (
            r_pos_mu,
            r_pos_sd,
            r_pos_skew,
        ) = df.select(
            pl.col(pos)
            .mean()
            .alias("p_mu")
            .item(),
            pl.col(pos)
            .std()
            .alias("s_sd")
            .item(),
            pl.col(pos)
            .skew()
            .alias("p_skew")
            .item(),
            # ]
            # if row_sig else [
            #     pl.lit(1.).alias("pos_mu").item(),
            #     pl.lit(0.).alias("pos_sd").item(),
            #     pl.lit(0.).alias("pos_skew").item(),
            # ])
        ).row(0, named=False)
        pos_r = dict(
            p_mu = r_pos_mu,
            p_sd = r_pos_sd,
            p_skew = r_pos_skew,
        )
    else:
        pos_r = {}

    r = {} if k is None else dict(key=k)
    perf = dict(
        mu = r_mu,
        sd = r_sd,
        rms = r_rms,
        skew=r_skew,
        shrp = r_shrp,
        sort = r_sort,
        dd=r_dd,
        calm = r_calm,
        **pos_r,
    )
    if round:
        perf = {
            k: f_round(v, round) for k, v in perf.items()
        }
    r = dict(
        **r,
        **kwargs,
        **perf,
    )
    if repr:
        for k, v in r.items():
            print(k, v)
        return
    return r