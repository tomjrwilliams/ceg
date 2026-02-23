from typing import overload, Literal
import numpy as np
import polars as pl
import numba as nb
from .logs import logger
from .pl import *

logger, log = logger(
    __file__,
    streams=dict(formatter=True),
)
# NOTE: threshold for sortino (== target return, eg. 0)

@nb.jit()
def backtest_rx_cum(
    rx: np.ndarray,
    null: np.ndarray,
    calendar: np.ndarray,
    signal: np.ndarray, # not lagged yet
    # signal_null: np.ndarray,
    scale: np.ndarray, # not lagged yet
    # scale_null: np.ndarray,
    rebal: np.ndarray,
    flows: np.ndarray, # not lagged yet
    cap: np.ndarray,
    fx: np.ndarray,
):
    res_cap = np.zeros(rx.shape, dtype=rx.dtype)
    res_pos = np.zeros(rx.shape, dtype=rx.dtype)
    res_pnl = np.zeros(rx.shape, dtype=rx.dtype)
    res_rx = np.zeros(rx.shape, dtype=rx.dtype)
    live: bool = False
    reb: bool = False
    pos: float = 0
    capital: float = 0
    t = 0
    while t < len(rx):
        if rebal[t]:
            reb = True

        # TODO: if rebal[t] = False, reset? (latest limit)
        # then have to disambiguate none from false
        
        # if not signal_null[t]:
        sig = signal[t]
        # if not scale_null[t]:
        sca = scale[t]

        if (
            null[t]
            or not calendar[t]
        ):
            res_cap[t] = np.nan
            res_pos[t] = np.nan
            res_pnl[t] = np.nan
            res_rx[t] = np.nan
            t += 1
            continue
        
        flow = flows[t]
        pnl = (rx[t] * pos) * fx[t]
        res_rx[t] = pnl / capital
        res_pnl[t] = pnl
        capital = capital + pnl + flow

        if not live:
            live = True
        if reb and sca != 0:
            pos = (capital * sig) / sca
            v_cap = cap[t]
            if v_cap >= 0:
                if pos > v_cap:
                    pos = v_cap
                elif pos < -v_cap:
                    pos = -v_cap
            reb = False
        res_pos[t] = pos
        res_cap[t] = capital
        t += 1
    return res_pos, res_rx, res_pnl, res_cap

@nb.jit()
def backtest_px_cum(
    px: np.ndarray, # always abs
    null: np.ndarray,
    calendar: np.ndarray,
    signal: np.ndarray,
    # signal_null: np.ndarray,
    scale: np.ndarray,
    # scale_null: np.ndarray,
    rebal: np.ndarray,
    flows: np.ndarray,
    cap: np.ndarray,
    fx: np.ndarray,
):
    res_rx = np.zeros(px.shape, dtype=px.dtype)
    live: bool = False
    v: float = 0.
    t = 0
    while t < len(px):
        if null[t]:
            res_rx[t] = np.nan
            t += 1
            continue
        elif not calendar[t]:
            res_rx[t] = np.nan
            # cant rebal on non trading, no signal
            t += 1
            continue
        else:
            if not live:
                live = True
                res_rx[t] = 0
            else:
                res_rx[t] = px[t] - v
            v = px[t]
            t += 1
    return backtest_rx_cum(
        rx=res_rx,
        null=null,
        calendar=calendar,
        signal=signal,
        # signal_null=signal_null,
        scale=scale,
        # scale_null=scale_null,
        rebal=rebal,
        flows=flows,
        cap=cap,
        fx=fx,
    )

# TODO: numba jit

@nb.jit()
def backtest_rx(
    rx: np.ndarray,
    null: np.ndarray,
    calendar: np.ndarray,
    signal: np.ndarray, # not lagged yet
    # signal_null: np.ndarray,
    scale: np.ndarray, # not lagged yet
    # scale_null: np.ndarray,
    rebal: np.ndarray,
    cap: np.ndarray,
    fx: np.ndarray,
    unit: float,
):
    res_pos = np.zeros(rx.shape, dtype=rx.dtype)
    res_rx = np.zeros(rx.shape, dtype=rx.dtype)
    live: bool = False
    reb: bool = False

    pos: float = 0

    sig: float = 0
    sca: float = 0

    t = 0
    while t < len(rx):
        if rebal[t]:
            reb = True

        # if not signal_null[t]:
        sig = signal[t]

        # if not scale_null[t]:
        sca = scale[t]

        # TODO: distinguish rebal first after including, versus rebal exactly
        # TODO: if rebal[t] = False, reset? (latest limit)
        # then have to disambiguate none from false
        if null[t]:
            res_pos[t] = np.nan
            res_rx[t] = np.nan
            t += 1
            continue
        elif not calendar[t]:
            res_pos[t] = np.nan
            res_rx[t] = np.nan
            # cant rebal on non trading, no signal
            # do we have returns on non cal days?
            t += 1
            continue
        elif sca == 0:
            if live:
                res_pos[t] = pos
                res_rx[t] = (rx[t] * pos) * fx[t]
            else:
                res_pos[t] = np.nan
                res_rx[t] = np.nan
            t += 1
        else:
            res_rx[t] = (rx[t] * pos) * fx[t]
            if not live:
                live = True
            if reb and sca != 0:
                pos = (unit * sig) / sca
                v_cap = cap[t]
                if v_cap >= 0:
                    if pos > v_cap:
                        pos = v_cap
                    elif pos < -v_cap:
                        pos = -v_cap
                        
                reb = False
                
            res_pos[t] = pos
            t += 1
    return res_pos, res_rx

@nb.jit()
def backtest_rx_pos(
    rx: np.ndarray,
    pos: np.ndarray,
    null_rx: np.ndarray,
    null_pos: np.ndarray,
):
    res_rx = np.zeros(rx.shape, dtype=rx.dtype)
    
    has_p: bool = False

    p = 0
    t = 0
    while t < len(rx):

        if has_p:
            if not null_rx[t]:
                res_rx[t] = rx[t] * p
            else:
                res_rx[t] = np.nan

            if not null_pos[t]:
                p = pos[t]

        else:
            res_rx[t] = np.nan
            if not null_pos[t]:
                p = pos[t]
                has_p = True

        t += 1
    return res_rx

@nb.jit()
def backtest_px(
    px: np.ndarray, # always abs
    null: np.ndarray,
    calendar: np.ndarray,
    signal: np.ndarray, # not lagged yet
    # signal_null: np.ndarray,
    scale: np.ndarray, # not lagged yet
    # scale_null: np.ndarray,
    rebal: np.ndarray,
    # boolean mask for first day gt
    # poss false for end cutoff
    cap: np.ndarray,
    fx: np.ndarray,
    unit: float,
):
    res_rx = np.zeros(px.shape, dtype=px.dtype)
    live: bool = False
    v: float = 0.
    t = 0
    while t < len(px):
        if null[t]:
            t += 1
            res_rx[t] = np.nan
            continue
        elif not calendar[t]:
            res_rx[t] = np.nan
            # cant rebal on non trading, no signal
            t += 1
            continue
        else:
            if not live:
                live = True
                res_rx[t] = 0
            else:
                res_rx[t] = px[t] - v
            v = px[t]
            t += 1
    return backtest_rx(
        rx=res_rx,
        null=null,
        calendar=calendar,
        signal=signal,
        # signal_null=signal_null,
        scale=scale,
        # scale_null=scale_null,
        rebal=rebal,
        cap=cap,
        fx=fx,
        unit=unit,
    )

def calendar_expr(
    calendar: str | pl.Expr | None = None,
    px: float | str | pl.Expr | None = None,
    rx: float | str | pl.Expr | None = None,
) -> pl.Expr:
    if calendar is not None:
        return expr(calendar, True)
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
    cap: float | str | pl.Expr | None = None,
    fx: float | str | pl.Expr | None = None,
    #
    unit: float | None = None,
    cum: bool = False,
):
    calendar = calendar_expr(calendar, px, rx)

    signal = expr(signal, 1.)
    scale = expr(scale, 1.)
    fx = expr(fx, 1.)
    cap = expr(cap, -1.)
    rebal = expr(rebal, True)
    
    df_vs = df.with_columns(
        signal=signal,
        scale=scale,
    ).select(
        calendar=calendar.cast(pl.Float64),
        signal=(
            pl.when(is_null(pl.col("signal")))
            .then(None)
            .otherwise("signal")
            .forward_fill()
            .fill_null(0)
            .alias("signal")
        ).cast(pl.Float64),
        scale=(
            pl.when(is_null(pl.col("scale")))
            .then(None)
            .otherwise("scale")
            .forward_fill()
            .fill_null(0)
            .alias("scale")
        ).cast(pl.Float64),
        rebal=rebal.cast(pl.Float64),
        cap=cap.cast(pl.Float64),
        fx=fx.cast(pl.Float64),
    )

    df_nd = df_vs.to_numpy()

    v_calendar = df_nd[:, 0].astype(bool)
    v_signal = df_nd[:, 1]
    v_scale = df_nd[:, 2]
    v_rebal = df_nd[:, 3].astype(bool)
    v_cap = df_nd[:, 4]
    v_fx = df_nd[:, 5]

    # v_calendar = df_vs.get_column("calendar").to_numpy()
    # v_signal = df_vs.get_column("signal").to_numpy()
    # v_scale = df_vs.get_column("scale").to_numpy()
    # v_signal, v_signal_null = to_numpy_with_null(
    #     df_vs.get_column("signal")
    # )
    # v_scale, v_scale_null = to_numpy_with_null(
    #     df_vs.get_column("scale")
    # )
    # v_rebal = df_vs.get_column("rebal").to_numpy()
    # v_cap = df_vs.get_column("cap").to_numpy()
    # v_fx = df_vs.get_column("fx").to_numpy()

    if px is None:
        assert rx is not None, dict(px=px, rx=rx)
        s_rx = df.select(rx=rx).get_column("rx")
        v_rx, v_null = to_numpy_with_null(s_rx)
        if cum:
            v_flows = (
                df.select(flows=flows)
                .get_column("flows")
                .to_numpy()
            )
            return backtest_rx_cum(
                v_rx,
                v_null,
                v_calendar,
                v_signal,
                # v_signal_null,
                v_scale,
                # v_scale_null,
                v_rebal,
                v_flows,
                v_cap,
                v_fx,
            )
        assert unit is not None, dict(cum=cum, unit=unit)
        return backtest_rx(
            v_rx,
            v_null,
            v_calendar,
            v_signal,
            # v_signal_null,
            v_scale,
            # v_scale_null,
            v_rebal,
            v_cap,
            v_fx,
            unit,
        )
    elif rx is None:
        assert px is not None, dict(px=px, rx=rx)
        s_px = df.select(px=px).get_column("px")
        v_px, v_null = to_numpy_with_null(s_px)
        if cum:
            v_flows = (
                df.select(flows=flows)
                .get_column("flows")
                .to_numpy()
            )
            return backtest_px_cum(
                v_px,
                v_null,
                v_calendar,
                v_signal,
                # v_signal_null,
                v_scale,
                # v_scale_null,
                v_rebal,
                v_flows,
                v_cap,
                v_fx,
            )
        assert unit is not None, dict(cum=cum, unit=unit)
        return backtest_px(
            v_px,
            v_null,
            v_calendar,
            v_signal,
            # v_signal_null,
            v_scale,
            # v_scale_null,
            v_rebal,
            v_cap,
            v_fx,
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
    cap: float | str | pl.Expr | None = None,
    fx: float | str | pl.Expr | None = None,
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
    cap: float | str | pl.Expr | None = None,
    fx: float | str | pl.Expr | None = None,
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
    cap: float | str | pl.Expr | None = None,
    fx: float | str | pl.Expr | None = None,
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
        cap=cap,
        fx=fx,
    )
    cal = calendar_expr(calendar, px, rx)
    assert cal is not None, calendar
    cal = cal.alias("CAL")

    def _null_mask(s: pl.Series | pl.Expr, k: str):
        s = cal_mask(s.fill_nan(None), "CAL")
        if isinstance(rx, str):
            s = null_mask(s, pl.col(rx))
        if isinstance(px, str):
            s = null_mask(s, pl.col(px))
        return s.alias(k)
    
    def _null_mask_self(k: str):
        # deals with inf etc.
        e = pl.col(k)
        return null_mask(e, e).alias(k)

    if cum:
        assert len(alias) == 4, alias
        assert len(v_res) == 4, len(v_res)
        # NOTE: cap here is capital
        a_pos, a_pct, a_pnl, a_cap = alias
        v_pos, v_pct, v_pnl, v_cap = v_res
        return df.with_columns(cal).with_columns(
            _null_mask(pl.Series(v_pos), a_pos),
            _null_mask(pl.Series(v_pct), a_pct),
            _null_mask(pl.Series(v_pnl), a_pnl),
            _null_mask(pl.Series(v_cap), a_cap),
        ).with_columns(
            _null_mask_self(a_pos),
            _null_mask_self(a_pct),
            _null_mask_self(a_pnl),
            _null_mask_self(a_cap),
        )
    else:
        assert len(alias) == 2, alias
        assert len(v_res) == 2, len(v_res)
        a_pos, a_rx = alias
        v_pos, v_rx = v_res
        return df.with_columns(cal).with_columns(
            _null_mask(pl.Series(v_pos), a_pos),
            _null_mask(pl.Series(v_rx), a_rx),
        ).with_columns(
            _null_mask_self(a_pos),
            _null_mask_self(a_rx),
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
    vol: str | None = None,
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
    v = f_vol(
        vol or rx,
        window=window,
        alpha=alpha,
        span=span,
        min_samples=min_samples,
        threshold=threshold,
    )
    return (mu * ann) / (v * ann_vol)


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
    with_pos: bool = True,
    with_keys: list[str] | None = None,
    sharpe_vols: dict[str, str | None] | None = None,
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
            (pos is not None or not with_pos)
            and rx is not None
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
    if pos is not None and pos in df.schema:
        df = df.with_columns(
            pl.col(pos).fill_nan(None),
        )
    else:
        assert not with_pos, df

    if sharpe_vols is None:
        sharpe_vols = {"shrp": None}

    perf = df.select(
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
        *[
            rx_sharpe(
                rx=rx,
                vol=rx_vol
            )
            .alias(sharpe_key)
            .item()
            for sharpe_key, rx_vol
            in sharpe_vols.items()
        ],
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
    ).row(0, named=True)

    if with_pos:
        assert pos is not None, (df, pos)
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

    if with_keys:
        r = {
            **r,
            **df.select((
                pl.col(_k)
                .first()
                .item()
                for _k in with_keys
            )).row(0, named=True)
        }

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


__all__ = [
    "backtest_rx_cum",
    "backtest_px_cum",
    "backtest_rx",
    "backtest_rx_pos",
    "backtest_px",
    "calendar_expr",
    "backtest",
    "with_backtest",
    "rx_mu",
    "rx_rms",
    "rx_vol",
    "rx_sharpe",
    "rx_hwm",
    "rx_dd",
    "rx_dd_max",
    "rx_calmar",
    "with_rx_mu",
    "with_rx_rms",
    "with_rx_vol",
    "with_rx_sharpe",
    "with_rx_hwm",
    "with_rx_dd",
    "with_rx_calmar",
    # "f_round",
    "perf_row",
]