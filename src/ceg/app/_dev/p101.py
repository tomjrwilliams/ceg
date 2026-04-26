from typing import Any, cast, TYPE_CHECKING, ClassVar
import itertools
from dataclasses import replace
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

import polars as pl

from ... import apis
from ...apis import assert_is

from ... import engine
from ...vis import explorer
from ...objs.dev import bars
from ...objs.dev import pfo
from ...objs.dev import agg
from ...objs.dev import calc

from ...objs.dev import strat

from .core import PROJECTS

PROJECT = "p101"

@dataclass(frozen=True)
class Project_101(strat.StratExplorer):
    PROJECT: ClassVar[str] = PROJECT

    @classmethod
    def load_(cls, kw: dict):
        return cls.load(**{
            k: v for k, v in kw.items()
            if k in [
                "root", "with_ui", "with_files", "test"
            ]
        })
    
    @classmethod
    def load(
        cls,
        root: str = "./__local__/apps",
        test: bool = False,
        with_ui: bool = True,
        with_files: bool = True,
    ):
        """
        >>> root = "./__local__/test"
        >>> proj = Project_101.load(root, test=True).reset()
        >>> nav = proj.prepare_app()
        >>> nav[0]()
        >>> nav[1]()
        """
        domain: strat.StratDomain

        self = (
            Project_101(root=root)
            .with_config(explorer.ExplorerConfig(PROJECT))
        )

        name = self.univ_name("es-tn")

        universes: dict[str, pfo.ContainerDaily] = {}
         univ, universes = (
            strat.StratDomain.new()
            .with_univ(
                name,
                depth=0,
                universes=universes,
            )
        )
        eng = univ.get_parent()

        FUT_ES = ("FUT", "ES")
        FUT_TN = ("FUT", "TN")
        FUT_GC = ("FUT", "GC")

        prices: dict[str, engine.Daily] = dict(
            ES = bars.FutDailyOrdinal.new(
                symbol="ES",
                ident = 1,
                field="close",
                adjust="raw",
                source="frd"
            ),
            TN = bars.FutDailyOrdinal.new(
                symbol="TN",
                ident = 1,
                field="close",
                adjust="raw",
                source="frd"
            ),
            GC = bars.FutDailyOrdinal.new(
                symbol="GC",
                ident = 1,
                field="close",
                adjust="raw",
                source="frd"
            ),
            # = EqDailyEtf.new(
            #     eng,
            #     symbol=params.asset.symbol,
            #     field="close",
            #     adjust="price",
            #     source="frd"
            # )
        )
        context.get().with_objs(
            *prices.values(), t=strat.StratDomain
        )

        strats_lo = dict(
            ES = strat.Strategy.cv(
                FUT_ES, span=48
            ),
            TN = strat.Strategy.cv(
                FUT_TN, span=48
            ),
            GC = strat.Strategy.cv(
                FUT_GC, span=48
            ),
        )

        cap_weight = .75
        vol_weight = .75

        strats = dict(
            ES_08_MR_LO = strat.Strategy.cv_osc(
                FUT_ES,
                span=48, 
                fast=8,
                ratio=1.5,
                activation="tanh",
                qile=cls.qile(5, 0, m=-1, f="z", w=.5),
                cap=strat.CapNotional(
                    val=10. * cap_weight
                ),
                rebal="3D",
                scale=vol_weight
            ),
            ES_64_MO_LO = strat.Strategy.cv_osc(
                FUT_ES,
                span=48, 
                fast=64,
                ratio=1.5,
                activation="tanh",
                qile=cls.qile(5, 3, 5, f="z", w=.5),
                cap=strat.CapNotional(
                    val=10. * cap_weight
                ),
                rebal="3D",
                scale=vol_weight
            ),
            TN_16_MO_LS = strat.Strategy.cv_osc(
                FUT_TN,
                span=48,
                fast=16,
                ratio=1.5,
                activation="tanh",
                qile=cls.qile(5, 4, 1, f="z", w=.5),
                cap=strat.CapNotional(
                    val=20. * cap_weight
                ),
                rebal="3D",
                scale=vol_weight
            ),
            GC_16_MO_LO = strat.Strategy.cv_osc(
                FUT_GC,
                span=48,
                fast=16,
                ratio=1.5,
                activation="tanh",
                qile=cls.qile(5, 3, 5, f="z", w=.5),
                cap=strat.CapNotional(
                    val=10. * cap_weight
                ),
                rebal="3D",
                scale=vol_weight
            ),
        )

        # TODO: re-test underlying strategies now weekly rebal works?
        # TODO: try spy, tlt as well (gold etf?)
        # TODO: try with futures=0.4 margin vs etf=1.0

        # TODO: eco signals, cond eco
        # TODO: z flip

        # TODO: wider equity / bond universe

        bt_lo = []
        for params_lo in strats_lo.values():
        univ, bt_lo = domain.with_backtest(
                univ, params_lo, into=bt_lo
            )
        
        bt_mo_raw = {}
        for params_mo in strats.values():
            assert params_mo.signal is not None, params_mo univ, bt_mo_raw = domain.with_backtest(
                univ,
                replace(
                    params_mo,
                    signal=replace(
                        params_mo.signal,
                        qile=None,
                        zile=None,
                        activation="tanh",
                    ),
                    cap=None,
                ),
                into=bt_mo_raw,
                key = lambda r: (
                    r.params.asset,
                    r.params.signal,
                    r.params.scale,
                    None,
                )
            )
        universes[univ.name] = univ
             univ, universes = universes.by_depth(
            name,
            depth=1,
            universes=universes
        )

        bt_mo = {}
        for params_mo in strats.values():
            params_mo = strat.rescale_cv(
                params_mo,
                bt_mo_raw, # type: ignore
                qile=None,
                zile=None,
                activation="tanh"
            )
            assert params_mo.signal is not None, params_mo
            params_mo = replace(
                params_mo,
                signal=replace(
                    params_mo.signal, activation=""
                )
            ) univ, bt_mo = domain.with_backtest(
                univ,
                params_mo,
                into=bt_mo,
                key = lambda r: (
                    r.params.asset,
                    r.params.signal,
                    r.params.scale,
                    None,
                )
            )

        universes[univ.name] = univ

        pfo_ident = strat.PfolioIdent(
            name, "ex", len(universes) - 1
        )

        (
            universes,
            pf_raw,
            pf_raw_pfo
        ) = strat.basket_pfo(
            list(bt_mo.values()),
            universes,
            pfo_ident,
            suffix="raw"
        )
        ( universes, pf_raw_ntl
        ) = strat.pfo_notionals(
            pf_raw_pfo,
            universes,
            pfo_ident,
            prices=prices,
            cap=10.,
        )
 pfo_nots = strat.pfo_notional_colls(
            pf_raw_pfo,
            pfo_ident,
            prices = prices,
            suffix="raw",
            f_agg="hgross",
        ) pfo_nots_net = strat.pfo_notional_colls(
            pf_raw_pfo,
            pfo_ident,
            prices = prices,
            suffix="raw-net",
            f_agg="hsum"
        )

        (
            universes,
            pf_ntl_cap_ratio,
            bt_final,
            pf,
            pf_pfo,
        ) = strat.cap_pfo_notionals(
            list(bt_mo.values()),
            pfo_nots,
            universes,
            pfo_ident,
            10.,
            suffix="cap",
        )
        (
            universes,
            pf_ntl_cap_ratio_test,
            bt_final_test,
            pf_test,
            pf_pfo_test,
        ) = strat.cap_pfo_notionals(
            list(bt_mo.values()),
            pf_raw_ntl,
            universes,
            pfo_ident,
            10.,
            suffix="test",
        )
        (
            universes,
            pf_ntl_net_cap_ratio,
            bt_net_cap,
            pf_net,
            pf_pfo_net,
        ) = strat.cap_pfo_notionals(
            list(bt_mo.values()),
            pfo_nots_net,
            universes,
            pfo_ident,
            10.,
            suffix="cap-net",
        )
 pfo_nots_cap = strat.pfo_notional_colls(
            pf_pfo,
            pfo_ident,
            prices = prices,
            suffix="cap",
        ) pfo_nots_net_cap = strat.pfo_notional_colls(
            pf_pfo_net,
            pfo_ident,
            prices = prices,
            suffix="cap-net",
        )
        if CAP_VARIANTS:
        univ, universes = universes.by_depth(
                name,
                depth=len(universes),
                universes=universes,
            )
            bt_norm = []
            for bt_mo_cap in bt_mo.values():
                univ, bt_norm = domain.with_backtest(
                    univ,
                    strat.Strategy(
                        asset=bt_mo_cap,
                        scale=strat.ScaleVol(
                            span=256*8,
                            rx=pf_raw.assume_res().rx
                        ),
                        rebal="3D"
                    ),
                    into=bt_norm,
                )    
            universes[univ.name] = univ
            
            pfo_ident = replace(
                pfo_ident, strat_depth=len(universes)-1
            )
            (
                universes,
                pf_norm,
                pf_norm_pfo
            ) = strat.basket_pfo(
                bt_norm,
                universes,
                pfo_ident,
                suffix="norm"
            )
 pfo_norm_nots = strat.pfo_notional_colls(
                pf_norm_pfo,
                pfo_ident,
                prices = prices,
                suffix="norm",
                eng=eng,
            )
            (
                universes,
                pf_ntl_norm_cap_ratio,
                bt_norm_cap,
                pf_norm_cap,
                pf_norm_cap_pfo,
            ) = strat.cap_pfo_notionals(
                bt_norm,
                pfo_norm_nots,
                universes,
                pfo_ident,
                10.,
                suffix="norm-cap",
            ) pfo_nots_norm_cap = strat.pfo_notional_colls(
                pf_norm_cap_pfo,
                pfo_ident,
                prices = prices,
                suffix="norm-cap",
                eng=eng,
            )
             univ, universes = universes.by_depth(
                name,
                depth=len(universes),
                universes=universes,
            )

            cap_span = 512
            pf_ntl_cap_ratio_mu = calc.DailyEMA.new(
                pf_ntl_cap_ratio,
                span=cap_span,
            )
            context.get().with_obj(
                pf_ntl_cap_ratio_mu, t=strat.StratDomain
            )

            bt_div = []
            for bt_mo_cap in bt_mo.values():
                univ, bt_norm = domain.with_backtest(
                    univ,
                    strat.Strategy(
                        asset=bt_mo_cap,
                        scale=strat.ScaleFactor(
                            pf_ntl_cap_ratio_mu.id,
                            f"cap_ratio_{cap_span}"
                        ),
                        rebal="3D"
                    ),
                    into=bt_div,
                )
            universes[univ.name] = univ
            
            pfo_ident = replace(
                pfo_ident, strat_depth=len(universes)-1
            )
            (
                universes,
                pf_div,
                pf_div_pfo
            ) = strat.basket_pfo(
                bt_div,
                universes,
                pfo_ident,
                suffix="div"
            )
 pfo_div_nots = strat.pfo_notional_colls(
                pf_div_pfo,
                pfo_ident,
                prices = prices,
                suffix="div",
                eng=eng,
            )
            (
                universes,
                pf_ntl_div_cap_ratio,
                bt_div_cap,
                pf_div_cap,
                pf_div_cap_pfo,
            ) = strat.cap_pfo_notionals(
                bt_norm,
                pfo_div_nots,
                universes,
                pfo_ident,
                10.,
                suffix="div-cap",
            ) pfo_nots_div_cap = strat.pfo_notional_colls(
                pf_div_cap_pfo,
                pfo_ident,
                prices = prices,
                suffix="div-cap",
                eng=eng,
            )

        if with_files:

            domain = assert_is(
                domain.sync(), strat.StratDomain
            )

            # TODO: params:
            # - signal
            # - scale
            # - asset specs
            # - caps, pfolio weights

            df_pfo = apis.pl.join_dfs(
                [
                    engine.daily_pl_df(
                        obj,
                        domain,
                        alias=key,
                    )
                    for obj, key in [
                        (
                            pfo_nots.pf_ntl_tot,
                            "pfo_ntl",
                        ),
                        (
                            pf_ntl_cap_ratio,
                            "pfo_ntl_cap_ratio",
                        ),
                        (
                            pfo_nots_cap.pf_ntl_tot,
                            "pfo_ntl_cap",
                        ),
                        (
                            pf_ntl_net_cap_ratio,
                            "pfo_ntl_net_cap_ratio",
                        ),
                        (
                            pfo_nots_net_cap.pf_ntl_tot,
                            "pfo_ntl_net_cap",
                        ),
                    ] + (
                        [] 
                        if not CAP_VARIANTS
                        else [
                            (
                                pfo_norm_nots.pf_ntl_tot,
                                "pfo_ntl_norm"
                            ),
                            (
                                pf_ntl_norm_cap_ratio,
                                "pfo_ntl_norm_cap_ratio",
                            ),
                            (
                                pfo_nots_norm_cap.pf_ntl_tot,
                                "pfo_ntl_norm_cap"
                            )
                        ]
                    )
                ] + sum([
                    [
                        engine.daily_pl_df(
                            obj,
                            alias=f"{key}_{suffix}"
                        )
                        for obj, key in zip(objs, keys)
                    ]
                    for objs, keys, suffix
                    in [
                        (
                            prices.values(),
                            prices.keys(),
                            "px"
                        ),
                        (
                            pfo_nots.pf_pos.values(),
                            prices.keys(),
                            "pos"
                        ),
                        (
                            pfo_nots.pf_ntl,
                            prices.keys(),
                            "ntl"
                        ),
                        (
                            pfo_nots_cap
                            .pf_pos
                            .values(),
                            prices.keys(),
                            "pos_cap",
                        ),
                        (
                            pfo_nots_cap.pf_ntl,
                            prices.keys(),
                            "ntl_cap"
                        ),

                        # TODO: norm, norm_cap
                        # TODO: trades, rebal mask
                    ]
                ], []),
                how="full",
                coalesce=True,
                on="date",
            ).sort(by="date")

            def get_obj(
                k: str | tuple[str, ...],
            ) -> engine.Daily | list[engine.Daily]:
                if isinstance(k, str):
                    return context.get().get(k, engine.Daily)
                else:
                    return [
                        context.get().get(kk, engine.Daily)
                        for kk in k
                    ]

            def get_pos(bt):
                bt = assert_is(bt, strat.Strategy)
                return bt.get_pos()
            
            def get_signal(bt):
                bt = assert_is(bt, strat.Strategy)
                return get_obj(assert_is(bt.signal))
            
            def get_scale(bt):
                bt = assert_is(bt, strat.Strategy)
                return get_obj(assert_is(bt.scale))

            df_strat = apis.pl.join_dfs(
                sum([
                    [
                        engine.daily_pl_df(
                            f_obj(bt),
                            alias=f"{key}_{suffix}",
                            method="prod"
                        )
                        for bt, key in zip(bts, keys)
                    ]
                    for bts, keys, suffix, f_obj
                    in [
                        (
                            list(bt_mo_raw.values()),
                            strats.keys(),
                            "scale_asset",
                            get_scale,
                        ),
                        (
                            list(bt_mo.values()),
                            strats.keys(),
                            "scale_strat",
                            get_scale,
                        ),
                        (
                            list(bt_mo_raw.values()),
                            strats.keys(),
                            "sig",
                            get_signal,
                        ),
                        (
                            list(bt_mo.values()),
                            strats.keys(),
                            "sig_qile",
                            get_signal,
                        ),
                        (
                            list(bt_mo.values()),
                            strats.keys(),
                            "pos",
                            get_pos,
                        ),
                        # (
                        #     bt_final,
                        #     strats.keys(),
                        #     "cap_pos",
                        #     get_pos,
                        # ),
                    ]
                ], []),
                how="full",
                coalesce=True,
                on="date",
            ).sort(by="date")

            df_rebalance = apis.pl.join_dfs(
                [
                    assert_is(pfo.rebalance_df(
                        assert_is(params.rebal, str),
                        domain,
                        df_index=df_strat,
                        alias=f"{key}_rebal",
                    ))
                    for key, params in strats.items()
                ],
                how="full",
                coalesce=True,
                on="date",
            )

            df_strat = df_strat.join(
                df_rebalance,
                how="full",
                coalesce=True,
                on="date",
            ).sort(by="date")

            # TODO: clear sheet first

            for sheet, df in {
                "pfo": df_pfo.select(
                    "date",
                    pl.selectors.numeric().fill_nan(None),
                    pl.selectors.boolean(),
                ).filter(
                    pl.any_horizontal(
                        pl.selectors.numeric()
                        .is_not_null()
                    )
                ),
                "strats": df_strat.select(
                    "date",
                    pl.selectors.numeric().fill_nan(None),
                    pl.selectors.boolean(),
                ).filter(
                    pl.any_horizontal(
                        pl.selectors.numeric()
                        .is_not_null()
                    )
                ),
            }.items():
                rs: list[list[Any]] = (
                    [list(df.schema.keys())]
                    + [
                        list(r.values())
                        for r in df.to_dicts()
                    ]
                )
                apis.gsheets.write(
                    apis.gsheets.CEG_PFOLIO, sheet, rs
                )

        if with_ui:
            self = self.with_page(
                self.page_name(0, name),
                type="daily",
                graph=True,
                t=Project_101
            )

            pf_res = (
                bt_lo
                + [
                    pf_raw,
                    pf,
                    pf_test,
                    pf_net,
                ]
                + (
                    [] if not CAP_VARIANTS
                    else [
                        pf_norm,
                        pf_norm_cap,
                        pf_div,
                        pf_div_cap
                    ]
                )
                + bt_final
                + bt_final_test
                + bt_net_cap
                + (
                    [] if not CAP_VARIANTS
                    else (
                        assert_is(bt_norm, list)
                        + assert_is(bt_norm_cap, list)
                        + assert_is(bt_div_cap, list)
                    )
                )
                + list(bt_mo.values())
                # + list(bt_mo_raw.values()),
            )

            self = self.with_perf_table(
                pf_res,
                t=Project_101
            )
            self = self.with_rx_graph(
                pf_res,
                t=Project_101
            )

        self = self.with_graph(
            t=Project_101,
            name=self.page_name(0, name)
        )
        
        return self

CAP_VARIANTS = False

PROJECTS[PROJECT] = Project_101.load_

__all__ = [
    "Project_101",
]