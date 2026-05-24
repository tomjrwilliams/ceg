"""DailyComposition: unified replacement for DailyAggregate + DailyCombination.

Holds inputs as typed Refs directly (vs the old integer indices into
Container.agg / Container.comb). Method-driven handler dispatch via the
existing HANDLERS registry in aggs.py.

DailyOutput: unified deriv (was DailyAggregateDeriv / DailyCombinationDeriv).
Thin shim over engine.Output specialised for Daily / chunked storage.
"""

from __future__ import annotations
import datetime as dt
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass
from dataclasses import replace

from frozendict import frozendict

import polars as pl

from .... import apis
from ....apis import t, assert_is
from ....engine import (
    VALUE,
    Daily,
    DailyChunked,
    DependsParentValue,
    Dependency,
    Event,
    Graph,
    HandleNewDailyFromDf,
    HandleNewValueFromPl,
    IdRef,
    AliasRef,
    MemberRef,
    Object,
    ObjectChange,
    Output,
    Ref,
    Snap,
    ValueUpdate,
    Handler,
    context,
    daily_pl_df,
)


TParamValue = bool | int | float | str | None


def _encode_ref_for_id(r: Ref | str) -> str:
    """Compact, deterministic string encoding of a Ref for use inside an id.

    Aliases are not session-stable so we don't allow them in id construction.
    """
    if isinstance(r, str):
        return r
    if isinstance(r, IdRef):
        return r.id
    if isinstance(r, MemberRef):
        return f"{_encode_ref_for_id(r.parent)}-{r.field}-{r.slot}"
    if isinstance(r, AliasRef):
        raise ValueError(
            "AliasRef in DailyComposition.refs is not stable for id construction;"
            " resolve to IdRef first."
        )
    raise ValueError(r)


def _ref_target_id(r: Ref | str) -> str:
    """The id-form target string for dependency wiring."""
    if isinstance(r, str):
        return r
    if isinstance(r, IdRef):
        return r.id
    if isinstance(r, AliasRef):
        # Resolved at wiring time via Graph.get; for now flag as not directly
        # encodable. Caller should resolve first.
        raise ValueError(
            "AliasRef in DailyComposition.peers needs to be resolved to an"
            " IdRef before dependency wiring."
        )
    if isinstance(r, MemberRef):
        # Member refs don't address a single graph id directly; would need to
        # be resolved through Graph for a Dependency. Defer until needed.
        raise ValueError(
            "MemberRef in DailyComposition.peers needs to be resolved before"
            " dependency wiring."
        )
    raise ValueError(r)


@dataclass(frozen=True)
class DailyComposition(DailyChunked, DependsParentValue):
    """A composition that combines values from multiple upstream refs.

    parent  — the namespace this composition belongs to (typically a Container).
              Depended on via DependsParentValue: the parent fires one
              ValueUpdate when all its members have first-published; that
              triggers this composition's handler.
    method  — name in HANDLERS registry; selects the calc.
    refs    — positional Ref inputs (replaces DailyAggregate.args).
    named   — named Ref inputs (replaces Combination.refs).
    params  — flat keyword args passed to the handler.
    peers   — explicit extra ValueUpdate dependencies, addressed by Ref.
    """

    parent: Object | str
    method: str

    refs:  tuple[IdRef | AliasRef | MemberRef | str, ...] = ()
    named: t.frozen.dict[str, IdRef | AliasRef | MemberRef | str] = t.frozen.dict()
    params: t.frozen.dict[str, TParamValue] | None = None
    peers: tuple[IdRef | AliasRef | MemberRef | str, ...] = ()

    version: int = 0
    type: Literal["DailyComposition"] = "DailyComposition"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyCompositionSnap"] = "DailyCompositionSnap"

    # ---- construction ----

    @classmethod
    def new(
        cls,
        parent: Object,
        method: str,
        refs: tuple[Ref | str, ...] = (),
        named: dict[str, Ref | str] | frozendict | None = None,
        params: dict[str, TParamValue] | frozendict | None = None,
        peers: tuple[Ref | str, ...] = (),
    ) -> "DailyComposition":
        refs_str = "-".join(_encode_ref_for_id(r) for r in refs)
        named_str = "-".join(
            f"{k}={_encode_ref_for_id(v)}"
            for k, v in (named or {}).items()
        )
        params_str = (
            "-".join(f"{k}={v}" for k, v in params.items())
            if params else ""
        )

        suffix = method
        if refs_str:   suffix += f"-{refs_str}"
        if named_str:  suffix += f"-{named_str}"
        if params_str: suffix += f"-{params_str}"

        return cls(
            parent=parent,
            id=parent.extend_id(suffix),
            method=method,
            refs=refs,
            named=frozendict(named or {}),
            params=frozendict(params) if params else None,
            peers=peers,
        )

    # ---- graph wiring ----

    def get_parent(self) -> Object:
        return context.get().get(self.parent, Object)

    def depends(self) -> list[Dependency]:
        return self.depends_parent() + [
            Dependency(
                self.id,
                _ref_target_id(p),
                events=("ValueUpdate",),
            )
            for p in self.peers
        ]

    def handler(self, event: Event, snap: Snap) -> Handler:
        parent = self.get_parent()
        parent_update = (
            isinstance(event, ValueUpdate) and event.obj == parent.id
        )
        if parent_update:
            # Lazy import to avoid circular: aggs.py imports composition.py
            # would be a loop; aggs HANDLERS is a runtime lookup.
            from .aggs import agg_handler
            return agg_handler(self.method, self.params)
        raise ValueError((self, event))

    # ---- fan-out children ----

    def deriv(
        self,
        key: str | None = None,
        index: int | None = None,
    ) -> "DailyOutput":
        return DailyOutput.new(self, key=key, index=index)


@dataclass(frozen=True)
class DailyOutput(Output, DailyChunked):
    """Fan-out child of a composition's handler.

    The handler computes the parent's value and writes the children's values
    as a side effect. DailyOutput has no deps of its own — Output.depends()
    returns [].
    """

    parent: Object | str
    version: int = 0
    type: Literal["DailyOutput"] = "DailyOutput"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyOutputSnap"] = "DailyOutputSnap"


# ---- df resolution + handler base ----

def _resolve_ref(r: Ref | str, g: Graph) -> Object:
    """Resolve a Ref or bare id-string to a Daily-shaped Object."""
    if isinstance(r, str):
        return assert_is(g.get(r, Object), Daily)
    return assert_is(g.get(r, Object), Daily)


def get_composition_dfs(
    refs: tuple[Ref | str, ...],
    g: Graph,
) -> tuple[list[pl.DataFrame], dt.date, dt.date]:
    """Resolve refs to Daily objects, join their dfs, return [single df], d_min, d_max.

    Each ref contributes one column (named by the resolved Object's id) to a
    single joined df. Compositions over multiple groups should be modelled as
    sub-Containers, each of which is itself a ref.
    """
    if not len(refs):
        raise ValueError("DailyComposition.refs is empty")

    objs = [_resolve_ref(r, g) for r in refs]

    df = (
        apis.pl.join_dfs(
            [daily_pl_df(o) for o in objs],
            on="date",
            align=True,
        )
        .sort(by="date")
        .select("date", *[o.id for o in objs])
    )

    d0, d1 = apis.pl.select_items(
        df,
        (pl.col("date").min(), pl.col("date").max()),
        dt.date,
    )
    return [df], d0, d1


@dataclass(frozen=True)
class HandleNewDailyComposition(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):
    """Handler base for DailyComposition.

    Subclasses override `agg_dfs(first, last, *dfs)` with the concrete
    calculation (e.g. h-sum, h-prod, rank-and-weight). The result is either:

      - a single pl.DataFrame with a VALUE column (the composition's own value), or
      - a tuple (df, k_is) — a df with multiple value columns plus a list of
        (key|index|tuple) entries that spawn DailyOutput children.
    """

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        *dfs: pl.DataFrame,
    ) -> pl.DataFrame | tuple[
        pl.DataFrame, list[str | int | tuple[str, int]]
    ]:
        raise NotImplementedError(self)

    def handle(
        self,
        event: ObjectChange,
        snap: "DailyComposition.Snap",
        obj: "DailyComposition",
    ):
        g = context.get(Graph)

        dfs, d0, d1 = get_composition_dfs(obj.refs, g)
        dfs = dfs + [
            daily_pl_df(_resolve_ref(p, g)) for p in obj.peers
        ]

        if snap.first and snap.last:
            if d0 == snap.first and d1 == snap.last:
                return frozendict().set(obj.id, (snap, ()))

        res = self.agg_dfs(d0, d1, *dfs)

        df = None
        snaps = frozendict()

        if isinstance(res, pl.DataFrame):
            df = res
            assert len(df) == ((d1 - d0).days + 1), (df, d1, d0)
            self.set_value_from_df(
                obj, df, VALUE, t=self.key_value_type(obj)
            )
        elif isinstance(res, tuple):
            df, k_is = res
            assert len(df) == ((d1 - d0).days + 1), (df, d1, d0)
            _snaps = {}

            def store_child(k_i):
                if isinstance(k_i, str):
                    k = col = k_i
                    i = None
                elif isinstance(k_i, int):
                    i = k_i
                    col = f"v_{i}"
                    k = None
                else:
                    k, i = k_i
                    col = f"{k}_{i}"
                child = obj.deriv(key=k, index=i)
                context.get().with_obj(child, force=True)
                child_snap = child.get_snap(Daily.Snap)
                self.set_value_from_df(
                    child, df, col, t=self.key_value_type(child)
                )
                _snaps[child.id] = (
                    replace(child_snap, first=d0, last=d1),
                    ValueUpdate(child.id),
                )

            list(map(store_child, k_is))
            for k, v in _snaps.items():
                assert k not in snaps, k
                snaps = snaps.set(k, v)
        else:
            raise ValueError((type(res), self))

        return ValueUpdate.from_snap(
            self.update_snap_from_df(
                snap,
                assert_is(df),
                d_min=d0,
                d_max=d1,
            ),
            into=snaps,
        )


__all__ = [
    "TParamValue",
    "DailyComposition",
    "DailyOutput",
    "get_composition_dfs",
    "HandleNewDailyComposition",
]
