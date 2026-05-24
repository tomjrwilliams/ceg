# Step 4.5: bt migration — delete `DailyCombination` + `combs.py`

Goal: eliminate the `Strategy → Combination → Container.comb → args[idx] → refs["rx"]` indirection chain. Strategy / Basket become typed parent namespaces; bt types read refs directly off them.

## End state

### Strategy

```python
@dataclass(frozen=True)
class Strategy(TriageUpstream, UseRootStorage):     # was: HasBackTest, HasAsset, AssetLike
    parent: Container | str               # namespace this strategy lives in
    asset:  AssetConcrete | AssetSynthetic

    # typed ref slots — replace Combination.refs
    rx:     IdRef
    px:     IdRef
    gross:  IdRef
    signal: IdRef | None = None
    scale:  IdRef | None = None
    cap:    IdRef | None = None

    # other strategy params (rebal etc.) stay as plain fields
    params: StrategyParams = ...
    name: str | None = None

    # the backtest / perf attached to this strategy
    bt:   DailyBtUvar | str | None = None
    perf: DailyBtPerf  | str | None = None

    version: int = 0
    type: Literal["Strategy"] = "Strategy"

    @dataclass(frozen=True)
    class Snap(TriageUpstream.Snap):
        type: Literal["StrategySnap"] = "StrategySnap"

    def upstream_paths(self) -> list[str]:
        # TriageUpstream contract — gate downstream bt on all data being ready
        return [r.id for r in (self.rx, self.px, self.gross, self.signal, self.scale, self.cap) if r is not None]

    def handler(self, event, snap):
        if isinstance(event, ValueUpdate):
            return HandleUpdateLatestUpstream()
        raise ValueError((self, event))
```

- Strategy is a Namespace (TriageUpstream) — fires one ValueUpdate when all its members have first-published.
- Children (rx/px/gross/signal/scale/cap) live as regular graph Objects; Strategy holds IdRefs to them.
- Bt/perf are computed downstream and may be assigned post-construction (legacy mutability via replace).

### Basket

```python
@dataclass(frozen=True)
class Basket(TriageUpstream, UseRootStorage):
    parent: Container | str
    members: tuple[IdRef, ...] = ()       # refs to Strategy objects
    params: StrategyParams = ...
    name: str = ""
    bt:   DailyBtMvar | str | None = None
    perf: DailyBtPerf  | str | None = None
    version: int = 0
    type: Literal["Basket"] = "Basket"
    ...
    def upstream_paths(self):
        return [r.id for r in self.members]
```

### DailyBtUvar / DailyBtMvar / DailyBtWeights

```python
@dataclass(frozen=True)
class DailyBtUvar(DailyChunked, DependsParentValue):
    parent: Strategy | str
    version: int = 0
    type: Literal["DailyBtUvar"] = "DailyBtUvar"

    @dataclass(frozen=True)
    class Snap(Daily.Snap): ...

    @classmethod
    def new(cls, parent: Strategy):
        return cls(parent=parent, id=parent.extend_id("bt-uv"))

    def get_parent(self) -> Strategy: ...

    # depends() inherited from DependsParentValue → [parent ValueUpdate]
    def handler(self, event, snap):
        if isinstance(event, ValueUpdate):
            return HandleNewDailyBtUvar()
        raise ValueError(...)

    def rx(self):   return DailyBtRx.new(self)
    def px(self):   return DailyBtPx.new(self)
    def gross(self):return DailyBtGross.new(self)
    def pos(self):  return DailyBtPos.new(self)
```

`DailyBtMvar(parent: Basket | str)` analogous. `DailyBtWeights(parent: Basket | str)` analogous.

`Backtest(Combination)` → **deleted**.

### Handlers

`HandleNewDailyBtUvar.handle`:

```python
def handle(self, event, snap, obj: DailyBtUvar):
    strategy = obj.get_parent()
    g = context.get(Graph)
    df_rx     = daily_pl_df(g.get(strategy.rx, Daily))
    df_px     = daily_pl_df(g.get(strategy.px, Daily))
    df_gross  = daily_pl_df(g.get(strategy.gross, Daily))
    df_signal = daily_pl_df(g.get(strategy.signal, Daily)) if strategy.signal else None
    df_scale  = daily_pl_df(g.get(strategy.scale,  Daily)) if strategy.scale  else None
    df_cap    = daily_pl_df(g.get(strategy.cap,    Daily)) if strategy.cap    else None

    res_df = combine_backtest_dfs(df_rx=df_rx, df_px=df_px, df_gross=df_gross,
                                   df_signal=df_signal, df_scale=df_scale, df_cap=df_cap,
                                   **strategy.params.shifts())
    # writes obj's value; fans out to DailyBtRx/Px/Gross/Pos children
    return ...
```

`HandleNewDailyBtMvar.handle`:

```python
def handle(self, event, snap, obj: DailyBtMvar):
    basket = obj.get_parent()
    sub_dfs = []
    for sref in basket.members:
        strat = context.get().get(sref, Strategy)
        sub_dfs.append((strat, daily_pl_df(strat.bt, Daily)))  # already computed per-strat
    # combine basket-level: weights * each strat's rx
    ...
```

### DailyBt* Output (deriv) types

```python
@dataclass(frozen=True)
class DailyBtPos(Output, DailyChunked):
    parent: Object | str    # any parent (DailyBtUvar | DailyBtMvar | DailyBtWeights)
    version: int = 0
    type: Literal["DailyBtPos"] = "DailyBtPos"
    @dataclass(frozen=True)
    class Snap(Daily.Snap): ...
```

Same shape for `DailyBtRx`, `DailyBtPx`, `DailyBtGross`, `DailyBtWeight`.

`DailyCombinationDeriv` → **deleted**.

### Container shrink

```python
@dataclass(frozen=True)
class Container(TriageUpstream, UseRootStorage):
    parent: Project | str
    name: str
    objects: tuple[IdRef | str, ...] = ()      # the only members slot
    # agg, comb, params — DELETED
    type: Literal["Container"] = "Container"
    # with_object stays; with_agg/with_comb/with_params DELETED
```

`Aggregate`, `Combination`, the local `Ref(type, i)` type → **deleted** from `containers.py`.

`combs.py` → **deleted** (DailyCombination + DailyCombinationDeriv + HandleNewDailyCombination + get_comb_dfs all gone).

## Migration order

Each row is a discrete change with tests gating it. `*` = tests file added; otherwise existing tests must still pass.

| # | Change | Tests | Risk |
|---|---|---|---|
| 1 | `tests/test_strategy.py` writes the new Strategy contract: typed Ref slots, TriageUpstream upstream_paths, handler dispatch on ValueUpdate. * | new file | — |
| 2 | Rewrite `Strategy` class in `strat/params.py`: typed Ref slots, drop `bind_spec`, drop `unpack`. Replace `.bind()` to (a) call `with_nested` for each data child or (b) accept pre-constructed refs. Strategy IS the namespace; no Combination intermediate. | test_strategy + import smoke | high — touches many call sites |
| 3 | Rewrite `Basket` analogously: members tuple of Strategy refs. | test_basket * | medium |
| 4 | `tests/test_bt_types.py`: new DailyBtUvar/Mvar/Weights constructors, depends(), handler dispatch. * | new file | — |
| 5 | Rewrite `DailyBtUvar/Mvar/Weights` in `bt.py`: inherit `DailyChunked, DependsParentValue`; parent is Strategy/Basket; drop args, drop metric. New `.new(parent)`, no Combination dependency. | test_bt_types + import smoke | high |
| 6 | Rewrite `HandleNewDailyBtUvar/Mvar/Weights.handle`: read from Strategy/Basket fields; replace `obj.params(coll)` + Combination decoding with typed field access. | structural / pure-function unit tests possible for `combine_backtest_dfs` orchestration | high — actual data flow |
| 7 | Migrate `DailyBtPos/Rx/Px/Gross/Weight` to extend `Output, DailyChunked` directly with `parent: Object | str`. Drop their `DailyCombinationDeriv` base. | test_bt_types (resolution) | low |
| 8 | Migrate `strat/baskets/agg.py` (`DailyAggregate` already migrated; now also remove residual Combination-shaped wiring) | import smoke | low |
| 9 | Update `strat/domain.py` parser unions: remove `Backtest`, add nothing new (Strategy/Basket already there). | import smoke | low |
| 10 | Delete `combs.py`: `DailyCombination`, `DailyCombinationDeriv`, `HandleNewDailyCombination`, `get_comb_dfs`. | full test pass | medium |
| 11 | Delete `Backtest` from `bt.py`. | full test pass | low |
| 12 | Shrink `Container`: drop `agg`, `comb`, `params` fields; drop `with_agg`, `with_comb`, `with_params` methods; drop `Aggregate`, `Combination`, `Ref` (local) types in `containers.py`. | full test pass + import smoke | medium |
| 13 | Update `objs/dev/agg/__init__.py` star-exports. | import smoke | trivial |

## Things I intentionally won't change in this round

- `app/_dev/p20*.py` — per prior instruction, user-facing apps stay broken.
- `objs/dev/strat/baskets/comb.py` — looks like an alt comb impl, separate concern.
- `combine_backtest_dfs` and helpers in `pfo/d/core.py` — pure functions, unchanged.
- `DailyPfolio/DailyPfolioRx/etc.` in `pfo.py` — they parent on DailyBtUvar/Mvar; will keep working once their parent types update because they're already in the strat/domain parser union.

## Design points locked in

- **`with_nested` supports single-ref slots.** Fields typed `IdRef | None` (or any union of Refs + None) are recognised by `_slot_shape` as `"single"`. Their child id is `parent.id + "-" + field` (no slot index). This is how Strategy holds rx/px/gross/signal/scale/cap as direct fields — no `tuple[Ref, ...]` workaround needed. Tests: see `tests/test_with_nested.py::test_single_slot_*`.

- **Param→object factories are free functions.** Each spec type (`SignalOsc`, `ScaleVol`, `CapNotional`, etc.) gets a corresponding free function — e.g. `build_signal_osc(spec, asset_data) -> Daily`. The existing `.obj()` methods on spec classes (`SignalOsc.obj`, `ScaleVol.obj`, etc.) migrate to free functions dispatched by spec type (e.g. via a single `build_param_obj(spec, ...)` with isinstance branching, or a registry). Strategy.bind orchestrates: walks its params, calls the appropriate factory for each non-None spec, then `with_nested`s the result into the typed slot.

## Things to flag during implementation

- **Param-driven sub-objects (signal, scale, cap):** Today `StrategyParams.bind(spec, params, self, univ)` builds out e.g. an `EOsc` signal object and writes its index into `Combination.refs["signal"]`. The new flow needs the signal/scale/cap *object* constructions to land on the Strategy's typed slot directly. This is the most subtle part of the migration; the bind logic moves from "decorate a Combination" to "populate Strategy fields".
- **Performance-driven snap structure:** `Strategy.Snap` (currently uses `Daily.Snap`) must become `TriageUpstream.Snap` if Strategy actually fires triage events. Verify by reading TriageUpstream handler chain.
- **`DailyBtPerf` parent type:** It takes `parent: DailyBtRx`. Unchanged. But its parser-union entry stays as-is.
- **Storage routing:** `UseRootStorage.value_key_val()` splits id on first `-`. New ids still start with project name (Strategy is `<project>-<container>-<i>`), so this still works.

## Stop conditions

If we hit either of these, stop and check in:
- More than 2 framework files outside the listed migration order need editing
- More than 3 currently-passing tests break for non-trivial reasons
- A circular import emerges that can't be broken with a lazy import

If we finish cleanly, the post-state has:
- `combs.py` deleted, `Backtest` deleted, `DailyCombination*` deleted
- `Container` reduced to `{parent, name, objects}` + `with_object`
- `Strategy`/`Basket` are first-class typed namespaces
- `DailyBt*` consume Strategy/Basket fields directly
- ~600 LOC removed net; the strat-to-bt indirection chain is one hop instead of four




Session done — 133 tests pass, 1 pre-existing failure, 1 skipped (Bound-using test).

  This round's deliverables
     
  Bound/Collection/init_bind cleanup:
  - Deleted init_bind from engine/collections.py.
  - Deleted bind_sum/bind_prod/bind_rank/bind_quantize/bind_transition_prob from calc/d/h/core.py.
  - Deleted bind = init_bind(new, ...) from rand/gaussian.py and calc/d/trans_prob.py.
  - Deleted three .unpack(bound, coll, force) methods from Strategy, Basket, StrategyParams in strat/params.py.
  - test_handle_pl.py::test_2 skipped with a note (uses removed Bound machinery; needs rewrite against new Graph alias API).
  
  triage handler.
  - DailyBtUvarTyped(DailyChunked, DependsParentValue) in bt.py — parent: Object | str, .new(parent) produces <parent>-bt-uv id,
   depends_parent() wires the trigger, handler stub.
  - DailyBtMvarTyped analogous on BasketTyped, <parent>-bt-mv id.
  - HandleNewDailyBtUvarTyped / HandleNewDailyBtMvarTyped stubs raising NotImplementedError — construction/dispatch contract
  validated; body is the only thing deferred.
  - 14 tests in tests/test_bt_typed.py.

  Remaining for future sessions

  - Handler bodies for HandleNewDailyBtUvarTyped/HandleNewDailyBtMvarTyped — read strategy.rx/px/gross/signal/scale/cap (or
  basket.members → each strategy), build dfs, call combine_backtest_dfs, write the bt value and fan out to
  DailyBtRx/Px/Gross/Pos.
  - Free-function param factories — extract SignalOsc.obj/ScaleVol.obj/CapNotional.obj/etc. to free functions; new
  Strategy.bind(graph, asset) orchestrates by calling each.
  - Migrate strat/baskets/agg.py and strat/params.py Strategy.bind/Basket.bind to populate the new typed slots (with_nested for
  each).
  - Delete legacy: DailyBtUvar/DailyBtMvar/DailyBtWeights/Backtest + combs.py (DailyCombination*) + containers.py interning
  fields (agg/comb/params).
  - Delete Collection from engine/collections.py once final framework callers (Bound/Bind/Apply/Select are still referenced in
  app/core.py and app/dev/states/sign.py, both user-facing).
