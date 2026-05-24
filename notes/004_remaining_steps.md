# Remaining steps after the typed-shape foundation

Status snapshot (as of the last session): 133 tests pass. The new typed
shapes (Ref / Output / DailyComposition / Graph.with_nested / Graph
aliases/tags/info/where / StrategyTyped / BasketTyped / DailyBtUvarTyped /
DailyBtMvarTyped) all exist and have tests. Legacy types (`Strategy`,
`Basket`, `DailyBtUvar`, `DailyBtMvar`, `DailyBtWeights`, `DailyCombination`,
`Container.agg`/`comb`/`params`, `Collection`/`Bound`/`Bind`/`Apply`/`Select`)
remain in place because user-facing callers still depend on them.

## Step A — Handler bodies for DailyBtUvarTyped / DailyBtMvarTyped

Both currently raise `NotImplementedError`. Once they read from typed parents,
the new bt path is fully runnable end-to-end.

### A1. `HandleNewDailyBtUvarTyped.handle`

```python
def handle(self, event, snap, obj: DailyBtUvarTyped):
    strat = obj.get_parent()                 # StrategyTyped
    g = context.get(Graph)

    df_rx     = daily_pl_df(g.get(strat.rx,    Daily))
    df_px     = daily_pl_df(g.get(strat.px,    Daily)) if strat.px    else None
    df_gross  = daily_pl_df(g.get(strat.gross, Daily)) if strat.gross else None
    df_signal = daily_pl_df(g.get(strat.signal, Daily)) if strat.signal else None
    df_scale  = daily_pl_df(g.get(strat.scale,  Daily)) if strat.scale  else None
    df_cap    = daily_pl_df(g.get(strat.cap,    Daily)) if strat.cap    else None

    res_df = combine_backtest_dfs(
        df_rx=df_rx, df_scale=df_scale, df_signal=df_signal,
        df_px=df_px, df_gross=df_gross, df_cap=df_cap,
        cap=strat.params.cap.val if strat.params.cap else None,
        # shifts → defaults inside combine_backtest_dfs
    )

    self.set_value_from_df(obj, res_df, VALUE, t=self.key_value_type(obj))
    # fan-out: DailyBtRx/Px/Gross/Pos children — same pattern as
    # HandleNewDailyComposition.handle (see composition.py).
    return ValueUpdate.from_snap(
        self.update_snap_from_df(snap, res_df), into=frozendict()
    )
```

Mix `HandleNewValueFromPl` + `HandleNewDailyFromDf` into the handler base
class so `set_value_from_df` / `update_snap_from_df` are inherited.

### A2. `HandleNewDailyBtMvarTyped.handle`

```python
def handle(self, event, snap, obj: DailyBtMvarTyped):
    basket = obj.get_parent()                 # BasketTyped
    g = context.get(Graph)

    member_dfs = []
    for sref in basket.members:
        strat = g.get(sref, StrategyTyped)
        # Each member already has its own DailyBtUvar attached
        # (via strat.bt: IdRef once Step B has wired things up).
        member_bt = g.get(strat.bt, DailyBtUvarTyped)
        member_dfs.append(daily_pl_df(member_bt))

    # Combine across members per basket.params.weight scheme.
    res_df = combine_mv_dfs(member_dfs, weight=basket.params.weight)
    self.set_value_from_df(obj, res_df, VALUE, t=self.key_value_type(obj))
    return ValueUpdate.from_snap(
        self.update_snap_from_df(snap, res_df), into=frozendict()
    )
```

Tests:
- A end-to-end test in `tests/test_bt_typed.py` that wires a StrategyTyped
  with synthetic rx/px/gross IdRefs into a graph, runs `graph.sync()`, and
  reads the bt's value. Use `DailyGaussian` as a noise source for inputs
  (mirrors `test_handle_pl.py::test_0`).
- A multi-strategy test for `DailyBtMvarTyped`.

## Step B — Free-function param factories

Today: `SignalOsc.obj(strategy)`, `ScaleVol.obj(strategy)`, etc. are
methods on the spec classes that construct the underlying Daily series
(EOsc, ERms, ...).

End state: free functions per spec type that take the spec, the asset's
data refs, and a graph; return `(graph, IdRef)`.

```python
# objs/dev/strat/factories.py
def build_scale(spec: TScale, asset: TAsset, graph: Graph) -> tuple[Graph, IdRef]:
    if isinstance(spec, ScaleVol):
        rx_obj = graph.get(IdRef(asset.rx().id), Daily)   # or however asset.rx() flows
        ema = DailyEMA.new(...)                            # or whichever calc
        graph = graph.with_obj(ema)
        return graph, IdRef(ema.id)
    if isinstance(spec, ScaleNotional):  ...
    if isinstance(spec, ScaleFactor):    ...
    raise ValueError(spec)

def build_signal(spec: TSignal, asset: TAsset, graph: Graph) -> tuple[Graph, IdRef]:
    if isinstance(spec, SignalOsc):
        # build EOsc + ERms chain
        ...
    ...

def build_cap(spec: TCap, asset: TAsset, graph: Graph) -> tuple[Graph, IdRef]:
    ...
```

Tests in `tests/test_factories.py`:
- Each factory: given a spec, returns an IdRef pointing at the right
  concrete Daily type registered in the graph.
- Each factory: idempotent on repeated call with same spec (returns same id).
- Each factory: distinct specs produce distinct ids.

Migration once factories exist:
- Delete `SignalOsc.obj` / `ScaleVol.obj` / `CapNotional.obj` etc. methods.

## Step C — Strategy.bind / Basket.bind rewrites

### C1. `StrategyTyped.bind`

```python
def bind(self, graph: Graph) -> tuple[Graph, StrategyTyped]:
    # 1. Build rx/px/gross from self.asset
    rx_obj    = self.asset.rx()
    px_obj    = self.asset.px()
    gross_obj = self.asset.gross()
    graph = graph.with_obj(rx_obj).with_obj(px_obj).with_obj(gross_obj)

    # 2. Build signal/scale/cap from self.params, via free-function factories
    signal_ref = None
    if self.params.signal is not None:
        graph, signal_ref = build_signal(self.params.signal, self.asset, graph)
    scale_ref = None
    if self.params.scale is not None:
        graph, scale_ref = build_scale(self.params.scale, self.asset, graph)
    cap_ref = None
    if self.params.cap is not None:
        graph, cap_ref = build_cap(self.params.cap, self.asset, graph)

    # 3. Update Strategy with typed slot refs
    self = replace(
        self,
        rx=IdRef(rx_obj.id), px=IdRef(px_obj.id), gross=IdRef(gross_obj.id),
        signal=signal_ref, scale=scale_ref, cap=cap_ref,
    )
    graph = graph.with_obj(self)

    # 4. Build bt and attach
    bt = DailyBtUvarTyped.new(parent=self)
    graph = graph.with_obj(bt)
    self = replace(self, bt=IdRef(bt.id))
    graph = graph.with_obj(self, force=True)

    return graph, self
```

### C2. `BasketTyped.bind`

```python
def bind(self, graph: Graph, member_strats: tuple[StrategyTyped, ...]) -> tuple[Graph, BasketTyped]:
    member_refs = []
    for s in member_strats:
        graph, s = s.bind(graph)
        member_refs.append(IdRef(s.id))
    self = replace(self, members=tuple(member_refs))
    graph = graph.with_obj(self)

    bt = DailyBtMvarTyped.new(parent=self)
    graph = graph.with_obj(bt)
    self = replace(self, bt=IdRef(bt.id))
    graph = graph.with_obj(self, force=True)
    return graph, self
```

Tests in `tests/test_strategy.py` / `tests/test_basket.py`:
- After `bind`, all six typed slots populated (where appropriate).
- After `bind`, `strategy.bt` is a valid IdRef resolving to a `DailyBtUvarTyped`.
- The bt's `depends_parent()` resolves correctly.
- After `bind`, `graph.sync()` runs without error (gated on Step A handlers).

## Step D — Migrate framework callers

### D1. `strat/baskets/agg.py`

Already partially migrated (off `DailyAggregate`/`DailyAggregateDeriv`). Two
call sites remaining that build strategies via the legacy `Strategy.bind_spec`
+ `Combination` flow. Rewrite each `for strat in strats: univ, _ = univ.with_object(strat.rx(), …)` chain to:

```python
strats_typed = [
    StrategyTyped.from_legacy(s)   # helper that converts old Strategy → StrategyTyped
    for s in strats
]
graph = context.get()
for s in strats_typed:
    graph, _ = s.bind(graph)
basket = BasketTyped(...)
graph, basket = basket.bind(graph, member_strats=tuple(strats_typed))
```

A small helper `StrategyTyped.from_legacy(old: Strategy) -> StrategyTyped`
makes the conversion ergonomic for the migration. Delete the helper after
all callers are converted.

### D2. `strat/project.py`

`StratExplorer.with_perf` and similar methods accept `list[Strategy | Basket]`.
Either:
- (a) update signatures to accept `list[StrategyTyped | BasketTyped]`, migrate
  call sites; or
- (b) make the legacy types unions with the typed types until callers move.

Pick (a) once all callers are migrated.

### D3. Tests

- `tests/test_strategy_bind.py` — Strategy.bind end-to-end.
- `tests/test_basket_bind.py` — Basket.bind end-to-end.

## Step E — Delete legacy

After Steps A–D are landed and tested:

1. **`DailyBtUvar`, `DailyBtMvar`, `DailyBtWeights`, `DailyBtXvar`, `Backtest`**
   in `bt.py` → delete.
2. **`HandleNewDailyBtUvar`, `HandleNewDailyBtMvar`, `HandleNewDailyBtWeights`**
   → delete.
3. **`combs.py`** → delete the file. Update `objs/dev/agg/__init__.py`
   star-import.
4. **`containers.py`** — strip `agg`, `comb`, `params` fields from
   `Container`. Remove `Aggregate`, `Combination`, local `Ref`. Remove
   `with_agg`, `with_comb`, `with_params`. Keep `objects` + `with_object`.
5. **`Strategy`, `Basket`, `Strategies`, `HasBackTest`** legacy classes in
   `strat/params.py` → delete. `Strategy.bind_spec`, `Basket.bind_specs`
   → delete.
6. **Rename** `StrategyTyped → Strategy`, `BasketTyped → Basket`,
   `DailyBtUvarTyped → DailyBtUvar`, etc. After deletion, the suffix becomes
   redundant.
7. Update `strat/domain.py` parser unions accordingly.

## Step F — Delete Collection / Bound / Bind / Apply / Select

After Step D:

1. `Collection`, `Bound`, `Bind`, `Apply`, `Select` etc. in
   `engine/collections.py` → delete.
2. `unpack_bound`, `bound`, `bind_params`, `where`, `select`, `Filter`,
   `Aliased`, `SelectLiteral`, `SelectWhere`, `SelectProduct`, `SelectZip`
   → delete.
3. `app/core.py` `AppConfig.with_configs` / `AppConfig.run` — rewrite
   against the new `Graph.with_obj(alias=, tag=, info=)` + `Graph.where()`
   API.
4. `app/dev/states/sign.py` and other `app/_dev/` scripts — caller updates.
5. Delete `engine/collections.py` entirely.
<!-- 
## Step G — Step 6: Auto-id factories

Independent of A–F. Per `notes/002_engine_unification.md`:

```python
class Object:
    @classmethod
    def role(cls, **fields) -> str:
        return camel_to_kebab(cls.__name__)
    @classmethod
    def encode_params(cls, **fields) -> str:
        return ""
    @classmethod
    def auto_id(cls, parent, **fields) -> str:
        suffix = cls.role(**fields)
        params = cls.encode_params(**fields)
        if params: suffix += "-" + params
        return parent.extend_id(suffix) if parent else suffix
```

Migrate `.new()` factories to call `auto_id` instead of bespoke
`cls.build_id(parent, ...)`. ~20 types touched, each ~3 LOC change.

Tests: `tests/test_auto_id.py` verifying determinism per type. -->

## Step H — Storage routing via parent chain

Today: `UseRootStorage` does `self.id.split("-", 1)[0]` to find the
"root" (project name). Convention-based.

Replace with: walk `self.parent` until finding a `Project`. Use that
project's name.

```python
def _find_project_root(obj, graph) -> str:
    current = obj
    while True:
        parent = current.parent
        if parent is None:
            if isinstance(current, Project):
                return current.name
            raise ValueError(obj)
        parent_obj = parent if isinstance(parent, Object) else graph.get(parent, Object)
        if isinstance(parent_obj, Project):
            return parent_obj.name
        current = parent_obj
```

Less brittle (no string-splitting convention), more explicit. Should be
the last move once Container/Project usage is stable.

## Stop conditions

Same as `notes/003_bt_migration.md`: stop and check in if more than two
files outside the current step need editing, more than three currently-
passing tests break for non-trivial reasons, or a circular import emerges
that needs lazy resolution.

## Recommended order

1. **G (auto-id)** — independent, low risk, demonstrates conventions before
   any further restructuring.
2. **A (handler bodies)** — unblocks Step B/C end-to-end testing.
3. **B (free factories)** — needs A so tests can exercise.
4. **C (bind rewrites)** — needs A + B.
5. **D (migrate framework callers)** — needs C.
6. **E (delete bt legacy)** — last, once D is verified.
7. **F (delete Collection)** — independent of bt; can interleave with D/E.
8. **H (storage routing)** — last, after Container is shrunk.

Total estimated effort: ~2,000 LOC modified, ~1,500 LOC deleted. Each step
A–H is one focused session.
