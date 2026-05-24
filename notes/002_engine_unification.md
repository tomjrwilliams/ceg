# Engine / Collections / Containers — unification design

## Goal

Collapse three near-overlapping registries (`Graph`, `Collection`, `Container`)
and four near-duplicate composition types (`Aggregate`, `Combination`,
`DailyAggregate`, `DailyCombination`) into a smaller, uniform surface. Replace
ad-hoc string references with a typed `Ref` family. Generate child ids
mechanically from parent + slot.

## End state

### `Ref` — typed reference family

```python
class Ref: ...                                 # abstract
class IdRef(Ref):     id: str                  # absolute graph id
class AliasRef(Ref):  name: str                # session-local alias  (was "o:foo")
class MemberRef(Ref): parent: Ref; field: str  # ref into another object's slot
                      slot: str | int          # (was container "obj_3" / "agg_2")
```

Replaces every stringly reference (`"o:foo"`, `"obj_3"`, `parent.extend_id`'d
suffixes, etc.). `Graph.get(ref)` dispatches on type.

Note: app-level `Data`/`Param` reference forms (pipeline outputs / UI params)
are out of scope for the engine — they stay in app code.

### `Graph` — single registry

```python
@dataclass(frozen=True)
class Graph:
    # persisted (encoded for multi-proc; consumed by sync)
    name: str | None
    objects:  frozendict[str, Object]
    depends:  frozendict[str, int]
    triggers: frozendict[str, frozendict[str, Dependency]]

    # session-local (not encoded; not consumed by sync)
    aliases: frozendict[str, Ref]            = frozendict()
    tags:    frozendict[str, frozenset[Ref]] = frozendict()
    info:    frozendict[str, frozendict]     = frozendict()

    # mutation — always returns (Graph, Ref)
    def with_obj(self, obj, *, alias=None, tag=(), info=None, force=False) -> (Graph, Ref): ...
    def with_nested(self, parent, field, child, *, index=None, key=None,
                    alias=None, tag=(), info=None) -> (Graph, Ref): ...
    def with_alias(self, ref, alias) -> Graph: ...
    def with_tag(self, ref, tag) -> Graph: ...
    def with_info(self, ref, info) -> Graph: ...

    # lookup
    def get(self, ref: Ref, t=None) -> Object: ...
    def get_id(self, id: str, t=None) -> Object: ...
    def get_alias(self, alias: str, t=None) -> Object: ...

    # query
    def where(self, *predicates, paths=None, type=None, tag=None,
              alias=None, **fields) -> Iterator[Object]: ...
```

`Collection` deleted. Aliases / tags / info live on the graph, session-local,
discarded on multi-proc serialisation.

### `with_nested` — uniform forward-reference primitive

Parent declares ref-bearing slots via regular dataclass fields with conventional
type annotations:

```python
members: tuple[Ref, ...]          = ()         # positional slot
outputs: frozendict[str, Ref]     = frozendict()  # named slot
```

`with_nested(parent, field, child, *, index|key, ...)`:

1. introspects parent's field annotation → tuple vs dict shape
2. derives child id: `parent.id + "-" + field + "-" + (index|key)`
3. stamps child id and parent ref onto child
4. inserts child id into parent's slot field
5. writes both back to graph; re-runs `parent.depends()`

Identity is by reference: parent's slot holds id strings, not embedded Objects.
Member content updates don't propagate ObjectChange events to the parent.

No default-slot convention; no marker base class. Multiple ref fields per
object are fine — caller picks the field. Detection via the existing
`obj_type_hints` machinery (extended to recognise `tuple[Ref, ...]` and
`frozendict[str, Ref]` field annotations).

Nesting: ref collections are flat at each level. Need grouping → make the group
its own Object referenced flatly.

### Query DSL

Three forms, all flowing through one compiled-predicate path:

```python
# lambda (escape hatch)
graph.where(lambda o: o.tags.symbol.startswith("V"))

# F-expression (preferred for typed sites)
graph.where(F.tags.symbol == "SPY", F.tags.adjust.in_(("pct", "raw")))

# string dot-paths (dynamic field names)
graph.where({"tags.symbol": "SPY"})

# session metadata
graph.where(tag="rx_containers", type=Container)
graph.where(alias=("cv-EQ-SPY",))
```

Shallow field kwargs (`graph.where(symbol="SPY")`) look at `obj.symbol`;
nested paths require dot-path or `F`.

### Composition unification

`Aggregate` + `Combination` + `DailyAggregate` + `DailyCombination` → single
`DailyComposition`:

```python
@dataclass(frozen=True)
class DailyComposition(DailyChunked, DependsParentValue):
    parent: Object | str                        # parent namespace
    method: str
    refs:   tuple[Ref, ...]           = ()      # positional inputs
    named:  frozendict[str, Ref]      = frozendict()  # named inputs
    params: frozendict[str, TParamValue] | None = None
    peers:  tuple[Ref, ...]           = ()
```

`DailyAggregateDeriv` + `DailyCombinationDeriv` → single `Output` with empty
`depends()`. Handlers that fan out emit `(field, child)` pairs; engine routes
each through `with_nested` instead of forcing them in.

### Namespace generalisation

`Container` and `Project` collapse into a `Namespace` base — an Object with
optional slot fields and optional `storage` config. `UseRootStorage` walks
`parent` chain to find the nearest `storage` instead of splitting id on `-`.

### Auto-id

```python
@dataclass(frozen=True)
class Object:
    @classmethod
    def role(cls, **fields) -> str:         # default: kebab(cls.__name__)
    @classmethod
    def encode_params(cls, **fields) -> str: # default: ""
    @classmethod
    def auto_id(cls, parent, **fields) -> str:
        return (parent.extend_id if parent else _identity)(
            cls.role(**fields)
            + (("-" + cls.encode_params(**fields)) if cls.encode_params(**fields) else "")
        )
```

Identity-derived types (`EqDailyEtf`, `DailyEMA`, ...) override
`role`/`encode_params` and call `auto_id` from their `.new()`. Slot-derived
types (children added via `with_nested`) get their id from
`parent.id + "-" + field + "-" + slot` — `auto_id` not used.

### Bind / Apply

`Bind` and `Apply` operate on `Graph`. The configuration knobs that exist
solely to thread Container's `(self', idx)` shape (`mutates_fields`,
`into_fields`, `unpack_fields`, `arg_order`) collapse. A single `into(ref)`
that, when present, calls `with_nested(ref, field, result)` instead of
`with_obj(result)`. Function signature drives positional/keyword; `Ref`
kwargs are auto-resolved against the graph.

## Sequencing

Each step lands independently, tests pass at every commit.

1. **Bug-fix PR** — duplicate `HandleNewDailyHSum`, broken
   `Aggregate.obj/agg`, unbound `i` in `Combination.with_objects`,
   `container`-undefined in `combs.py:215`, `("parent,")` typo in
   `Bind.unpack_bound_kw`, `snap.merge_updates` missing `self`, `kw[v] = v` in
   `set_storage_defaults`, REDIS branch typo in `dependency_key_val`.
2. **`Ref` family** alongside strings; both work.
3. **`Output` unifies `*Deriv`** types.
4. **`DailyComposition` unifies aggs/combs**; migrate `aggs.py` / `combs.py`.
5. **`Namespace` base** generalises `Project`/`Container`; storage routing via
   parent chain.
6. **Auto-id** — `role` / `encode_params` / `auto_id`; migrate `.new()`
   factories incrementally.
7. **`Graph` absorbs `Collection`** — aliases / tags / info, `with_nested_*`,
   `where` / `F`. Migrate `sign.py`. Delete `Collection`.

Steps 1–4 can all land before any of 5–7.

## Migration notes

- Existing stored ids change format (slot-derived ids now include field name).
  One-time rebuild from config required at cutover. Not transparent.
- `version: int` field is presently a no-op (not in id, doesn't bust cache).
  Either include in `encode_params` or remove during step 6.
- `alias_parents` extended to recurse into `tuple[Ref, ...]` and
  `frozendict[str, Ref]` (currently misses `frozendict`).

## Deferred / out of scope

- Per-object storage flavour mixins (`DailyChunked`, `UseRootDataFrameStorage`)
  — orthogonal to namespace.
- Multi-process `EncodedContext` / `Context.encode/decode` — still TODO; build
  on top of unified Graph once stable.
- `Daily.Snap` `first`/`last` date convention — orthogonal.
- App-level `Data` / `Param` references — UI concerns.
