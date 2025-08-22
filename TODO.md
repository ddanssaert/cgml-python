# CGML v1.3 Alignment TODO

This document lists all outstanding work to align the Python engine with the CGML v1.3 specification. Each item states whether the engine must change, the spec should be amended, or both. Items are grouped by subsystem.

Legend
- [ ] = not started
- [~] = partial
- [x] = done

## 1) Loader, Schema, Modularity
- [x] Enforce cgml_version equals "1.3" (reject others; clear error).
- [x] Add meta.rng support to Meta model: { deterministic: bool, seed?: int }.
- [x] Implement !inherit directive with merge/override rules:
  - [x] Shallow object merge by key; child overrides parent.
  - [x] Array identity merge: rules by id; components.decks by name; components.zones by name; component_types entries by type/name; transitions by (from,to,id?).
  - [x] Deletion via disabled: true.
  - [x] Rule refinement via extends: <rule_id> (advisory validation).
- [ ] After resolving imports and inheritance, validate merged document against schema (cgml.schema.json).
- [x] Expand Components models:
  - [x] ZoneInstance: owner_scope (player|team|global).
  - [x] VariableInstance: scope (global|per_player|per_team), computed: bool, expression: Operand/Condition tree.
  - [x] ComponentTypeDef for zone_types: default_face (up|down), allows_reorder: bool.
- [ ] Improve validation errors: include file path, pointer to failing key.

## 2) Game State & Core Semantics
- [x] Add Card.face: "up" | "down" (default from zone default_face).
- [x] Carry type-level settings into Zone instances (ordering, visibility, default_face, allows_reorder).
- [x] Enforce zone ordering semantics: unordered, fifo, lifo, shuffled.
- [x] Enforce zone visibility semantics (owner/others/all: all | count_only | hidden | top_card_only).
- [x] Implement owner(<card>) helper semantics (derive from containing zone owner/owner_scope).

## 3) Path / Selector Language
- [x] Implement documented anchors: $currentPlayer, $activeState, $currentPhase, $turnOrder.
- [x] Add selector filters: [by_id=...], [current], [opponent], [team=...].
- [x] Add path functions: top(<zone|list>), bottom(<zone|list>), all(<zone>), count(<zone|list>), owner(<card>), rank(<card>).
- [x] Support ref placeholders in path strings: ref:<name>.
- [x] Keep $.shared_zones only if documented; otherwise remove or alias through $.zones (spec-align).
- [x] Remove support for non-$ dotted paths; require $-rooted selectors (align to spec).
- [x] Add support for group_by operator for aggregating/grouping lists.
- [x] Add support for distinct operator for deduplicating lists.

## 4) Expression / Operator Engine
- [x] Ensure top-level Condition nodes always resolve to boolean. Avoid returning raw values from max/min/sum/count in evaluate_condition.
- [x] Implement missing core operators per spec:
  - [x] any(list,predicate?), all(list,predicate?)
  - [x] len
  - [x] mul, div, mod, avg
  - [x] contains, in, exists
  - [~] canPerform: dry-run validation of an action spec
- [x] list constructor (list) – already partially supported; add schema docs if needed.
- [x] rank_value: use correct deck context (zone.of_deck or originating deck type); raise if ambiguous.
- [x] top/all/bottom as operands are supported; unify with path function equivalents.
- [x] Add support for distinct operator as per v1.3.
- [x] Add support for group_by operator as per v1.3.

## 5) Actions: Movement, Visibility, Search, Random, Structure
- Movement & dealing
  - [x] DEAL (runtime) to a target zone (single player), not round-robin.
  - [x] DEAL_ROUND_ROBIN (runtime & setup), respecting order and count.
  - [x] MOVE_ALL (runtime) – already present.
  - [ ] DEAL_ALL (runtime) – add if required by spec (present as standardized).
- Visibility & face state
  - [ ] REVEAL (respect visibility semantics; change effective visibility mask).
  - [ ] CONCEAL (reverse of REVEAL for given audience).
  - [ ] FLIP (toggle Card.face for target set).
  - [ ] PEEK (temporary visibility for specific players).
  - [ ] LOOK (non-mutating read access for a player).
- Ordering
  - [ ] REORDER (enforce allows_reorder and ordering rules; implement by: top|bottom|custom sequence).
- Search/Random
  - [ ] CHOOSE_RANDOM (use RNG policy; deterministic with seed when configured).
  - [ ] SEARCH_ZONE with filter; supply $.card context in predicate.
  - [ ] MILL (move N from deck to discard).
  - [ ] REVEAL_MATCHING (compute set; apply visibility change).
- Flow control / structure
  - [x] FOR_EACH_PLAYER: implement order parameter and simultaneous semantics; remove shorthand that fans-out the next action; only explicit do is allowed. [x] Shorthand removed in engine.
  - [x] PARALLEL: execute branches and join per wait: all; deterministic order.
  - [x] IF: implement with then/else blocks using engine condition evaluation.
  - [ ] FOR_EACH: Implement generic FOR_EACH action for arbitrary lists (binds ref:item for each iteration).
- Flow modifiers
  - [ ] SET_PHASE
  - [ ] SKIP_TURN
  - [ ] EXTRA_TURN
  - [ ] REVERSE_ORDER
  - [ ] INSERT_PHASE
  - [ ] REMOVE_PHASE
- Player input
  - [ ] REQUEST_INPUT with options (path/value) and multiselect; supply store_as.
  - [ ] REQUEST_INPUT: support dynamic filtering and options via operator expressions (distinct, filter, etc.).
- Action plumbing
  - [ ] Return values from actions and support store_as on each step.
  - [ ] Support ref usage in subsequent operands and path placeholders (ref:<name>).
  - [ ] on_failure policy per action (continue | abort | rollback). At minimum, support continue/abort; treat rollback as abort until transactions exist.

## 6) Rule System and Events
- Rule fields
  - [ ] timing: pre | post | replace handling relative to event default behavior.
  - [ ] priority: higher first, deterministic tie-breaker.
  - [ ] once_per: phase | turn | game; track counters.
  - [ ] enabled_when predicate to enable/disable rules.
- Triggers
  - [ ] on.state.enter.<State>, on.state.exit.<State>.
  - [ ] on.phase.<Phase> (exists) – ensure fired on phase entry.
  - [ ] on.turn.begin, on.turn.end.
  - [ ] on.draw, on.play, on.discard, on.move (instrument actions to emit these).
  - [ ] timing: replace to replace default event behavior (e.g., replace draw).
- Event context
  - [ ] Provide standard event context object (fields: $.card, $.from, $.to, $.player, etc.) for every event/trigger type.
  - [ ] Ensure dynamic referencing of event context in path selectors during rule evaluation.
## 7) Flow, Turn Order, Simultaneity, Transitions, Win Condition
- [ ] Apply flow.player_order: clockwise | counterclockwise | simultaneous across per-turn phases.
- [ ] Simultaneity semantics: batch operations; deterministic tie-breakers (seating order, then player id) when needed.
- [ ] Transitions: add id and priority; resolution order: document order, then priority, then tie-breaker.
- [ ] Evaluate transitions at defined checkpoints (end of phase/state) consistently.
- [ ] Win condition: implement flow.win_condition.evaluator to compute winner(s)/ranking/reason; use at GameOver.

## 8) Determinism & RNG
- [x] Honor meta.rng settings; seed global PRNG(s) from meta.rng.seed when deterministic.
- [x] Route all random choices (shuffle, choose_random, random policy selection) through seeded PRNG.
- [x] Remove nondeterministic random.choice in simulator for action selection when deterministic is enabled (or make selection deterministic).

## 9) Error Handling & Diagnostics
- [ ] Implement canPerform operator as dry-run check for actions.
- [ ] Structured error reporting for action failures: include rule id, action index, reason.
- [ ] Enforce schema-valid parameters pre-execution; surface clear messages.

## 10) Non-Standard Behavior (remove)
- [x] Action name matching: enforce exact-case per spec; remove case-insensitive lookup and remove SET_GAME_STATE alias support (accept only SET_STATE).
- [x] Remove FOR_EACH_PLAYER without do (next-action fan-out). Only explicit do is supported.
- [x] Remove support for non-$ dotted paths; require $-rooted selectors exclusively.
- [x] Remove auto rank comparison casting in comparisons; require explicit rank_value for rank comparisons.
- [x] Prohibit string-based rank comparisons; require explicit rank_value for rank comparisons (§12.5, §15, §19).
- [x] Remove $.shared_zones as a separate path root; provide $.zones only (maintain internal alias if needed but not exposed).

## 11) Spec Feedback / Proposed Updates
- [x] Document list operator explicitly in §12 since examples use it.
- [x] Define event context schema for on.draw/on.move/etc (fields and anchors like $.card).
- [x] Clarify DEAL vs DEAL_ROUND_ROBIN semantics with examples in setup/runtime.
- [x] Confirm spec disallows plain dotted paths and separate $.shared_zones root (engine will not support them).
- [x] Standardize generic FOR_EACH action over arbitrary lists with ref:item and FOR_EACH examples.
- [x] Standardize distinct and group_by operators in the operator vocabulary, with usage and engine requirements.
- [x] Document dynamic REQUEST_INPUT options and filtering via arbitrary operator expressions.
- [x] Clarify rank comparison: engines must require rank_value where relevant; no implicit coercion/casting.

## 12) Tests & Conformance Suite
- [ ] Build test fixtures for examples in README (examples/*.yaml) to validate operators/actions.
- [ ] Add integration tests for provided games: war.yml, high_card.yml, gofish.yml.
- [ ] Add determinism tests (fixed seed => identical outcomes).
- [ ] Visibility/face state tests ensuring enforcement for owners/others/all.
- [ ] Transition priority & once_per tests.

## 13) Documentation & Developer Experience
- [ ] Update README with engine-supported subset until full alignment, including any temporary extensions.
- [ ] Provide migration notes (e.g., DEAL behavior change; FOR_EACH shorthand removal; rank comparisons).
- [ ] Add developer docs for writing actions and emitting events.
- [ ] Document new FOR_EACH (arbitrary list) control structure and ref:item pattern in README and samples.
- [ ] Document distinct/group_by operators in Expression Language section.
- [ ] Add examples for dynamic REQUEST_INPUT options using operator expressions.

---

Notes
- Current gaps include: inheritance/merge, RNG determinism, visibility/face state, many actions, operators, path filters/anchors, rule timing/priority/once_per, events, win condition, and simultaneity semantics.
- Where marked as “spec feedback,” we’ll open issues/PRs to clarify the wording; engine will not ship non-spec conveniences.
