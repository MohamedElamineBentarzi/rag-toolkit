# DR-0003 — The tuner: a Template Method, and no cache of its own

**Status:** accepted · **Milestone:** v0.8 (PR 3 of 4) · **Builds on:** DR-0002
(evaluators score data) · **Breaking:** no (new surface) · **Audience:**
maintainer + coding agents

---

## 0. TL;DR

> **`Tuner.run()` is a Template Method over one abstract primitive,
> `iter_candidates(space)`.** Which combinations, in what order — that is the
> only thing a tuning strategy decides. Build, index, retrieve, generate,
> score, time, log: once, in the base.
>
> **The tuner has no cache.** ARCHITECTURE §6.2's key formula is already
> materialized by the blob parse cache and `CachingEmbedder`. The tuner's
> contribution to §6.2's outcome is **enumeration order**.
>
> **`index_ms` / `query_ms` are split**, because total latency partly measures
> which trial ran first.

## 1. The Template Method, a third time

`Parser.parse()` wraps `iter_pages()`. `Chunker.chunk()` wraps `iter_spans()`.
`Tuner.run()` wraps `iter_candidates()`. The same shape for the same reason:
the bookkeeping is identical for every strategy and getting it wrong is silent.

What lives in `run()`, once: two-phase screening, cost collection, trial
identity, error isolation, logging, ranking. What lives in a strategy:

```python
class GridTuner(Tuner):
    def iter_candidates(self, space):
        return space.expand()
```

That is the whole of `grid`. `random` is a dozen lines. Neither can get
two-phase screening wrong, because neither implements it.

The alternative — `run()` abstract, each tuner driving its own loop — was
rejected on the same grounds as DR-0002 §2.1: a rule that must be re-typed per
implementation is a rule that will be broken by the third implementation, and
"only judge the top 5" is a rule with a bill attached.

**Order is part of the strategy, not a detail.** `iter_candidates` returns an
*ordered* iterator because adjacency decides cache warmth (§3). That is why the
primitive is `iter_candidates` and not `candidates() -> set`.

## 2. No cache of its own

ARCHITECTURE §6.2 specifies:

```
cache_key(stage N) = sha256(dataset/source hashes + fingerprint(1..N))
```

That formula is **already implemented**, twice, and shipped before this
milestone existed:

| §6.2 stage | Where it already lives | Key |
|---|---|---|
| parse | `IndexingPipeline._parse` / `_load_parsed` | `parsed/{content_hash}/{parser_fingerprint}.md` |
| embed | `CachingEmbedder` | text × embedder fingerprint |

Building a `StageCache` keyed on (content hash × fingerprint) would be a second
implementation of §7.2's layout, and the two would drift the first time either
changed. So the tuner adds **no cache**. What it adds is **order**:
`SearchSpace.expand()` enumerates with the earliest stage varying slowest, so
candidates sharing a parse/embed prefix run back to back and inherit a warm
cache instead of thrashing it.

This is a refinement of §6.2's *mechanism*, not its *outcome* — the outcome
("24 combos ⇒ 1 parse") is what PR 4's benchmark must confirm honestly, with
real numbers, per rule 2 of the roadmap. Early evidence from PR 2 (two trials,
parse 29 ms → 1.8 ms, `cache_hits` all true on the second) says the bet is
sound; a number in a release note still has to earn it.

**Consequence — the ordering is load-bearing.** `STAGE_KINDS` declares stages in
pipeline order and `dimensions()` returns that order, deliberately *not*
alphabetical: sorted alphabetically, `parser` falls after `generator`, the most
expensive stage would vary fastest, and a grid would re-parse the corpus on
nearly every trial. That is a one-word change away at all times, which is why it
has a test (`test_dimensions_are_in_pipeline_order_not_alphabetical`).

## 3. `index_ms` / `query_ms`: total latency lies

Found by running PR 2's stack, not by thinking about it: two trials, the second
15× "cheaper" — because it ran second and hit a warm parse cache. The caching
that makes tuning affordable also makes `latency_ms` partly a measure of
**running order**.

The split quarantines it, and it works because the confound has a boundary:

- **`index_ms`** (parse, store_raw, store_parsed, chunk, enrich) — one-time per
  corpus, and cache-confounded across a run. Read it beside `cache_hits`.
- **`query_ms`** (retrieve, refine, generate) — untouched by the parse/embed
  caches, paid per question forever, and what a waiting user actually feels.

So `Leaderboard.to_table` defaults to `query_ms`. That default is load-bearing:
ranking on `latency_ms` inside a tuning run partly ranks by luck.

Rejected: *subtracting* cached stages from cost (invents a number nobody
measured), and *cold-starting every trial* (throws away the caching that makes
tuning possible, to make one column prettier).

## 4. Two phases, one code path

§7.3 requires screening all candidates on IR metrics and judging only the top-N.
The implementation detail worth recording: **phase 2 re-runs a finalist from its
spec** rather than keeping phase 1's pipelines alive.

- Keeping 24 live `ChunkIndex` objects to save a handful of cache-warm re-runs
  breaks the rule that memory never scales with the workload (AGENTS.md §2.3).
- Re-running proves config-as-data: if a spec cannot reconstruct a pipeline, the
  spec was never reproducible, and the trial log was a lie. The rebuild is the
  assertion.
- A finalist's trial is therefore **one complete run**, not two halves stitched
  together — `_run_one(generate=True)` is the same code path as phase 1 with one
  flag flipped, so phase 2 cannot drift from phase 1's notion of a trial.

Also settled here:

- **A failed candidate is a result.** It is recorded with its error and the run
  continues; one invalid combination must not take an overnight grid's other 23
  results with it. Only `RagBlocksError` is caught — a `KeyboardInterrupt` or a
  bug in the tuner must still stop the world.
- **Nothing scored on `screen_by` ⇒ no finalists ⇒ no judge.** An unlabeled
  dataset earns no ranking; judging an arbitrary five would spend money to
  rank noise (DR-0002 §4's honest-absence rule, applied to a phase).
- **A trial's identity is what ran, not what was asked for.** `trial_id =
  sha256(resolved describes)[:16]`, so two spellings of the same pipeline are
  one trial and a component `version` bump is a new one. Only a candidate that
  failed *before* it was built falls back to hashing its search spec.

## 5. `SearchSpace` is data; `PipelineBuilder` is wiring

Neither is a `Component`. A search space has no behavior to swap (DR-0002 §7
made the same call about `EvalDataset`); a builder is glue, like the pipelines
it builds. Neither gets a `kind`, a registry slot, or a fingerprint. What
identifies a trial is the components' fingerprints — never the glue.

**The list/tuple rule.** In `choice(...)`, a **list is a grid axis** and a
**tuple is one value**: `chunk_chars=[512, 1024]` is two trials,
`k_values=(1, 5, 10)` is one config. The distinction has to live somewhere,
because parameters whose value is genuinely a sequence are common (`k_values`,
`weights`) and "expand every list" would make those untunable and silently
wrong. This matches §6.1's notation and the codebase's existing habit of tuples
for literal sequences.

**Why `PipelineBuilder` exists at all** — the one new abstraction in v0.8 with
no precedent, so the argument is stated rather than assumed. `registry.create`
turns data into components, which is enough for a chunker. It is not enough for
a pipeline: `ChunkIndex` is *"wired from live, stateful backends — never built
by `registry.create` alone"* (DR-0001 v2), and `IndexRetriever` enforces it —
build it by name and it raises *"must be built with index=, not by name
alone"*. Something must hold the live store, assemble the index, and inject it
into the retriever. Writing pipelines by hand, that something is you; for 24
combinations, it is this class.

Its rules:

- **A fresh store per trial** (`store_factory` is a factory, called per build).
  Sharing one would let trial 1's chunks answer trial 2's query, and every
  number after that is fiction.
- **A shared blob store**, deliberately. It is content-addressed by (hash ×
  fingerprint), so it cannot contaminate — and sharing it is the entire reason
  a grid parses once.
- **Stated limits.** It builds what `SearchSpace` can describe. It cannot build
  a `FusionRetriever` (composed of *other retrievers*; no sensible flat
  spelling), and says so rather than leaking a `TypeError`.
- **The seam is the callable, not the class.** The tuner depends on
  `PipelineFactory = Callable[[dict], RagPipeline]`; `PipelineBuilder` is its
  default, never a requirement (Dependency Inversion).

## 6. Rejected alternatives

- **`run()` abstract per tuner** — §1.
- **A `StageCache` component** — §2. It would re-implement §7.2's key layout.
- **`bayesian` / successive-halving tuners** — §7.3 says "later", and both need
  the trial log to have accumulated real runs before they can be tuned
  themselves. YAGNI; the seam is `iter_candidates` and it is already open.
- **A `SearchSpace` component / `EvalDataset` kind** — a taxonomy entry
  wrapping a list (DR-0002 §7).
- **Unseeded `RandomTuner`** — a tuning run that cannot be reproduced is not
  evidence. `seed=0` by default; pass `seed=None` to opt into irreproducibility.
- **Sampling with replacement** when `n_trials` exceeds the space — it would
  bill twice for the same pipeline. Return the whole (shuffled) space instead.

## 7. Consequences

- `RagPipeline.ask_with_context()` was added (and `ask()` now delegates to it):
  evaluation needs both halves of a run, and the alternative — hand-rolling
  `query()` + `generate()` — silently skips the "generate" trace event, so a
  trial under-reports the one stage that costs money. A caller should never
  have to reimplement a pipeline to observe it.
- `CostCollector` gained `index_ms`/`query_ms`; `Leaderboard.to_table` defaults
  to `query_ms`.
- PR 4 owes this DR a number: the §6.2 claim ("24 combos ⇒ 1 parse, 2 chunk
  runs, 4 embed runs") gets measured on the committed benchmark and reported
  honestly, including if prefix-ordering does not deliver it.
