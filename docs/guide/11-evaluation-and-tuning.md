# 11 — Evaluation and tuning

*Previous: [10 — Recipes](10-recipes.md)*

You have a pipeline. Is it any good? Would a different chunk size be better?
Is the cross-encoder earning the 180 ms it costs you?

This chapter is about answering those questions with numbers instead of
opinions — and about the ways a number can lie to you, which is most of what
this subsystem's design is defending against.

## The shape of it

Four pieces, each usable alone:

| Piece | What it does |
|---|---|
| `Evaluator` | Scores outcomes. `ir` (recall/MRR/nDCG), `answer-match` (token overlap), `ragas` (LLM judge). |
| `SearchSpace` | Declares what to try, as data. |
| `Tuner` | Runs the combinations and ranks them. `grid`, `random`. |
| `TrialLog` / `Leaderboard` | The record, and the view over it. |

The rule that shapes everything: **an evaluator scores data the pipeline
already produced; it never runs the pipeline** (DR-0002). The run loop belongs
to the tuner. That is why `recall@k` is a pure function you can check by hand,
and why the RAGAS adapter is a translator with no orchestration in it.

## Scoring one pipeline

```python
import rag_blocks as rk
from rag_blocks.evaluation import EvalOutcome, EvalSample

rag = rk.RagPipeline(chunker=rk.MarkdownChunker())
rag.index(rk.Source.from_path("handbook.pdf"))

dataset = [
    EvalSample(
        question="How long is the internship?",
        relevant_doc_ids=(doc_id,),
        reference_answer="A minimum of sixteen weeks.",
    ),
]

outcomes = []
for sample in dataset:
    answer, context = rag.ask_with_context(sample.question, k=5)
    outcomes.append(
        EvalOutcome(sample=sample, retrieved=tuple(context), answer=answer)
    )

print(rk.RetrievalEvaluator(k_values=(1, 5)).evaluate(outcomes).metrics)
# {'mrr': 1.0, 'ndcg@1': 1.0, 'ndcg@5': 1.0, 'recall@1': 1.0, 'recall@5': 1.0}
```

Use `ask_with_context`, not `query()` + `generate()` by hand: it returns both
halves *and* keeps the pipeline's own tracing intact, so cost attribution
doesn't silently lose the generation step.

### Labeling: the choice that decides what you can tune

`EvalSample` takes ground truth at two granularities, and picking wrong is a
trap that never announces itself:

- **`relevant_chunk_ids`** — exact, and **chunker-locked**. `Chunk.id` is
  `{doc_id}:{index}`, so the same id means a *different passage* under a
  different chunker, and may not exist at all under a coarser one. Valid only
  while the chunker is fixed. The score stays plausible and becomes wrong.
- **`relevant_doc_ids`** — coarser, and survives any chunking, because a
  document's identity is its content hash rather than a cut decision.

**If you want to tune chunk size, label by document.** That is what the
committed benchmark does.

### Unlabeled samples are skipped, not zeroed

A sample an evaluator can't score joins no average and gets an empty
`per_sample` entry. `0.0` would read as *"the pipeline failed this question"*
when the truth is *"we never asked"*.

The consequence is yours to watch: **aggregates are means over the labeled
subset**. Two labeled rows out of thirty produce a confident-looking number
computed from two rows. `per_sample` is what reveals that, which is why it
exists.

## Searching a space

```python
from rag_blocks.evaluation import choice

space = rk.SearchSpace(
    chunker=[choice("fixed", chunk_chars=[400, 800], overlap_chars=[0, 100]),
             choice("markdown-aware")],
    embedder=[choice("hashing", dimensions=[128, 512])],
    refine=[[], [choice("score-threshold", min_score=0.1)]],
)
print(len(space))   # 20 combinations
```

**A list is a grid axis; a tuple is one value.** `chunk_chars=[400, 800]` means
two trials. `k_values=(1, 5, 10)` means *one* trial configured with that tuple.
Parameters whose value is genuinely a sequence are common, and expanding them
would make those untunable — so the distinction has to live somewhere, and this
is where.

Chain stages (`refine`, `enrich`) take a list of chains, and `[]` — no refiners
— is a real candidate. It is the baseline a cross-encoder has to beat to earn
its latency.

## Running it

```python
board = rk.GridTuner(screen_by="ndcg@5", finalists=3).run(
    space,
    dataset,
    sources,
    evaluators=[rk.RetrievalEvaluator(k_values=(1, 5)), rk.AnswerMatchEvaluator()],
    log=rk.TrialLog("runs/today.jsonl"),
    k=5,
)
print(board.to_table(by="ndcg@5"))
```

**Two phases, because the two evaluator families differ in cost by orders of
magnitude.** Every candidate is screened on the free retrieval metrics; only
the top `finalists` re-run with generation and face the judge. A 40-combination
grid with an LLM judge costs 40 screens and 5 verdicts, not 40 verdicts.

A candidate that raises is recorded with its error and the run continues — one
invalid combination must not take an overnight grid's other 39 results with it.
Check for them:

```python
failed = [t for t in board.trials if "error" in t.metadata]
```

### It reuses work automatically

A 24-combination grid does **not** parse your corpus 24 times. `SearchSpace`
enumerates with the earliest stage varying slowest, so trials sharing a
parse/embed prefix run back to back and hit the caches that already exist (the
blob parse cache, `CachingEmbedder`). Give the builder a blob store and it is
automatic:

```python
from rag_blocks.evaluation import PipelineBuilder

build = PipelineBuilder(blob_store=rk.LocalBlobStore()).build
board = rk.GridTuner().run(space, dataset, sources, evaluators=[...], build=build)
```

Measured on the committed benchmark: **12 combinations, 1 parse.**

## Reading the results honestly

### Rank on `query_ms`, not `latency_ms`

`Trial.cost` splits latency deliberately:

- **`index_ms`** — one-time per corpus, and **cache-confounded**: within a run
  the first trial pays for the parse and the rest inherit it, so this partly
  measures *which trial ran first*. Read it next to `cache_hits`.
- **`query_ms`** — untouched by the tuner's caching, paid per question forever,
  and what a waiting user actually feels.

`to_table` defaults to `query_ms` for exactly this reason.

### `api_usd` is never guessed

No price table ships with this library, and none will. Vendor prices drift, and
a plausible wrong number is worse than none — someone will make a real decision
on it. Supply prices and you get `api_usd`; supply none and **the key is
absent**, not `0.0`:

```python
board = tuner.run(..., prices={"input_tokens": 3.0 / 1e6, "output_tokens": 15.0 / 1e6})
```

### The marginals are the actual answer

A winner tells you what to ship. A marginal tells you **why**, and which parts
of your pipeline are carrying their weight:

```python
for m in board.marginal("chunker", by="ndcg@5"):
    print(m)
# chunker=markdown-aware: +0.0300 quality for -6.8 cost (n=4)
# chunker=fixed(chunk_chars=400,overlap_chars=0): -0.0282 quality for +4.1 cost (n=4)
```

Averaged over every other choice, that is what picking each option was worth —
quality *and* price. Read `n` before believing one: with `n=1`, "averaged over
everything else" is an average over one thing.

## The LLM judge

`AnswerMatchEvaluator` (token overlap) is free and deterministic but cannot see
that "Berlin" and "The capital is Berlin" agree, nor whether an answer is
*faithful* to its context. `RagasEvaluator` can, at a price:

```python
judge = rk.RagasEvaluator(
    llm=my_wrapped_llm,                 # a ragas/LangChain LLM object
    cache=rk.LocalBlobStore(),          # verdicts are memoized — see below
    judge_model="gpt-4o-2024-08-06",    # WHO judged: cache key + fingerprint
)
```

Install with `pip install 'rag-blocks[ragas]'`. Everything else in this
subsystem is dependency-free — the judge is the only part you opt into.

Three things worth knowing:

1. **The LLM is injected, not configured.** Building one from a model string
   would drag LangChain into a library whose core is stdlib-only. Credentials
   therefore live inside the vendor's own resolution — there is deliberately no
   `api_key` field here. Left to itself, ragas defaults to OpenAI and reads
   `OPENAI_API_KEY`.
2. **`judge_model` does real work.** The LLM arrives as an opaque object, so it
   is invisible to `describe()`. That label *is* the judge's identity: it is
   part of every cache key and of the fingerprint. **Change the judge, change
   the label** — or the cache will serve one model's verdicts as another's.
3. **Verdicts are cached** by (question, answer, judge-model). Re-running a
   leaderboard to reformat a table must not re-bill you. It is also what lets a
   judged evaluator be reproducible at all.

## The trial log

JSONL is the truth; SQLite is a disposable index over it.

```python
log = rk.TrialLog("runs/today.jsonl")
board = rk.Leaderboard(log.read())

log.query("SELECT trial_id FROM trials WHERE json_extract(metrics,'$.\"ndcg@5\"') > 0.9")
log.rebuild()   # lost or stale .db? regenerate it from the log
```

Every trial carries the full `describe()` of every stage, so a single line
reconstructs what ran — without the code that ran it. Secrets are already gone:
`describe()` redacts them at every depth, which is why a trial log is safe to
commit.

## The committed benchmark

`benchmarks/baseline/` is the regression baseline every later milestone reruns
with one component swapped:

    python benchmarks/baseline/run.py

Hermetic — no vendor, no key, no network. Read its `README.md` before quoting
anything from it: four documents and 28 questions make its numbers *indicative*,
not authoritative.

## Extending

A custom evaluator is a class and a decorator:

```python
@registry.register
class MyEvaluator(rk.Evaluator):
    name = "my-metric"
    stage = "retrieval"          # or "generation" — the tuner reads this

    def evaluate(self, outcomes):
        scores = [{"my_metric": ...} for o in outcomes]
        return rk.MetricReport(metrics=self._aggregate(scores),
                               per_sample=tuple(scores))
```

Call `assert_evaluator_contract` from its tests and you inherit every guarantee
the rest of the suite relies on. Use `self._aggregate` rather than your own
mean — it is what keeps "a skipped sample is not a zero" true everywhere.

A custom tuner implements one method:

```python
@registry.register
class MyTuner(rk.Tuner):
    name = "my-search"

    def iter_candidates(self, space):
        yield from ...   # WHICH combinations, in what order
```

Everything else — two-phase screening, cost, logging, error isolation — comes
from `Tuner.run`. Order matters: adjacent trials sharing a prefix inherit a warm
cache.

---

*See also: [DR-0002](../decisions/DR-0002-evaluator-contract.md) (why evaluators
score data), [DR-0003](../decisions/DR-0003-tuning-and-caching.md) (the Template
Method, and why the tuner has no cache).*
