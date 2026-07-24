# DR-0005 — Corpus owns all storage: self-managed persistence goes pure

**Status:** REJECTED — kept the asymmetry instead (see the decision note below) ·
**Amends:** would have amended DR-0004 D1/D3 + Gap 2 · **Audience:** maintainer +
coding agents · **Method:** closes an invariant DR-0004 stated but did not
actually hold.

> ## Decision (why this was NOT adopted)
>
> We kept the asymmetry deliberately. Making the Corpus own *all* storage (pure
> `snapshot`/`restore`, a Corpus-owned state store) would force every
> self-managed representation's persistence through one blob-shaped seam the
> Corpus controls — reintroducing, in a new place, exactly the rigidity the
> self-managed family exists to avoid. The chosen model instead:
>
> - **The Corpus** owns the shared `VectorStore` and does the efficient
>   single-pass multi-vector `upsert` for everything that *can* share a database.
> - **Vector-backed reps** stay pure mathematical projections (no storage).
> - **Self-managed reps (BM25)** hold their **own isolated `BlobStore`**, because
>   an inverted index cannot be merged into a vector payload — they do their own
>   side-write and own their own persistence.
>
> Keeping the `VectorStore` in the Corpus protects single-pass write efficiency
> for what can share; letting BM25 own its blob store lets it live in its
> necessary isolation. The one real gap DR-0004 left — that a self-managed rep's
> store couldn't be supplied from a flat spec or the Studio — was closed *without*
> this refactor: the builder now resolves a nested `store` sub-spec by type (rep →
> index → store, via `_build_component`), and the Studio exposes a `BlobStore`
> input port on a self-managed representation so you can wire a blob store block
> straight into BM25. See `tump_docs/STORAGE-DESIGN-QUESTION-FOR-EXPERT.md` for
> the options weighed; this is "Option B, made explicit."
>
> The rest of this document is the rejected proposal, kept for the record.

---

**(rejected proposal follows)** · **Would have amended:** DR-0004 D1/D3 + Gap 2
(the self-managed persistence path only) · **Would have kept:** everything else
in DR-0004 · **Breaking:** yes.

---

## 0. TL;DR

DR-0004's headline invariant is *"the `Corpus` is the single owner of storage;
a `Representation` holds no store and does no I/O."* That is literally true for
the **vector-backed** family (dense, sparse) — their vectors go through the
Corpus's one `VectorStore`. It is **not** true for the **self-managed** family:
`LexicalRepresentation.persist()` calls into a `BM25Index` that *holds its own
`BlobStore`* and writes itself (`storage/bm25_index.py:62,105`). Storage
ownership leaked out of the Corpus and into a representation's encoder — the
exact coupling D1 rejected, hiding one level down.

The visible symptoms:

- **Asymmetry** — dense/sparse persist *through the Corpus*; BM25 persists
  *directly to a blob store it secretly owns*. Two philosophies for one concept.
- **A dead dependency** — that `BlobStore` is a *component* dependency, so a flat
  spec / the Studio can't supply it. The builder never injects one
  (`evaluation/builder.py:248` creates the index with its flat params only), so
  `store` defaults to `None` and `persist()`/`load()` are **silent no-ops**:
  BM25 built from a spec forgets its index between runs.

The fix keeps the part that is good — the Corpus's **polymorphic loop that never
names BM25** — and moves only the *ownership*:

> **A representation never holds a store — self-managed ones included.** The
> self-managed family stops owning a persistence backend and instead exposes a
> **pure, I/O-free** pair, `snapshot() -> bytes | None` / `restore(bytes)`. The
> **`Corpus` becomes the single owner of *all* storage**: the `VectorStore` for
> vectors *and* a `BlobStore` (its "state store") for self-managed state. Its
> `persist`/`load` iterate representations polymorphically and do the I/O —
> exactly as `add`/`search` already do.

`Corpus.persist()` goes from delegating I/O to *doing* it, without ever learning
what BM25 is:

```python
# before (DR-0004): the rep owns the write — invariant violated one level down
for rep in self._self_managed:
    rep.persist()                       # -> BM25Index.persist() -> its own store

# after (DR-0005): the Corpus owns the write — the rep only serializes itself
for rep in self._self_managed:
    blob = rep.snapshot()               # pure: in-memory state -> bytes | None
    if blob is not None:
        self._state_store.put(self._state_key(rep), blob)
```

Same loop, same OCP (a new representation slots in untouched), now honoring the
invariant DR-0004 only claimed.

---

## 1. Root cause (where the invariant leaks, precisely)

DR-0004 §7 says, verbatim: *"no store reference on a representation, no `bind`
step."* Grep the shipped tree:

1. `storage/bm25_index.py:63` — `def __init__(self, store: Optional[BlobStore] = None, ...)`.
   A representation's encoder **holds a store**.
2. `storage/bm25_index.py:105-115` — `persist()`/`load()` call
   `self._store.put/get`. A representation's encoder **does I/O**.
3. `indexing/representation.py:266` — `LexicalRepresentation.persist()` forwards
   to it. The representation participates in that I/O.
4. `indexing/corpus.py:169-172` — `Corpus.persist()` does `store.persist()` for
   vectors but `rep.persist()` for self-managed — **two ownership models in one
   method**.
5. `evaluation/builder.py:248` — the nested-encoder resolver injects the encoder
   but *not* a store, so from a spec the dependency is unfillable and defaults to
   `None`. Persistence is structurally impossible through the supported path.

The defect is narrow and it is *ownership*, not mechanism. Self-managed **search**
legitimately lives on the representation (no store speaks BM25 — DR-0004 D2/Gap
2, preserved). Self-managed **storage** should not: "the Corpus is the single
storage owner" must be literally true, or it is not an invariant.

---

## 2. Decisions

### D1 — A representation holds no store. Self-managed ones expose pure state.

The self-managed family keeps its runtime state (BM25 must hold its in-memory
index to score — "self-managed" *means* it searches itself) but loses its
*persistence backend*. It gains a pure serialization pair on the base
`Representation`:

```python
# indexing/representation.py
class Representation(Component):
    # ... encode_corpus / encode_query (vector-backed) ...
    # ... ingest / search (self-managed) ...

    def snapshot(self) -> Optional[bytes]:
        """Serialize this representation's durable state to opaque bytes, or
        None when it has none to persist. PURE: no I/O, no store. Vector-backed
        reps return None (their state lives in the Corpus's VectorStore);
        self-managed reps serialize their in-memory index. The Corpus writes the
        bytes wherever it owns storage — the rep never learns where."""
        return None

    def restore(self, blob: bytes) -> None:
        """Rehydrate in-memory state from `snapshot()` bytes. PURE: no I/O.
        Default no-op (vector-backed have nothing to restore)."""
```

`persist()`/`load()` are **removed from `Representation`** — persistence is the
Corpus's job now, and a representation exposes only *what* to persist, never
*where*. The distinction the DR insists on: a self-managed rep is **stateful at
runtime** (it holds its index) yet **storeless for durability** (it owns no
backend, does no I/O). DR-0004's "no store, no I/O" was always about durability;
D1 makes it hold for *every* family.

### D2 — `LexicalRepresentation` / `BM25Index`: serialize, don't store.

```python
# indexing/representation.py
class LexicalRepresentation(Representation):   # self-managed
    def __init__(self, index: LexicalIndex, space: Optional[str] = None): ...
    def ingest(self, chunks):  self._index.add(chunks)
    def search(self, text, k, filters=None):  return self._index.search(text, k, filters)
    def snapshot(self):  return self._index.serialize()      # bytes
    def restore(self, blob):  self._index.deserialize(blob)

# storage/bm25_index.py — the store dependency is DELETED
class BM25Index(LexicalIndex):
    def __init__(self, config=None, **overrides):            # no `store=`
        ...
    def serialize(self) -> bytes:   ...   # was the body of persist()
    def deserialize(self, blob: bytes) -> None:  ...   # was the body of load()
```

`LexicalIndex` swaps its `persist()`/`load()` (store-touching) for
`serialize() -> bytes` / `deserialize(bytes)` (pure). The `namespace` config and
the `store=` constructor arg on `BM25Index` are gone — a namespace *key* is the
Corpus's concern now (D3), not the leaf's.

### D3 — `Corpus`: the single owner of *all* storage.

```python
# indexing/corpus.py
class Corpus(Component):
    def __init__(
        self,
        store: VectorStore,
        representations: Sequence[Representation],
        state_store: Optional[BlobStore] = None,     # NEW: self-managed state
    ) -> None: ...

    def persist(self) -> None:
        self._store.persist()                        # vectors (unchanged)
        if self._state_store is not None:
            for rep in self._self_managed:           # <-- same polymorphic loop
                blob = rep.snapshot()                # pure
                if blob is not None:
                    self._state_store.put(self._state_key(rep), blob)

    def load(self) -> None:
        """Rehydrate self-managed reps from the state store (no-op without one,
        or when a rep has no saved state)."""
        if self._state_store is None:
            return
        for rep in self._self_managed:
            key = self._state_key(rep)
            if self._state_store.exists(key):
                rep.restore(self._state_store.get(key))

    def _state_key(self, rep: Representation) -> str:
        return f"corpus/{rep.space}/state"           # keyed by space, not by kind
```

The Corpus now owns *both* backends it needs and does *every* write and read for
both families. `add`/`search`/`fetch` are untouched. The loop is the DR-0004 loop
— it never names BM25, so a new self-managed representation (a different term
index, a phonetic index) implements `snapshot`/`restore` and slots in with **zero
Corpus edits**. OCP, preserved and now consistent.

When `state_store is None`, self-managed reps run **in-memory / ephemeral** —
the same graceful degradation the code has today, but now explicit and owned by
the Corpus instead of hidden in an encoder's `None` default.

### D4 — Builder injects the state store into the Corpus (owner, not leaf).

The pipeline already threads one `BlobStore` (parse-cache + raw capture). The
builder hands that same store to the Corpus as its `state_store` — dependency
injection at the **owner**, matching how it already injects the `VectorStore`:

```python
# evaluation/builder.py (_build, sketch)
blob_store = self._create("blob_store", spec["blob_store"], None) if "blob_store" in spec else self.blob_store
corpus = Corpus(store, reps, state_store=blob_store) if reps else None
# blob_store still passed to RagPipeline for the parser; ONE backend, two key
# namespaces ("corpus/…/state" vs the parse-cache keys).
```

The nested-encoder resolver (`_build_representation`) is **simplified**: it no
longer has to (and never could) inject a store into the encoder. `BM25Index` is
built from its flat params alone, as a pure component should be.

### D5 — Studio: storage attaches to the Corpus, uniformly.

The `Corpus` node gains a second, **optional** infrastructure input —
`BlobStore` — beside its `VectorStore` input:

```
CORPUS_NODE in: ["Representation", "VectorStore", "BlobStore"]   # BlobStore optional
```

Now *all* storage wires to the one storage owner: `VectorStore → Corpus` (vectors)
and `BlobStore → Corpus` (self-managed state). A single `blob_store` block can
fan out to both the parser (cache) and the Corpus (state). There is **no
BlobStore → representation wire** and no per-encoder store field — the thing the
maintainer flagged as unclean. `BlobStore` and `VectorStore` are already optional
corpus inputs (a corpus with only vector-backed reps needs no state store), so
completeness stays as DR's studio rules define it.

### D6 — Identity unchanged; the state store is a sink, not identity.

`Corpus.fingerprint()` still folds the `VectorStore` fingerprint and `{space:
rep.fingerprint()}` (DR-0004 D8). The `state_store` is a *persistence sink* — like
the parser's blob cache, it is *where* durable state lives, not *what* the
component is — so it does **not** enter the fingerprint. A rep's identity is its
config + encoder, exactly as before; moving where its bytes are written changes
nothing about which cached encodings are valid. The DR-0004 acceptance tripwire
(D8) carries over verbatim, strengthened: **no store reference on any
representation, self-managed included** — if one reappears, the invariant leaked
again.

---

## 3. The honest caveat (name it, like DR-0004 named ColBERT)

This cleanly covers self-managed representations whose durable state is a
**serializable in-memory structure** — BM25 and every classic term index (tf/df
tables, an inverted list, a trie). Their state *is* bytes; the Corpus owning
those bytes is exactly right.

It does **not** cover a hypothetical self-managed representation backed by an
**external service it must hold a live connection to** (an Elasticsearch- or
Solr-backed rep). Such a thing owns *external infrastructure*, not a serializable
blob — its `snapshot()` would be `None` (nothing local to persist) yet it must
hold a client, re-opening the "rep holds a backend" question in a genuinely
different shape. That is **out of scope**: none ships, and forcing an ES
connection through a blob `snapshot` would be a worse abstraction than admitting
it is a different beast. If one is ever proposed, it gets its own DR (likely: a
distinct "externally-backed representation" contract, not a store on the base
class). Do not pre-build it (YAGNI).

So: DR-0005 makes **serializable** self-managed persistence clean and uniform; it
does not claim to have solved externally-backed representations.

---

## 4. Invariants (the audit)

1. **Single storage owner (now literal).** The `Corpus` owns the `VectorStore`
   *and* the state store and performs every durable read/write for both
   families. No representation holds a store or does persistence I/O (D1/D3).
2. **Polymorphic, OCP-preserving loop.** `persist`/`load` iterate representations
   without naming any kind; a new self-managed rep implements `snapshot`/`restore`
   and needs zero Corpus edits (D3).
3. **Self-managed search unchanged.** BM25 still holds its in-memory index and
   answers `search` itself; the retriever still cannot tell a term-scored space
   from a vector one (DR-0004 D2/Invariant 3, untouched).
4. **Single write pass, query/corpus parity, per-rep caching** — all DR-0004
   invariants carry over unchanged (D6).

---

## 5. Migration plan (ordered; each step lands green)

1. **`storage/lexical_index.py`** — replace the store-touching `persist()`/`load()`
   with pure `serialize() -> bytes` / `deserialize(bytes)` on the `LexicalIndex`
   interface (default: raise `NotImplementedError` or a sensible empty state).
2. **`storage/bm25_index.py`** — delete the `store=` param and the `namespace`
   config; rename `_serialize`/`_deserialize` to the public `serialize`/
   `deserialize`; drop `persist`/`load`/`_key`. Unit test: round-trip
   `serialize`→`deserialize` reproduces search results.
3. **`indexing/representation.py`** — add `snapshot`/`restore` to `Representation`
   (default `None`/no-op); implement them on `LexicalRepresentation` over the
   index's serialize/deserialize; remove `Representation.persist` and
   `LexicalRepresentation.persist`.
4. **`indexing/corpus.py`** — add `state_store: Optional[BlobStore] = None`;
   rewrite `persist` to write `rep.snapshot()`; add `load`; add `_state_key`.
   Contract test: build a corpus with a memory blob store, `add` + `persist`,
   construct a fresh corpus on the same blob store, `load`, assert BM25 search
   matches (persistence now works end-to-end from the public API).
5. **`evaluation/builder.py`** — pass `state_store=blob_store` into `Corpus`;
   simplify `_build_representation` (no store injection). Test: a spec with a
   `lexical` rep + a `blob_store` produces a corpus whose BM25 persists.
6. **`studio/manifest.py`** — add `"BlobStore"` to `CORPUS_NODE["in"]` (optional);
   Studio auto-wiring may fan an existing blob store to the Corpus. Update the
   studio validity note (BlobStore is an optional corpus input, like VectorStore).
7. **Docs/tests/examples** — update DR-0004 §7.6 (see §6 below), the storage
   guide, and any example that relied on `BM25Index(store=…)`. CHANGELOG
   (breaking): `BM25Index(store=…)` → the Corpus's `state_store=`;
   `Corpus(store, reps)` → `Corpus(store, reps, state_store=…)`.

---

## 6. AGENTS.md §7.6 addendum (paste-ready; amends the DR-0004 persistence note)

> **Storage ownership (DR-0005, amends DR-0004 Gap 2).** A `Representation` holds
> **no store and does no durable I/O — self-managed reps included.** Vector-backed
> reps encode into the Corpus's `VectorStore`; self-managed reps (BM25) hold their
> index in memory to *search*, and expose their durable state only as a **pure**
> `snapshot() -> bytes | None` / `restore(bytes)` — never a store, never I/O. The
> **`Corpus` is the single owner of ALL storage**: the `VectorStore` for vectors
> and an optional `state_store: BlobStore` for self-managed state. `Corpus.persist`
> writes `rep.snapshot()` and `Corpus.load` calls `rep.restore(...)`, iterating
> reps **polymorphically** (it never names BM25 — a new self-managed kind needs
> zero Corpus edits). The builder injects the pipeline's blob store into the
> Corpus as its `state_store`; in the Studio, `BlobStore` wires to the **Corpus**
> (never to a representation). Self-managed reps whose state is a serializable
> in-memory structure are fully covered; an externally-backed rep (holds a live ES
> connection) is out of scope and needs its own DR. Do not re-litigate: no store
> on any representation, no persistence I/O on any representation, storage attaches
> to the Corpus.

---

## 7. Non-goals (YAGNI fences)

No externally-backed self-managed representation (§3 caveat — a future DR if ever
needed); no separate "state store" *kind* distinct from `BlobStore` (a blob store
is exactly the put/get/exists interface serialized state wants); no async persist;
no per-space state store override in v1 (one Corpus state store; revisit only with
a real need); no change to vector-backed storage, search routing, the single-pass
write, or fingerprint identity (all DR-0004, untouched).

---

## 8. Vocabulary (additions to DR-0004 §8)

| Term | Meaning |
|---|---|
| **State store** | The `BlobStore` the `Corpus` owns for self-managed representation state. Optional; `None` ⇒ self-managed reps run ephemeral. |
| **`snapshot()` / `restore()`** | A representation's **pure**, I/O-free serialization pair. Vector-backed: `None`/no-op. Self-managed: its in-memory index ⇄ bytes. The Corpus does the reading/writing. |
| **Storeless for durability** | A representation may be stateful at *runtime* (BM25 holds its index to search) yet own **no** durable backend and do **no** persistence I/O — the property DR-0005 makes hold for every family. |
