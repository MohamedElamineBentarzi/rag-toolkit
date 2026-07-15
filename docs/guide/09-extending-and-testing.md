# 09 · Add your own part

Everything in the toolkit is a swappable part, including the ones you write. A new
capability — a custom embedder, a company-specific reader, your own clean-up step
— is just a new class you register. You never edit the toolkit's files to add one.

## Five steps to a new part

1. Subclass the base for the stage you're filling (`Parser`, `Chunker`,
   `Embedder`, `Refiner`, `VectorStore`, …).
2. Give it a `name` (unique within that stage) and a `version`.
3. Optionally add a nested `Config` dataclass for settings.
4. Implement the one method that stage requires.
5. Add the `@registry.register` decorator.

Then it works anywhere that stage is used — by name or by import.

### Example: a custom embedder

```python
from rag_blocks import registry, Embedder

@registry.register
class ConstantEmbedder(Embedder):
    name = "constant"
    version = "0.1.0"

    @property
    def dimensions(self) -> int:
        return 8

    def embed_texts(self, texts):
        return [[1.0] * 8 for _ in texts]

    def embed_query(self, text):
        return [1.0] * 8

emb = registry.create("embedder", "constant")
```

### Example: a custom clean-up step (refiner)

A refiner takes the current candidate chunks and returns a reordered/trimmed list.
You don't have to trim to `k` — the pipeline does the final trim:

```python
from rag_blocks import registry, Refiner

@registry.register
class LengthRefiner(Refiner):
    """Prefer longer passages."""
    name = "length"
    version = "0.1.0"

    def refine(self, query, candidates, k):
        return sorted(candidates, key=lambda sc: len(sc.chunk.text), reverse=True)
```

### Example: a custom generator

The base handles the numbering and citation matching; you only write how the text
is produced. `packed.texts` and `packed.citations` are the chunks you were given:

```python
from rag_blocks import registry, Generator

@registry.register
class TemplateGenerator(Generator):
    name = "template"
    version = "0.1.0"

    def _complete(self, query, packed):
        if not packed.citations:
            return ("No context.", {})
        return (f"Based on the sources: {packed.texts[0]} [1]", {})
```

### Example: a custom write destination (sink)

To also send chunks somewhere else during indexing, you don't even need a base
class — anything with `add(chunks)` and `persist()` works. Here's a tiny index
that just collects "urgent" chunks:

```python
class KeywordAlertIndex:
    def __init__(self): self.hits = []
    def add(self, chunks):
        self.hits += [c for c in chunks if "urgent" in c.text.lower()]
    def persist(self): ...

rag = RagPipeline(chunk_index=index, extra_sinks=[KeywordAlertIndex()])
```

## Settings, secrets, and add-on dependencies

A few rules keep custom parts safe and cache-correct:

- **Settings.** Put configurable values in a nested `Config` dataclass. Passing an
  unknown setting fails immediately with a clear error, rather than being silently
  ignored.
- **Bump `version` when you change behavior.** The toolkit caches results by a
  fingerprint of a part's settings; changing the version is how old cached results
  get correctly thrown out.
- **Secrets.** Never fetch a secret inside your part. Accept an `api_key` (or
  similar) as a setting and fall back to the vendor's own environment variable.
  Any setting whose name contains `key`, `token`, `secret`, `password`, or
  `credential` is automatically hidden from logs and fingerprints — so rotating a
  key never breaks your cache and never leaks into a log.
- **Heavy dependencies.** If your part needs a big library, import it *inside* the
  method that uses it and raise a clear "install this add-on" message if it's
  missing — so people who don't use your part never have to install its
  dependency.

## Proving it works: contract tests

Every stage comes with a **contract test** that checks your part behaves the way
the rest of the toolkit expects. This is the important safety net: passing it
means your part will slot in and every other part can rely on it.

```python
from tests.contract_checks import assert_embedder_contract, assert_refiner_contract
assert_embedder_contract(ConstantEmbedder())
assert_refiner_contract(LengthRefiner())
```

What each one checks, in plain terms:

| Contract test | Checks that your part… |
|---|---|
| `assert_parser_contract` | reads pages in order, with sane page/offset info, the same way each time |
| `assert_chunker_contract` | produces gap-free pieces whose text matches the document, with pages filled |
| `assert_enricher_contract` | keeps the document link; any *added* chunks are marked synthetic with a parent-derived id |
| `assert_embedder_contract` | returns one equal-length vector per input, and nothing for empty input |
| `assert_vector_store_contract` | validates its schema, returns nearest-first, and doesn't duplicate on re-add |
| `assert_lexical_index_contract` | ranks term matches first and respects filters |
| `assert_index_contract` | can search every form it holds, and fails loudly on an unknown one |
| `assert_retriever_contract` | labels results, returns them best-first, and respects `k` |
| `assert_refiner_contract` | returns a reordered list drawn from what it was given |
| `assert_generator_contract` | cites only the chunks it was given, and handles empty context without crashing |
| `assert_blob_store_contract` | stores and returns bytes exactly, and reports missing keys quietly |

### One rule worth calling out: added chunks must be marked

If your enricher *generates* chunks (a summary, a Q&A pair) rather than just
tweaking existing ones, each generated chunk must be marked `synthetic`, given an
id derived from its parent (like `parent-id#aug0`), and carry a position index.
This keeps generated chunks from being mistaken for real, contiguous pieces of the
document — for instance, so the neighbor-expander doesn't splice a summary into the
middle of a page. The enricher contract test enforces this for you.

## Two kinds of tests

- **Hermetic tests** (the default) run with no network, no API keys, and give the
  same result every time. They're possible because every part can be tested
  through the same swap-points production uses — plug in a fake OCR engine, a fake
  model call, an in-memory store.
- **Integration tests** exercise the real vendors (Qdrant, a real embedding model,
  Claude, MinIO). They're marked separately and only run when you ask:

```bash
pytest                                   # fast, hermetic — the default
pytest -m integration tests/integration  # the real backends
```

Keep your own parts the same way: hermetic by default, real vendors behind the
integration marker. And run `ruff check` and `mypy` — both are expected to pass.

Next: **[10 · Recipes](10-recipes.md)** — complete setups you can copy.
