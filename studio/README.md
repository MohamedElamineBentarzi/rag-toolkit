# rag-blocks Studio

A visual, n8n-style builder for rag-blocks pipelines. Drag blocks onto a canvas,
connect them by their **data contracts** (only compatible ports connect, with
instant feedback), configure each from an auto-generated form, and **export the
exact JSON `load_spec()` loads**.

It is **optional** and fully **static**: the app has no runtime Python. The
component palette comes from a manifest introspected out of the registry, so a
newly registered component appears with no hand-editing.

```
rag_blocks/
  cli.py                    # `rag-blocks studio` entry point
  studio/manifest.py        # build_manifest(): registry -> manifest dict (shipped)
  studio/server.py          # stdlib static server + live /blocks.json
studio/
  tools/build_manifest.py   # dev-only: writes app/public/blocks.json for `npm run dev`
  app/                      # the Vite + React + @xyflow/react source
```

## Run it — from a `pip install` (end users)

The app ships pre-built inside the wheel, with a stdlib launcher. No Node needed.

```bash
pip install rag-blocks
rag-blocks studio            # generates the manifest from YOUR install, opens the browser
# rag-blocks studio --port 8000 --no-browser
```

`/blocks.json` is generated fresh at launch from the installed registry — so it
reflects exactly the components you have, including your own registered plugins.

## Run it — from a source checkout (developing Studio)

```bash
python studio/tools/build_manifest.py    # writes studio/app/public/blocks.json
cd studio/app && npm install && npm run dev   # http://localhost:5173
```

`npm run manifest` (from `studio/app`) re-runs the manifest step.

## Releasing (bundling the app into the wheel)

The wheel includes the built app only when it's present at build time (hatch
`artifacts = ["rag_blocks/studio/_dist/**"]`). The release step:

```bash
cd studio/app && npm ci && npm run build      # -> studio/app/dist
cp -r studio/app/dist ../../rag_blocks/studio/_dist
cd ../.. && python -m build                    # wheel now bundles the app + CLI
```

`rag_blocks/studio/_dist/` is a gitignored build artifact — never committed.

## Use it

- **Drag** blocks from the left palette onto the canvas (or click them).
- New blocks **auto-connect** into the pipeline by contract type as you drop
  them, so nothing lands as an orphan — you can still rewire anything.
- **Connect** an output port to an input port. A connection only forms when the
  contract types match (`Document` → `Document`, `Chunk[]` → `Chunk[]`, …);
  incompatible ports are refused and dimmed while you drag.
- Representation blocks fan into the **Corpus** node, which exposes **one index
  output per representation**. Wire an index into a retriever to pick which
  representation it reads — an `index` retriever takes one, a `hybrid` takes as
  many as you give it. The wiring *is* the selection; there's no pick-list.
- A **self-managed** representation (BM25/`lexical`) keeps its *own* index, so it
  sprouts an optional **BlobStore** input port. Wire a `blob_store` block into it
  to persist that index (it nests inside the encoder on export); leave it
  unwired and it runs in-memory, rebuilt each run. The Corpus still owns the
  shared vector store for the dense/sparse reps — the asymmetry is intentional
  (see `docs/decisions/DR-0005`).
- **Configure** the selected block on the right; read its **Info** tab for the
  docstring and every parameter.
- **Export spec** downloads `pipeline.json`. Load it back in Python:

  ```python
  import rag_blocks as rk
  rag = rk.PipelineBuilder().build(rk.load_spec("pipeline.json"))
  ```

- **Import** re-opens a saved `pipeline.json` onto the canvas.

## What it deliberately doesn't do (v1)

- Composite retrievers (`fusion`/`hyde`/`multi-query`) configure their nested
  sub-retrievers in the inspector (the canvas keeps one clean retriever node);
  each sub-retriever picks its representation from the Corpus indexes wired in.
- No live "run this pipeline" preview and no server-side validation — those need
  an optional Python bridge that isn't part of the static v1.

Secrets never enter the exported spec (§7.4): credential fields are shown as
password inputs and dropped on export; the environment supplies them.
