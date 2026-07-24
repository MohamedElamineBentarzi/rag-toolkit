// The shape of blocks.json, emitted by studio/tools/build_manifest.py.
// Keep this in sync with that script — it is the one contract between the
// Python introspection and the app.

export type ParamType = "str" | "int" | "float" | "bool" | "enum" | "json";

export interface ParamSpec {
  name: string;
  type: ParamType;
  default: unknown;
  required: boolean;
  choices?: (string | number)[];
  /** A credential (§7.4): shown as a password field, dropped on export. */
  secret?: boolean;
}

export interface ComponentSpec {
  /** The spec stage this fills: "chunker", "representations", "refine", …, or an
   *  encoder registry kind ("embedder"/"sparse_encoder"/"lexical_index") for a
   *  nested block. */
  kind: string;
  /** The implementation name, e.g. "fixed". */
  name: string;
  version: string;
  doc: string;
  /** Its constructor takes the live corpus (retrievers, NeighborExpander). */
  takes_index: boolean;
  /** False when it needs another component/callable a flat spec can't carry. */
  exportable: boolean;
  not_exportable_reason?: string;
  /** A composite retriever: its sub-retrievers nest under `inner` (one) or
   *  `retrievers` (a list). Configured in the inspector, not by graph edges. */
  composite?: "inner" | "retrievers";
  /** A representation wraps an encoder: which nested param holds it and which
   *  registry kind to offer (e.g. {param:"embedder", kind:"embedder"}). The
   *  encoder is picked in the inspector, not by a graph edge. */
  encoder?: { param: string; kind: string };
  /** A self-managed encoder (BM25) that keeps its own isolated persistence
   *  backend: which nested param takes a BlobStore. The Studio wires a BlobStore
   *  block into the representation that mounts this encoder. */
  store_slot?: { param: string; kind: string };
  /** An encoder block — offered inside a representation's inspector, not on the
   *  top-level palette. */
  nested?: boolean;
  /** Shapes the query with an LLM — wired from the pipeline's generator. */
  needs_llm?: boolean;
  params: ParamSpec[];
}

export interface StageSpec {
  kind: string;
  in: string[];
  out: string;
  chain?: boolean;
  single?: boolean;
  /** True for the synthetic Corpus node. */
  synthetic?: boolean;
  /** Input port types that accept many edges (the Corpus's Representation
   *  fan-in). */
  many_in?: string[];
}

export interface Manifest {
  types: Record<string, { color: string }>;
  stages: StageSpec[];
  components: ComponentSpec[];
}
