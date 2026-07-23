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
  /** The spec stage this fills: "chunker", "embedder", "refine", … */
  kind: string;
  /** The implementation name, e.g. "fixed". */
  name: string;
  version: string;
  doc: string;
  /** Its constructor takes the live index (retrievers, NeighborExpander). */
  takes_index: boolean;
  /** False when it needs another component/callable a flat spec can't carry. */
  exportable: boolean;
  not_exportable_reason?: string;
  /** A composite retriever: its sub-retrievers nest under `inner` (one) or
   *  `retrievers` (a list). Configured in the inspector, not by graph edges. */
  composite?: "inner" | "retrievers";
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
  /** True for the one synthetic node (the ChunkIndex). */
  synthetic?: boolean;
}

export interface Manifest {
  types: Record<string, { color: string }>;
  stages: StageSpec[];
  components: ComponentSpec[];
}
