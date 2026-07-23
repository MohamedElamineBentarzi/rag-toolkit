import dagre from "@dagrejs/dagre";
import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge, SubRetriever } from "../graph/model";
import { handleId } from "../graph/ports";

interface RetrieverEntry {
  name: string;
  params?: Record<string, unknown>;
  inner?: RetrieverEntry;
  retrievers?: RetrieverEntry[];
}

// Rebuild a canvas graph from a flat spec — the inverse of compileSpec, closing
// the round trip with save_spec/load_spec. The spec has no positions or edges,
// so we reconstruct the pipeline wiring by contract type and auto-lay-it-out
// with dagre. Endpoints (Source, Query) stay unconnected, as on a fresh build.

const REPRESENTATIONS = ["embedder", "sparse", "lexical"];
const NODE_W = 180;
const NODE_H = 68;

let importCounter = 0;

export function importSpec(
  spec: Record<string, unknown>,
  mIndex: ManifestIndex,
): { nodes: BlockNode[]; edges: BlockEdge[] } {
  const nodes: BlockNode[] = [];
  const edges: BlockEdge[] = [];
  const mk = (kind: string, name: string, params: Record<string, unknown>): BlockNode => {
    const comp = mIndex.component(kind, name);
    const full: Record<string, unknown> = {};
    for (const p of comp?.params ?? []) full[p.name] = p.default;
    Object.assign(full, params);
    const node: BlockNode = {
      id: `imp-${kind}-${name}-${++importCounter}`,
      type: "block",
      position: { x: 0, y: 0 },
      data: { kind, name, params: full },
    };
    nodes.push(node);
    return node;
  };
  const connect = (a: BlockNode, type: string, b: BlockNode) =>
    edges.push({
      id: `e-${a.id}-${b.id}`,
      source: a.id,
      target: b.id,
      sourceHandle: handleId("out", type),
      targetHandle: handleId("in", type),
      style: { stroke: mIndex.typeColor(type) },
    });

  const single = (kind: string): BlockNode | null => {
    const e = spec[kind] as { name: string; params?: Record<string, unknown> } | undefined;
    return e ? mk(kind, e.name, e.params ?? {}) : null;
  };
  const chain = (kind: string): BlockNode[] => {
    const list = (spec[kind] as { name: string; params?: Record<string, unknown> }[]) ?? [];
    return list.map((e) => mk(kind, e.name, e.params ?? {}));
  };
  // A retriever, possibly composite: its nested sub-retrievers land in node.data
  // (configured in the inspector), not as separate graph nodes.
  const subFromSpec = (e: RetrieverEntry): SubRetriever => {
    const params: Record<string, unknown> = {};
    for (const p of mIndex.component("retriever", e.name)?.params ?? []) params[p.name] = p.default;
    Object.assign(params, e.params ?? {});
    return { name: e.name, params };
  };
  const buildRetriever = (e: RetrieverEntry | undefined): BlockNode | null => {
    if (!e) return null;
    const node = mk("retriever", e.name, e.params ?? {});
    if (e.inner) node.data.inner = subFromSpec(e.inner);
    if (e.retrievers) node.data.retrievers = e.retrievers.map(subFromSpec);
    return node;
  };

  // Build nodes.
  const parser = single("parser");
  const chunker = single("chunker");
  const enrichers = chain("enrich");
  const reps = REPRESENTATIONS.map((k) => single(k)).filter((n): n is BlockNode => !!n);
  const store = single("vector_store");
  const retriever = buildRetriever(spec["retriever"] as RetrieverEntry | undefined);
  const refiners = chain("refine");
  const generator = single("generator");
  const blob = single("blob_store");

  const retrieverTakesIndex = mIndex.component(
    retriever?.data.kind ?? "", retriever?.data.name ?? "",
  )?.takes_index;
  const needsIndex = reps.length > 0 || !!store || !!retrieverTakesIndex;
  const index = needsIndex ? mk("index", "ChunkIndex", {}) : null;
  if (index) index.data.synthetic = true;

  // Wire the backbone by type.
  if (blob && parser) connect(blob, "BlobStore", parser);
  if (parser && chunker) connect(parser, "Document", chunker);
  let chunkTail = chunker;
  for (const e of enrichers) {
    if (chunkTail) connect(chunkTail, "Chunk[]", e);
    chunkTail = e;
  }
  if (chunkTail) for (const r of reps) connect(chunkTail, "Chunk[]", r);
  if (index) {
    for (const r of reps) connect(r, "Representation", index);
    if (store) connect(store, "Store", index);
    if (retriever) connect(index, "Index", retriever);
  }
  let scoredTail = retriever;
  for (const r of refiners) {
    if (scoredTail) connect(scoredTail, "ScoredChunk[]", r);
    scoredTail = r;
  }
  if (scoredTail && generator) connect(scoredTail, "ScoredChunk[]", generator);

  layout(nodes, edges);
  return { nodes, edges };
}

// dagre left-to-right auto-layout so an imported spec reads like a pipeline.
function layout(nodes: BlockNode[], edges: BlockEdge[]): void {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 40, ranksep: 90 });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of edges) g.setEdge(e.source, e.target);
  dagre.layout(g);
  for (const n of nodes) {
    const p = g.node(n.id);
    n.position = { x: p.x - NODE_W / 2, y: p.y - NODE_H / 2 };
  }
}
