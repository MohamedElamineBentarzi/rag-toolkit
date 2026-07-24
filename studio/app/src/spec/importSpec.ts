import dagre from "@dagrejs/dagre";
import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge, SubRetriever } from "../graph/model";
import { handleId } from "../graph/ports";
import { corpusArity, repSpace } from "../graph/corpus";
import { repStoreSlot } from "../graph/store-slot";

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
  const connect = (a: BlockNode, type: string, b: BlockNode, space?: string) =>
    edges.push({
      id: `e-${a.id}-${b.id}-${type}${space ? `#${space}` : ""}`,
      source: a.id,
      target: b.id,
      sourceHandle: handleId("out", type, space),
      targetHandle: handleId("in", type),
      style: { stroke: mIndex.typeColor(type) },
    });

  const single = (kind: string): BlockNode | null => {
    const e = spec[kind] as { name: string; params?: Record<string, unknown> } | undefined;
    return e ? mk(kind, e.name, e.params ?? {}) : null;
  };
  // A representation, whose wrapped encoder rides nested in its params under the
  // manifest-named param (`embedder`/`index`/…) — lifted back into node.data.encoder.
  const mkRep = (e: { name: string; params?: Record<string, unknown> }): BlockNode => {
    const comp = mIndex.component("representations", e.name);
    const encParam = comp?.encoder?.param;
    const flat: Record<string, unknown> = {};
    for (const p of comp?.params ?? []) flat[p.name] = p.default;
    for (const [k, v] of Object.entries(e.params ?? {})) if (k !== encParam) flat[k] = v;
    const node = mk("representations", e.name, {});
    node.data.params = flat;
    if (encParam && comp?.encoder) {
      const enc = e.params?.[encParam] as { name: string; params?: Record<string, unknown> } | undefined;
      if (enc) {
        const encParams: Record<string, unknown> = {};
        for (const p of mIndex.component(comp.encoder.kind, enc.name)?.params ?? []) encParams[p.name] = p.default;
        Object.assign(encParams, enc.params ?? {});
        node.data.encoder = { name: enc.name, params: encParams };
      }
    }
    return node;
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
  const repEntries = (spec["representations"] as { name: string; params?: Record<string, unknown> }[]) ?? [];
  const reps = repEntries.map(mkRep);
  const store = single("vector_store");
  const retriever = buildRetriever(spec["retriever"] as RetrieverEntry | undefined);
  const refiners = chain("refine");
  const generator = single("generator");
  const blob = single("blob_store");

  const needsCorpus = reps.length > 0 || !!store;
  const corpus = needsCorpus ? mk("corpus", "Corpus", {}) : null;
  if (corpus) {
    corpus.type = "corpus";
    corpus.data.synthetic = true;
  }

  // Wire the backbone by type.
  if (blob && parser) connect(blob, "BlobStore", parser);
  if (parser && chunker) connect(parser, "Document", chunker);
  let chunkTail = chunker;
  for (const e of enrichers) {
    if (chunkTail) connect(chunkTail, "Chunk[]", e);
    chunkTail = e;
  }
  if (chunkTail) for (const r of reps) connect(chunkTail, "Chunk[]", r);
  if (corpus) {
    for (const r of reps) connect(r, "Representation", corpus);
    if (store) connect(store, "VectorStore", corpus);
    // Wire the corpus indexes the retriever actually reads — one edge per space,
    // recovered from its representation params (base) or its sub-retrievers'
    // (composite). This is the inverse of compile's "wiring is the selection".
    if (retriever) {
      const available = reps.map(repSpace);
      for (const space of retrieverSpaces(retriever, mIndex, available))
        if (available.includes(space)) connect(corpus, "Corpus", retriever, space);
    }
  }
  let scoredTail = retriever;
  for (const r of refiners) {
    if (scoredTail) connect(scoredTail, "ScoredChunk[]", r);
    scoredTail = r;
  }
  if (scoredTail && generator) connect(scoredTail, "ScoredChunk[]", generator);

  // A self-managed rep whose encoder carries a nested store sub-spec: lift it
  // out into a blob_store block wired into the rep (the inverse of compile).
  for (const rep of reps) {
    const slot = repStoreSlot(rep, mIndex);
    const enc = rep.data.encoder;
    if (!slot || !enc) continue;
    const spec = enc.params[slot.param];
    if (spec && typeof spec === "object" && !Array.isArray(spec) && "name" in spec) {
      const s = spec as { name: string; params?: Record<string, unknown> };
      delete enc.params[slot.param];
      connect(mk("blob_store", s.name, s.params ?? {}), "BlobStore", rep);
    }
  }

  layout(nodes, edges);
  return { nodes, edges };
}

// Which corpus indexes a retriever reads, from its spec params:
//   single (index)  -> its one `representation`
//   multi  (hybrid) -> its `representations`, or all when omitted ("fuse all")
//   pool (composite)-> the union its sub-retrievers each name, or all
function retrieverSpaces(
  retriever: BlockNode,
  mIndex: ManifestIndex,
  available: string[],
): string[] {
  const arity = corpusArity(mIndex.component("retriever", retriever.data.name));
  if (arity === "single") {
    const r = retriever.data.params?.representation;
    return typeof r === "string" && r ? [r] : available.slice(0, 1);
  }
  if (arity === "multi") {
    const rs = retriever.data.params?.representations;
    return Array.isArray(rs) ? (rs as string[]) : available;
  }
  const subs = [retriever.data.inner, ...(retriever.data.retrievers ?? [])].filter(
    Boolean,
  ) as SubRetriever[];
  const used: string[] = [];
  for (const s of subs) {
    const r = s.params?.representation;
    if (typeof r === "string" && r && !used.includes(r)) used.push(r);
  }
  return used.length ? used : available;
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
