import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge, SubRetriever } from "../graph/model";
import { corpusArity, corpusSpaces, findCorpus, wiredSpaces } from "../graph/corpus";
import { repStoreSlot, wiredBlobStore } from "../graph/store-slot";

interface SpecEntry {
  name: string;
  params: Record<string, unknown>;
  inner?: SpecEntry;
  retrievers?: SpecEntry[];
}

// Compile the canvas graph into the flat `{stage: {name, params}}` spec that
// load_spec()/PipelineBuilder consume. The graph is richer than the spec (it has
// positions, an explicit index node, edges); compiling throws all that away and
// keeps only what the spec encodes: which component per stage, in what order for
// chains, with which non-default params.

const SINGLE_STAGES = [
  "parser", "chunker", "retriever", "generator", "vector_store", "blob_store",
];
const CHAIN_STAGES = ["enrich", "refine"];

export type Spec = Record<string, unknown>;

export function compileSpec(
  nodes: BlockNode[],
  edges: BlockEdge[],
  mIndex: ManifestIndex,
): Spec {
  const spec: Spec = {};

  for (const kind of SINGLE_STAGES) {
    // The top-level blob_store is the pipeline's parse-cache backend. A blob
    // store wired *only* into a representation is that rep's private BM25
    // persistence (nested in its encoder), not a pipeline stage — skip it here.
    const node =
      kind === "blob_store"
        ? topLevelBlobStore(nodes, edges)
        : nodes.find((n) => n.data.kind === kind);
    if (node) spec[kind] = entry(node, mIndex, nodes, edges);
  }
  for (const kind of CHAIN_STAGES) {
    const chain = orderChain(nodes.filter((n) => n.data.kind === kind), edges);
    if (chain.length) spec[kind] = chain.map((n) => entry(n, mIndex, nodes, edges));
  }
  // Representations: a generic list, each carrying its wrapped encoder as a
  // nested sub-spec (DR-0004) — no hardcoded embedder/sparse/lexical keys.
  const reps = nodes.filter((n) => n.data.kind === "representations");
  if (reps.length) spec.representations = reps.map((n) => repEntry(n, mIndex, nodes, edges));
  return spec;
}

/** The blob store that is the pipeline's own (parse-cache) backend: one wired to
 *  the parser, or — failing that — one that isn't a representation's private
 *  store. A blob store wired only into a rep is nested in that rep, not here. */
function topLevelBlobStore(
  nodes: BlockNode[],
  edges: BlockEdge[],
): BlockNode | undefined {
  const blobs = nodes.filter((n) => n.data.kind === "blob_store");
  const parser = nodes.find((n) => n.data.kind === "parser");
  const wiredToParser = (b: BlockNode) =>
    !!parser && edges.some((e) => e.source === b.id && e.target === parser.id);
  const isRepPrivate = (b: BlockNode) =>
    edges.some(
      (e) =>
        e.source === b.id &&
        e.targetHandle === "in:BlobStore" &&
        nodes.some((n) => n.id === e.target && n.data.kind === "representations"),
    );
  return blobs.find(wiredToParser) ?? blobs.find((b) => !isRepPrivate(b));
}

/** A representation entry: its flat params (e.g. `space`) plus the wrapped
 *  encoder nested under the param the manifest names (`embedder`/`index`/…). A
 *  self-managed encoder (BM25) also nests the BlobStore wired into the rep, so
 *  it owns its own persistence (the deliberate asymmetry). */
function repEntry(
  node: BlockNode,
  mIndex: ManifestIndex,
  nodes: BlockNode[],
  edges: BlockEdge[],
): SpecEntry {
  const comp = mIndex.component(node.data.kind, node.data.name);
  const params = cleanParams(node.data.kind, node.data.name, node.data.params, mIndex);
  if (comp?.encoder && node.data.encoder) {
    const enc = node.data.encoder;
    const encParams = cleanParams(comp.encoder.kind, enc.name, enc.params, mIndex);
    const slot = repStoreSlot(node, mIndex);
    if (slot) {
      const blob = wiredBlobStore(node.id, nodes, edges);
      if (blob) {
        encParams[slot.param] = {
          name: blob.data.name,
          params: cleanParams("blob_store", blob.data.name, blob.data.params, mIndex),
        };
      }
    }
    params[comp.encoder.param] = { name: enc.name, params: encParams };
  }
  return { name: node.data.name, params };
}

/** One `{name, params}` entry, plus the nested sub-retrievers of a composite.
 *  A base retriever's representation choice is read off its wired Corpus index
 *  edges, not its params — the wiring is the selection (DR-0004 studio). */
function entry(
  node: BlockNode,
  mIndex: ManifestIndex,
  nodes: BlockNode[],
  edges: BlockEdge[],
): SpecEntry {
  const comp = mIndex.component(node.data.kind, node.data.name);
  const params = cleanParams(node.data.kind, node.data.name, node.data.params, mIndex);

  if (node.data.kind === "retriever") {
    delete params.representation; // wired, not typed
    delete params.representations;
    const arity = corpusArity(comp);
    const spaces = wiredSpaces(node.id, edges);
    if (arity === "single") {
      if (spaces.length) params.representation = spaces[0];
    } else if (arity === "multi" && spaces.length) {
      // Omit when it's every index — that's the builder's default ("fuse all").
      const corpus = findCorpus(nodes);
      const all = corpus ? corpusSpaces(corpus.id, nodes, edges) : [];
      const isAll = spaces.length === all.length && spaces.every((s) => all.includes(s));
      if (!isAll) params.representations = spaces;
    }
  }

  const out: SpecEntry = { name: node.data.name, params };
  if (comp?.composite === "inner" && node.data.inner) {
    out.inner = subEntry(node.data.inner, mIndex);
  } else if (comp?.composite === "retrievers" && node.data.retrievers?.length) {
    out.retrievers = node.data.retrievers.map((r) => subEntry(r, mIndex));
  }
  return out;
}

function subEntry(sub: SubRetriever, mIndex: ManifestIndex): SpecEntry {
  return { name: sub.name, params: cleanParams("retriever", sub.name, sub.params, mIndex) };
}

/** Params minus secrets and minus anything left at its default (an omitted param
 *  already means "the default" to the builder). */
function cleanParams(
  kind: string,
  name: string,
  raw: Record<string, unknown>,
  mIndex: ManifestIndex,
): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  for (const p of mIndex.component(kind, name)?.params ?? []) {
    if (p.secret) continue; // §7.4: credentials never enter a spec
    const value = raw[p.name];
    if (value === undefined || value === null) continue;
    if (JSON.stringify(value) === JSON.stringify(p.default)) continue; // omit defaults
    params[p.name] = value;
  }
  return params;
}

/** Order chain nodes along their edges: the head is the one with no incoming
 *  edge from another node of the same kind; then follow the line. */
function orderChain(chainNodes: BlockNode[], edges: BlockEdge[]): BlockNode[] {
  if (chainNodes.length <= 1) return chainNodes;
  const ids = new Set(chainNodes.map((n) => n.id));
  const nextOf = new Map<string, string>();
  const hasIncoming = new Set<string>();
  for (const e of edges) {
    if (ids.has(e.source) && ids.has(e.target)) {
      nextOf.set(e.source, e.target);
      hasIncoming.add(e.target);
    }
  }
  const order: BlockNode[] = [];
  const seen = new Set<string>();
  const byId = new Map(chainNodes.map((n) => [n.id, n]));
  // Start from every head (no in-kind predecessor) for determinism, then walk.
  const heads = chainNodes.filter((n) => !hasIncoming.has(n.id));
  for (const head of heads) {
    let cur: string | undefined = head.id;
    while (cur && !seen.has(cur)) {
      seen.add(cur);
      order.push(byId.get(cur)!);
      cur = nextOf.get(cur);
    }
  }
  for (const n of chainNodes) if (!seen.has(n.id)) order.push(n); // stragglers
  return order;
}
