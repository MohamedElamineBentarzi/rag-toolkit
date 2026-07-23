import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge, SubRetriever } from "../graph/model";

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
  "parser", "chunker", "embedder", "sparse", "lexical", "retriever", "generator",
  "vector_store", "blob_store",
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
    const node = nodes.find((n) => n.data.kind === kind);
    if (node) spec[kind] = entry(node, mIndex);
  }
  for (const kind of CHAIN_STAGES) {
    const chain = orderChain(nodes.filter((n) => n.data.kind === kind), edges);
    if (chain.length) spec[kind] = chain.map((n) => entry(n, mIndex));
  }
  return spec;
}

/** One `{name, params}` entry, plus the nested sub-retrievers of a composite. */
function entry(node: BlockNode, mIndex: ManifestIndex): SpecEntry {
  const comp = mIndex.component(node.data.kind, node.data.name);
  const out: SpecEntry = {
    name: node.data.name,
    params: cleanParams(node.data.kind, node.data.name, node.data.params, mIndex),
  };
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
