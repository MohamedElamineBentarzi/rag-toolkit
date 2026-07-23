import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge } from "../graph/model";

// Compile the canvas graph into the flat `{stage: {name, params}}` spec that
// load_spec()/PipelineBuilder consume. The graph is richer than the spec (it has
// positions, an explicit index node, edges); compiling throws all that away and
// keeps only what the spec encodes: which component per stage, in what order for
// chains, with which non-default params.

const SINGLE_STAGES = [
  "parser", "chunker", "embedder", "sparse", "lexical", "retriever", "generator",
  "store", "blob_store",
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

/** One `{name, params}` entry: params minus secrets and minus anything left at
 *  its default (an omitted param already means "the default" to the builder). */
function entry(node: BlockNode, mIndex: ManifestIndex): { name: string; params: Record<string, unknown> } {
  const comp = mIndex.component(node.data.kind, node.data.name);
  const params: Record<string, unknown> = {};
  for (const p of comp?.params ?? []) {
    if (p.secret) continue; // §7.4: credentials never enter a spec
    const value = node.data.params[p.name];
    if (value === undefined || value === null) continue;
    if (JSON.stringify(value) === JSON.stringify(p.default)) continue; // omit defaults
    params[p.name] = value;
  }
  return { name: node.data.name, params };
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
