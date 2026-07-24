import type { BlockNode, BlockEdge, BlockData } from "./model";
import type { ComponentSpec } from "../manifest/types";
import { handleId, parseHandle } from "./ports";

// The Corpus is the pivot of the graph: representations fan into it, and it
// exposes one *index* output per representation it holds. A retriever selects
// which representation(s) it reads by which of those index ports it's wired to —
// there is no representation pick-list any more, the wiring *is* the choice.
// These helpers are the one place that maps between representation nodes, their
// spaces, and the "out:Corpus#<space>" index ports.

/** A representation's space — its `space` param, defaulting to its name. */
export function repSpace(node: { data: BlockData }): string {
  const s = node.data.params?.space;
  return (typeof s === "string" && s) || node.data.name;
}

/** The distinct representation spaces feeding a corpus, in wiring order — one
 *  index output port per entry. */
export function corpusSpaces(
  corpusId: string,
  nodes: BlockNode[],
  edges: BlockEdge[],
): string[] {
  const spaces: string[] = [];
  for (const e of edges) {
    if (e.target !== corpusId || e.targetHandle !== "in:Representation") continue;
    const rep = nodes.find((n) => n.id === e.source && n.data.kind === "representations");
    if (!rep) continue;
    const s = repSpace(rep);
    if (!spaces.includes(s)) spaces.push(s);
  }
  return spaces;
}

/** The source handle id of a corpus's index port for `space`. */
export function corpusOutHandle(space: string): string {
  return handleId("out", "Corpus", space);
}

/** The single corpus node, if present. */
export function findCorpus(nodes: BlockNode[]): BlockNode | undefined {
  return nodes.find((n) => n.data.kind === "corpus");
}

// How many representations a retriever reads:
//   single — exactly one (an `index` retriever, param `representation`)
//   multi  — one or more (a `hybrid` retriever, param `representations`)
//   pool   — any number, offered to its nested sub-retrievers (a composite)
export type CorpusArity = "single" | "multi" | "pool";

export function corpusArity(comp: ComponentSpec | undefined): CorpusArity {
  if (comp?.params.some((p) => p.name === "representation")) return "single";
  if (comp?.params.some((p) => p.name === "representations")) return "multi";
  return "pool";
}

/** The representation spaces wired into a retriever (its Corpus index edges). */
export function wiredSpaces(nodeId: string, edges: BlockEdge[]): string[] {
  const spaces: string[] = [];
  for (const e of edges) {
    if (e.target !== nodeId || e.targetHandle !== "in:Corpus") continue;
    const p = parseHandle(e.sourceHandle);
    if (p?.space && !spaces.includes(p.space)) spaces.push(p.space);
  }
  return spaces;
}

/** Drop corpus index edges whose space no longer exists (a representation was
 *  removed, or its space renamed) — keeps the graph honest after any change. */
export function pruneCorpusEdges(nodes: BlockNode[], edges: BlockEdge[]): BlockEdge[] {
  const corpus = findCorpus(nodes);
  if (!corpus) return edges;
  const live = new Set(corpusSpaces(corpus.id, nodes, edges));
  return edges.filter((e) => {
    if (e.source !== corpus.id) return true;
    const p = parseHandle(e.sourceHandle);
    return !p?.space || live.has(p.space);
  });
}
