import type { Connection } from "@xyflow/react";
import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge, Problem } from "./model";
import { parseHandle, type Port } from "./ports";
import { corpusArity, corpusSpaces, findCorpus, repSpace } from "./corpus";

// ---------------------------------------------------------------------------
// Live connection validation — the headline feature. A connection is valid iff
// the source's output type equals the target's input type. That single rule,
// read off the typed handle ids, is what makes only compatible blocks connect.
// ---------------------------------------------------------------------------

export function connectionType(
  connection: Connection,
): { source: string; target: string } | null {
  const src = parseHandle(connection.sourceHandle);
  const tgt = parseHandle(connection.targetHandle);
  if (!src || !tgt || src.dir !== "out" || tgt.dir !== "in") return null;
  return { source: src.type, target: tgt.type };
}

/** Which target handles legitimately accept more than one edge: representations
 *  fan into the Corpus, and a hybrid/composite retriever reads many Corpus
 *  indexes. A single `index` retriever, and everything else, is one-in. */
function acceptsMany(
  targetNode: BlockNode,
  port: Port,
  mIndex: ManifestIndex | null,
): boolean {
  const kind = targetNode.data.kind;
  if (kind === "corpus" && port.type === "Representation") return true;
  if (kind === "retriever" && port.type === "Corpus") {
    return corpusArity(mIndex?.component("retriever", targetNode.data.name)) !== "single";
  }
  return false;
}

export function isValidConnection(
  connection: Connection,
  nodes: BlockNode[],
  edges: BlockEdge[],
  mIndex: ManifestIndex | null = null,
): boolean {
  if (connection.source === connection.target) return false; // no self-loop

  const types = connectionType(connection);
  if (!types) return false;
  if (types.source !== types.target) return false; // the type rule

  const targetNode = nodes.find((n) => n.id === connection.target);
  const tgt = parseHandle(connection.targetHandle);
  if (!targetNode || !tgt) return false;

  // One edge per input handle, except the fan-in ports above.
  if (!acceptsMany(targetNode, tgt, mIndex)) {
    const taken = edges.some(
      (e) => e.target === connection.target && e.targetHandle === connection.targetHandle,
    );
    if (taken) return false;
  }
  // No duplicate of the exact same edge (same source port -> same target port).
  return !edges.some(
    (e) =>
      e.source === connection.source &&
      e.target === connection.target &&
      e.sourceHandle === connection.sourceHandle &&
      e.targetHandle === connection.targetHandle,
  );
}

/** Can a still-dragging connection from `pendingSourceType` legally land on an
 *  input handle of `portType`? Drives the green/dim handle highlight. */
export function portIsCompatible(
  pendingSourceType: string | null,
  portType: string,
): boolean {
  return pendingSourceType != null && pendingSourceType === portType;
}

// ---------------------------------------------------------------------------
// Structural problems — completeness, not just per-edge typing. A pipeline is
// "valid" only when it is actually *connected*: every block's required inputs
// are wired and its output is consumed, all the way to a real sink (a generator
// answering, or — retrieval-only — a retriever). Recomputed on every change.
// ---------------------------------------------------------------------------

// Infrastructure inputs that are optional — a parser needs no blob store, a
// corpus needs no vector store. Everything else on a block's input list is a
// required connection.
const OPTIONAL_INPUT_TYPES = new Set(["BlobStore", "VectorStore"]);

export function computeProblems(
  nodes: BlockNode[],
  edges: BlockEdge[],
  mIndex: ManifestIndex,
): Problem[] {
  const problems: Problem[] = [];
  const err = (message: string) => problems.push({ level: "error", message });
  const warn = (message: string) => problems.push({ level: "warn", message });

  const blocks = nodes.filter((n) => n.data.kind !== "endpoint" && n.data.kind !== "corpus");
  const has = (kind: string) => nodes.some((n) => n.data.kind === kind);
  const hasInto = (target: string, type: string) =>
    edges.some((e) => e.target === target && e.targetHandle === `in:${type}`);
  const outConsumed = (source: string, type: string) =>
    edges.some((e) => e.source === source && e.sourceHandle === `out:${type}`);

  // Duplicate single-slot stages: two chunkers can't both be "the" chunker.
  const counts = new Map<string, number>();
  for (const n of blocks) {
    const stage = mIndex.stage(n.data.kind);
    if (stage?.single) counts.set(n.data.kind, (counts.get(n.data.kind) ?? 0) + 1);
  }
  for (const [kind, n] of counts) {
    if (n > 1)
      err(`${n} ${kind} blocks — a pipeline has at most one. Remove the extras or make them a chain.`);
  }

  // Every block's required inputs must be wired (endpoints supply Source/Query
  // automatically; BlobStore/Store are optional).
  for (const n of blocks) {
    const stage = mIndex.stage(n.data.kind);
    for (const type of stage?.in ?? []) {
      if (OPTIONAL_INPUT_TYPES.has(type)) continue;
      if (!hasInto(n.id, type)) err(`${n.data.kind}:${n.data.name} needs its ${type} input connected.`);
    }
  }

  // A representation missing the encoder it wraps (dense needs an embedder, …).
  for (const n of blocks) {
    if (n.data.kind !== "representations") continue;
    const comp = mIndex.component(n.data.kind, n.data.name);
    if (comp?.encoder && !n.data.encoder)
      err(`${n.data.name} needs ${comp.encoder.kind} — pick one in the inspector.`);
  }

  // Two representations can't share a space — the corpus addresses each by it,
  // and its output ports are keyed by it (so a clash silently hides one).
  const spaceCounts = new Map<string, number>();
  for (const n of blocks) {
    if (n.data.kind !== "representations") continue;
    const s = repSpace(n);
    spaceCounts.set(s, (spaceCounts.get(s) ?? 0) + 1);
  }
  for (const [s, count] of spaceCounts) {
    if (count > 1)
      err(`Two representations share the name "${s}" — rename one (its "space") so each is distinct.`);
  }

  // The sink of the pipeline: a generator answers; without one, the retriever is
  // the terminal (retrieval-only). Its dangling ScoredChunk[] is allowed then.
  const hasGenerator = has("generator");

  // Every block's output must feed something, except the allowed terminal.
  for (const n of blocks) {
    const stage = mIndex.stage(n.data.kind);
    const out = stage?.out;
    if (!out) continue;
    if (outConsumed(n.id, out)) continue;
    if (n.data.kind === "vector_store" || n.data.kind === "blob_store") {
      warn(`${n.data.kind}:${n.data.name} isn't connected to anything — it won't be used.`);
    } else if (out === "ScoredChunk[]" && !hasGenerator) {
      // retrieval-only terminal: the retriever/last refiner is the end.
    } else {
      err(`${n.data.kind}:${n.data.name} output isn't connected to anything.`);
    }
  }

  // A corpus with no representations, or one that feeds no retriever.
  const corpus = findCorpus(nodes);
  if (corpus) {
    if (corpusSpaces(corpus.id, nodes, edges).length === 0)
      err("The Corpus has no representations — connect at least one.");
    else if (!edges.some((e) => e.source === corpus.id))
      err("The Corpus isn't feeding a retriever — wire one of its indexes into a retriever.");
  }

  // There has to be somewhere for the pipeline to end.
  if (blocks.length > 0 && !has("retriever") && !hasGenerator)
    err("A pipeline needs a retriever (and, to answer, a generator).");

  if (hasCycle(nodes, edges)) err("The graph has a cycle.");

  return problems;
}

function hasCycle(nodes: BlockNode[], edges: BlockEdge[]): boolean {
  const adj = new Map<string, string[]>();
  for (const n of nodes) adj.set(n.id, []);
  for (const e of edges) adj.get(e.source)?.push(e.target);

  const state = new Map<string, 0 | 1 | 2>(); // 0 unseen, 1 in-stack, 2 done
  const visit = (id: string): boolean => {
    if (state.get(id) === 1) return true;
    if (state.get(id) === 2) return false;
    state.set(id, 1);
    for (const next of adj.get(id) ?? []) if (visit(next)) return true;
    state.set(id, 2);
    return false;
  };
  return nodes.some((n) => visit(n.id));
}
