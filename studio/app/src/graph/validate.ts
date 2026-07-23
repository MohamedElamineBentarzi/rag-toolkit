import type { Connection } from "@xyflow/react";
import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge, Problem } from "./model";
import { acceptsMany, parseHandle } from "./ports";

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

export function isValidConnection(
  connection: Connection,
  nodes: BlockNode[],
  edges: BlockEdge[],
): boolean {
  if (connection.source === connection.target) return false; // no self-loop

  const types = connectionType(connection);
  if (!types) return false;
  if (types.source !== types.target) return false; // the type rule

  const targetNode = nodes.find((n) => n.id === connection.target);
  const tgt = parseHandle(connection.targetHandle);
  if (!targetNode || !tgt) return false;

  // One edge per input handle, except the ChunkIndex's Representation fan-in.
  if (!acceptsMany(targetNode.data.kind, tgt)) {
    const taken = edges.some(
      (e) => e.target === connection.target && e.targetHandle === connection.targetHandle,
    );
    if (taken) return false;
  }
  // No duplicate of the exact same edge.
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
// Structural problems — what per-edge typing can't catch, recomputed on every
// change so feedback stays instant even for non-type mistakes.
// ---------------------------------------------------------------------------

export function computeProblems(
  nodes: BlockNode[],
  edges: BlockEdge[],
  mIndex: ManifestIndex,
): Problem[] {
  const problems: Problem[] = [];

  // Duplicate single-slot stages: two chunkers can't both be "the" chunker.
  const counts = new Map<string, number>();
  for (const n of nodes) {
    if (n.data.kind === "index") continue;
    const stage = mIndex.stage(n.data.kind);
    if (stage?.single) counts.set(n.data.kind, (counts.get(n.data.kind) ?? 0) + 1);
  }
  for (const [kind, n] of counts) {
    if (n > 1)
      problems.push({
        level: "error",
        message: `${n} ${kind} blocks — a pipeline has at most one. Remove the extras or make them a chain.`,
      });
  }

  // An index-backed block whose Index input isn't wired to a ChunkIndex node.
  for (const n of nodes) {
    const comp = mIndex.component(n.data.kind, n.data.name);
    if (!comp?.takes_index) continue;
    const wired = edges.some(
      (e) => e.target === n.id && e.targetHandle === "in:Index",
    );
    if (!wired)
      problems.push({
        level: "error",
        message: `${n.data.kind}:${n.data.name} needs the index — connect a ChunkIndex to its Index port.`,
      });
  }

  // Representations present but nothing collects them into an index.
  const reps = nodes.filter((n) =>
    ["embedder", "sparse", "lexical"].includes(n.data.kind),
  );
  const hasIndex = nodes.some((n) => n.data.kind === "index");
  if (reps.length > 0 && !hasIndex)
    problems.push({
      level: "warn",
      message: "Representation blocks aren't feeding a ChunkIndex node yet.",
    });

  if (hasCycle(nodes, edges))
    problems.push({ level: "error", message: "The graph has a cycle." });

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
