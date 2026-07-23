import type { BlockNode, BlockEdge } from "./model";
import type { ManifestIndex } from "../manifest/load";
import { handleId } from "./ports";

// The three pipeline endpoints — where data enters and leaves. They are always
// on the canvas (seeded on load, non-deletable) so the shape of a RAG pipeline
// reads at a glance: Source documents in, a Query in, an Answer out. They are
// not components and never export to a spec; they exist to anchor the flow.
export interface EndpointDef {
  id: string;
  endpoint: string;
  type: string; // its contract type: Source | Query | Answer
  dir: "in" | "out"; // out = it produces (Source/Query); in = it receives (Answer)
  label: string;
  position: { x: number; y: number };
}

export const ENDPOINTS: EndpointDef[] = [
  { id: "ep-source", endpoint: "source", type: "Source", dir: "out", label: "Source documents", position: { x: -320, y: 30 } },
  { id: "ep-query", endpoint: "query", type: "Query", dir: "out", label: "Query", position: { x: -320, y: 300 } },
  { id: "ep-answer", endpoint: "answer", type: "Answer", dir: "in", label: "Answer", position: { x: 1220, y: 150 } },
];

const ENDPOINT_IDS = new Set(ENDPOINTS.map((e) => e.id));

export function isEndpointId(id: string): boolean {
  return ENDPOINT_IDS.has(id);
}

/** Fresh endpoint nodes — seeded on load, on clear, and after import. */
export function endpointNodes(): BlockNode[] {
  return ENDPOINTS.map((e) => ({
    id: e.id,
    type: "endpoint",
    position: { ...e.position },
    deletable: false,
    data: {
      kind: "endpoint",
      name: e.endpoint,
      params: {},
      epType: e.type,
      epDir: e.dir,
      label: e.label,
    },
  }));
}

/** Edges tying the endpoints to the pipeline: Source -> parser, Query ->
 *  retriever, generator -> Answer, for whichever of those blocks exist. */
export function endpointEdges(nodes: BlockNode[], mIndex: ManifestIndex): BlockEdge[] {
  const find = (kind: string) => nodes.find((n) => n.data.kind === kind);
  const edges: BlockEdge[] = [];
  const link = (source: string, type: string, target: string) =>
    edges.push({
      id: `epw-${source}-${target}`,
      source,
      target,
      sourceHandle: handleId("out", type),
      targetHandle: handleId("in", type),
      style: { stroke: mIndex.typeColor(type) },
    });

  const parser = find("parser");
  if (parser) link("ep-source", "Source", parser.id);
  const retriever = find("retriever");
  if (retriever) link("ep-query", "Query", retriever.id);
  const generator = find("generator");
  if (generator) link(generator.id, "Answer", "ep-answer");
  return edges;
}

/** Add edges not already present (by endpoint identity), so re-wiring on every
 *  change stays idempotent. */
export function mergeEdges(base: BlockEdge[], add: BlockEdge[]): BlockEdge[] {
  const key = (e: BlockEdge) => `${e.source}|${e.target}|${e.sourceHandle}|${e.targetHandle}`;
  const seen = new Set(base.map(key));
  return [...base, ...add.filter((e) => !seen.has(key(e)))];
}
