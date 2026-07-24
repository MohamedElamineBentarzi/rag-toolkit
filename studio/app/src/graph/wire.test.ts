import { describe, it, expect } from "vitest";
import { autoWireNewNode } from "./wire";
import { computeProblems } from "./validate";
import { pruneCorpusEdges, wiredSpaces } from "./corpus";
import { endpointNodes, endpointEdges, mergeEdges } from "./endpoints";
import { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge } from "./model";
import type { Manifest } from "../manifest/types";
import manifestJson from "../../public/blocks.json";

const mIndex = new ManifestIndex(manifestJson as unknown as Manifest);

let n = 0;
function make(kind: string, name: string): BlockNode {
  const params: Record<string, unknown> = {};
  for (const p of mIndex.component(kind, name)?.params ?? []) params[p.name] = p.default;
  const node: BlockNode = { id: `${kind}-${++n}`, type: "block", position: { x: 0, y: 0 }, data: { kind, name, params } };
  if (mIndex.component(kind, name)?.encoder) node.data.encoder = { name: "hashing", params: {} };
  return node;
}

// A tiny stand-in for store.addNode: create the node, spawn the Corpus when
// something needs it, auto-wire, then re-tie the endpoints — exactly the store's
// sequence, so this exercises the real interactive build.
function add(
  state: { nodes: BlockNode[]; edges: BlockEdge[] },
  kind: string,
  name: string,
): { nodes: BlockNode[]; edges: BlockEdge[] } {
  const node = make(kind, name);
  const next = [...state.nodes, node];
  if ((kind === "representations" || kind === "vector_store") && !next.some((x) => x.data.kind === "corpus")) {
    next.push({ id: `corpus-${++n}`, type: "corpus", position: { x: 0, y: 0 }, data: { kind: "corpus", name: "Corpus", params: {}, synthetic: true } });
  }
  const wired = autoWireNewNode(node, next, state.edges, mIndex);
  let edges = state.edges.filter((e) => !wired.remove.includes(e.id));
  edges = mergeEdges(edges, wired.add);
  edges = mergeEdges(edges, endpointEdges(next, mIndex));
  return { nodes: next, edges: pruneCorpusEdges(next, edges) };
}

describe("autoWireNewNode (drop -> auto-connect)", () => {
  it("builds a complete, error-free pipeline as blocks are dropped in order", () => {
    let s = { nodes: endpointNodes(), edges: [] as BlockEdge[] };
    for (const [k, nm] of [
      ["parser", "plaintext"],
      ["chunker", "fixed"],
      ["representations", "dense"],
      ["retriever", "index"],
      ["generator", "extractive"],
    ] as const) {
      s = add(s, k, nm);
    }
    const errors = computeProblems(s.nodes, s.edges, mIndex).filter((p) => p.level === "error");
    expect(errors).toEqual([]);
    // The index retriever picked up the dense index by wiring, not a param.
    const retr = s.nodes.find((x) => x.data.kind === "retriever")!;
    expect(wiredSpaces(retr.id, s.edges)).toEqual(["dense"]);
  });

  it("hands every index to a hybrid retriever, and re-threads a spliced refiner", () => {
    let s = { nodes: endpointNodes(), edges: [] as BlockEdge[] };
    s = add(s, "parser", "plaintext");
    s = add(s, "chunker", "fixed");
    s = add(s, "representations", "dense");
    s = add(s, "representations", "lexical");
    s = add(s, "retriever", "hybrid");
    s = add(s, "generator", "extractive");
    const retr = s.nodes.find((x) => x.data.kind === "retriever")!;
    expect(wiredSpaces(retr.id, s.edges).sort()).toEqual(["dense", "lexical"]);

    // Dropping a refiner splices between retriever and generator (no dangling).
    s = add(s, "refine", "score-threshold");
    const errors = computeProblems(s.nodes, s.edges, mIndex).filter((p) => p.level === "error");
    expect(errors).toEqual([]);
    const refine = s.nodes.find((x) => x.data.kind === "refine")!;
    const gen = s.nodes.find((x) => x.data.kind === "generator")!;
    // retriever -> refine -> generator
    expect(s.edges.some((e) => e.source === retr.id && e.target === refine.id)).toBe(true);
    expect(s.edges.some((e) => e.source === refine.id && e.target === gen.id)).toBe(true);
    expect(s.edges.some((e) => e.source === retr.id && e.target === gen.id)).toBe(false);
  });

  it("wires a new index into an existing hybrid retriever", () => {
    let s = { nodes: endpointNodes(), edges: [] as BlockEdge[] };
    s = add(s, "chunker", "fixed");
    s = add(s, "representations", "dense");
    s = add(s, "retriever", "hybrid");
    s = add(s, "representations", "lexical"); // added after the retriever
    const retr = s.nodes.find((x) => x.data.kind === "retriever")!;
    expect(wiredSpaces(retr.id, s.edges).sort()).toEqual(["dense", "lexical"]);
  });
});
