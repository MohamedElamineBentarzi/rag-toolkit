import { describe, it, expect } from "vitest";
import type { Connection } from "@xyflow/react";
import { isValidConnection, computeProblems } from "./validate";
import { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge } from "./model";
import type { Manifest } from "../manifest/types";
import manifestJson from "../../public/blocks.json";

const mIndex = new ManifestIndex(manifestJson as unknown as Manifest);

function node(id: string, kind: string, name: string): BlockNode {
  return { id, type: "block", position: { x: 0, y: 0 }, data: { kind, name, params: {} } };
}
function conn(source: string, sourceType: string, target: string, targetType: string): Connection {
  return { source, target, sourceHandle: `out:${sourceType}`, targetHandle: `in:${targetType}` };
}

describe("isValidConnection (the type rule)", () => {
  const nodes = [node("c", "chunker", "fixed"), node("e", "representations", "dense"), node("p", "parser", "docling")];

  it("allows matching contract types", () => {
    expect(isValidConnection(conn("c", "Chunk[]", "e", "Chunk[]"), nodes, [])).toBe(true);
  });

  it("refuses mismatched types", () => {
    expect(isValidConnection(conn("p", "Document", "e", "Chunk[]"), nodes, [])).toBe(false);
  });

  it("refuses a self-loop", () => {
    expect(isValidConnection(conn("c", "Chunk[]", "c", "Chunk[]"), nodes, [])).toBe(false);
  });

  it("refuses a second edge into a single input", () => {
    const edges: BlockEdge[] = [
      { id: "x", source: "c", target: "e", sourceHandle: "out:Chunk[]", targetHandle: "in:Chunk[]" },
    ];
    expect(isValidConnection(conn("c", "Chunk[]", "e", "Chunk[]"), nodes, edges)).toBe(false);
  });

  it("lets many representations fan into the corpus", () => {
    const withCorpus = [...nodes, node("cx", "corpus", "Corpus"), node("e2", "representations", "lexical")];
    const edges: BlockEdge[] = [
      { id: "x", source: "e", target: "cx", sourceHandle: "out:Representation", targetHandle: "in:Representation" },
    ];
    expect(
      isValidConnection(conn("e2", "Representation", "cx", "Representation"), withCorpus, edges),
    ).toBe(true);
  });
});

describe("computeProblems (structural)", () => {
  const isError = (ps: ReturnType<typeof computeProblems>, re: RegExp) =>
    ps.some((p) => p.level === "error" && re.test(p.message));

  it("flags duplicate single-slot stages", () => {
    const nodes = [node("c1", "chunker", "fixed"), node("c2", "chunker", "fixed")];
    const ps = computeProblems(nodes, [], mIndex);
    expect(isError(ps, /2 chunker blocks/)).toBe(true);
  });

  it("flags a retriever with no Corpus wired in", () => {
    const nodes = [node("r", "retriever", "index")];
    const ps = computeProblems(nodes, [], mIndex);
    expect(isError(ps, /retriever:index needs its Corpus/)).toBe(true);
  });

  it("flags a representation missing its encoder", () => {
    const nodes = [node("e", "representations", "dense")]; // no encoder picked
    const ps = computeProblems(nodes, [], mIndex);
    expect(isError(ps, /needs embedder/)).toBe(true);
  });

  it("flags two representations that share a space", () => {
    const a = encoded("a", "representations", "dense"); // space -> "dense"
    const b = encoded("b", "representations", "dense"); // space -> "dense" too
    const ps = computeProblems([a, b], [], mIndex);
    expect(isError(ps, /share the name "dense"/)).toBe(true);
  });

  it("is happy when two dense reps have distinct spaces", () => {
    const a = encoded("a", "representations", "dense");
    const b = encoded("b", "representations", "dense");
    b.data.params = { space: "dense-2" };
    const ps = computeProblems([a, b], [], mIndex);
    expect(ps.some((p) => /share the name/.test(p.message))).toBe(false);
  });

  it("flags a corpus that no retriever reads", () => {
    // dense rep -> corpus, but nothing consumes the corpus index.
    const nodes = [
      encoded("e", "representations", "dense"),
      node("cx", "corpus", "Corpus"),
    ];
    const edges: BlockEdge[] = [
      { id: "x", source: "e", target: "cx", sourceHandle: "out:Representation", targetHandle: "in:Representation" },
    ];
    const ps = computeProblems(nodes, edges, mIndex);
    expect(isError(ps, /Corpus isn't feeding a retriever/)).toBe(true);
  });

  it("allows a retrieval-only pipeline (no generator) to be complete", () => {
    // chunker -> dense -> corpus#dense -> index retriever <- Query.  No generator.
    const nodes = [
      node("c", "chunker", "fixed"),
      encoded("e", "representations", "dense"),
      node("cx", "corpus", "Corpus"),
      node("r", "retriever", "index"),
    ];
    const edges: BlockEdge[] = [
      { id: "1", source: "c", target: "e", sourceHandle: "out:Chunk[]", targetHandle: "in:Chunk[]" },
      { id: "2", source: "e", target: "cx", sourceHandle: "out:Representation", targetHandle: "in:Representation" },
      { id: "3", source: "cx", target: "r", sourceHandle: "out:Corpus#dense", targetHandle: "in:Corpus" },
      { id: "4", source: "ep-query", target: "r", sourceHandle: "out:Query", targetHandle: "in:Query" },
      // chunker needs a Document source in a real graph; here we only assert the
      // retriever's dangling ScoredChunk[] output is *not* treated as an error.
    ];
    const ps = computeProblems(nodes, edges, mIndex);
    expect(ps.some((p) => /output isn't connected/.test(p.message))).toBe(false);
  });
});

// A representation with its encoder already picked (so the "needs encoder" rule
// doesn't fire in tests that aren't about it).
function encoded(id: string, kind: string, name: string): BlockNode {
  const n = node(id, kind, name);
  n.data.encoder = { name: "hashing", params: {} };
  return n;
}
