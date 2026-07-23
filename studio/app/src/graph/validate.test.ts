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
  const nodes = [node("c", "chunker", "fixed"), node("e", "embedder", "hashing"), node("p", "parser", "docling")];

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

  it("lets many representations fan into the index", () => {
    const withIndex = [...nodes, node("idx", "index", "ChunkIndex"), node("e2", "embedder", "hashing")];
    const edges: BlockEdge[] = [
      { id: "x", source: "e", target: "idx", sourceHandle: "out:Representation", targetHandle: "in:Representation" },
    ];
    expect(
      isValidConnection(conn("e2", "Representation", "idx", "Representation"), withIndex, edges),
    ).toBe(true);
  });
});

describe("computeProblems (structural)", () => {
  it("flags duplicate single-slot stages", () => {
    const nodes = [node("c1", "chunker", "fixed"), node("c2", "chunker", "fixed")];
    const ps = computeProblems(nodes, [], mIndex);
    expect(ps.some((p) => p.level === "error" && /chunker/.test(p.message))).toBe(true);
  });

  it("flags an index-backed block that isn't wired to the index", () => {
    const nodes = [node("r", "retriever", "index"), node("idx", "index", "ChunkIndex")];
    const ps = computeProblems(nodes, [], mIndex);
    expect(ps.some((p) => /needs the index/.test(p.message))).toBe(true);
  });

  it("is happy with a wired index-backed retriever", () => {
    const nodes = [node("r", "retriever", "index"), node("idx", "index", "ChunkIndex")];
    const edges: BlockEdge[] = [
      { id: "x", source: "idx", target: "r", sourceHandle: "out:Index", targetHandle: "in:Index" },
    ];
    const ps = computeProblems(nodes, edges, mIndex);
    expect(ps.some((p) => /needs the index/.test(p.message))).toBe(false);
  });
});
