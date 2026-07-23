import { describe, it, expect } from "vitest";
import { compileSpec } from "./compile";
import { importSpec } from "./importSpec";
import { ManifestIndex } from "../manifest/load";
import type { BlockNode } from "../graph/model";
import type { Manifest } from "../manifest/types";
import manifestJson from "../../public/blocks.json";

const mIndex = new ManifestIndex(manifestJson as unknown as Manifest);

function node(id: string, kind: string, name: string, params: Record<string, unknown>): BlockNode {
  return { id, type: "block", position: { x: 0, y: 0 }, data: { kind, name, params } };
}

describe("compileSpec (graph -> spec)", () => {
  it("emits one entry per single stage, omitting default params", () => {
    const nodes = [
      node("c", "chunker", "fixed", { chunk_chars: 200, overlap_chars: 200 }), // overlap is default
      node("e", "embedder", "hashing", { dimensions: 256 }), // default
      node("g", "generator", "extractive", { max_context_chars: 4000 }), // default
    ];
    const spec = compileSpec(nodes, [], mIndex);
    expect(spec).toEqual({
      chunker: { name: "fixed", params: { chunk_chars: 200 } }, // only the non-default
      embedder: { name: "hashing", params: {} },
      generator: { name: "extractive", params: {} },
    });
  });

  it("never exports a secret param", () => {
    const nodes = [node("g", "generator", "anthropic", { api_key: "sk-leak", max_tokens: 999 })];
    const spec = compileSpec(nodes, [], mIndex) as { generator: { params: Record<string, unknown> } };
    expect(spec.generator.params).not.toHaveProperty("api_key");
    expect(spec.generator.params).toHaveProperty("max_tokens", 999);
  });

  it("orders a refine chain along its edges", () => {
    const nodes = [
      node("r2", "refine", "keyword", {}),
      node("r1", "refine", "score-threshold", { min_score: 0.1 }),
    ];
    const edges = [
      { id: "e", source: "r1", target: "r2", sourceHandle: "out:ScoredChunk[]", targetHandle: "in:ScoredChunk[]" },
    ];
    const spec = compileSpec(nodes, edges, mIndex) as { refine: { name: string }[] };
    expect(spec.refine.map((r) => r.name)).toEqual(["score-threshold", "keyword"]);
  });
});

describe("infrastructure blocks (store + blob_store)", () => {
  it("emits them, dropping the store's and blob store's secrets", () => {
    const nodes = [
      node("s", "vector_store", "qdrant", { url: "http://x:6333", api_key: "sk-leak" }),
      node("b", "blob_store", "minio", { endpoint: "s3.example.com:9000", access_key: "AK", secret_key: "SK" }),
    ];
    const spec = compileSpec(nodes, [], mIndex) as {
      vector_store: { name: string; params: Record<string, unknown> };
      blob_store: { params: Record<string, unknown> };
    };
    expect(spec.vector_store).toEqual({ name: "qdrant", params: { url: "http://x:6333" } });
    expect(spec.blob_store.params).not.toHaveProperty("access_key");
    expect(spec.blob_store.params).not.toHaveProperty("secret_key");
    expect(spec.blob_store.params).toHaveProperty("endpoint", "s3.example.com:9000");
  });
});

describe("round trip (import -> compile)", () => {
  it("reproduces a spec through the graph and back", () => {
    const spec = {
      chunker: { name: "fixed", params: { chunk_chars: 200 } },
      embedder: { name: "hashing", params: { dimensions: 128 } },
      refine: [{ name: "score-threshold", params: { min_score: 0.1 } }],
      generator: { name: "extractive", params: {} },
    };
    const { nodes, edges } = importSpec(spec, mIndex);
    const back = compileSpec(nodes, edges, mIndex);
    expect(back).toEqual(spec);
  });

  it("round-trips a composite: hyde wrapping an index retriever", () => {
    const spec = {
      embedder: { name: "hashing", params: { dimensions: 128 } },
      retriever: { name: "hyde", params: {}, inner: { name: "index", params: { representation: "dense" } } },
      generator: { name: "anthropic", params: {} },
    };
    const { nodes, edges } = importSpec(spec, mIndex);
    expect(compileSpec(nodes, edges, mIndex)).toEqual(spec);
  });

  it("round-trips fusion wrapping two index retrievers", () => {
    const spec = {
      embedder: { name: "hashing", params: {} },
      lexical: { name: "bm25", params: {} },
      retriever: {
        name: "fusion",
        params: {},
        retrievers: [
          { name: "index", params: { representation: "dense" } },
          { name: "index", params: { representation: "lexical" } },
        ],
      },
      generator: { name: "extractive", params: {} },
    };
    const { nodes, edges } = importSpec(spec, mIndex);
    expect(compileSpec(nodes, edges, mIndex)).toEqual(spec);
  });

  it("round-trips a spec with a vector store and blob store", () => {
    const spec = {
      parser: { name: "plaintext", params: {} },
      chunker: { name: "fixed", params: { chunk_chars: 200 } },
      embedder: { name: "hashing", params: { dimensions: 128 } },
      vector_store: { name: "memory", params: {} },
      blob_store: { name: "local", params: { root: "/data" } },
      generator: { name: "extractive", params: {} },
    };
    const { nodes, edges } = importSpec(spec, mIndex);
    expect(compileSpec(nodes, edges, mIndex)).toEqual(spec);
  });
});
