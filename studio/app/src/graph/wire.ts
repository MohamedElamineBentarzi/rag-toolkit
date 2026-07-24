import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge } from "./model";
import { handleId } from "./ports";
import { corpusArity, corpusSpaces, findCorpus, repSpace, wiredSpaces } from "./corpus";

// Auto-wiring: when a block lands on the canvas we connect it into the pipeline
// by contract type, so a new block is never a floating orphan the user has to
// hand-wire. This mirrors how importSpec reconstructs a spec's backbone, but
// scoped to the one new node — and it can *re-thread* a chain (drop an enricher
// between the chunker and the representations, a refiner between the retriever
// and the generator) rather than leaving a dangling middle block.

/** A typed edge; `space` tags a Corpus index port ("out:Corpus#dense"). */
export function makeTypedEdge(
  source: string,
  type: string,
  target: string,
  mIndex: ManifestIndex,
  space?: string,
): BlockEdge {
  return {
    id: `e-${source}-${target}-${type}${space ? `#${space}` : ""}`,
    source,
    target,
    sourceHandle: handleId("out", type, space),
    targetHandle: handleId("in", type),
    style: { stroke: mIndex.typeColor(type) },
  };
}

export interface WireResult {
  add: BlockEdge[];
  remove: string[]; // edge ids to drop (chain re-threading)
}

export function autoWireNewNode(
  node: BlockNode,
  nodes: BlockNode[],
  edges: BlockEdge[],
  mIndex: ManifestIndex,
): WireResult {
  const others = nodes.filter((n) => n.id !== node.id);
  const find = (kind: string) => others.find((n) => n.data.kind === kind);
  const all = (kind: string) => others.filter((n) => n.data.kind === kind);
  const inputFree = (target: string, type: string) =>
    !edges.some((e) => e.target === target && e.targetHandle === `in:${type}`);
  const corpus = findCorpus(others);

  const add: BlockEdge[] = [];
  const remove: string[] = [];
  const edge = (source: string, target: string, type: string, space?: string) =>
    add.push(makeTypedEdge(source, type, target, mIndex, space));

  switch (node.data.kind) {
    case "parser": {
      const blob = find("blob_store");
      if (blob) edge(blob.id, node.id, "BlobStore");
      const chunker = find("chunker");
      if (chunker && inputFree(chunker.id, "Document")) edge(node.id, chunker.id, "Document");
      break;
    }

    case "chunker": {
      const parser = find("parser");
      if (parser) edge(parser.id, node.id, "Document");
      // Feed the head of the chunk flow: an enricher if present, else the reps.
      const enrich = find("enrich");
      if (enrich && inputFree(enrich.id, "Chunk[]")) {
        edge(node.id, enrich.id, "Chunk[]");
      } else {
        for (const r of all("representations"))
          if (inputFree(r.id, "Chunk[]")) edge(node.id, r.id, "Chunk[]");
      }
      break;
    }

    case "enrich": {
      // Splice into the Chunk[] flow: tail -> newEnrich -> (tail's old consumers).
      spliceChain(node, "Chunk[]", chunkTail(others, edges), edges, add, remove, mIndex);
      break;
    }

    case "representations": {
      const tail = chunkTail(others, edges);
      if (tail && inputFree(node.id, "Chunk[]")) edge(tail.id, node.id, "Chunk[]");
      if (corpus) {
        edge(node.id, corpus.id, "Representation");
        // Hand the new index to the retrievers that read many; give a lone
        // `index` retriever its one representation only if it has none yet.
        const space = repSpace(node);
        for (const r of all("retriever")) {
          const arity = corpusArity(mIndex.component("retriever", r.data.name));
          if (arity === "single") {
            if (wiredSpaces(r.id, edges).length === 0) edge(corpus.id, r.id, "Corpus", space);
          } else {
            edge(corpus.id, r.id, "Corpus", space);
          }
        }
      }
      break;
    }

    case "vector_store": {
      if (corpus) edge(node.id, corpus.id, "VectorStore");
      break;
    }

    case "blob_store": {
      const parser = find("parser");
      if (parser) edge(node.id, parser.id, "BlobStore");
      break;
    }

    case "retriever": {
      if (corpus) {
        const spaces = corpusSpaces(corpus.id, others, edges);
        const arity = corpusArity(mIndex.component("retriever", node.data.name));
        const chosen = arity === "single" ? spaces.slice(0, 1) : spaces;
        for (const space of chosen) edge(corpus.id, node.id, "Corpus", space);
      }
      // Feed the head of the scored flow: a refiner if present, else generator.
      const refine = find("refine");
      if (refine && inputFree(refine.id, "ScoredChunk[]")) {
        edge(node.id, refine.id, "ScoredChunk[]");
      } else {
        const gen = find("generator");
        if (gen && inputFree(gen.id, "ScoredChunk[]")) edge(node.id, gen.id, "ScoredChunk[]");
      }
      break;
    }

    case "refine": {
      spliceChain(node, "ScoredChunk[]", scoredTail(others, edges), edges, add, remove, mIndex);
      break;
    }

    case "generator": {
      const tail = scoredTail(others, edges);
      if (tail && inputFree(node.id, "ScoredChunk[]")) edge(tail.id, node.id, "ScoredChunk[]");
      break;
    }
  }

  return { add, remove };
}

// Insert `node` inline on a `type` flow: connect tail -> node, then re-point
// everything the tail used to feed so it now comes through node.
function spliceChain(
  node: BlockNode,
  type: string,
  tail: BlockNode | undefined,
  edges: BlockEdge[],
  add: BlockEdge[],
  remove: string[],
  mIndex: ManifestIndex,
): void {
  if (!tail) return;
  add.push(makeTypedEdge(tail.id, type, node.id, mIndex));
  for (const e of edges) {
    if (e.source !== tail.id || e.sourceHandle !== `out:${type}`) continue;
    remove.push(e.id);
    add.push(makeTypedEdge(node.id, type, e.target, mIndex));
  }
}

/** The end of the Chunk[] flow: the enricher feeding nothing downstream, or the
 *  chunker if there are no enrichers. */
function chunkTail(nodes: BlockNode[], edges: BlockEdge[]): BlockNode | undefined {
  return tailOf("enrich", "chunker", "Chunk[]", nodes, edges);
}

/** The end of the ScoredChunk[] flow: the last refiner, or the retriever. */
function scoredTail(nodes: BlockNode[], edges: BlockEdge[]): BlockNode | undefined {
  return tailOf("refine", "retriever", "ScoredChunk[]", nodes, edges);
}

function tailOf(
  chainKind: string,
  headKind: string,
  type: string,
  nodes: BlockNode[],
  edges: BlockEdge[],
): BlockNode | undefined {
  const chain = nodes.filter((n) => n.data.kind === chainKind);
  if (chain.length) {
    const feedsAnother = new Set(
      edges
        .filter((e) => e.targetHandle === `in:${type}` && chain.some((c) => c.id === e.target))
        .map((e) => e.source),
    );
    return chain.find((c) => !feedsAnother.has(c.id)) ?? chain[chain.length - 1];
  }
  return nodes.find((n) => n.data.kind === headKind);
}
