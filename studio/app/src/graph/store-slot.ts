import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge, BlockData } from "./model";

// A self-managed representation (BM25/lexical) keeps its OWN isolated blob store
// for its inverted-index side-write — the Corpus owns the shared vector store,
// this owns its persistence (the deliberate, documented asymmetry). In the
// Studio that shows up as an optional `BlobStore` input port on the
// representation whose chosen encoder declares a store slot; wiring a blob_store
// block into it is how you give BM25 somewhere durable to persist.

/** The store slot of a representation node's *currently chosen* encoder, if that
 *  encoder keeps its own persistence backend (BM25 does; an embedder doesn't). */
export function repStoreSlot(
  node: { data: BlockData },
  mIndex: ManifestIndex,
): { param: string } | null {
  if (node.data.kind !== "representations") return null;
  const comp = mIndex.component("representations", node.data.name);
  const enc = node.data.encoder;
  if (!comp?.encoder || !enc) return null;
  const encComp = mIndex.component(comp.encoder.kind, enc.name);
  return encComp?.store_slot ? { param: encComp.store_slot.param } : null;
}

/** The blob_store node wired into a representation's BlobStore port, if any. */
export function wiredBlobStore(
  nodeId: string,
  nodes: BlockNode[],
  edges: BlockEdge[],
): BlockNode | undefined {
  const edge = edges.find(
    (e) => e.target === nodeId && e.targetHandle === "in:BlobStore",
  );
  if (!edge) return undefined;
  return nodes.find((n) => n.id === edge.source && n.data.kind === "blob_store");
}
