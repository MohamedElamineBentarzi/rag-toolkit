import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type NodeChange,
  type OnConnectStartParams,
} from "@xyflow/react";
import type { Manifest } from "../manifest/types";
import { ManifestIndex } from "../manifest/load";
import type { BlockNode, BlockEdge, BlockData, Problem } from "./model";
import { handleId, parseHandle } from "./ports";
import { computeProblems, isValidConnection } from "./validate";
import { endpointEdges, endpointNodes, isEndpointId, mergeEdges } from "./endpoints";

const REPRESENTATION_KINDS = ["embedder", "sparse", "lexical"];

interface StudioState {
  manifest: Manifest | null;
  mIndex: ManifestIndex | null;
  nodes: BlockNode[];
  edges: BlockEdge[];
  selectedId: string | null;
  /** The output type of a connection currently being dragged (drives the
   *  green/dim handle highlight). */
  pendingSourceType: string | null;
  problems: Problem[];

  setManifest: (m: Manifest) => void;
  addNode: (kind: string, name: string, position?: { x: number; y: number }) => void;
  updateParams: (id: string, params: Record<string, unknown>) => void;
  updateData: (id: string, patch: Partial<BlockData>) => void;
  select: (id: string | null) => void;
  deleteSelected: () => void;
  onNodesChange: (changes: NodeChange<BlockNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<BlockEdge>[]) => void;
  onConnect: (c: Connection) => void;
  onConnectStart: (params: OnConnectStartParams) => void;
  onConnectEnd: () => void;
  setGraph: (nodes: BlockNode[], edges: BlockEdge[]) => void;
  clear: () => void;
}

let counter = 0;
const nextId = (kind: string, name: string) => `${kind}-${name}-${++counter}`;

export const useStudio = create<StudioState>((set, get) => {
  // Recompute the problems list after any structural change, in one place.
  const withProblems = (nodes: BlockNode[], edges: BlockEdge[]) => {
    const mIndex = get().mIndex;
    return { nodes, edges, problems: mIndex ? computeProblems(nodes, edges, mIndex) : [] };
  };

  return {
    manifest: null,
    mIndex: null,
    nodes: [],
    edges: [],
    selectedId: null,
    pendingSourceType: null,
    problems: [],

    setManifest: (m) =>
      set({ manifest: m, mIndex: new ManifestIndex(m), nodes: endpointNodes() }),

    addNode: (kind, name, position) => {
      const { mIndex, nodes, edges } = get();
      if (!mIndex) return;
      const comp = mIndex.component(kind, name);
      const params: Record<string, unknown> = {};
      for (const p of comp?.params ?? []) params[p.name] = p.default;

      const node: BlockNode = {
        id: nextId(kind, name),
        type: "block",
        position: position ?? tile(nodes.length),
        data: { kind, name, params },
      };

      // Composite retriever: seed its nested sub-retriever(s) with a base one.
      if (comp?.composite === "inner") {
        const base = mIndex.baseRetrievers()[0];
        if (base) node.data.inner = { name: base.name, params: mIndex.defaultParams("retriever", base.name) };
      } else if (comp?.composite === "retrievers") {
        node.data.retrievers = [];
      }

      // Ensure the synthetic ChunkIndex exists whenever something that attaches
      // to it appears: a representation, a vector store, or an index-backed
      // block. The blob store attaches to the parser instead, not the index.
      const next = [...nodes, node];
      const needsIndex =
        REPRESENTATION_KINDS.includes(kind) || kind === "vector_store" || comp?.takes_index;
      if (needsIndex && !next.some((n) => n.data.kind === "index")) {
        next.push(makeIndexNode(next.length));
      }

      // Auto-wire so a new block is never an orphan on the canvas — a sensible
      // default edge the user can still rewire: representations and the store
      // feed the index; index-backed blocks read from it; the blob store backs
      // the parser (where caching + raw capture happen).
      let nextEdges = edges;
      const indexNode = next.find((n) => n.data.kind === "index");
      if (kind === "blob_store") {
        const parser = next.find((n) => n.data.kind === "parser");
        if (parser) nextEdges = addEdge(makeEdge(node.id, "BlobStore", parser.id, mIndex), nextEdges);
      } else if (indexNode) {
        if (REPRESENTATION_KINDS.includes(kind)) {
          nextEdges = addEdge(makeEdge(node.id, "Representation", indexNode.id, mIndex), nextEdges);
        } else if (kind === "vector_store") {
          nextEdges = addEdge(makeEdge(node.id, "Store", indexNode.id, mIndex), nextEdges);
        } else if (comp?.takes_index) {
          nextEdges = addEdge(makeEdge(indexNode.id, "Index", node.id, mIndex), nextEdges);
        }
      }
      // Tie the endpoints in too (Source->parser, Query->retriever, generator->
      // Answer) so adding a parser/retriever/generator connects to its terminal.
      nextEdges = mergeEdges(nextEdges, endpointEdges(next, mIndex));
      set({ ...withProblems(next, nextEdges), selectedId: node.id });
    },

    updateParams: (id, params) =>
      set((s) => ({
        nodes: s.nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, params } } : n,
        ),
      })),

    updateData: (id, patch) =>
      set((s) => ({
        nodes: s.nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, ...patch } } : n,
        ),
      })),

    select: (id) => set({ selectedId: id }),

    deleteSelected: () => {
      const { selectedId, nodes, edges } = get();
      if (!selectedId || isEndpointId(selectedId)) return; // endpoints are fixed
      const next = nodes.filter((n) => n.id !== selectedId);
      const nextEdges = edges.filter(
        (e) => e.source !== selectedId && e.target !== selectedId,
      );
      set({ ...withProblems(next, nextEdges), selectedId: null });
    },

    onNodesChange: (changes) =>
      set((s) => {
        // Endpoints can be moved but never removed (belt-and-suspenders beside
        // their `deletable: false`).
        const safe = changes.filter(
          (c) => !(c.type === "remove" && isEndpointId(c.id)),
        );
        const nodes = applyNodeChanges(safe, s.nodes);
        return withProblems(nodes, s.edges);
      }),

    onEdgesChange: (changes) =>
      set((s) => {
        const edges = applyEdgeChanges(changes, s.edges);
        return withProblems(s.nodes, edges);
      }),

    onConnect: (c) =>
      set((s) => {
        if (!isValidConnection(c, s.nodes, s.edges)) return {};
        const src = parseHandle(c.sourceHandle);
        const edge: BlockEdge = {
          ...c,
          id: `e-${c.source}-${c.target}-${c.sourceHandle}-${c.targetHandle}`,
          style: src ? { stroke: s.mIndex?.typeColor(src.type) } : undefined,
        };
        const edges = addEdge(edge, s.edges);
        return withProblems(s.nodes, edges);
      }),

    onConnectStart: (params) => {
      const port = parseHandle(params.handleId);
      set({ pendingSourceType: port?.dir === "out" ? port.type : null });
    },
    onConnectEnd: () => set({ pendingSourceType: null }),

    setGraph: (nodes, edges) => {
      // An imported graph never carries the endpoints — prepend them and wire
      // them to the pipeline, so the terminals are always present and connected.
      const mIndex = get().mIndex;
      const all = [...endpointNodes(), ...nodes];
      const wired = mIndex ? mergeEdges(edges, endpointEdges(all, mIndex)) : edges;
      set({ ...withProblems(all, wired), selectedId: null });
    },

    clear: () =>
      set({ nodes: endpointNodes(), edges: [], selectedId: null, problems: [] }),
  };
});

// Click-to-add tiles nodes into a loose grid so they never stack; a dropped
// node uses the drop point instead.
function tile(i: number): { x: number; y: number } {
  return { x: 60 + (i % 4) * 260, y: 70 + Math.floor(i / 4) * 168 };
}

function makeEdge(
  source: string,
  type: string,
  target: string,
  mIndex: ManifestIndex,
): BlockEdge {
  return {
    id: `e-${source}-${target}-${type}`,
    source,
    target,
    sourceHandle: handleId("out", type),
    targetHandle: handleId("in", type),
    style: { stroke: mIndex.typeColor(type) },
  };
}

function makeIndexNode(order: number): BlockNode {
  return {
    id: nextId("index", "index"),
    type: "block",
    position: tile(order),
    data: { kind: "index", name: "ChunkIndex", params: {}, synthetic: true },
  };
}
