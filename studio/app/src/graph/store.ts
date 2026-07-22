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
import type { BlockNode, BlockEdge, Problem } from "./model";
import { parseHandle } from "./ports";
import { computeProblems, isValidConnection } from "./validate";

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

    setManifest: (m) => set({ manifest: m, mIndex: new ManifestIndex(m) }),

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

      // Ensure the synthetic ChunkIndex exists whenever a representation or an
      // index-backed block appears — the two things that need it.
      let next = [...nodes, node];
      const needsIndex = REPRESENTATION_KINDS.includes(kind) || comp?.takes_index;
      if (needsIndex && !next.some((n) => n.data.kind === "index")) {
        next.push(makeIndexNode(next.length));
      }
      set({ ...withProblems(next, edges), selectedId: node.id });
    },

    updateParams: (id, params) =>
      set((s) => ({
        nodes: s.nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, params } } : n,
        ),
      })),

    select: (id) => set({ selectedId: id }),

    deleteSelected: () => {
      const { selectedId, nodes, edges } = get();
      if (!selectedId) return;
      const next = nodes.filter((n) => n.id !== selectedId);
      const nextEdges = edges.filter(
        (e) => e.source !== selectedId && e.target !== selectedId,
      );
      set({ ...withProblems(next, nextEdges), selectedId: null });
    },

    onNodesChange: (changes) =>
      set((s) => {
        const nodes = applyNodeChanges(changes, s.nodes);
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

    setGraph: (nodes, edges) =>
      set({ ...withProblems(nodes, edges), selectedId: null }),

    clear: () => set({ nodes: [], edges: [], selectedId: null, problems: [] }),
  };
});

// Click-to-add tiles nodes into a loose grid so they never stack; a dropped
// node uses the drop point instead.
function tile(i: number): { x: number; y: number } {
  return { x: 60 + (i % 4) * 260, y: 70 + Math.floor(i / 4) * 168 };
}

function makeIndexNode(order: number): BlockNode {
  return {
    id: nextId("index", "index"),
    type: "block",
    position: tile(order),
    data: { kind: "index", name: "ChunkIndex", params: {}, synthetic: true },
  };
}
