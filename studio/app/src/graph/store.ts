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
import { parseHandle } from "./ports";
import { computeProblems, isValidConnection } from "./validate";
import { endpointEdges, endpointNodes, isEndpointId, mergeEdges } from "./endpoints";
import { autoWireNewNode } from "./wire";
import { pruneCorpusEdges, repSpace } from "./corpus";

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
  deleteEdge: (id: string) => void;
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
  // Recompute after any structural change, in one place: prune now-invalid
  // corpus index edges, then recompute the problems list.
  const withProblems = (nodes: BlockNode[], rawEdges: BlockEdge[]) => {
    const mIndex = get().mIndex;
    const edges = pruneCorpusEdges(nodes, rawEdges);
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

      // A representation is addressed by its *space* — the corpus keys one output
      // port per space, so two spaces can't collide. The space defaults to the
      // rep's type name ("dense"), so a second `dense` would clash: auto-name it
      // "dense-2", "dense-3", … The user can rename it in the inspector.
      if (kind === "representations") {
        const used = new Set(
          nodes.filter((n) => n.data.kind === "representations").map((n) => repSpace(n)),
        );
        if (used.has(name)) {
          let i = 2;
          while (used.has(`${name}-${i}`)) i++;
          params.space = `${name}-${i}`;
        }
      }

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
      // Representation: seed its wrapped encoder with the first available one of
      // the right kind (dense→an embedder, lexical→a lexical index).
      if (comp?.encoder) {
        const enc = (mIndex.componentsByKind.get(comp.encoder.kind) ?? []).find((c) => c.exportable);
        if (enc) node.data.encoder = { name: enc.name, params: mIndex.defaultParams(comp.encoder.kind, enc.name) };
      }

      // Ensure the synthetic Corpus exists whenever something that attaches to it
      // appears: a representation or a vector store. (A corpus with no
      // representations has no index ports, so we don't spawn an empty one for a
      // retriever — adding a representation creates it and wires it up.)
      const next = [...nodes, node];
      const needsCorpus = kind === "representations" || kind === "vector_store";
      if (needsCorpus && !next.some((n) => n.data.kind === "corpus")) {
        next.push(makeCorpusNode(next.length));
      }

      // Auto-wire the new block into the pipeline by contract type, so it's never
      // a floating orphan — a sensible default the user can still rewire.
      const wired = autoWireNewNode(node, next, edges, mIndex);
      let nextEdges = edges.filter((e) => !wired.remove.includes(e.id));
      nextEdges = mergeEdges(nextEdges, wired.add);
      // Tie the endpoints in too (Source->parser, Query->retriever, generator->
      // Answer) so adding a parser/retriever/generator connects to its terminal.
      nextEdges = mergeEdges(nextEdges, endpointEdges(next, mIndex));
      set({ ...withProblems(next, nextEdges), selectedId: node.id });
    },

    updateParams: (id, params) =>
      set((s) => {
        const nodes = s.nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, params } } : n,
        );
        // A representation's `space` param drives the corpus index ports, so a
        // rename can invalidate a wired edge — reconcile and re-check.
        return withProblems(nodes, s.edges);
      }),

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

    // Click a connection to remove it. Endpoint wiring re-appears only when a
    // block is next added (mergeEdges), so a deliberate cut otherwise stays cut.
    deleteEdge: (id) =>
      set((s) => withProblems(s.nodes, s.edges.filter((e) => e.id !== id))),

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
        if (!isValidConnection(c, s.nodes, s.edges, s.mIndex)) return {};
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

function makeCorpusNode(order: number): BlockNode {
  return {
    id: nextId("corpus", "corpus"),
    type: "corpus",
    position: tile(order),
    data: { kind: "corpus", name: "Corpus", params: {}, synthetic: true },
  };
}
