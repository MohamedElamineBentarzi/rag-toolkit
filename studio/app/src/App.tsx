import { useCallback, useEffect, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Connection,
  type IsValidConnection,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useStudio } from "./graph/store";
import { BlockNode } from "./graph/BlockNode";
import { EndpointNode } from "./graph/EndpointNode";
import { isValidConnection as checkConnection } from "./graph/validate";
import type { BlockNode as BlockNodeType, BlockEdge } from "./graph/model";
import { Palette } from "./panels/Palette";
import { Inspector } from "./panels/Inspector";
import { StatusPill } from "./panels/StatusPill";
import { loadManifest } from "./manifest/load";
import { compileSpec } from "./spec/compile";
import { validateSpec } from "./spec/validateSpec";
import { importSpec } from "./spec/importSpec";
import { stageAccent } from "./theme/tokens";

// Defined once, outside the component: React Flow warns (and rerenders) if
// nodeTypes is a fresh object each render.
const nodeTypes = { block: BlockNode, endpoint: EndpointNode };

export default function App() {
  return (
    <ReactFlowProvider>
      <Studio />
    </ReactFlowProvider>
  );
}

function Studio() {
  const setManifest = useStudio((s) => s.setManifest);
  const nodes = useStudio((s) => s.nodes);
  const edges = useStudio((s) => s.edges);
  const onNodesChange = useStudio((s) => s.onNodesChange);
  const onEdgesChange = useStudio((s) => s.onEdgesChange);
  const onConnect = useStudio((s) => s.onConnect);
  const onConnectStart = useStudio((s) => s.onConnectStart);
  const onConnectEnd = useStudio((s) => s.onConnectEnd);
  const addNode = useStudio((s) => s.addNode);
  const select = useStudio((s) => s.select);
  const { screenToFlowPosition, fitView } = useReactFlow();

  // The panels are sliding glass drawers, toggled from the toolbar.
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  useEffect(() => {
    loadManifest()
      .then((m) => {
        setManifest(m);
        setTimeout(() => fitView({ padding: 0.25 }), 60); // frame the endpoints
      })
      .catch((e) => alert(String(e)));
  }, [setManifest, fitView]);

  // Only offer a connection to React Flow if the contract types match — this is
  // where the "invalid connections refuse to form" behavior lives.
  const isValidConnection: IsValidConnection<BlockEdge> = useCallback(
    (c) => checkConnection(c as Connection, useStudio.getState().nodes, useStudio.getState().edges),
    [],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const raw = event.dataTransfer.getData("application/rag-block");
      if (!raw) return;
      const { kind, name } = JSON.parse(raw);
      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      addNode(kind, name, position);
    },
    [screenToFlowPosition, addNode],
  );

  return (
    <div className="app">
      <Toolbar
        leftOpen={leftOpen}
        rightOpen={rightOpen}
        onToggleLeft={() => setLeftOpen((v) => !v)}
        onToggleRight={() => setRightOpen((v) => !v)}
        onImported={() => setTimeout(() => fitView({ padding: 0.2 }), 0)}
      />
      <div className="stage">
        <aside className={`drawer left ${leftOpen ? "open" : ""}`}>
          <Palette />
        </aside>

        <div className="canvas" onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
          <ReactFlow<BlockNodeType, BlockEdge>
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onConnectStart={(_, p) => onConnectStart(p)}
            onConnectEnd={onConnectEnd}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => select(node.id)}
            onPaneClick={() => select(null)}
            fitView
            proOptions={{ hideAttribution: true }}
            defaultEdgeOptions={{ animated: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={22}
              size={1}
              color="rgba(255,255,255,0.06)"
            />
            <Controls showInteractive={false} position="bottom-left" />
            <MiniMap
              pannable
              zoomable
              nodeStrokeWidth={0}
              nodeColor={(n) => stageAccent[(n.data as { kind: string }).kind] ?? "#5b5b74"}
              maskColor="rgba(8,8,14,0.6)"
              style={{ background: "transparent" }}
            />
          </ReactFlow>
        </div>

        <aside className={`drawer right ${rightOpen ? "open" : ""}`}>
          <Inspector />
        </aside>
      </div>
    </div>
  );
}

function Toolbar(props: {
  leftOpen: boolean;
  rightOpen: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  onImported: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const setGraph = useStudio((s) => s.setGraph);
  const clear = useStudio((s) => s.clear);
  const deleteSelected = useStudio((s) => s.deleteSelected);

  const onExport = () => {
    const { nodes, edges, mIndex, manifest, problems } = useStudio.getState();
    if (!mIndex || !manifest) return;
    const spec = compileSpec(nodes, edges, mIndex);
    const errors = [
      ...problems.filter((p) => p.level === "error").map((p) => p.message),
      ...validateSpec(spec, manifest),
    ];
    if (errors.length) {
      alert("Fix these before exporting:\n\n- " + errors.join("\n- "));
      return;
    }
    download("pipeline.json", JSON.stringify(spec, null, 2));
  };

  const onImportFile = async (file: File) => {
    const { mIndex, manifest } = useStudio.getState();
    if (!mIndex || !manifest) return;
    let spec: unknown;
    try {
      spec = JSON.parse(await file.text());
    } catch {
      alert("That file isn't valid JSON.");
      return;
    }
    const errors = validateSpec(spec, manifest);
    if (errors.length) {
      alert("Not a valid pipeline spec:\n\n- " + errors.join("\n- "));
      return;
    }
    const { nodes, edges } = importSpec(spec as Record<string, unknown>, mIndex);
    setGraph(nodes, edges);
    props.onImported();
  };

  return (
    <div className="toolbar">
      <button
        className={`icon-btn ${props.leftOpen ? "active" : ""}`}
        onClick={props.onToggleLeft}
        title="Toggle blocks panel"
        aria-label="Toggle blocks panel"
      >
        <PanelIcon side="left" />
      </button>
      <div className="title">
        <span className="dot" />
        <span className="brand">rag-blocks</span>
        <span className="sub">studio</span>
      </div>
      <StatusPill />
      <button onClick={deleteSelected}>Delete</button>
      <button onClick={() => { if (confirm("Clear the canvas?")) clear(); }}>Clear</button>
      <button onClick={() => fileRef.current?.click()}>Import</button>
      <button className="primary" onClick={onExport}>Export spec</button>
      <button
        className={`icon-btn ${props.rightOpen ? "active" : ""}`}
        onClick={props.onToggleRight}
        title="Toggle inspector panel"
        aria-label="Toggle inspector panel"
      >
        <PanelIcon side="right" />
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="application/json,.json"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onImportFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function PanelIcon({ side }: { side: "left" | "right" }) {
  const x = side === "left" ? 5.5 : 10.5;
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="1.5" y="2.5" width="13" height="11" rx="2.2" />
      <line x1={x} y1="2.5" x2={x} y2="13.5" />
    </svg>
  );
}

function download(filename: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
