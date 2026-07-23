import type { CSSProperties } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useStudio } from "./store";
import { inputPorts, outputPort, type Port } from "./ports";
import { portIsCompatible } from "./validate";
import type { BlockNode as BlockNodeType } from "./model";
import { stageAccent } from "../theme/tokens";

// One renderer for every block. Ports are derived from the node's stage in the
// manifest, so a new component needs no new node code. Input handles sit on the
// left, the single output on the right, each filled with its contract-type
// color; while a connection drags, compatible inputs glow and the rest dim.
export function BlockNode({ id, data, selected }: NodeProps<BlockNodeType>) {
  const mIndex = useStudio((s) => s.mIndex);
  const pending = useStudio((s) => s.pendingSourceType);
  if (!mIndex) return null;

  const stage = mIndex.stage(data.kind);
  const inputs = stage ? inputPorts(stage) : [];
  const output = stage ? outputPort(stage) : null;
  const accent = stageAccent[data.kind] ?? "#8b8b9e";

  return (
    <div
      className={`blocknode ${selected ? "selected" : ""} ${data.synthetic ? "synthetic" : ""}`}
      style={{ ["--stage" as string]: accent } as CSSProperties}
    >
      <div className="body">
        <div className="kind">{data.kind}</div>
        <div className="name">{data.name}</div>
      </div>

      {inputs.map((port, i) => (
        <TypedHandle
          key={port.id}
          port={port}
          dir="target"
          color={mIndex.typeColor(port.type)}
          top={pct(i, inputs.length)}
          highlight={pending ? (portIsCompatible(pending, port.type) ? "compatible" : "incompatible") : null}
        />
      ))}
      {output && (
        <TypedHandle
          port={output}
          dir="source"
          color={mIndex.typeColor(output.type)}
          top="50%"
          highlight={null}
          nodeId={id}
        />
      )}
    </div>
  );
}

function TypedHandle(props: {
  port: Port;
  dir: "source" | "target";
  color: string;
  top: string;
  highlight: "compatible" | "incompatible" | null;
  nodeId?: string;
}) {
  const { port, dir, color, top, highlight } = props;
  const side = dir === "target" ? Position.Left : Position.Right;
  return (
    <>
      <Handle
        id={port.id}
        type={dir}
        position={side}
        className={highlight ?? ""}
        style={{ top, background: color }}
        title={port.type}
      />
      <span
        className="port-label"
        style={{ top, [dir === "target" ? "left" : "right"]: 16 }}
      >
        {port.type}
      </span>
    </>
  );
}

// Evenly space N handles down the card's height.
function pct(i: number, n: number): string {
  return `${((i + 1) / (n + 1)) * 100}%`;
}
