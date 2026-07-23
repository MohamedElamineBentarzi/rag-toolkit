import type { CSSProperties } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useStudio } from "./store";
import type { BlockNode as BlockNodeType } from "./model";

// A pipeline endpoint: a distinct pill (not a card) with one typed handle, so
// it reads as a terminal, not a component. Source/Query emit; Answer receives.
export function EndpointNode({ data }: NodeProps<BlockNodeType>) {
  const mIndex = useStudio((s) => s.mIndex);
  const type = String(data.epType);
  const isOut = data.epDir === "out";
  const color = mIndex?.typeColor(type) ?? "#8b8b9e";

  return (
    <div
      className={`endpoint ${isOut ? "out" : "in"}`}
      style={{ ["--ep" as string]: color } as CSSProperties}
    >
      {!isOut && <span className="ep-dot" />}
      <div className="ep-text">
        <div className="ep-label">{String(data.label)}</div>
        <div className="ep-type">{type}</div>
      </div>
      {isOut && <span className="ep-dot" />}
      <Handle
        id={`${data.epDir}:${type}`}
        type={isOut ? "source" : "target"}
        position={isOut ? Position.Right : Position.Left}
        style={{ background: color }}
      />
    </div>
  );
}
