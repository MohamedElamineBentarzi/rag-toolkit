import type { CSSProperties } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useStudio } from "./store";
import { inputPorts, outputPort, handleId, type Port } from "./ports";
import { portIsCompatible } from "./validate";
import { corpusArity, repSpace } from "./corpus";
import { repStoreSlot } from "./store-slot";
import type { BlockNode as BlockNodeType } from "./model";
import type { StageSpec } from "../manifest/types";
import type { ManifestIndex } from "../manifest/load";
import { stageAccent } from "../theme/tokens";

// The label shown on an *input* port. A port that fans in many of a type reads
// with array notation (`Representation[]`), so "accepts several" is visible at a
// glance; and a retriever's Corpus input reads as the representation(s) it
// selects — one for an `index`, many for a `hybrid`/composite — since that's
// what wiring an index into it means (there's no bare "Corpus" concept for the
// user to decode).
export function inputPortLabel(
  kind: string,
  name: string,
  port: Port,
  stage: StageSpec,
  mIndex: ManifestIndex,
): string {
  if (kind === "retriever" && port.type === "Corpus") {
    return corpusArity(mIndex.component("retriever", name)) === "single"
      ? "Representation"
      : "Representation[]";
  }
  if (stage.many_in?.includes(port.type)) return `${port.type}[]`;
  return port.type;
}

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
  // A representation reads by its *space* (its name), with its type as the label
  // — so two `dense` reps are told apart by "dense" vs "dense-2" at a glance.
  const isRep = data.kind === "representations";

  // A self-managed representation (BM25) sprouts an optional BlobStore input for
  // its own persistence — wire a blob_store block into it to persist its index.
  if (repStoreSlot({ data }, mIndex)) {
    inputs.push({ dir: "in", type: "BlobStore", id: handleId("in", "BlobStore") });
  }

  return (
    <div
      className={`blocknode ${selected ? "selected" : ""} ${data.synthetic ? "synthetic" : ""}`}
      style={{ ["--stage" as string]: accent } as CSSProperties}
    >
      <div className="body">
        <div className="kind">{isRep ? data.name : data.kind}</div>
        <div className="name">{isRep ? repSpace({ data }) : data.name}</div>
      </div>

      {inputs.map((port, i) => (
        <TypedHandle
          key={port.id}
          port={port}
          dir="target"
          color={mIndex.typeColor(port.type)}
          top={pct(i, inputs.length)}
          label={stage ? inputPortLabel(data.kind, data.name, port, stage, mIndex) : port.type}
          highlight={pending ? (portIsCompatible(pending, port.type) ? "compatible" : "incompatible") : null}
        />
      ))}
      {output && (
        <TypedHandle
          port={output}
          dir="source"
          color={mIndex.typeColor(output.type)}
          top="50%"
          label={output.type}
          highlight={null}
          nodeId={id}
        />
      )}
    </div>
  );
}

export function TypedHandle(props: {
  port: Port;
  dir: "source" | "target";
  color: string;
  top: string;
  highlight: "compatible" | "incompatible" | null;
  label?: string;
  nodeId?: string;
}) {
  const { port, dir, color, top, highlight, label } = props;
  const side = dir === "target" ? Position.Left : Position.Right;
  const text = label ?? port.type;
  return (
    <>
      <Handle
        id={port.id}
        type={dir}
        position={side}
        className={highlight ?? ""}
        style={{ top, background: color }}
        title={text}
      />
      <span
        className={`port-label ${dir}`}
        // Sit just outside the card — to the left of an input, right of an
        // output — so the hint never overlaps the block's own content.
        style={{ top, [dir === "target" ? "right" : "left"]: "calc(100% + 12px)" }}
      >
        {text}
      </span>
    </>
  );
}

// Evenly space N handles down the card's height.
export function pct(i: number, n: number): string {
  return `${((i + 1) / (n + 1)) * 100}%`;
}
