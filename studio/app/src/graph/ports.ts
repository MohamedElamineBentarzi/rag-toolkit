import type { StageSpec } from "../manifest/types";

// One handle per typed port. The handle id carries the contract type so the
// connection validator can compare types without any extra lookup:
//   target (input)  ->  "in:Document"
//   source (output) ->  "out:Chunk[]"
export type PortDir = "in" | "out";

export interface Port {
  dir: PortDir;
  type: string;
  id: string;
}

export function handleId(dir: PortDir, type: string): string {
  return `${dir}:${type}`;
}

export function parseHandle(id: string | null | undefined): Port | null {
  if (!id) return null;
  const i = id.indexOf(":");
  if (i < 0) return null;
  const dir = id.slice(0, i) as PortDir;
  const type = id.slice(i + 1);
  if (dir !== "in" && dir !== "out") return null;
  return { dir, type, id };
}

/** Input ports of a node's stage (Query/Source endpoints included — they just
 *  stay unconnected, supplied at runtime). */
export function inputPorts(stage: StageSpec): Port[] {
  return stage.in.map((type) => ({ dir: "in", type, id: handleId("in", type) }));
}

export function outputPort(stage: StageSpec): Port {
  return { dir: "out", type: stage.out, id: handleId("out", stage.out) };
}

/** The one target handle that legitimately accepts many edges: representations
 *  fan into the ChunkIndex. Everything else is one-in. */
export function acceptsMany(kind: string, port: Port): boolean {
  return kind === "index" && port.type === "Representation";
}
