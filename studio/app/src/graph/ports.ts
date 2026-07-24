import type { StageSpec } from "../manifest/types";

// One handle per typed port. The handle id carries the contract type so the
// connection validator can compare types without any extra lookup:
//   target (input)  ->  "in:Document"
//   source (output) ->  "out:Chunk[]"
// A handle may also carry a *space* after a "#", to distinguish otherwise
// identical ports on one node — the Corpus emits one "out:Corpus#<space>" port
// per representation it holds, so wiring a specific index to a retriever is
// what selects that representation:
//   source (index) ->  "out:Corpus#dense"
export type PortDir = "in" | "out";

export interface Port {
  dir: PortDir;
  type: string;
  /** The representation space, for a Corpus index port. */
  space?: string;
  id: string;
}

export function handleId(dir: PortDir, type: string, space?: string): string {
  return space ? `${dir}:${type}#${space}` : `${dir}:${type}`;
}

export function parseHandle(id: string | null | undefined): Port | null {
  if (!id) return null;
  const hash = id.indexOf("#");
  const space = hash >= 0 ? id.slice(hash + 1) : undefined;
  const core = hash >= 0 ? id.slice(0, hash) : id;
  const i = core.indexOf(":");
  if (i < 0) return null;
  const dir = core.slice(0, i) as PortDir;
  const type = core.slice(i + 1);
  if (dir !== "in" && dir !== "out") return null;
  return { dir, type, space, id };
}

/** Input ports of a node's stage (Query/Source endpoints included — they just
 *  stay unconnected, supplied at runtime). */
export function inputPorts(stage: StageSpec): Port[] {
  return stage.in.map((type) => ({ dir: "in", type, id: handleId("in", type) }));
}

export function outputPort(stage: StageSpec): Port {
  return { dir: "out", type: stage.out, id: handleId("out", stage.out) };
}
