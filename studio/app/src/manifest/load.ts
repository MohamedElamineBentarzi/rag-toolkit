import type { Manifest, StageSpec, ComponentSpec } from "./types";

// Fetch the generated manifest from public/. Static: no server, just a file.
export async function loadManifest(): Promise<Manifest> {
  const res = await fetch("blocks.json");
  if (!res.ok) {
    throw new Error(
      `Could not load blocks.json (${res.status}). Generate it first: ` +
        `python studio/tools/build_manifest.py`,
    );
  }
  return (await res.json()) as Manifest;
}

/** Index helpers over a manifest, so lookups read clearly at call sites. */
export class ManifestIndex {
  readonly stagesByKind: Map<string, StageSpec>;
  readonly componentsByKind: Map<string, ComponentSpec[]>;

  constructor(readonly manifest: Manifest) {
    this.stagesByKind = new Map(manifest.stages.map((s) => [s.kind, s]));
    this.componentsByKind = new Map();
    for (const c of manifest.components) {
      const list = this.componentsByKind.get(c.kind) ?? [];
      list.push(c);
      this.componentsByKind.set(c.kind, list);
    }
  }

  stage(kind: string): StageSpec | undefined {
    return this.stagesByKind.get(kind);
  }

  component(kind: string, name: string): ComponentSpec | undefined {
    return this.componentsByKind.get(kind)?.find((c) => c.name === name);
  }

  typeColor(type: string): string {
    return this.manifest.types[type]?.color ?? "#8b8b9e";
  }

  /** Base (non-composite) retrievers a composite can wrap. */
  baseRetrievers(): ComponentSpec[] {
    return (this.componentsByKind.get("retriever") ?? []).filter(
      (c) => c.exportable && !c.composite,
    );
  }

  /** A component's params filled with their defaults. */
  defaultParams(kind: string, name: string): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const p of this.component(kind, name)?.params ?? []) out[p.name] = p.default;
    return out;
  }
}
