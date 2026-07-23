import type { Manifest } from "../manifest/types";

// A TypeScript mirror of rag_blocks.evaluation.builder.validate_spec — the same
// structural gate, so the app rejects a spec the Python side would reject,
// before it ever writes or reads a file. Structure only: an unknown component
// name or a bad param stays the builder's to catch (we don't have the registry
// here). Kept deliberately close to the Python messages.

export function validateSpec(spec: unknown, manifest: Manifest): string[] {
  const errors: string[] = [];
  if (spec === null || typeof spec !== "object" || Array.isArray(spec)) {
    return [`spec must be a mapping of stage -> entry, got ${jstype(spec)}`];
  }
  const known = new Set(manifest.stages.filter((s) => !s.synthetic).map((s) => s.kind));
  const chains = new Set(manifest.stages.filter((s) => s.chain).map((s) => s.kind));

  const obj = spec as Record<string, unknown>;
  const unknown = Object.keys(obj).filter((k) => !known.has(k));
  if (unknown.length) errors.push(`unknown stage(s) ${JSON.stringify(unknown.sort())}`);

  for (const [stage, value] of Object.entries(obj)) {
    if (!known.has(stage)) continue;
    if (chains.has(stage)) {
      if (!Array.isArray(value)) {
        errors.push(`${stage}= must be a chain (a list of entries, [] for none), got ${jstype(value)}`);
        continue;
      }
      value.forEach((entry) => validateEntry(stage, entry, errors));
    } else {
      validateEntry(stage, value, errors);
    }
  }
  return errors;
}

function validateEntry(stage: string, entry: unknown, errors: string[]): void {
  if (entry === null || typeof entry !== "object" || Array.isArray(entry) || !("name" in entry)) {
    errors.push(`${stage} entry must be {"name": ..., "params": {...}}`);
    return;
  }
  const params = (entry as Record<string, unknown>).params;
  if (params !== undefined && (typeof params !== "object" || params === null || Array.isArray(params))) {
    errors.push(`${stage} params must be a mapping`);
  }
}

function jstype(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  return typeof v;
}
