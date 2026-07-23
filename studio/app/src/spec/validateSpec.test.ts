import { describe, it, expect } from "vitest";
import { validateSpec } from "./validateSpec";
import type { Manifest } from "../manifest/types";
import manifestJson from "../../public/blocks.json";

const manifest = manifestJson as unknown as Manifest;

// Mirrors tests/evaluation/test_spec_io.py's validate_spec cases — the TS gate
// must reject exactly what the Python one rejects.
describe("validateSpec", () => {
  it("accepts a well-formed spec", () => {
    const spec = { chunker: { name: "fixed", params: { chunk_chars: 200 } } };
    expect(validateSpec(spec, manifest)).toEqual([]);
  });

  it("rejects an unknown stage", () => {
    expect(validateSpec({ chunkerr: { name: "fixed" } }, manifest).join()).toMatch(/unknown stage/);
  });

  it("rejects a malformed entry", () => {
    expect(validateSpec({ chunker: "fixed" }, manifest).join()).toMatch(/"name"/);
  });

  it("rejects a bare chain entry", () => {
    expect(validateSpec({ refine: { name: "keyword" } }, manifest).join()).toMatch(/must be a chain/);
  });

  it("rejects non-object params", () => {
    expect(validateSpec({ chunker: { name: "fixed", params: [1] } }, manifest).join()).toMatch(/params must be a mapping/);
  });

  it("rejects a non-object spec", () => {
    expect(validateSpec(["fixed"], manifest).join()).toMatch(/mapping/);
  });
});
