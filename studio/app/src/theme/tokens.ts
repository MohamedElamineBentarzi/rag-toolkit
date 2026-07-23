// n8n-ish dark palette. Kept as one small token set so the whole app reads from
// a single place; the per-type port colors live in the manifest (theme of the
// data, not the chrome).
export const theme = {
  bg: "#14141c",
  grid: "#2a2a3a",
  surface: "#1d1d29",
  surfaceRaised: "#24243422",
  panel: "#191922",
  border: "#33334a",
  borderStrong: "#454563",
  text: "#e6e6f0",
  textDim: "#9a9ab0",
  accent: "#ff6d5a", // n8n coral
  accentDim: "#ff6d5a55",
  good: "#43b581",
  bad: "#e0525b",
  warn: "#e0a458",
} as const;

// Stage accent bar colors — one per spec stage, so a block's role is legible at
// a glance even before you read its title.
export const stageAccent: Record<string, string> = {
  parser: "#4f9dde",
  chunker: "#43b581",
  enrich: "#3fae9f",
  embedder: "#c586f0",
  sparse: "#a06cf0",
  lexical: "#e0a458",
  index: "#e0a458",
  retriever: "#e06c9f",
  refine: "#e0868f",
  generator: "#d4d44a",
};
