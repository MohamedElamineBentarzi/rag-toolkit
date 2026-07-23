import { useState } from "react";
import { useStudio } from "../graph/store";
import type { Problem } from "../graph/model";

// The pipeline's validity, lifted out of the canvas onto the top bar: a colored
// status pill (green valid / amber warnings / red errors) that opens a popover
// spelling out what's wrong and how to fix it. This is the single place to look
// before exporting.
export function StatusPill() {
  const problems = useStudio((s) => s.problems);
  const hasBlocks = useStudio((s) => s.nodes.some((n) => n.data.kind !== "endpoint"));
  const [open, setOpen] = useState(false);

  const errors = problems.filter((p) => p.level === "error");
  const warns = problems.filter((p) => p.level === "warn");
  const state: "empty" | "error" | "warn" | "ok" = !hasBlocks
    ? "empty"
    : errors.length
      ? "error"
      : warns.length
        ? "warn"
        : "ok";

  const label = {
    empty: "Empty",
    error: `${errors.length} ${errors.length === 1 ? "issue" : "issues"}`,
    warn: `${warns.length} ${warns.length === 1 ? "warning" : "warnings"}`,
    ok: "Valid",
  }[state];

  return (
    <div className="status-wrap">
      <button
        className={`status ${state}`}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        title="Pipeline status"
      >
        <span className="status-dot" />
        <span>{label}</span>
      </button>
      {open && (
        <>
          <div className="status-backdrop" onClick={() => setOpen(false)} />
          <StatusPopover state={state} problems={problems} />
        </>
      )}
    </div>
  );
}

function StatusPopover({
  state,
  problems,
}: {
  state: "empty" | "error" | "warn" | "ok";
  problems: Problem[];
}) {
  return (
    <div className="status-popover glass" role="dialog">
      <h5>Pipeline status</h5>
      {state === "empty" && (
        <div className="status-msg">Drag blocks from the left panel to start building.</div>
      )}
      {state === "ok" && (
        <div className="status-msg good">✓ Everything checks out — ready to export.</div>
      )}
      {(state === "error" || state === "warn") &&
        problems.map((p, i) => (
          <div key={i} className={`row ${p.level}`}>
            <span className="tag">{p.level}</span>
            <span>{p.message}</span>
          </div>
        ))}
    </div>
  );
}
