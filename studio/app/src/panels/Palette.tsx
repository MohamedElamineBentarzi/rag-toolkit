import { useState } from "react";
import type { CSSProperties } from "react";
import { useStudio } from "../graph/store";
import { stageAccent } from "../theme/tokens";
import type { ComponentSpec } from "../manifest/types";

// The block library as collapsible sections, one per stage, plus a search box —
// so it stays compact (no long scroll) and a block is a click or a type away.
// Sections default collapsed except the first; searching flattens + reveals
// every match.
export function Palette() {
  const manifest = useStudio((s) => s.manifest);
  const mIndex = useStudio((s) => s.mIndex);
  const addNode = useStudio((s) => s.addNode);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({});
  if (!manifest || !mIndex) return <div className="palette" />;

  const q = query.trim().toLowerCase();
  const stages = manifest.stages.filter(
    (s) => !s.synthetic && (mIndex.componentsByKind.get(s.kind)?.length ?? 0) > 0,
  );

  const sections = stages
    .map((stage, idx) => {
      const comps = (mIndex.componentsByKind.get(stage.kind) ?? []).filter(
        (c) => !q || c.name.toLowerCase().includes(q),
      );
      return { kind: stage.kind, comps, isOpen: q ? true : (open[stage.kind] ?? idx === 0) };
    })
    .filter((s) => s.comps.length > 0);

  return (
    <div className="palette">
      <div className="palette-search">
        <SearchIcon />
        <input
          placeholder="Search blocks…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {query && (
          <button className="clear" onClick={() => setQuery("")} aria-label="Clear search">
            ×
          </button>
        )}
      </div>

      <div className="palette-groups">
        {sections.length === 0 && <div className="palette-empty">No blocks match “{query}”.</div>}
        {sections.map(({ kind, comps, isOpen }) => (
          <div className="group" key={kind}>
            <button
              className={`group-header ${isOpen ? "open" : ""}`}
              style={{ ["--stage" as string]: stageAccent[kind] ?? "#8b8b9e" } as CSSProperties}
              onClick={() => setOpen((o) => ({ ...o, [kind]: !isOpen }))}
            >
              <span className="chev" />
              <span className="gdot" />
              <span className="gname">{kind}</span>
              <span className="gcount">{comps.length}</span>
            </button>
            {isOpen && (
              <div className="group-body">
                {comps.map((c) => (
                  <PaletteBlock
                    key={c.name}
                    comp={c}
                    onAdd={() => c.exportable && addNode(c.kind, c.name)}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function PaletteBlock({ comp, onAdd }: { comp: ComponentSpec; onAdd: () => void }) {
  const disabled = !comp.exportable;
  return (
    <div
      className={`block ${disabled ? "disabled" : ""}`}
      style={{ ["--stage" as string]: stageAccent[comp.kind] ?? "#8b8b9e" } as CSSProperties}
      draggable={!disabled}
      onClick={disabled ? undefined : onAdd}
      onDragStart={(e) => {
        e.dataTransfer.setData("application/rag-block", JSON.stringify({ kind: comp.kind, name: comp.name }));
        e.dataTransfer.effectAllowed = "move";
      }}
      title={disabled ? comp.not_exportable_reason : comp.doc.split("\n")[0]}
    >
      <span className="name">{comp.name}</span>
    </div>
  );
}

function SearchIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="7" cy="7" r="4.5" />
      <line x1="10.5" y1="10.5" x2="14" y2="14" />
    </svg>
  );
}
