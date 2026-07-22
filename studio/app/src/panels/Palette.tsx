import type { CSSProperties } from "react";
import { useStudio } from "../graph/store";
import { stageAccent } from "../theme/tokens";
import type { ComponentSpec } from "../manifest/types";

// The block library, grouped by stage in pipeline order. Drag a block onto the
// canvas (or click it). Non-exportable blocks (composites a flat spec can't
// carry) are shown disabled with the reason, so the palette never hides what
// exists — it explains what won't export.
export function Palette() {
  const manifest = useStudio((s) => s.manifest);
  const mIndex = useStudio((s) => s.mIndex);
  const addNode = useStudio((s) => s.addNode);
  if (!manifest || !mIndex) return <div className="palette" />;

  return (
    <div className="palette">
      {manifest.stages
        .filter((s) => !s.synthetic)
        .map((stage) => {
          const comps = mIndex.componentsByKind.get(stage.kind) ?? [];
          if (!comps.length) return null;
          return (
            <div key={stage.kind}>
              <h4>{stage.kind}</h4>
              {comps.map((c) => (
                <PaletteBlock key={c.name} comp={c} onAdd={() => c.exportable && addNode(c.kind, c.name)} />
              ))}
            </div>
          );
        })}
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
