import type { CSSProperties } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useStudio } from "./store";
import { inputPorts } from "./ports";
import { portIsCompatible } from "./validate";
import { corpusOutHandle, corpusSpaces } from "./corpus";
import { TypedHandle, pct, inputPortLabel } from "./BlockNode";
import type { BlockNode as BlockNodeType } from "./model";
import { stageAccent } from "../theme/tokens";

// The Corpus, rendered specially: representations fan into it on the left, and
// it exposes one labelled *index* output per representation it holds on the
// right — a list, not a single opaque port. Wiring a specific index to a
// retriever is what selects that representation, so the ports carry the meaning
// and their labels are always shown (not hover-only like ordinary ports).
export function CorpusNode({ id, selected }: NodeProps<BlockNodeType>) {
  const mIndex = useStudio((s) => s.mIndex);
  const pending = useStudio((s) => s.pendingSourceType);
  const nodes = useStudio((s) => s.nodes);
  const edges = useStudio((s) => s.edges);
  if (!mIndex) return null;

  const stage = mIndex.stage("corpus");
  const inputs = stage ? inputPorts(stage) : [];
  const spaces = corpusSpaces(id, nodes, edges);
  const corpusColor = mIndex.typeColor("Corpus");
  const repColor = stageAccent["representations"] ?? corpusColor;
  const accent = stageAccent["corpus"] ?? "#8b8b9e";

  return (
    <div
      className={`blocknode corpus ${selected ? "selected" : ""}`}
      style={{ ["--stage" as string]: accent } as CSSProperties}
    >
      <div className="body">
        <div className="kind">corpus</div>
        <div className="name">Corpus</div>
      </div>

      {inputs.map((port, i) => (
        <TypedHandle
          key={port.id}
          port={port}
          dir="target"
          color={mIndex.typeColor(port.type)}
          top={pct(i, inputs.length)}
          label={stage ? inputPortLabel("corpus", "corpus", port, stage, mIndex) : port.type}
          highlight={pending ? (portIsCompatible(pending, port.type) ? "compatible" : "incompatible") : null}
        />
      ))}

      <div className="corpus-indexes">
        {spaces.length === 0 ? (
          <div className="corpus-empty">connect representations →</div>
        ) : (
          <div className="corpus-cap">representations</div>
        )}
        {spaces.map((space) => (
          <div className="corpus-index" key={space}>
            <span className="ci-dot" style={{ background: repColor }} />
            <span className="ci-name">{space}</span>
            <Handle
              id={corpusOutHandle(space)}
              type="source"
              position={Position.Right}
              style={{ top: "50%", background: repColor }}
              title={`${space} — a representation held in the corpus; wire it into a retriever`}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
