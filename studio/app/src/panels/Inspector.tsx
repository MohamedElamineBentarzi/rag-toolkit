import { useState } from "react";
import { useStudio } from "../graph/store";
import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, SubRetriever } from "../graph/model";
import type { ComponentSpec, ParamSpec } from "../manifest/types";

// The right drawer: configure the selected block, or read how it works. Both
// tabs are generated from the manifest — the form fields from each param's type,
// the info from the component's docstring — so a new component is configurable
// and documented with zero UI work.
export function Inspector() {
  const [tab, setTab] = useState<"config" | "info">("config");
  const selectedId = useStudio((s) => s.selectedId);
  const node = useStudio((s) => s.nodes.find((n) => n.id === s.selectedId));
  const mIndex = useStudio((s) => s.mIndex);
  const updateParams = useStudio((s) => s.updateParams);
  const updateData = useStudio((s) => s.updateData);

  if (!node || !mIndex) {
    return (
      <div className="inspector">
        <div className="empty">Select a block to configure it.</div>
      </div>
    );
  }
  if (node.data.kind === "endpoint") {
    return (
      <div className="inspector">
        <div className="empty">
          <b>{String(node.data.label)}</b> is a pipeline endpoint — where{" "}
          {node.data.epDir === "in" ? "the answer comes out" : "data goes in"}. It's always on
          the canvas and isn't part of the exported spec.
        </div>
      </div>
    );
  }
  if (node.data.kind === "index") {
    return (
      <div className="inspector">
        <div className="empty">
          The <b>ChunkIndex</b> is wired from the representation blocks that feed it — it has
          no params of its own.
        </div>
      </div>
    );
  }

  const comp = mIndex.component(node.data.kind, node.data.name);
  if (!comp) return <div className="inspector" />;

  return (
    <div className="inspector">
      <div className="tabs">
        <button className={tab === "config" ? "active" : ""} onClick={() => setTab("config")}>
          Configure
        </button>
        <button className={tab === "info" ? "active" : ""} onClick={() => setTab("info")}>
          Info
        </button>
      </div>
      <div className="head">
        <div className="cname">{comp.name}</div>
        <div className="meta">
          {comp.kind} · v{comp.version}
        </div>
      </div>

      {tab === "config" ? (
        <>
          <ConfigForm
            key={selectedId ?? ""}
            comp={comp}
            params={node.data.params}
            onChange={(params) => updateParams(node.id, params)}
          />
          {comp.composite && (
            <CompositeEditor
              comp={comp}
              node={node}
              mIndex={mIndex}
              onPatch={(patch) => updateData(node.id, patch)}
            />
          )}
        </>
      ) : (
        <InfoView comp={comp} />
      )}
    </div>
  );
}

// A composite retriever's nested sub-retriever(s), configured here rather than
// as separate graph nodes — so the canvas keeps its clean single-output
// retriever ports (progressive disclosure; the nesting is the rare case).
function CompositeEditor({
  comp,
  node,
  mIndex,
  onPatch,
}: {
  comp: ComponentSpec;
  node: BlockNode;
  mIndex: ManifestIndex;
  onPatch: (patch: { inner?: SubRetriever; retrievers?: SubRetriever[] }) => void;
}) {
  const bases = mIndex.baseRetrievers();
  return (
    <div className="composite">
      <div className="composite-head">
        {comp.composite === "inner" ? "Inner retriever" : "Fused retrievers"}
      </div>
      {comp.needs_llm && (
        <div className="hint llm">
          Shapes the query with the pipeline&rsquo;s generator LLM — add an LLM generator
          (e.g. anthropic) for this to run.
        </div>
      )}
      {comp.composite === "inner" && (
        <SubRetrieverEditor
          value={node.data.inner ?? defaultSub(mIndex)}
          bases={bases}
          mIndex={mIndex}
          onChange={(sub) => onPatch({ inner: sub })}
        />
      )}
      {comp.composite === "retrievers" && (
        <RetrieverList
          value={node.data.retrievers ?? []}
          bases={bases}
          mIndex={mIndex}
          onChange={(list) => onPatch({ retrievers: list })}
        />
      )}
    </div>
  );
}

function RetrieverList({
  value,
  bases,
  mIndex,
  onChange,
}: {
  value: SubRetriever[];
  bases: ComponentSpec[];
  mIndex: ManifestIndex;
  onChange: (list: SubRetriever[]) => void;
}) {
  return (
    <div>
      {value.map((sub, i) => (
        <SubRetrieverEditor
          key={i}
          value={sub}
          bases={bases}
          mIndex={mIndex}
          onChange={(s) => onChange(value.map((v, j) => (j === i ? s : v)))}
          onRemove={() => onChange(value.filter((_, j) => j !== i))}
        />
      ))}
      {value.length === 0 && <div className="hint">Add at least one retriever to fuse.</div>}
      <button
        className="sub-add"
        onClick={() => onChange([...value, defaultSub(mIndex)])}
      >
        + add retriever
      </button>
    </div>
  );
}

function SubRetrieverEditor({
  value,
  bases,
  mIndex,
  onChange,
  onRemove,
}: {
  value: SubRetriever;
  bases: ComponentSpec[];
  mIndex: ManifestIndex;
  onChange: (sub: SubRetriever) => void;
  onRemove?: () => void;
}) {
  const comp = mIndex.component("retriever", value.name);
  return (
    <div className="sub-retriever">
      <div className="sub-head">
        <select
          value={value.name}
          onChange={(e) =>
            onChange({ name: e.target.value, params: mIndex.defaultParams("retriever", e.target.value) })
          }
        >
          {bases.map((b) => (
            <option key={b.name} value={b.name}>
              {b.name}
            </option>
          ))}
        </select>
        {onRemove && (
          <button className="sub-remove" onClick={onRemove} aria-label="Remove">
            ×
          </button>
        )}
      </div>
      {comp && comp.params.length > 0 && (
        <ConfigForm comp={comp} params={value.params} onChange={(params) => onChange({ name: value.name, params })} />
      )}
    </div>
  );
}

function defaultSub(mIndex: ManifestIndex): SubRetriever {
  const base = mIndex.baseRetrievers()[0];
  return base
    ? { name: base.name, params: mIndex.defaultParams("retriever", base.name) }
    : { name: "", params: {} };
}

function ConfigForm({
  comp,
  params,
  onChange,
}: {
  comp: ComponentSpec;
  params: Record<string, unknown>;
  onChange: (p: Record<string, unknown>) => void;
}) {
  if (!comp.params.length) {
    return <div className="empty">No parameters.</div>;
  }
  const set = (name: string, value: unknown) => onChange({ ...params, [name]: value });
  return (
    <div>
      {comp.params.map((p) => (
        <Field key={p.name} p={p} value={params[p.name]} onChange={(v) => set(p.name, v)} />
      ))}
    </div>
  );
}

function Field({ p, value, onChange }: { p: ParamSpec; value: unknown; onChange: (v: unknown) => void }) {
  return (
    <div className="field">
      <label>
        {p.name}
        {p.required ? " *" : ""} {p.secret ? "🔒" : ""}
      </label>
      <FieldInput p={p} value={value} onChange={onChange} />
      {p.secret && <div className="hint">Secret — set via environment; never saved to the spec.</div>}
    </div>
  );
}

function FieldInput({ p, value, onChange }: { p: ParamSpec; value: unknown; onChange: (v: unknown) => void }) {
  switch (p.type) {
    case "bool":
      return <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />;
    case "int":
      return (
        <input
          type="number"
          step={1}
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => onChange(e.target.value === "" ? null : parseInt(e.target.value, 10))}
        />
      );
    case "float":
      return (
        <input
          type="number"
          step="any"
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => onChange(e.target.value === "" ? null : parseFloat(e.target.value))}
        />
      );
    case "enum":
      return (
        <select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
          {(p.choices ?? []).map((c) => (
            <option key={String(c)} value={String(c)}>
              {String(c)}
            </option>
          ))}
        </select>
      );
    case "json":
      return <JsonInput value={value} onChange={onChange} />;
    default:
      return (
        <input
          type={p.secret ? "password" : "text"}
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
        />
      );
  }
}

// A JSON param (dict/list) edited as text; invalid JSON is flagged but not
// silently dropped.
function JsonInput({ value, onChange }: { value: unknown; onChange: (v: unknown) => void }) {
  const [text, setText] = useState(() => JSON.stringify(value ?? null));
  const [bad, setBad] = useState(false);
  return (
    <>
      <textarea
        rows={3}
        value={text}
        style={bad ? { borderColor: "var(--bad)" } : undefined}
        onChange={(e) => {
          setText(e.target.value);
          try {
            onChange(JSON.parse(e.target.value));
            setBad(false);
          } catch {
            setBad(true);
          }
        }}
      />
      {bad && <div className="hint" style={{ color: "var(--bad)" }}>Invalid JSON</div>}
    </>
  );
}

function InfoView({ comp }: { comp: ComponentSpec }) {
  return (
    <div>
      <table className="paramtable">
        <tbody>
          {comp.params.map((p) => (
            <tr key={p.name}>
              <td className="k">{p.name}</td>
              <td className="t">{p.type}</td>
              <td className="t">{p.choices ? p.choices.join(" | ") : fmt(p.default)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!comp.exportable && (
        <div className="hint" style={{ color: "var(--warn)", marginTop: 8 }}>
          Not exportable: {comp.not_exportable_reason}
        </div>
      )}
      <div className="doc">{comp.doc || "No documentation."}</div>
    </div>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}
