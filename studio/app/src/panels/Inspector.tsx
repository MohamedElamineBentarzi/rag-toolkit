import { useState } from "react";
import { useStudio } from "../graph/store";
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

  if (!node || !mIndex) {
    return (
      <div className="inspector">
        <div className="empty">Select a block to configure it.</div>
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
        <ConfigForm
          key={selectedId ?? ""}
          comp={comp}
          params={node.data.params}
          onChange={(params) => updateParams(node.id, params)}
        />
      ) : (
        <InfoView comp={comp} />
      )}
    </div>
  );
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
