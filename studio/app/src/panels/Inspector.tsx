import { useState } from "react";
import { useStudio } from "../graph/store";
import type { ManifestIndex } from "../manifest/load";
import type { BlockNode, SubRetriever } from "../graph/model";
import type { ComponentSpec, ParamSpec } from "../manifest/types";
import { corpusArity, wiredSpaces } from "../graph/corpus";

// The right drawer: configure the selected block, or read how it works. Both
// tabs are generated from the manifest — the form fields from each param's type,
// the info from the component's docstring — so a new component is configurable
// and documented with zero UI work.
export function Inspector() {
  const [tab, setTab] = useState<"config" | "info">("config");
  const selectedId = useStudio((s) => s.selectedId);
  const node = useStudio((s) => s.nodes.find((n) => n.id === s.selectedId));
  const edges = useStudio((s) => s.edges);
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
  if (node.data.kind === "corpus") {
    return (
      <div className="inspector">
        <div className="empty">
          The <b>Corpus</b> is wired from the representation blocks that feed it and the vector
          store that backs it — it has no params of its own.
        </div>
      </div>
    );
  }

  const comp = mIndex.component(node.data.kind, node.data.name);
  if (!comp) return <div className="inspector" />;

  // A retriever selects its representation(s) by wiring Corpus index ports, so
  // `spaces` is what's actually wired in — the pool its sub-retrievers pick from,
  // and the read-only list a base retriever shows instead of a param field.
  const isRetriever = node.data.kind === "retriever";
  const arity = isRetriever ? corpusArity(comp) : "pool";
  const spaces = isRetriever ? wiredSpaces(node.id, edges) : [];
  // A base retriever's representation param is wired, not typed — hide it. A
  // component that wraps an engine (docling's OCR) hides its raw name + config
  // params too; the EngineEditor renders them as a picker + sub-form.
  const hide = isRetriever && arity !== "pool" ? ["representation", "representations"] : [];
  if (comp.engine_slot) hide.push(comp.engine_slot.name_param, comp.engine_slot.config_param);

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
            spaces={spaces}
            hide={hide}
            mIndex={mIndex}
            onChange={(params) => updateParams(node.id, params)}
          />
          {isRetriever && arity !== "pool" && <WiredReps arity={arity} spaces={spaces} />}
          {comp.engine_slot && (
            <EngineEditor
              slot={comp.engine_slot}
              params={node.data.params}
              mIndex={mIndex}
              onChange={(params) => updateParams(node.id, params)}
            />
          )}
          {comp.encoder && (
            <EncoderEditor
              comp={comp}
              node={node}
              mIndex={mIndex}
              onPatch={(patch) => updateData(node.id, patch)}
            />
          )}
          {comp.composite && (
            <CompositeEditor
              comp={comp}
              node={node}
              mIndex={mIndex}
              spaces={spaces}
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

// A base retriever's representations, read-only: they come from the Corpus index
// ports wired into it, not a field here. Wire more indexes on the canvas to add
// them (a single `index` retriever takes exactly one).
function WiredReps({ arity, spaces }: { arity: "single" | "multi"; spaces: string[] }) {
  return (
    <div className="composite">
      <div className="composite-head">Representations</div>
      {spaces.length === 0 ? (
        <div className="hint">
          Wire a Corpus index into this retriever&rsquo;s <b>Corpus</b> port to choose{" "}
          {arity === "single" ? "its representation" : "which representations to fuse"}.
        </div>
      ) : (
        <ul className="wired-reps">
          {spaces.map((s) => (
            <li key={s}>
              <span className="wr-dot" />
              {s}
            </li>
          ))}
        </ul>
      )}
      {arity === "multi" && (
        <div className="hint">Wire more Corpus indexes to fuse them; all wired are used.</div>
      )}
    </div>
  );
}

// A component's wrapped engine picked here, not typed as a name string + raw
// JSON. The chosen engine's name lives in `name_param`, its config (a normal
// form over the engine's own params — secrets as password fields) in
// `config_param`. This is what surfaces the OCR engines in the Studio: the
// DoclingParser's `ocr_engine` becomes a dropdown of the installed OCR engines.
function EngineEditor({
  slot,
  params,
  mIndex,
  onChange,
}: {
  slot: { kind: string; name_param: string; config_param: string };
  params: Record<string, unknown>;
  mIndex: ManifestIndex;
  onChange: (params: Record<string, unknown>) => void;
}) {
  const options = (mIndex.componentsByKind.get(slot.kind) ?? []).filter((c) => c.exportable);
  const name = params[slot.name_param];
  const current = typeof name === "string" ? name : "";
  const cfg = (params[slot.config_param] as Record<string, unknown>) ?? {};
  const engComp = current ? mIndex.component(slot.kind, current) : undefined;

  const pick = (value: string) =>
    onChange({
      ...params,
      [slot.name_param]: value || null,
      [slot.config_param]: value ? mIndex.defaultParams(slot.kind, value) : {},
    });

  return (
    <div className="composite">
      <div className="composite-head">{slot.kind} engine</div>
      <div className="sub-retriever">
        <div className="sub-head">
          <select value={current} onChange={(e) => pick(e.target.value)}>
            <option value="">— built-in / none —</option>
            {options.map((o) => (
              <option key={o.name} value={o.name}>
                {o.name}
              </option>
            ))}
          </select>
        </div>
        {engComp && engComp.params.length > 0 && (
          <ConfigForm
            comp={engComp}
            params={cfg}
            mIndex={mIndex}
            onChange={(next) => onChange({ ...params, [slot.config_param]: next })}
          />
        )}
      </div>
    </div>
  );
}

// The `auto` parser's `routes` map — file format → parser — as a dropdown per
// format instead of a JSON blob. Rows come from the current value (falling back
// to the default routes), each picking from the real parsers (never `auto`).
function RouteTable({
  p,
  value,
  mIndex,
  onChange,
}: {
  p: ParamSpec;
  value: unknown;
  mIndex: ManifestIndex;
  onChange: (v: unknown) => void;
}) {
  const routes = (
    value && typeof value === "object" && !Array.isArray(value) ? value : p.default
  ) as Record<string, string>;
  const parsers = (mIndex.componentsByKind.get("parser") ?? []).filter(
    (c) => c.name !== "auto" && c.exportable,
  );
  return (
    <div className="field">
      <label>
        routes <span className="field-note">file format → parser</span>
      </label>
      <div className="routes">
        {Object.keys(routes).map((fmt) => (
          <div className="route-row" key={fmt}>
            <span className="route-key">{fmt}</span>
            <select value={routes[fmt] ?? ""} onChange={(e) => onChange({ ...routes, [fmt]: e.target.value })}>
              {parsers.map((pp) => (
                <option key={pp.name} value={pp.name}>
                  {pp.name}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
}

// The `auto` parser's `parser_configs` — per-parser config overrides — as a
// collapsible config sub-form for each parser its routes actually use, instead
// of nested JSON. A parser that wraps an engine (docling's OCR) gets its picker
// here too, so OCR is configurable *through* auto.
function ParserConfigs({
  value,
  routes,
  mIndex,
  onChange,
}: {
  value: unknown;
  routes: unknown;
  mIndex: ManifestIndex;
  onChange: (v: unknown) => void;
}) {
  const configs = (
    value && typeof value === "object" && !Array.isArray(value) ? value : {}
  ) as Record<string, Record<string, unknown>>;
  const routeMap = (routes && typeof routes === "object" ? routes : {}) as Record<string, string>;
  const used = [...new Set(Object.values(routeMap))].filter((n) => n && n !== "auto");
  if (!used.length) return null;
  return (
    <div className="field">
      <label>
        parser settings <span className="field-note">per routed parser</span>
      </label>
      {used.map((parser) => {
        const comp = mIndex.component("parser", parser);
        if (!comp || comp.params.length === 0) return null;
        return (
          <ParserConfigSection
            key={parser}
            comp={comp}
            params={configs[parser] ?? {}}
            mIndex={mIndex}
            onChange={(cfg) => onChange({ ...configs, [parser]: cfg })}
          />
        );
      })}
    </div>
  );
}

function ParserConfigSection({
  comp,
  params,
  mIndex,
  onChange,
}: {
  comp: ComponentSpec;
  params: Record<string, unknown>;
  mIndex: ManifestIndex;
  onChange: (cfg: Record<string, unknown>) => void;
}) {
  const [open, setOpen] = useState(false);
  const hide = comp.engine_slot ? [comp.engine_slot.name_param, comp.engine_slot.config_param] : [];
  return (
    <div className="composite">
      <button className={`composite-head toggle ${open ? "open" : ""}`} onClick={() => setOpen((o) => !o)}>
        <span className="chev" /> {comp.name}
      </button>
      {open && (
        <div className="sub-retriever">
          <ConfigForm comp={comp} params={params} hide={hide} mIndex={mIndex} onChange={onChange} />
          {comp.engine_slot && (
            <EngineEditor slot={comp.engine_slot} params={params} mIndex={mIndex} onChange={onChange} />
          )}
        </div>
      )}
    </div>
  );
}

// A representation's wrapped encoder (the embedder/index it mounts), configured
// here rather than as a separate graph node — the encoder has no meaning apart
// from the representation that owns it, so it nests in the inspector (DR-0004 D7).
function EncoderEditor({
  comp,
  node,
  mIndex,
  onPatch,
}: {
  comp: ComponentSpec;
  node: BlockNode;
  mIndex: ManifestIndex;
  onPatch: (patch: { encoder?: SubRetriever }) => void;
}) {
  const slot = comp.encoder!;
  const options = mIndex.encoderChoices(slot.kind);
  const value = node.data.encoder;
  const encComp = value ? mIndex.component(slot.kind, value.name) : undefined;
  if (options.length === 0) {
    const family = slot.kind.replace(/_/g, " ");
    return (
      <div className="composite">
        <div className="composite-head">{slot.kind}</div>
        <div className="hint">
          No {family} is installed in this build yet, so <b>{comp.name}</b> can&rsquo;t be built.
          {slot.kind === "sparse_encoder" && (
            <> Use <b>lexical</b> (bm25) for keyword retrieval in the meantime.</>
          )}
        </div>
      </div>
    );
  }
  return (
    <div className="composite">
      <div className="composite-head">{slot.kind}</div>
      <div className="sub-retriever">
        <div className="sub-head">
          <select
            value={value?.name ?? ""}
            onChange={(e) =>
              onPatch({ encoder: { name: e.target.value, params: mIndex.defaultParams(slot.kind, e.target.value) } })
            }
          >
            {!value && <option value="">— pick a {slot.kind} —</option>}
            {options.map((o) => (
              <option key={o.name} value={o.name}>
                {o.name}
              </option>
            ))}
          </select>
        </div>
        {value && encComp && encComp.params.length > 0 && (
          <ConfigForm
            comp={encComp}
            params={value.params}
            mIndex={mIndex}
            onChange={(params) => onPatch({ encoder: { name: value.name, params } })}
          />
        )}
        {encComp?.store_slot && (
          <div className="hint">
            Keeps its own index. Wire a <b>BlobStore</b> into this block&rsquo;s BlobStore port
            to persist it — otherwise it runs in-memory (rebuilt each run).
          </div>
        )}
      </div>
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
  spaces,
  onPatch,
}: {
  comp: ComponentSpec;
  node: BlockNode;
  mIndex: ManifestIndex;
  spaces: string[];
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
          spaces={spaces}
          onChange={(sub) => onPatch({ inner: sub })}
        />
      )}
      {comp.composite === "retrievers" && (
        <RetrieverList
          value={node.data.retrievers ?? []}
          bases={bases}
          mIndex={mIndex}
          spaces={spaces}
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
  spaces,
  onChange,
}: {
  value: SubRetriever[];
  bases: ComponentSpec[];
  mIndex: ManifestIndex;
  spaces: string[];
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
          spaces={spaces}
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
  spaces,
  onChange,
  onRemove,
}: {
  value: SubRetriever;
  bases: ComponentSpec[];
  mIndex: ManifestIndex;
  spaces: string[];
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
        <ConfigForm comp={comp} params={value.params} spaces={spaces} mIndex={mIndex} onChange={(params) => onChange({ name: value.name, params })} />
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
  spaces = [],
  hide = [],
  mIndex,
  onChange,
}: {
  comp: ComponentSpec;
  params: Record<string, unknown>;
  spaces?: string[];
  hide?: string[];
  mIndex: ManifestIndex;
  onChange: (p: Record<string, unknown>) => void;
}) {
  if (!comp.params.length) {
    return <div className="empty">No parameters.</div>;
  }
  const visible = comp.params.filter((p) => !hide.includes(p.name));
  if (!visible.length) return null; // e.g. an `index` retriever, wired not typed
  const set = (name: string, value: unknown) => onChange({ ...params, [name]: value });
  return (
    <div>
      {visible.map((p) => {
        // A dispatcher map (auto's routes / parser_configs) is a structured form,
        // not raw JSON.
        if (p.map_value_kind)
          return <RouteTable key={p.name} p={p} value={params[p.name]} mIndex={mIndex} onChange={(v) => set(p.name, v)} />;
        if (p.config_map_kind)
          return <ParserConfigs key={p.name} value={params[p.name]} routes={params.routes} mIndex={mIndex} onChange={(v) => set(p.name, v)} />;
        return <Field key={p.name} p={p} value={params[p.name]} spaces={spaces} onChange={(v) => set(p.name, v)} />;
      })}
    </div>
  );
}

function Field({ p, value, spaces, onChange }: { p: ParamSpec; value: unknown; spaces: string[]; onChange: (v: unknown) => void }) {
  return (
    <div className="field">
      <label>
        {p.name}
        {p.required ? " *" : ""} {p.secret ? "🔒" : ""}
      </label>
      {p.secret ? (
        // The Studio only exports a spec, and secrets never enter one (§7.4), so
        // there is nothing to type here — the value comes from the environment
        // at run time. Show that instead of a dead input.
        <div className="secret-note">From the environment at run time — never entered here or saved to the spec.</div>
      ) : (
        <FieldInput p={p} value={value} spaces={spaces} onChange={onChange} />
      )}
    </div>
  );
}

function FieldInput({ p, value, spaces, onChange }: { p: ParamSpec; value: unknown; spaces: string[]; onChange: (v: unknown) => void }) {
  // A sub-retriever's representation selector, driven by the spaces wired into
  // the composite's Corpus port — a pick-list, never a typed-in magic string.
  // Falls through to the plain widgets when no index is wired yet.
  if (spaces.length && p.name === "representation") {
    return <RepresentationSelect spaces={spaces} value={value} onChange={onChange} />;
  }
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

// One representation, picked from the corpus's wired spaces.
function RepresentationSelect({
  spaces,
  value,
  onChange,
}: {
  spaces: string[];
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const current = typeof value === "string" ? value : "";
  return (
    <select value={current} onChange={(e) => onChange(e.target.value || null)}>
      {!current && <option value="">— pick a representation —</option>}
      {spaces.map((s) => (
        <option key={s} value={s}>
          {s}
        </option>
      ))}
      {current && !spaces.includes(current) && (
        <option value={current}>{current} (not connected)</option>
      )}
    </select>
  );
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
