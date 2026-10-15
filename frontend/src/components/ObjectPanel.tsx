import type { Fix, ObjectView } from "../types";

interface Props {
  objects: ObjectView[];
  selected: string | null;
  busy: boolean;
  onSelect: (name: string) => void;
  onReview: (name: string, decision: "approve" | "reject") => void;
  onFix: (name: string, fixId: string) => void;
  applied: Fix[];
}

export function statusClass(o: ObjectView): string {
  if (o.decision === "approved") return "approved";
  if (o.decision === "rejected") return "rejected";
  if (o.status === "checking") return "checking";
  if (o.findings.some((f) => f.severity === "error")) return "error";
  if (o.findings.length) return "warning";
  return o.status === "ok" ? "ok" : "pending";
}

export function ObjectPanel({ objects, selected, busy, onSelect, onReview, onFix, applied }: Props) {
  const current = objects.find((o) => o.name === selected) ?? null;
  return (
    <aside className="panel">
      <ul className="object-list">
        {objects.map((o) => (
          <li key={o.name} className={`${statusClass(o)} ${o.name === selected ? "selected" : ""}`}>
            <button onClick={() => onSelect(o.name)}>
              <span className="dot" />
              {o.name}
              <span className="count">{o.findings.length ? o.findings.length : ""}</span>
            </button>
          </li>
        ))}
      </ul>
      {current && (
        <div className="details">
          <h2>{current.name}</h2>
          <p className="meta">
            {current.movable ? "movable" : "fixed"} · {current.status} · {current.decision}
          </p>
          {current.findings.length === 0 && current.status === "ok" && <p>No findings.</p>}
          {current.findings.map((f, i) => (
            <div key={i} className={`finding ${f.severity}`}>
              <strong>{f.rule}</strong>
              <p>{f.message}</p>
              {/* stays enabled while running: a second press is exactly what the server deduplicates */}
              {f.fix && (
                <button onClick={() => onFix(current.name, f.fix!.fix_id)}>
                  Apply fix & rerun: {f.fix.label}
                </button>
              )}
            </div>
          ))}
          {applied.length > 0 && (
            <div className="applied">
              <h3>Applied fixes</h3>
              {applied.map((fix) => (
                <div key={fix.fix_id}>
                  <span>{fix.label}</span>
                  <button onClick={() => onFix(current.name, fix.fix_id)}>Re-apply</button>
                </div>
              ))}
            </div>
          )}
          <div className="actions">
            <button disabled={busy || current.decision === "approved"} onClick={() => onReview(current.name, "approve")}>
              Approve
            </button>
            <button disabled={busy || current.decision === "rejected"} onClick={() => onReview(current.name, "reject")}>
              Reject
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
