import type { Metrics } from "../types";

export function MetricsFooter({ metrics }: { metrics: Metrics | null }) {
  const m = metrics;
  const cell = (label: string, value: string) => (
    <div className="metric">
      <span className="label">{label}</span>
      <span className="value">{value}</span>
    </div>
  );
  return (
    <footer className="metrics">
      {cell("Time to usable scene", m?.time_to_usable_s == null ? "—" : `${m.time_to_usable_s} s`)}
      {cell("Manual fixes", m ? String(m.manual_fixes) : "—")}
      {cell("Reruns", m ? String(m.reruns) : "—")}
      {cell("Cost", m ? `$${m.cost.toFixed(4)}` : "—")}
      {cell("Saved by idempotency", m ? `$${m.saved_by_idempotency.toFixed(4)}` : "—")}
      <div className="rate-card">Rate card (fake): compile $0.001/run · static $0.0005/object · simulation $0.01/s</div>
    </footer>
  );
}
