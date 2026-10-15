import { STAGES, STAGE_LABELS, type StageView, type Stage } from "../types";

export function StageProgress({ stages }: { stages: Record<Stage, StageView> }) {
  return (
    <ol className="stages">
      {STAGES.map((stage) => {
        const view = stages[stage];
        return (
          <li key={stage} className={`stage ${view.status}`}>
            <span className="stage-name">{STAGE_LABELS[stage]}</span>
            <span className="stage-meta">
              {view.status === "done" && `${view.durationMs} ms · $${view.cost?.toFixed(4)}`}
              {view.status === "running" && "running…"}
              {view.status === "failed" && "failed"}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
