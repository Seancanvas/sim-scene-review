export type Stage = "load" | "static" | "simulation";
export const STAGES: Stage[] = ["load", "static", "simulation"];
export const STAGE_LABELS: Record<Stage, string> = {
  load: "Load & compile",
  static: "Static asset checks",
  simulation: "Simulation check",
};

export type Severity = "error" | "warning";
export interface Fix { fix_id: string; label: string }
export interface Finding { object: string; rule: string; severity: Severity; message: string; fix: Fix | null }

export type ObjectStatus = "pending" | "checking" | "ok" | "has_issues";
export type Decision = "pending" | "approved" | "rejected";
export interface ObjectView {
  name: string;
  movable: boolean;
  status: ObjectStatus;
  decision: Decision;
  findings: Finding[];
}

export interface GeomView {
  type: string;
  size: number[];
  pos: number[];
  quat: number[]; // w, x, y, z as MuJoCo emits it
  rgba: number[];
  vertices?: number[];
  faces?: number[];
}
export interface GeometryObject { name: string; movable: boolean; geoms: GeomView[] }

export interface Metrics {
  time_to_usable_s: number | null;
  manual_fixes: number;
  reruns: number;
  cost: number;
  saved_by_idempotency: number;
}

export interface Snapshot {
  scene_id: string;
  version: string;
  current_run_id: string | null;
  objects: ObjectView[];
  geometry: GeometryObject[];
  metrics: Metrics;
}

export type StageStatus = "pending" | "running" | "done" | "failed";
export interface StageView { status: StageStatus; durationMs?: number; cost?: number }

export interface ViewState {
  sceneId: string | null;
  runId: string | null;
  running: boolean;
  stages: Record<Stage, StageView>;
  objects: Record<string, ObjectView>;
  order: string[];
  geometry: GeometryObject[];
  metrics: Metrics | null;
  error: string | null;
  selected: string | null;
  /** object name -> fixes applied in this page session, oldest first */
  appliedFixes: Record<string, Fix[]>;
}

export type RunEvent =
  | { type: "stage_started"; data: { stage: Stage } }
  | { type: "geometry"; data: { objects: GeometryObject[] } }
  | { type: "finding"; data: { finding: Finding } }
  | { type: "object_status"; data: { object: string; status: ObjectStatus } }
  | { type: "stage_completed"; data: { stage: Stage; duration_ms: number; cost: number } }
  | { type: "run_completed"; data: { metrics: Metrics } }
  | { type: "run_failed"; data: { stage: string; code: string; message: string } };

export const EVENT_TYPES: RunEvent["type"][] = [
  "stage_started", "geometry", "finding", "object_status", "stage_completed", "run_completed", "run_failed",
];
