import type { GeometryObject, Metrics, ObjectView, RunEvent, Snapshot, Stage, StageView, ViewState } from "./types";
import { STAGES } from "./types";

const freshStages = (): Record<Stage, StageView> =>
  Object.fromEntries(STAGES.map((s) => [s, { status: "pending" }])) as Record<Stage, StageView>;

export function initialState(): ViewState {
  return {
    sceneId: null, runId: null, running: false, stages: freshStages(),
    objects: {}, order: [], geometry: [], metrics: null, error: null, selected: null,
  };
}

/** A snapshot taken mid-run has objects still "checking"; treat the scene as
 * running so the page resubscribes to the run's stream after a reload. The
 * server replays every event; the pipeline clears an object's findings the
 * moment it goes back to "checking", so nothing is counted twice. */
export function fromSnapshot(snapshot: Snapshot): ViewState {
  const running = snapshot.current_run_id !== null && snapshot.objects.some((o) => o.status === "checking");
  return {
    ...initialState(),
    sceneId: snapshot.scene_id,
    runId: snapshot.current_run_id,
    running,
    objects: Object.fromEntries(snapshot.objects.map((o) => [o.name, o])),
    order: snapshot.objects.map((o) => o.name),
    geometry: snapshot.geometry,
    metrics: snapshot.metrics,
  };
}

/**
 * A new run starts and the stages reset. A full run clears every object's findings. An object
 * rerun keeps the previous findings on screen (marked checking) until the run completes and the
 * snapshot replaces them, so the fix button stays pressable — a second press is what the server
 * deduplicates.
 */
export function beginRun(state: ViewState, runId: string, onlyObject: string | null): ViewState {
  const objects = { ...state.objects };
  for (const name of onlyObject ? [onlyObject] : state.order) {
    if (objects[name]) {
      objects[name] = { ...objects[name], status: "checking", findings: onlyObject ? objects[name].findings : [] };
    }
  }
  return { ...state, runId, running: true, stages: freshStages(), error: null, objects };
}

export function select(state: ViewState, name: string | null): ViewState {
  return { ...state, selected: name };
}

export function applyReview(state: ViewState, object: ObjectView, metrics: Metrics): ViewState {
  return { ...state, metrics, objects: { ...state.objects, [object.name]: object } };
}

function withGeometry(state: ViewState, geometry: GeometryObject[]): ViewState {
  const objects = { ...state.objects };
  const order = [...state.order];
  for (const g of geometry) {
    if (g.name === "world" || objects[g.name]) continue;
    objects[g.name] = { name: g.name, movable: g.movable, status: "pending", decision: "pending", findings: [] };
    order.push(g.name);
  }
  return { ...state, geometry, objects, order };
}

export function reduceEvent(state: ViewState, event: RunEvent): ViewState {
  switch (event.type) {
    case "stage_started":
      return { ...state, stages: { ...state.stages, [event.data.stage]: { status: "running" } } };
    case "stage_completed":
      return {
        ...state,
        stages: {
          ...state.stages,
          [event.data.stage]: { status: "done", durationMs: event.data.duration_ms, cost: event.data.cost },
        },
      };
    case "geometry":
      return withGeometry(state, event.data.objects);
    case "finding": {
      const f = event.data.finding;
      const target = state.objects[f.object];
      if (!target) return state;
      return { ...state, objects: { ...state.objects, [f.object]: { ...target, findings: [...target.findings, f] } } };
    }
    case "object_status": {
      const target = state.objects[event.data.object];
      if (!target) return state;
      return { ...state, objects: { ...state.objects, [event.data.object]: { ...target, status: event.data.status } } };
    }
    case "run_completed":
      return { ...state, running: false, metrics: event.data.metrics };
    case "run_failed": {
      const stages = { ...state.stages };
      const stage = event.data.stage as Stage;
      if (stages[stage]) stages[stage] = { status: "failed" };
      return { ...state, running: false, error: event.data.message, stages };
    }
  }
}
