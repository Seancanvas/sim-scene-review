import { describe, expect, it } from "vitest";
import { applyReview, beginRun, fromSnapshot, initialState, reduceEvent, select } from "./reducer";
import type { Finding, Metrics, Snapshot } from "./types";

const metrics: Metrics = { time_to_usable_s: null, manual_fixes: 0, reruns: 0, cost: 0, saved_by_idempotency: 0 };
const snapshot: Snapshot = {
  scene_id: "scn_1",
  version: "v1",
  current_run_id: null,
  objects: [
    { name: "table", movable: false, status: "pending", decision: "pending", findings: [] },
    { name: "mug", movable: true, status: "pending", decision: "pending", findings: [] },
  ],
  geometry: [],
  metrics,
};
const finding: Finding = { object: "mug", rule: "zero_mass", severity: "error", message: "m", fix: null };

describe("reducer", () => {
  it("hydrates from a snapshot keeping object order", () => {
    const s = fromSnapshot(snapshot);
    expect(s.sceneId).toBe("scn_1");
    expect(s.order).toEqual(["table", "mug"]);
    expect(s.metrics).toEqual(metrics);
    expect(s.running).toBe(false);
  });

  it("hydrates a scene whose run is still going as running, so the stream resumes", () => {
    const mid: Snapshot = {
      ...snapshot,
      current_run_id: "run_9",
      objects: snapshot.objects.map((o) => ({ ...o, status: "checking" as const })),
    };
    const s = fromSnapshot(mid);
    expect(s.running).toBe(true);
    expect(s.runId).toBe("run_9");
  });

  it("beginRun for a full run clears every object's findings", () => {
    let s = fromSnapshot(snapshot);
    s = reduceEvent(s, { type: "finding", data: { finding } });
    s = beginRun(s, "run_3", null);
    expect(s.objects.mug.findings).toEqual([]);
    expect(s.objects.mug.status).toBe("checking");
  });

  it("beginRun resets stages, keeps the rerun object's findings and marks it checking", () => {
    let s = fromSnapshot(snapshot);
    s = reduceEvent(s, { type: "finding", data: { finding } });
    s = beginRun(s, "run_2", "mug");
    expect(s.runId).toBe("run_2");
    expect(s.running).toBe(true);
    expect(s.stages.load.status).toBe("pending");
    expect(s.objects.mug.findings).toEqual([finding]);
    expect(s.objects.mug.status).toBe("checking");
  });

  it("tracks stage progress", () => {
    let s = beginRun(fromSnapshot(snapshot), "run_1", null);
    s = reduceEvent(s, { type: "stage_started", data: { stage: "load" } });
    expect(s.stages.load.status).toBe("running");
    s = reduceEvent(s, { type: "stage_completed", data: { stage: "load", duration_ms: 12, cost: 0.001 } });
    expect(s.stages.load).toEqual({ status: "done", durationMs: 12, cost: 0.001 });
  });

  it("appends findings and updates object status", () => {
    let s = beginRun(fromSnapshot(snapshot), "run_1", null);
    s = reduceEvent(s, { type: "object_status", data: { object: "mug", status: "checking" } });
    s = reduceEvent(s, { type: "finding", data: { finding } });
    s = reduceEvent(s, { type: "object_status", data: { object: "mug", status: "has_issues" } });
    expect(s.objects.mug.findings).toEqual([finding]);
    expect(s.objects.mug.status).toBe("has_issues");
  });

  it("creates objects it has not seen when geometry arrives", () => {
    let s = beginRun(initialState(), "run_1", null);
    s = reduceEvent(s, { type: "geometry", data: { objects: [
      { name: "world", movable: false, geoms: [] },
      { name: "new", movable: true, geoms: [] },
    ] } });
    expect(s.order).toEqual(["new"]);
    expect(s.geometry).toHaveLength(2);
  });

  it("run_completed stores metrics and stops running", () => {
    let s = beginRun(fromSnapshot(snapshot), "run_1", null);
    s = reduceEvent(s, { type: "run_completed", data: { metrics: { ...metrics, cost: 0.03 } } });
    expect(s.running).toBe(false);
    expect(s.metrics?.cost).toBe(0.03);
  });

  it("run_failed marks the stage and records the message", () => {
    let s = beginRun(fromSnapshot(snapshot), "run_1", null);
    s = reduceEvent(s, { type: "stage_started", data: { stage: "load" } });
    s = reduceEvent(s, { type: "run_failed", data: { stage: "load", code: "compile_error", message: "bad" } });
    expect(s.stages.load.status).toBe("failed");
    expect(s.error).toBe("bad");
    expect(s.running).toBe(false);
  });

  it("select and applyReview", () => {
    let s = select(fromSnapshot(snapshot), "mug");
    expect(s.selected).toBe("mug");
    s = applyReview(s, { ...snapshot.objects[1], decision: "approved" }, { ...metrics, time_to_usable_s: 3 });
    expect(s.objects.mug.decision).toBe("approved");
    expect(s.metrics?.time_to_usable_s).toBe(3);
  });
});
