import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { MetricsFooter } from "./components/MetricsFooter";
import { ObjectPanel } from "./components/ObjectPanel";
import { SceneInput } from "./components/SceneInput";
import { StageProgress } from "./components/StageProgress";
import { applyReview, beginRun, fromSnapshot, initialState, reduceEvent, select } from "./reducer";
import type { RunEvent, ViewState } from "./types";
import { useRunStream } from "./useRunStream";

const sceneFromHash = () => new URLSearchParams(location.hash.slice(1)).get("scene");

export default function App() {
  const [state, setState] = useState<ViewState>(initialState);
  const [notice, setNotice] = useState<string | null>(null);

  const fail = useCallback((message: string) => setNotice(message), []);

  const hydrate = useCallback(
    (sceneId: string) =>
      api.snapshot(sceneId).then((snap) => setState(fromSnapshot(snap))).catch((e) => fail(e.message)),
    [fail],
  );

  useEffect(() => {
    const id = sceneFromHash();
    if (id) hydrate(id);
  }, [hydrate]);

  const onEvent = useCallback(
    (event: RunEvent) => {
      setState((s) => reduceEvent(s, event));
      // The run's final state lives on the server; re-read it so decisions and
      // the version are authoritative after every run.
      if (event.type === "run_completed" || event.type === "run_failed") {
        const id = sceneFromHash();
        if (id) api.snapshot(id).then((snap) => setState((s) => ({ ...fromSnapshot(snap), stages: s.stages, running: false, selected: s.selected, error: s.error }))).catch(() => undefined);
      }
    },
    [],
  );
  useRunStream(state.running ? state.runId : null, onEvent);

  const startRun = (sceneId: string) =>
    api.startRun(sceneId).then((r) => setState((s) => beginRun(s, r.run_id, null))).catch((e) => fail(e.message));

  const onScene = (sceneId: string) => {
    location.hash = `scene=${sceneId}`;
    setNotice(null);
    hydrate(sceneId).then(() => startRun(sceneId));
  };

  const onFix = (name: string, fixId: string) => {
    if (!state.sceneId) return;
    api
      .fix(state.sceneId, name, fixId)
      .then((r) => {
        if (r.cached) setNotice(`Already ran this fix on this version — reusing run ${r.run_id}, nothing charged.`);
        else setState((s) => beginRun(s, r.run_id, name));
      })
      .catch((e) => fail(e.message));
  };

  const onReview = (name: string, decision: "approve" | "reject") => {
    if (!state.sceneId) return;
    api.review(state.sceneId, name, decision).then((r) => setState((s) => applyReview(s, r.object, r.metrics))).catch((e) => fail(e.message));
  };

  const objects = state.order.map((n) => state.objects[n]);

  return (
    <div className="app">
      <header>
        <h1>Sim-Scene Review</h1>
        <SceneInput busy={state.running} onScene={onScene} onError={fail} />
        {state.sceneId && (
          <button className="secondary" disabled={state.running} onClick={() => startRun(state.sceneId!)}>
            Run again
          </button>
        )}
      </header>
      {(notice || state.error) && <div className="notice">{state.error ?? notice}</div>}
      <StageProgress stages={state.stages} />
      <main>
        <div className="viewport-placeholder">3D viewport arrives in the next task</div>
        <ObjectPanel
          objects={objects}
          selected={state.selected}
          busy={state.running}
          onSelect={(n) => setState((s) => select(s, n))}
          onReview={onReview}
          onFix={onFix}
        />
      </main>
      <MetricsFooter metrics={state.metrics} />
    </div>
  );
}
