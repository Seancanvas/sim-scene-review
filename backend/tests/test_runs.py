import threading
import time

import pytest

from app import runs
from app.models import Finding, ObjectInfo
from app.runs import Registry, RunRecord, SceneRecord, idempotency_key, rerun_cost

OBJECTS = [ObjectInfo("a", 1, True, (1,)), ObjectInfo("b", 2, False, (2,))]


def test_events_get_increasing_ids_and_replay_after():
    run = RunRecord("run_1", "scn_1")
    run.emit("stage_started", {"stage": "load"})
    run.emit("finding", {"x": 1})
    assert [e.id for e in run.events_after(0)] == [1, 2]
    assert [e.type for e in run.events_after(1)] == ["finding"]
    assert run.last_id == 2


def test_wait_for_returns_when_finished():
    run = RunRecord("run_1", "scn_1")
    run.finish()
    assert run.wait_for(0, timeout=1.0) == []
    assert run.done


def test_wait_for_wakes_on_emit_from_another_thread():
    run = RunRecord("run_1", "scn_1")
    threading.Timer(0.05, lambda: run.emit("ping", {})).start()
    started = time.monotonic()
    events = run.wait_for(0, timeout=5.0)
    assert [e.type for e in events] == ["ping"]
    assert time.monotonic() - started < 1.0


def test_idempotency_key_is_deterministic_and_specific():
    assert idempotency_key("v1", "a", "fix") == idempotency_key("v1", "a", "fix")
    assert idempotency_key("v1", "a", "fix") != idempotency_key("v2", "a", "fix")
    assert idempotency_key("v1", "a", "fix") != idempotency_key("v1", "b", "fix")


def test_metrics_report_time_to_usable_only_when_all_approved():
    clock = [1000.0]
    reg = Registry(now=lambda: clock[0])
    scene = reg.add_scene("<mujoco/>", "v1", OBJECTS)
    assert scene.metrics()["time_to_usable_s"] is None
    scene.objects["a"].decision = "approved"
    scene.mark_usable_if_done()
    assert scene.metrics()["time_to_usable_s"] is None
    clock[0] = 1012.5
    scene.objects["b"].decision = "approved"
    scene.mark_usable_if_done()
    assert scene.metrics()["time_to_usable_s"] == 12.5


def test_snapshot_serialises_findings():
    scene = SceneRecord("scn_1", "<mujoco/>", "v1", OBJECTS, created_at=0.0)
    scene.objects["a"].findings = [Finding("a", "zero_mass", "error", "m")]
    snap = scene.snapshot()
    assert snap["scene_id"] == "scn_1"
    assert snap["objects"][0]["findings"][0]["rule"] == "zero_mass"
    assert snap["metrics"]["cost"] == 0.0
    assert snap["current_run_done"] is True


def test_snapshot_reports_whether_the_current_run_is_done():
    scene = SceneRecord("scn_1", "<mujoco/>", "v1", OBJECTS, created_at=0.0)
    run = RunRecord("run_1", "scn_1")
    scene.current_run_id, scene.current_run = run.run_id, run
    assert scene.snapshot()["current_run_done"] is False
    run.finish()
    assert scene.snapshot()["current_run_done"] is True


def test_registry_evicts_expired_and_oldest(monkeypatch):
    monkeypatch.setattr(runs, "MAX_SCENES", 2)
    clock = [0.0]
    reg = Registry(now=lambda: clock[0])
    first = reg.add_scene("<mujoco/>", "v1", [])
    clock[0] = 10.0
    second = reg.add_scene("<mujoco/>", "v1", [])
    third = reg.add_scene("<mujoco/>", "v1", [])
    assert reg.scene(first.scene_id) is None  # oldest dropped at the cap
    assert reg.scene(second.scene_id) is not None
    clock[0] = 10.0 + runs.SCENE_TTL_SECONDS + 1
    reg.add_scene("<mujoco/>", "v1", [])
    assert reg.scene(second.scene_id) is None and reg.scene(third.scene_id) is None


def test_registry_drops_runs_with_their_scene(monkeypatch):
    monkeypatch.setattr(runs, "MAX_SCENES", 1)
    reg = Registry()
    scene = reg.add_scene("<mujoco/>", "v1", [])
    run = reg.add_run(scene.scene_id)
    reg.add_scene("<mujoco/>", "v1", [])
    assert reg.run(run.run_id) is None


def test_registry_keeps_only_recent_runs_per_scene(monkeypatch):
    monkeypatch.setattr(runs, "MAX_RUNS_PER_SCENE", 2)
    reg = Registry()
    scene = reg.add_scene("<mujoco/>", "v1", [])
    first = reg.add_run(scene.scene_id)
    second = reg.add_run(scene.scene_id)
    third = reg.add_run(scene.scene_id)
    assert reg.run(first.run_id) is None
    assert reg.run(second.run_id) is not None and reg.run(third.run_id) is not None


def test_rerun_cost_matches_rate_card():
    assert rerun_cost() == pytest.approx(0.001 + 0.0005 + 0.01 * 2.0)
