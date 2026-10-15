import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import api, pipeline
from app.main import create_app
from app.runs import rerun_cost


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def sse_events(client, run_id, last_id=None):
    headers = {"Last-Event-ID": str(last_id)} if last_id is not None else {}
    events = []
    with client.stream("GET", f"/api/runs/{run_id}/events", headers=headers) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        current = {}
        for line in res.iter_lines():
            if line.startswith("id: "):
                current["id"] = int(line[4:])
            elif line.startswith("event: "):
                current["type"] = line[7:]
            elif line.startswith("data: "):
                current["data"] = json.loads(line[6:])
            elif line == "" and current:
                events.append(current)
                current = {}
    return events


def create_tabletop(client):
    res = client.post("/api/scenes", json={"sample": "tabletop"})
    assert res.status_code == 201, res.text
    return res.json()["scene_id"]


def gated_execute(monkeypatch):
    """Hold every pipeline run until the returned event is set, so a test can
    make its second request while the first is still in progress."""
    gate = threading.Event()
    real = pipeline.execute

    def wait_then_run(scene, run, **kwargs):
        gate.wait(10.0)
        real(scene, run, **kwargs)

    monkeypatch.setattr(pipeline, "execute", wait_then_run)
    return gate


def start_and_finish(client, scene_id):
    run_id = client.post(f"/api/scenes/{scene_id}/runs").json()["run_id"]
    events = sse_events(client, run_id)
    assert events[-1]["type"] == "run_completed"
    return run_id, events


def test_health_and_samples(client):
    assert client.get("/api/health").json() == {"ok": True}
    assert client.get("/api/samples").json() == [{"id": "tabletop", "label": "Tabletop with planted issues"}]


def test_unknown_sample_uses_error_shape(client):
    res = client.post("/api/scenes", json={"sample": "nope"})
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "sample_not_found"


def test_validation_error_uses_error_shape(client):
    res = client.post("/api/scenes", json={})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "validation_error"


def test_unknown_api_path_and_wrong_method_use_error_shape(client):
    assert client.get("/api/nope").json()["error"]["code"] == "http_404"
    assert client.get("/api/scenes").json()["error"]["code"] == "http_405"


def test_frontend_served_without_swallowing_api_errors(tmp_path, monkeypatch):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>t</title>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "favicon.svg").write_text("<svg/>")
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path))
    with TestClient(create_app()) as c:
        res = c.get("/")
        assert res.status_code == 200 and "<title>t</title>" in res.text
        assert c.get("/assets/app.js").status_code == 200
        assert c.get("/favicon.svg").status_code == 200
        res = c.get("/api/scenes")
        assert res.status_code == 405 and res.json()["error"]["code"] == "http_405"
        res = c.get("/api/nope")
        assert res.status_code == 404 and res.json()["error"]["code"] == "http_404"
        res = c.get("/missing.txt")
        assert res.status_code == 404 and res.json()["error"]["code"] == "http_404"


def test_root_file_never_leaves_dist(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>")
    (tmp_path / "secret.txt").write_text("nope")
    monkeypatch.setenv("FRONTEND_DIST", str(dist))
    with TestClient(create_app()) as c:
        for path in ("/..%5Csecret.txt", "/..%2Fsecret.txt"):
            res = c.get(path)
            assert res.status_code == 404, path
            assert res.json()["error"]["code"] == "http_404"
            assert "nope" not in res.text


def test_unexpected_exception_uses_error_shape():
    app = create_app()

    async def boom():
        raise RuntimeError("boom")

    app.add_api_route("/api/boom", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        res = c.get("/api/boom")
    assert res.status_code == 500
    assert res.json() == {"error": {"code": "internal_error", "message": "internal error"}}


def test_upload_rejects_non_mjcf(client):
    res = client.post("/api/scenes/upload", files={"file": ("page.xml", b"<html><body>hi</body></html>", "text/xml")})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "not_mjcf"


def test_upload_rejects_include_and_binary(client):
    res = client.post("/api/scenes/upload", files={"file": ("s.xml", b'<mujoco><include file="x.xml"/></mujoco>', "text/xml")})
    assert res.json()["error"]["code"] == "include_not_allowed"
    res = client.post("/api/scenes/upload", files={"file": ("s.xml", b"\xff\xfe\x00", "application/octet-stream")})
    assert res.json()["error"]["code"] == "not_utf8"


def test_upload_refused_by_declared_length_before_reading(client):
    # TestClient forwards the header as given, so a 10-byte body can claim 5 MB
    res = client.post(
        "/api/scenes/upload",
        content=b"x" * 10,
        headers={"Content-Length": "5000000", "Content-Type": "multipart/form-data; boundary=x"},
    )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "too_large"


def test_upload_without_file_field_is_422(client):
    res = client.post("/api/scenes/upload", files={"other": ("s.xml", b"<mujoco/>", "text/xml")})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "validation_error"


def test_upload_accepts_mjcf(client):
    xml = b'<mujoco><worldbody><body name="b" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/></body></worldbody></mujoco>'
    res = client.post("/api/scenes/upload", files={"file": ("s.xml", xml, "text/xml")})
    assert res.status_code == 201
    snap = client.get(f"/api/scenes/{res.json()['scene_id']}").json()
    assert [o["name"] for o in snap["objects"]] == ["b"]


def test_scene_creation_does_not_block_the_loop(client, monkeypatch):
    real = api.compile_scene

    def slow(xml):
        time.sleep(0.5)
        return real(xml)

    monkeypatch.setattr(api, "compile_scene", slow)
    results = []
    worker = threading.Thread(target=lambda: results.append(client.post("/api/scenes", json={"sample": "tabletop"})))
    worker.start()
    time.sleep(0.1)  # let the create reach compile_scene
    started = time.monotonic()
    assert client.get("/api/health").json() == {"ok": True}
    assert time.monotonic() - started < 0.2
    worker.join(5.0)
    assert results and results[0].status_code == 201


def test_full_run_streams_in_order(client):
    scene_id = create_tabletop(client)
    _, events = start_and_finish(client, scene_id)
    ids = [e["id"] for e in events]
    assert ids == list(range(1, len(ids) + 1))
    assert events[0]["type"] == "stage_started"
    assert any(e["type"] == "finding" and e["data"]["finding"]["object"] == "mug_a" for e in events)
    snap = client.get(f"/api/scenes/{scene_id}").json()
    assert snap["current_run_id"] is not None
    assert snap["objects"][0]["name"] == "table"


def test_events_replay_from_last_event_id(client):
    scene_id = create_tabletop(client)
    run_id, events = start_and_finish(client, scene_id)
    tail = sse_events(client, run_id, last_id=len(events) - 2)
    assert [e["id"] for e in tail] == [len(events) - 1, len(events)]


def test_events_replay_past_end_closes(client):
    scene_id = create_tabletop(client)
    run_id, events = start_and_finish(client, scene_id)
    assert sse_events(client, run_id, last_id=len(events)) == []
    assert sse_events(client, run_id, last_id=len(events) + 10) == []


def test_events_unknown_run_is_404(client):
    res = client.get("/api/runs/run_missing/events")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "run_not_found"


def test_fix_is_idempotent_and_free_the_second_time(client):
    scene_id = create_tabletop(client)
    start_and_finish(client, scene_id)
    first = client.post(f"/api/scenes/{scene_id}/objects/mug_a/fix", json={"fix_id": "recompute_mass"})
    assert first.status_code == 202 and first.json()["cached"] is False
    events = sse_events(client, first.json()["run_id"])
    assert events[-1]["type"] == "run_completed"
    cost_before = client.get(f"/api/scenes/{scene_id}").json()["metrics"]["cost"]
    second = client.post(f"/api/scenes/{scene_id}/objects/mug_a/fix", json={"fix_id": "recompute_mass"})
    assert second.status_code == 202 and second.json() == {"run_id": first.json()["run_id"], "cached": True}
    metrics = client.get(f"/api/scenes/{scene_id}").json()["metrics"]
    assert metrics["cost"] == cost_before
    assert metrics["saved_by_idempotency"] == pytest.approx(rerun_cost(), abs=1e-4)
    assert metrics["manual_fixes"] == 1


def test_fix_errors(client):
    scene_id = create_tabletop(client)
    assert client.post(f"/api/scenes/{scene_id}/objects/ghost/fix", json={"fix_id": "recompute_mass"}).status_code == 404
    res = client.post(f"/api/scenes/{scene_id}/objects/mug_a/fix", json={"fix_id": "nope"})
    assert res.status_code == 400 and res.json()["error"]["code"] == "unknown_fix"


def test_review_and_time_to_usable(client):
    scene_id = create_tabletop(client)
    start_and_finish(client, scene_id)
    assert client.post(f"/api/scenes/{scene_id}/objects/ghost/review", json={"decision": "approve"}).status_code == 404
    res = client.post(f"/api/scenes/{scene_id}/objects/mug_b/review", json={"decision": "reject"})
    assert res.json()["object"]["decision"] == "rejected"
    for name in client.get(f"/api/scenes/{scene_id}").json()["objects"]:
        res = client.post(f"/api/scenes/{scene_id}/objects/{name['name']}/review", json={"decision": "approve"})
    assert res.json()["metrics"]["time_to_usable_s"] is not None


def test_failed_fix_is_not_cached(client, monkeypatch):
    scene_id = create_tabletop(client)
    start_and_finish(client, scene_id)

    def failing(scene, run, **kwargs):
        run.emit("run_failed", {"stage": "load", "code": "internal_error", "message": "boom"})
        run.finish()

    with monkeypatch.context() as m:
        m.setattr("app.api.pipeline.execute", failing)
        first = client.post(f"/api/scenes/{scene_id}/objects/mug_a/fix", json={"fix_id": "recompute_mass"})
        assert sse_events(client, first.json()["run_id"])[-1]["type"] == "run_failed"
    second = client.post(f"/api/scenes/{scene_id}/objects/mug_a/fix", json={"fix_id": "recompute_mass"})
    assert second.status_code == 202 and second.json()["cached"] is False
    assert second.json()["run_id"] != first.json()["run_id"]
    assert sse_events(client, second.json()["run_id"])[-1]["type"] == "run_completed"
    assert client.get(f"/api/scenes/{scene_id}").json()["metrics"]["saved_by_idempotency"] == 0


def test_concurrent_full_runs_conflict(client, monkeypatch):
    scene_id = create_tabletop(client)
    gate = gated_execute(monkeypatch)
    first = client.post(f"/api/scenes/{scene_id}/runs")
    second = client.post(f"/api/scenes/{scene_id}/runs")
    gate.set()
    assert second.status_code == 409 and second.json()["error"]["code"] == "run_in_progress"
    assert sse_events(client, first.json()["run_id"])[-1]["type"] == "run_completed"
    assert client.post(f"/api/scenes/{scene_id}/runs").status_code == 202


def test_fix_during_run_conflicts_but_repeat_is_cached(client, monkeypatch):
    scene_id = create_tabletop(client)
    start_and_finish(client, scene_id)
    gate = gated_execute(monkeypatch)
    first = client.post(f"/api/scenes/{scene_id}/objects/mug_a/fix", json={"fix_id": "recompute_mass"})
    other = client.post(f"/api/scenes/{scene_id}/objects/plate/fix", json={"fix_id": "enable_collision"})
    again = client.post(f"/api/scenes/{scene_id}/objects/mug_a/fix", json={"fix_id": "recompute_mass"})
    gate.set()
    assert other.status_code == 409 and other.json()["error"]["code"] == "run_in_progress"
    assert again.status_code == 202 and again.json()["cached"] is True
    sse_events(client, first.json()["run_id"])


def test_last_event_id_is_clamped(client):
    scene_id = create_tabletop(client)
    run_id, events = start_and_finish(client, scene_id)
    assert len(sse_events(client, run_id, last_id=-5)) == len(events)
    with client.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "abc"}) as res:
        assert sum(1 for line in res.iter_lines() if line.startswith("id: ")) == len(events)
