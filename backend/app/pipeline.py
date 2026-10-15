"""Run the three stages for a scene and stream events into a RunRecord.
Runs on a worker thread; it only touches the scene under scene.lock."""
from __future__ import annotations

import logging
import time
from typing import Optional

from .checks.sim_rules import simulate
from .checks.static_rules import run_static_rules
from .geometry import export_geometry
from .loader import SceneError, compile_scene
from .runs import (
    RATE_COMPILE,
    RATE_SIM_PER_SECOND,
    RATE_STATIC_PER_OBJECT,
    ObjectState,
    RunRecord,
    SceneRecord,
)

log = logging.getLogger("sim_scene")


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def execute(scene: SceneRecord, run: RunRecord, xml: Optional[str] = None, only: Optional[str] = None) -> None:
    saved = {}
    try:
        _execute(scene, run, xml, only, saved)
    except Exception:  # noqa: BLE001 - the client must always see an ending
        log.exception("run %s failed", run.run_id)
        with scene.lock:
            for name, (status, findings) in saved.items():
                state = scene.objects.get(name)
                if state is not None:
                    state.status, state.findings = status, findings
        for name, (status, _) in saved.items():
            run.emit("object_status", {"object": name, "status": status})
        run.emit("run_failed", {"stage": "internal", "code": "internal_error", "message": "internal error; see server log"})
    finally:
        run.finish()


def _execute(scene, run, xml, only, saved):
    started = time.monotonic()
    run.emit("stage_started", {"stage": "load"})
    try:
        compiled = compile_scene(xml if xml is not None else scene.xml)
    except SceneError as e:
        run.emit("run_failed", {"stage": "load", "code": e.code, "message": e.message})
        return
    if only is not None and all(o.name != only for o in compiled.objects):
        run.emit("run_failed", {"stage": "load", "code": "object_missing_after_fix", "message": f"{only} is not in the fixed scene"})
        return
    with scene.lock:
        if xml is not None:
            scene.xml, scene.version = compiled.xml, compiled.version
            scene.objects = {
                o.name: scene.objects.get(o.name) or ObjectState(o.name, o.movable)
                for o in compiled.objects
            }
        scene.geometry = export_geometry(compiled.model, compiled.data, compiled.objects)
        scene.charge(RATE_COMPILE)
        targets = [o for o in compiled.objects if only is None or o.name == only]
        for obj in targets:
            saved[obj.name] = (scene.objects[obj.name].status, scene.objects[obj.name].findings)
            # clear now, not at the end: a snapshot taken mid-run must not carry
            # findings that the stream is about to replay
            scene.objects[obj.name].status = "checking"
            scene.objects[obj.name].findings = []
    run.emit("geometry", {"objects": scene.geometry})
    for obj in targets:
        run.emit("object_status", {"object": obj.name, "status": "checking"})
    run.emit("stage_completed", {"stage": "load", "duration_ms": _ms(started), "cost": RATE_COMPILE})

    started = time.monotonic()
    run.emit("stage_started", {"stage": "static"})
    static = [
        f for f in run_static_rules(compiled.model, compiled.data, compiled.objects)
        if only is None or f.object == only
    ]
    cost = RATE_STATIC_PER_OBJECT * len(targets)
    with scene.lock:
        scene.charge(cost)
    for finding in static:
        run.emit("finding", {"finding": finding.to_dict()})
    run.emit("stage_completed", {"stage": "static", "duration_ms": _ms(started), "cost": cost})

    started = time.monotonic()
    run.emit("stage_started", {"stage": "simulation"})
    result = simulate(compiled.model, compiled.data, compiled.objects)
    dynamic = [f for f in result.findings if only is None or f.object == only]
    cost = RATE_SIM_PER_SECOND * result.simulated_seconds
    with scene.lock:
        scene.charge(cost)
    for finding in dynamic:
        run.emit("finding", {"finding": finding.to_dict()})
    run.emit("stage_completed", {"stage": "simulation", "duration_ms": _ms(started), "cost": cost})

    findings = static + dynamic
    with scene.lock:
        for obj in targets:
            state = scene.objects[obj.name]
            state.findings = [f for f in findings if f.object == obj.name]
            state.status = "has_issues" if state.findings else "ok"
            if only is not None:
                state.decision = "pending"
        if only is not None:
            scene.reruns += 1
        statuses = {obj.name: scene.objects[obj.name].status for obj in targets}
        metrics = scene.metrics()
    for name, status in statuses.items():
        run.emit("object_status", {"object": name, "status": status})
    run.emit("run_completed", {"metrics": metrics})
