"""In-memory records: scenes, runs, event buffers, idempotency, pricing, metrics.
Nothing persists; a restart empties everything (documented trade-off)."""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .checks.sim_rules import SIM_SECONDS

RATE_COMPILE = 0.001
RATE_STATIC_PER_OBJECT = 0.0005
RATE_SIM_PER_SECOND = 0.01
MAX_SCENES = 200
MAX_RUNS_PER_SCENE = 20
SCENE_TTL_SECONDS = 2 * 60 * 60


def rerun_cost() -> float:
    return RATE_COMPILE + RATE_STATIC_PER_OBJECT + RATE_SIM_PER_SECOND * SIM_SECONDS


@dataclass
class Event:
    id: int
    type: str
    data: dict


class RunRecord:
    def __init__(self, run_id: str, scene_id: str, object_name: Optional[str] = None):
        self.run_id = run_id
        self.scene_id = scene_id
        self.object_name = object_name
        self.done = False
        self._events = []
        self._cond = threading.Condition()

    def emit(self, type: str, data: dict) -> None:
        with self._cond:
            self._events.append(Event(len(self._events) + 1, type, data))
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self.done = True
            self._cond.notify_all()

    def events_after(self, last_id: int) -> list:
        with self._cond:
            return self._events[last_id:]

    def wait_for(self, last_id: int, timeout: float) -> list:
        """Block until an event after last_id exists, the run is done, or
        timeout passes. Meant to be called from a worker thread."""
        with self._cond:
            self._cond.wait_for(lambda: len(self._events) > last_id or self.done, timeout)
            return self._events[last_id:]

    @property
    def last_id(self) -> int:
        with self._cond:
            return len(self._events)


@dataclass
class ObjectState:
    name: str
    movable: bool
    status: str = "pending"  # pending | checking | ok | has_issues
    decision: str = "pending"  # pending | approved | rejected
    findings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "movable": self.movable,
            "status": self.status,
            "decision": self.decision,
            "findings": [f.to_dict() for f in self.findings],
        }


class SceneRecord:
    def __init__(self, scene_id: str, xml: str, version: str, objects, created_at: float):
        self.scene_id = scene_id
        self.xml = xml
        self.version = version
        self.objects = {o.name: ObjectState(o.name, o.movable) for o in objects}
        self.geometry = []
        self.created_at = created_at
        self.usable_at: Optional[float] = None
        self.cost = 0.0
        self.saved = 0.0
        self.manual_fixes = 0
        self.reruns = 0
        self.fix_runs = {}  # idempotency key -> run_id
        self.current_run_id: Optional[str] = None
        self.lock = threading.Lock()
        self._now = time.time

    def charge(self, amount: float) -> None:
        self.cost += amount

    def mark_usable_if_done(self) -> None:
        if self.objects and self.usable_at is None and all(
            o.decision == "approved" for o in self.objects.values()
        ):
            self.usable_at = self._now()

    def metrics(self) -> dict:
        return {
            "time_to_usable_s": None if self.usable_at is None else round(self.usable_at - self.created_at, 1),
            "manual_fixes": self.manual_fixes,
            "reruns": self.reruns,
            "cost": round(self.cost, 4),
            "saved_by_idempotency": round(self.saved, 4),
        }

    def snapshot(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "version": self.version,
            "current_run_id": self.current_run_id,
            "objects": [o.to_dict() for o in self.objects.values()],
            "geometry": self.geometry,
            "metrics": self.metrics(),
        }


def idempotency_key(version: str, object_name: str, fix_id: str) -> str:
    return hashlib.sha256(f"{version}:{object_name}:{fix_id}".encode("utf-8")).hexdigest()[:16]


class Registry:
    def __init__(self, now=time.time):
        self._scenes = {}
        self._runs = {}
        self._lock = threading.Lock()
        self._now = now

    def add_scene(self, xml: str, version: str, objects) -> SceneRecord:
        with self._lock:
            self._evict()
            scene = SceneRecord(f"scn_{secrets.token_hex(8)}", xml, version, objects, self._now())
            scene._now = self._now
            self._scenes[scene.scene_id] = scene
            return scene

    def scene(self, scene_id: str) -> Optional[SceneRecord]:
        with self._lock:
            return self._scenes.get(scene_id)

    def add_run(self, scene_id: str, object_name: Optional[str] = None) -> RunRecord:
        with self._lock:
            mine = [r for r in self._runs.values() if r.scene_id == scene_id]  # oldest first
            for stale in mine[: max(0, len(mine) + 1 - MAX_RUNS_PER_SCENE)]:
                self._runs.pop(stale.run_id, None)
            run = RunRecord(f"run_{secrets.token_hex(8)}", scene_id, object_name)
            self._runs[run.run_id] = run
            return run

    def run(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def _evict(self) -> None:
        now = self._now()
        for scene_id, scene in list(self._scenes.items()):
            if now - scene.created_at > SCENE_TTL_SECONDS:
                self._drop(scene_id)
        while len(self._scenes) >= MAX_SCENES:
            oldest = min(self._scenes.values(), key=lambda s: s.created_at)
            self._drop(oldest.scene_id)

    def _drop(self, scene_id: str) -> None:
        self._scenes.pop(scene_id, None)
        self._runs = {rid: r for rid, r in self._runs.items() if r.scene_id != scene_id}
