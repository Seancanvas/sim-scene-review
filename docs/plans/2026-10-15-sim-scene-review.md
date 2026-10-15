# Sim-Scene Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public single-service web app that checks a MuJoCo (MJCF) scene in three streamed stages, lets a person approve, reject or fix-and-rerun each object, and reports cost and time-to-usable-scene.

**Architecture:** FastAPI compiles the scene with MuJoCo, runs pure-function rules over the compiled model, steps a 2 s simulation, and streams findings over resumable SSE from an in-memory run buffer; fixes edit the scene through `MjSpec` and are compiled fresh. A Vite + React + React Three Fiber frontend draws the geometry MuJoCo compiled and folds events into view state with a pure reducer. One Docker image serves both.

**Tech Stack:** Python 3.9-compatible code (local venv is 3.9, image is 3.12), `mujoco==3.3.7`, FastAPI, uvicorn, pytest; TypeScript, Vite, React 19, @react-three/fiber, @react-three/drei, three, Vitest; Docker, Render.

**Spec:** `docs/specs/2026-10-15-sim-scene-review-design.md` — read it first; this plan argues from it.

## Global Constraints

- Backend code must run on Python 3.9: `from __future__ import annotations` at the top of every module, no `match`, no `X | Y` unions at runtime, `Optional[...]` from `typing`.
- `mujoco==3.3.7` everywhere (local venv at `D:\sim-scene-review\.venv`, Docker image).
- No code from any other repository may be copied in. Everything here is written from scratch.
- Uploads: XML only, ≤ 1 MB (`1_000_000` bytes); reject `<include>` and any `file=` attribute before parsing.
- Simulation: at most 2 s simulated time and 10 s wall time per run.
- Rate card: compile `0.001` per run, static `0.0005` per object, simulation `0.01` per simulated second.
- Every HTTP error body is `{"error": {"code": str, "message": str}}` with a matching status.
- Memory limits: 200 scenes, 2-hour TTL.
- Commands below assume the repo root `D:\sim-scene-review`; the backend venv is `.venv/Scripts/python`.
- Every commit ends with the two trailer lines shown in Task 1 step 7.

## Review Focus

Inputs the spec implies but is silent on; each has a test pinned to the task that owns the code.

1. **Valid XML that is not MJCF** (`<html>…</html>`) → `400 not_mjcf`, never a 500. — Task 9, `test_upload_rejects_non_mjcf`.
2. **A fix whose result does not compile** → `run_failed`, and the scene's version and XML stay what they were. — Task 8, `test_object_rerun_with_broken_xml_keeps_version`.
3. **Event stream for an unknown run id** → `404` JSON, not a hanging stream. — Task 9, `test_events_unknown_run_is_404`.
4. **Reconnect with `Last-Event-ID` at or past the last event of a finished run** → the stream closes immediately with nothing to replay. — Task 9, `test_events_replay_past_end_closes`.
5. **A body with no geoms** (inertial only) → every static rule returns nothing, no exception. — Task 3, `test_empty_body_produces_no_findings`.

---

### Task 1: Backend scaffold and loader

**Files:**
- Create: `backend/requirements.txt`, `backend/requirements-dev.txt`, `backend/pyproject.toml`
- Create: `backend/app/__init__.py`, `backend/app/models.py`, `backend/app/loader.py`
- Create: `backend/tests/__init__.py`, `backend/tests/scenes.py`, `backend/tests/test_loader.py`

**Interfaces:**
- Produces: `models.Fix(fix_id, label)`, `models.Finding(object, rule, severity, message, fix=None).to_dict()`, `models.ObjectInfo(name, body_id, movable, geom_ids)`.
- Produces: `loader.SceneError(code, message)`, `loader.validate_xml(text) -> None`, `loader.canonicalise(text) -> str`, `loader.scene_version(xml) -> str`, `loader.compile_scene(xml) -> CompiledScene(xml, version, model, data, objects)`, `loader.MAX_UPLOAD_BYTES`.
- Produces for tests: `tests.scenes.wrap(worldbody_xml, extra="") -> str`.

- [ ] **Step 1: Dependencies and pytest config**

`backend/requirements.txt`:
```
mujoco==3.3.7
numpy>=1.26,<3
fastapi>=0.115,<1
uvicorn[standard]>=0.30
python-multipart>=0.0.9
```

`backend/requirements-dev.txt`:
```
-r requirements.txt
pytest>=8
httpx>=0.27
```

`backend/pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Run: `.venv/Scripts/python -m pip install -r backend/requirements-dev.txt`
Expected: ends with `Successfully installed ...` (mujoco 3.3.7 is already present).

Then pin what resolved on Python 3.9 so the 3.12 image runs the same versions (newer uvicorn and numpy have dropped 3.9, so the ranges above would resolve differently there):

Run: `.venv/Scripts/python -m pip freeze | grep -iE "^(fastapi|uvicorn|numpy|pydantic|pydantic_core|starlette|python-multipart|httpx|pytest)=="`

Replace each `>=` line in both requirements files with the exact `==` version printed (keep `mujoco==3.3.7`, keep `[standard]` on uvicorn). numpy 2.0.x is the last line with Python 3.9 wheels and also ships 3.12 wheels, so the pins hold in both environments.

- [ ] **Step 2: Models**

`backend/app/__init__.py` and `backend/tests/__init__.py`: empty files.

`backend/app/models.py`:
```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

# "error" blocks approval in spirit; "warning" is advice. The UI colours by this.
Severity = str


@dataclass(frozen=True)
class Fix:
    fix_id: str
    label: str


@dataclass(frozen=True)
class Finding:
    object: str
    rule: str
    severity: Severity
    message: str
    fix: Optional[Fix] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObjectInfo:
    name: str
    body_id: int
    movable: bool
    geom_ids: tuple
```

- [ ] **Step 3: Test helper and failing loader tests**

`backend/tests/scenes.py`:
```python
"""Tiny MJCF builders for tests. `extra` lands before <worldbody> (assets, options)."""


def wrap(worldbody: str, extra: str = "") -> str:
    return (
        "<mujoco>"
        f"{extra}"
        '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
        f"{worldbody}</worldbody></mujoco>"
    )
```

`backend/tests/test_loader.py`:
```python
import pytest

from app.loader import SceneError, canonicalise, compile_scene, scene_version, validate_xml
from tests.scenes import wrap

BALL = '<body name="ball" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/></body>'


def test_rejects_include():
    with pytest.raises(SceneError) as e:
        validate_xml(wrap('<include file="other.xml"/>'))
    assert e.value.code == "include_not_allowed"


def test_rejects_file_attribute():
    with pytest.raises(SceneError) as e:
        validate_xml(wrap(BALL, extra='<asset><mesh name="m" file="m.stl"/></asset>'))
    assert e.value.code == "file_reference_not_allowed"


def test_rejects_oversize():
    with pytest.raises(SceneError) as e:
        validate_xml("<mujoco>" + " " * 1_000_001 + "</mujoco>")
    assert e.value.code == "too_large"


def test_rejects_non_mjcf():
    with pytest.raises(SceneError) as e:
        validate_xml("<html><body>hello</body></html>")
    assert e.value.code == "not_mjcf"


def test_compile_error_is_scene_error():
    massless = (
        '<body name="b" pos="0 0 1"><joint type="free"/>'
        '<inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/>'
        '<geom type="box" size="0.1 0.1 0.1"/></body>'
    )
    with pytest.raises(SceneError) as e:
        compile_scene(wrap(massless))
    assert e.value.code == "compile_error"
    assert "mass" in e.value.message


def test_lists_objects_and_movability():
    xml = wrap(
        '<body name="table"><geom type="box" size="1 1 0.1"/></body>'
        '<body name="cup" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/>'
        '<body name="handle" pos="0.05 0 0"><geom type="sphere" size="0.01"/></body></body>'
    )
    scene = compile_scene(xml)
    assert [o.name for o in scene.objects] == ["table", "cup", "handle"]
    assert [o.movable for o in scene.objects] == [False, True, True]
    assert all(len(o.geom_ids) == 1 for o in scene.objects)
    assert scene.version == scene_version(xml)


def test_canonicalise_names_unnamed_bodies_and_is_stable():
    xml = canonicalise(wrap('<body pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/></body>'))
    assert 'name="body_1"' in xml
    assert canonicalise(xml) == xml
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_loader.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.loader'`.

- [ ] **Step 4: Loader**

`backend/app/loader.py`:
```python
"""Validate, canonicalise and compile MJCF. Everything downstream works on a
compiled MjModel, so the browser and the checks see exactly one scene."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import mujoco

from .models import ObjectInfo

MAX_UPLOAD_BYTES = 1_000_000

_INCLUDE = re.compile(r"<\s*include\b", re.IGNORECASE)
_FILE_ATTR = re.compile(r"\bfile\s*=", re.IGNORECASE)


class SceneError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_xml(text: str) -> None:
    if len(text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise SceneError("too_large", "scene must be 1 MB or smaller")
    if "<mujoco" not in text:
        raise SceneError("not_mjcf", "root element must be <mujoco>")
    if _INCLUDE.search(text):
        raise SceneError("include_not_allowed", "<include> is not allowed")
    if _FILE_ATTR.search(text):
        raise SceneError(
            "file_reference_not_allowed",
            "file= attributes are not allowed; inline mesh vertices instead",
        )


def scene_version(xml: str) -> str:
    return hashlib.sha256(xml.encode("utf-8")).hexdigest()[:16]


def canonicalise(text: str) -> str:
    """Parse, name every unnamed body, and let MuJoCo re-serialise. Fixes
    address bodies by name, so a name must exist for each one."""
    validate_xml(text)
    try:
        spec = mujoco.MjSpec.from_string(text)
    except ValueError as e:
        raise SceneError("parse_error", str(e))
    for index, body in enumerate(spec.bodies):
        if index > 0 and not body.name:
            body.name = f"body_{index}"
    return spec.to_xml()


@dataclass
class CompiledScene:
    xml: str
    version: str
    model: mujoco.MjModel
    data: mujoco.MjData
    objects: list


def compile_scene(xml: str) -> CompiledScene:
    try:
        model = mujoco.MjSpec.from_string(xml).compile()
    except ValueError as e:
        raise SceneError("compile_error", str(e))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return CompiledScene(xml, scene_version(xml), model, data, list_objects(model))


def list_objects(model: mujoco.MjModel) -> list:
    objects = []
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body_{body}"
        geoms = tuple(int(g) for g in range(model.ngeom) if model.geom_bodyid[g] == body)
        # body_weldid is 0 when nothing between this body and the world can move
        movable = bool(model.body_weldid[body] != 0)
        objects.append(ObjectInfo(name, body, movable, geoms))
    return objects
```

- [ ] **Step 5: Run the loader tests**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_loader.py -q`
Expected: `7 passed`.

- [ ] **Step 6: Make sure the compile-error test really exercises MuJoCo**

Run: `cd backend && ../.venv/Scripts/python -c "from app.loader import compile_scene; from tests.scenes import wrap; compile_scene(wrap('<body name=\"b\" pos=\"0 0 1\"><joint type=\"free\"/><inertial pos=\"0 0 0\" mass=\"0\" diaginertia=\"0 0 0\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></body>'))"`
Expected: traceback ending in `app.loader.SceneError: Error: mass and inertia of moving bodies must be larger than mjMINVAL ...`.

- [ ] **Step 7: Commit**

```bash
git add backend
git commit -F - <<'EOF'
feat(backend): validate, canonicalise and compile MJCF scenes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 2: Sample scene loader

The sample `backend/app/samples/tabletop.xml` is already committed. This task makes it loadable by name and pins what it contains.

**Files:**
- Create: `backend/app/samples/__init__.py`
- Test: `backend/tests/test_samples.py`

**Interfaces:**
- Produces: `samples.SAMPLES: dict[str, str]` (id → label), `samples.load_sample(name) -> str`.

- [ ] **Step 1: Failing test**

`backend/tests/test_samples.py`:
```python
from app.loader import compile_scene
from app.samples import SAMPLES, load_sample

EXPECTED = {
    "table": False,
    "mug_a": True,
    "mug_b": True,
    "plate": True,
    "box_small": True,
    "cabinet": False,
    "cabinet_door": True,
    "tray_mm": True,
}


def test_tabletop_is_listed():
    assert "tabletop" in SAMPLES


def test_tabletop_compiles_with_expected_objects():
    scene = compile_scene(load_sample("tabletop"))
    assert {o.name: o.movable for o in scene.objects} == EXPECTED
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_samples.py -q`
Expected: FAIL, `No module named 'app.samples'`.

- [ ] **Step 2: Sample loader**

`backend/app/samples/__init__.py`:
```python
from __future__ import annotations

from pathlib import Path

SAMPLES = {"tabletop": "Tabletop with planted issues"}


def load_sample(name: str) -> str:
    return Path(__file__).with_name(f"{name}.xml").read_text(encoding="utf-8")
```

- [ ] **Step 3: Run**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_samples.py -q`
Expected: `2 passed`.

- [ ] **Step 4: Commit**

```bash
git add backend/app/samples/__init__.py backend/tests/test_samples.py
git commit -F - <<'EOF'
feat(backend): load built-in sample scenes by name

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 3: Static rules

**Files:**
- Create: `backend/app/checks/__init__.py`, `backend/app/checks/static_rules.py`
- Test: `backend/tests/test_static_rules.py`

**Interfaces:**
- Consumes: `loader.compile_scene`, `models.Finding/Fix/ObjectInfo`.
- Produces: `static_rules.run_static_rules(model, data, objects) -> list[Finding]`; individual rules `zero_mass`, `no_collision`, `default_friction`, `unlimited_joint`, `implausible_scale`, each `(model, data, obj) -> list[Finding]`; `STATIC_RULES` tuple. Fix ids produced: `recompute_mass`, `enable_collision`, `limit_joints`, `scale_mm_to_m` (Task 5 implements them under exactly these names).

- [ ] **Step 1: Failing tests**

`backend/app/checks/__init__.py`: empty.

`backend/tests/test_static_rules.py`:
```python
from app.checks import static_rules
from app.checks.static_rules import (
    default_friction,
    implausible_scale,
    no_collision,
    run_static_rules,
    unlimited_joint,
    zero_mass,
)
from app.loader import compile_scene
from tests.scenes import wrap


def findings_for(body_xml, rule=None):
    scene = compile_scene(wrap(body_xml))
    if rule is None:
        return run_static_rules(scene.model, scene.data, scene.objects)
    return [f for obj in scene.objects for f in rule(scene.model, scene.data, obj)]


def ball(geom="", inertial=""):
    return (
        f'<body name="ball" pos="0 0 1"><joint type="free"/>{inertial}'
        f'<geom type="sphere" size="0.05" {geom}/></body>'
    )


MASSLESS = '<inertial pos="0 0 0" mass="1e-7" diaginertia="1e-10 1e-10 1e-10"/>'


def test_zero_mass_fires_for_massless_movable():
    found = findings_for(ball(inertial=MASSLESS), zero_mass)
    assert [f.rule for f in found] == ["zero_mass"]
    assert found[0].severity == "error"
    assert found[0].fix.fix_id == "recompute_mass"


def test_zero_mass_ignores_static_body():
    static = '<body name="post"><inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/><geom type="sphere" size="0.05"/></body>'
    assert findings_for(static, zero_mass) == []


def test_no_collision_fires_when_all_geoms_disabled():
    found = findings_for(ball(geom='contype="0" conaffinity="0"'), no_collision)
    assert [f.rule for f in found] == ["no_collision"]
    assert found[0].fix.fix_id == "enable_collision"


def test_no_collision_quiet_when_one_geom_collides():
    two = (
        '<body name="b" pos="0 0 1"><joint type="free"/>'
        '<geom type="sphere" size="0.05" contype="0" conaffinity="0"/>'
        '<geom type="sphere" size="0.02" pos="0.05 0 0"/></body>'
    )
    assert findings_for(two, no_collision) == []


def test_default_friction_warns_on_movable_object():
    found = findings_for(ball(), default_friction)
    assert [(f.rule, f.severity) for f in found] == [("default_friction", "warning")]
    assert found[0].fix is None


def test_default_friction_quiet_when_set():
    assert findings_for(ball(geom='friction="0.8 0.005 0.0001"'), default_friction) == []


DOOR = '<body name="door" pos="0 0 1"><joint name="h" type="hinge" axis="0 0 1" {attrs}/><geom type="box" size="0.1 0.01 0.1"/></body>'


def test_unlimited_joint_fires_for_hinge_without_range():
    found = findings_for(DOOR.format(attrs=""), unlimited_joint)
    assert [f.rule for f in found] == ["unlimited_joint"]
    assert "'h'" in found[0].message
    assert found[0].fix.fix_id == "limit_joints"


def test_unlimited_joint_quiet_with_range():
    assert findings_for(DOOR.format(attrs='range="-1 1"'), unlimited_joint) == []


def test_unlimited_joint_ignores_free_joint():
    assert findings_for(ball(), unlimited_joint) == []


def test_scale_fires_with_fix_for_millimetre_box():
    tray = '<body name="tray" pos="0 0 5"><joint type="free"/><geom type="box" size="150 100 5"/></body>'
    found = findings_for(tray, implausible_scale)
    assert [(f.rule, f.severity) for f in found] == [("implausible_scale", "error")]
    assert found[0].fix.fix_id == "scale_mm_to_m"


def test_scale_warns_without_fix_when_simply_huge():
    huge = '<body name="huge"><geom type="box" size="20000 20000 20000"/></body>'
    found = findings_for(huge, implausible_scale)
    assert [(f.rule, f.severity) for f in found] == [("implausible_scale", "warning")]
    assert found[0].fix is None


def test_scale_warns_for_tiny_object():
    # not ball(): that helper already sets size, and a duplicate attribute is invalid XML
    dot = '<body name="dot" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.001"/></body>'
    found = findings_for(dot, implausible_scale)
    assert [(f.rule, f.severity) for f in found] == [("implausible_scale", "warning")]


def test_empty_body_produces_no_findings():
    empty = '<body name="empty" pos="0 0 1"><joint type="free"/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/></body>'
    assert findings_for(empty) == []


def test_broken_rule_becomes_check_error(monkeypatch):
    def explode(model, data, obj):
        raise RuntimeError("boom")

    monkeypatch.setattr(static_rules, "STATIC_RULES", (explode,))
    found = findings_for(ball())
    assert [(f.rule, f.severity) for f in found] == [("check_error", "warning")]
    assert "explode" in found[0].message and "boom" in found[0].message
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_static_rules.py -q`
Expected: FAIL, `No module named 'app.checks.static_rules'`.

- [ ] **Step 2: Rules**

`backend/app/checks/static_rules.py`:
```python
"""Static checks over a compiled model. Each rule is a pure function
(model, data, obj) -> list[Finding]. A rule that raises is reported as a
check_error finding for that object; it never sinks the run."""
from __future__ import annotations

import mujoco
import numpy as np

from ..models import Finding, Fix

MASS_EPS = 1e-6
DEFAULT_FRICTION = np.array([1.0, 0.005, 0.0001])
MAX_EXTENT_M = 20.0
MIN_EXTENT_M = 0.005


def zero_mass(model, data, obj):
    mass = float(model.body_mass[obj.body_id])
    if not obj.movable or mass > MASS_EPS:
        return []
    return [
        Finding(
            obj.name,
            "zero_mass",
            "error",
            f"{obj.name} can move but weighs {mass:.1e} kg; contacts with it cannot be resolved.",
            Fix("recompute_mass", "Recompute mass and inertia from geometry at 1000 kg/m³"),
        )
    ]


def no_collision(model, data, obj):
    if not obj.geom_ids:
        return []
    if any(model.geom_contype[g] or model.geom_conaffinity[g] for g in obj.geom_ids):
        return []
    return [
        Finding(
            obj.name,
            "no_collision",
            "error",
            f"every geom on {obj.name} has contype=0 and conaffinity=0, so nothing can touch it.",
            Fix("enable_collision", "Enable collision (contype=1, conaffinity=1)"),
        )
    ]


def default_friction(model, data, obj):
    if not obj.movable or not obj.geom_ids:
        return []
    if not all(np.allclose(model.geom_friction[g], DEFAULT_FRICTION) for g in obj.geom_ids):
        return []
    return [
        Finding(
            obj.name,
            "default_friction",
            "warning",
            f"{obj.name} uses MuJoCo's default friction (1, 0.005, 0.0001); "
            "set measured values before relying on sim-to-real.",
        )
    ]


_LIMITABLE = {
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
}


def unlimited_joint(model, data, obj):
    out = []
    for j in range(model.njnt):
        kind = _LIMITABLE.get(int(model.jnt_type[j]))
        if model.jnt_bodyid[j] != obj.body_id or kind is None or model.jnt_limited[j]:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        out.append(
            Finding(
                obj.name,
                "unlimited_joint",
                "error",
                f"{kind} joint '{jname}' on {obj.name} has no range, so it can move without bound.",
                Fix("limit_joints", "Add ±90° (hinge) / ±0.5 m (slide) limits to unlimited joints"),
            )
        )
    return out


def implausible_scale(model, data, obj):
    # longest side of each geom's bounding box (geom_aabb holds centre + half sizes)
    extents = [
        2.0 * float(model.geom_aabb[g, 3:].max())
        for g in obj.geom_ids
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
    ]
    if not extents:
        return []
    biggest = max(extents)
    if biggest > MAX_EXTENT_M:
        fix = None
        if MIN_EXTENT_M <= biggest / 1000.0 <= MAX_EXTENT_M:
            fix = Fix("scale_mm_to_m", "Scale geometry by 0.001 (millimetres → metres)")
        return [
            Finding(
                obj.name,
                "implausible_scale",
                "error" if fix else "warning",
                f"{obj.name} spans about {biggest:.0f} m; it looks like it was modelled in millimetres.",
                fix,
            )
        ]
    if biggest < MIN_EXTENT_M:
        return [
            Finding(
                obj.name,
                "implausible_scale",
                "warning",
                f"{obj.name} spans about {biggest * 1000:.2f} mm; check the units it was exported in.",
            )
        ]
    return []


STATIC_RULES = (zero_mass, no_collision, default_friction, unlimited_joint, implausible_scale)


def run_static_rules(model, data, objects):
    findings = []
    for obj in objects:
        for rule in STATIC_RULES:
            try:
                findings.extend(rule(model, data, obj))
            except Exception as e:  # noqa: BLE001 - one broken rule must not sink the run
                findings.append(
                    Finding(obj.name, "check_error", "warning", f"{rule.__name__} failed: {e}")
                )
    return findings
```

- [ ] **Step 3: Run**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_static_rules.py -q`
Expected: `14 passed`.

- [ ] **Step 4: Commit**

```bash
git add backend/app/checks backend/tests/test_static_rules.py
git commit -F - <<'EOF'
feat(backend): static asset rules over the compiled model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 4: Simulation rules

**Files:**
- Create: `backend/app/checks/sim_rules.py`
- Test: `backend/tests/test_sim_rules.py`

**Interfaces:**
- Produces: `sim_rules.simulate(model, data, objects) -> SimResult(findings: list[Finding], simulated_seconds: float)`, constants `SIM_SECONDS`, `WALL_SECONDS`, `EXPLODE_SPEED`, `FLOOR_DROP_M`; helpers `lowest_point(model, data, obj) -> float`, `initial_penetrations(model, data, objects) -> list[Finding]`.

- [ ] **Step 1: Failing tests**

`backend/tests/test_sim_rules.py`:
```python
import numpy as np

from app.checks import sim_rules
from app.checks.sim_rules import lowest_point, simulate
from app.loader import compile_scene
from tests.scenes import wrap


def run(body_xml, extra=""):
    scene = compile_scene(wrap(body_xml, extra=extra))
    return simulate(scene.model, scene.data, scene.objects)


def rules(result):
    return sorted((f.object, f.rule) for f in result.findings)


def ball(z, geom=""):
    return f'<body name="ball" pos="0 0 {z}"><joint type="free"/><geom type="sphere" size="0.05" {geom}/></body>'


def test_resting_object_is_clean():
    result = run(ball(0.05))
    assert result.findings == []
    assert result.simulated_seconds >= sim_rules.SIM_SECONDS


def test_initial_penetration_names_the_movable_object():
    result = run(ball(0.03))
    assert rules(result) == [("ball", "initial_penetration")]
    assert "mm inside" in result.findings[0].message


def test_fell_through_floor_when_collision_off():
    assert rules(run(ball(1.0, 'contype="0" conaffinity="0"'))) == [("ball", "fell_through_floor")]


def test_exploded_under_absurd_gravity():
    result = run(ball(1.0, 'contype="0" conaffinity="0"'), extra='<option gravity="0 0 -300"/>')
    assert ("ball", "exploded") in rules(result)


def test_unstable_detects_nan(monkeypatch):
    def poison(model, data):
        data.qvel[:] = float("nan")

    monkeypatch.setattr(sim_rules.mujoco, "mj_step", poison)
    assert rules(run(ball(1.0))) == [("ball", "unstable")]


def test_wall_clock_cap_stops_stepping(monkeypatch):
    monkeypatch.setattr(sim_rules, "WALL_SECONDS", 0.0)
    result = run(ball(1.0))
    assert result.simulated_seconds == 0.0


def test_lowest_point_uses_box_corners_not_bounding_sphere():
    scene = compile_scene(wrap('<body name="slab" pos="0 0 0.01"><joint type="free"/><geom type="box" size="1 1 0.01"/></body>'))
    assert abs(lowest_point(scene.model, scene.data, scene.objects[0])) < 1e-9
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_sim_rules.py -q`
Expected: FAIL, `No module named 'app.checks.sim_rules'`.

- [ ] **Step 2: Simulation rules**

`backend/app/checks/sim_rules.py`:
```python
"""Simulation checks: step the scene for a short time and look at what moved."""
from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from ..models import Finding

SIM_SECONDS = 2.0
WALL_SECONDS = 10.0
# Free fall for SIM_SECONDS reaches 9.81 * 2 = 19.6 m/s, just under this. Raise
# SIM_SECONDS and a body falling through the floor also counts as exploded.
EXPLODE_SPEED = 20.0
PENETRATION_M = 0.001
FLOOR_DROP_M = 0.5
QACC_LIMIT = 1e10

_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
)


@dataclass
class SimResult:
    findings: list
    simulated_seconds: float


def lowest_point(model, data, obj) -> float:
    """World z of the lowest bounding-box corner over the object's geoms. The
    bounding sphere would put a flat slab metres below where it really is."""
    zs = []
    for g in obj.geom_ids:
        centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        corners = (centre + _CORNERS * half) @ rot.T + data.geom_xpos[g]
        zs.append(float(corners[:, 2].min()))
    return min(zs) if zs else float("inf")


def floor_height(model, data) -> float:
    planes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    return min(float(data.geom_xpos[g][2]) for g in planes) if planes else 0.0


def initial_penetrations(model, data, objects):
    by_body = {o.body_id: o for o in objects if o.movable}
    worst = {}
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist >= -PENETRATION_M:
            continue
        for g in (contact.geom1, contact.geom2):
            obj = by_body.get(int(model.geom_bodyid[g]))
            if obj is not None:
                worst[obj.name] = min(worst.get(obj.name, 0.0), float(contact.dist))
    return [
        Finding(
            name,
            "initial_penetration",
            "error",
            f"{name} starts {-dist * 1000:.0f} mm inside another geom; move it so the surfaces touch instead of overlap.",
        )
        for name, dist in worst.items()
    ]


def _dofs(model, obj):
    start = int(model.body_dofadr[obj.body_id])
    return slice(start, start + int(model.body_dofnum[obj.body_id]))


def simulate(model, data, objects) -> SimResult:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    findings = initial_penetrations(model, data, objects)
    floor_z = floor_height(model, data)
    movable = [o for o in objects if o.movable]
    exploded, unstable = set(), set()
    started = time.monotonic()
    while data.time < SIM_SECONDS and time.monotonic() - started < WALL_SECONDS:
        mujoco.mj_step(model, data)
        for obj in movable:
            dofs = _dofs(model, obj)
            qvel, qacc = data.qvel[dofs], data.qacc[dofs]
            if np.isnan(qvel).any() or (qacc.size and np.abs(qacc).max() > QACC_LIMIT):
                unstable.add(obj.name)
            elif np.linalg.norm(data.cvel[obj.body_id, 3:]) > EXPLODE_SPEED:
                exploded.add(obj.name)
        if movable and len(unstable) == len(movable):
            break
    for obj in movable:
        if obj.name in unstable:
            findings.append(
                Finding(obj.name, "unstable", "error",
                        f"{obj.name} produced NaN or unbounded accelerations; check its mass, inertia and contacts.")
            )
            continue
        if obj.name in exploded:
            findings.append(
                Finding(obj.name, "exploded", "error",
                        f"{obj.name} exceeded {EXPLODE_SPEED:.0f} m/s; usually an overlap or a bad inertia launching it.")
            )
        if lowest_point(model, data, obj) < floor_z - FLOOR_DROP_M:
            findings.append(
                Finding(obj.name, "fell_through_floor", "error",
                        f"{obj.name} ended more than {FLOOR_DROP_M:.1f} m below the floor; it has no collision with it.")
            )
    return SimResult(findings, float(data.time))
```

- [ ] **Step 3: Run**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_sim_rules.py -q`
Expected: `7 passed`. If `test_unstable_detects_nan` reports `exploded` as well, the NaN branch is not being reached first: check that the `elif` order in the loop matches the code above.

- [ ] **Step 4: Commit**

```bash
git add backend/app/checks/sim_rules.py backend/tests/test_sim_rules.py
git commit -F - <<'EOF'
feat(backend): simulation rules with wall-clock cap

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 5: Fixes through MjSpec

**Files:**
- Create: `backend/app/fixes.py`
- Test: `backend/tests/test_fixes.py`

**Interfaces:**
- Consumes: `loader.SceneError`, `loader.canonicalise`, `loader.compile_scene`.
- Produces: `fixes.apply_fix(xml, object_name, fix_id) -> str` (canonical XML), `fixes.FIXES: dict[str, callable]` with keys `recompute_mass`, `enable_collision`, `limit_joints`, `scale_mm_to_m`. Raises `SceneError("unknown_fix")` / `SceneError("unknown_object")`.

- [ ] **Step 1: Failing tests**

`backend/tests/test_fixes.py`:
```python
import math

import pytest

from app.checks.static_rules import implausible_scale, no_collision, unlimited_joint, zero_mass
from app.fixes import apply_fix
from app.loader import SceneError, canonicalise, compile_scene
from tests.scenes import wrap

SPHERE_MASS = 4.0 / 3.0 * math.pi * 0.05**3 * 1000.0


def fixed(body_xml, name, fix_id, rule):
    xml = apply_fix(canonicalise(wrap(body_xml)), name, fix_id)
    scene = compile_scene(xml)
    return [f for o in scene.objects for f in rule(scene.model, scene.data, o)], scene


def test_recompute_mass_drops_explicit_inertial():
    body = ('<body name="ball" pos="0 0 1"><joint type="free"/>'
            '<inertial pos="0 0 0" mass="1e-7" diaginertia="1e-10 1e-10 1e-10"/>'
            '<geom type="sphere" size="0.05"/></body>')
    found, scene = fixed(body, "ball", "recompute_mass", zero_mass)
    assert found == []
    assert scene.model.body_mass[1] == pytest.approx(SPHERE_MASS, rel=1e-6)


def test_recompute_mass_overrides_explicit_geom_mass():
    body = '<body name="ball" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05" mass="1e-7"/></body>'
    found, scene = fixed(body, "ball", "recompute_mass", zero_mass)
    assert found == []
    assert scene.model.body_mass[1] == pytest.approx(SPHERE_MASS, rel=1e-6)


def test_enable_collision():
    body = '<body name="ball" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05" contype="0" conaffinity="0"/></body>'
    found, scene = fixed(body, "ball", "enable_collision", no_collision)
    assert found == []
    assert scene.model.geom_contype[1] == 1 and scene.model.geom_conaffinity[1] == 1


def test_limit_joints_hinge_and_slide():
    body = ('<body name="arm" pos="0 0 1"><joint name="h" type="hinge" axis="0 0 1"/>'
            '<joint name="s" type="slide" axis="1 0 0"/><geom type="box" size="0.1 0.01 0.1"/></body>')
    found, scene = fixed(body, "arm", "limit_joints", unlimited_joint)
    assert found == []
    # MjSpec.to_xml() keeps 6 significant digits, so -1.5708 is what comes back
    assert scene.model.jnt_range[0].tolist() == pytest.approx([-math.pi / 2, math.pi / 2], abs=1e-4)
    assert scene.model.jnt_range[1].tolist() == pytest.approx([-0.5, 0.5])


def test_limit_joints_keeps_existing_range():
    # MJCF angles default to degrees; the compiled range is in radians.
    body = ('<body name="arm" pos="0 0 1"><joint name="h" type="hinge" axis="0 0 1" range="-20 20"/>'
            '<joint name="s" type="slide" axis="1 0 0"/><geom type="box" size="0.1 0.01 0.1"/></body>')
    found, scene = fixed(body, "arm", "limit_joints", unlimited_joint)
    assert found == []
    assert scene.model.jnt_range[0].tolist() == pytest.approx([-math.radians(20), math.radians(20)], abs=1e-4)


def test_scale_mm_to_m():
    body = '<body name="tray" pos="0 0 5"><joint type="free"/><geom type="box" size="150 100 5" pos="10 0 0"/></body>'
    found, scene = fixed(body, "tray", "scale_mm_to_m", implausible_scale)
    assert found == []
    assert scene.model.geom_size[1].tolist() == pytest.approx([0.15, 0.1, 0.005])
    assert scene.model.geom_pos[1].tolist() == pytest.approx([0.01, 0.0, 0.0])


def test_unknown_fix_and_object():
    xml = canonicalise(wrap('<body name="ball" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/></body>'))
    with pytest.raises(SceneError) as e:
        apply_fix(xml, "ball", "nope")
    assert e.value.code == "unknown_fix"
    with pytest.raises(SceneError) as e:
        apply_fix(xml, "ghost", "enable_collision")
    assert e.value.code == "unknown_object"
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_fixes.py -q`
Expected: FAIL, `No module named 'app.fixes'`.

- [ ] **Step 2: Fixes**

`backend/app/fixes.py`:
```python
"""Suggested fixes. Each edits the scene through MjSpec and returns new
canonical XML. The caller compiles that XML fresh: compiling the edited spec
in place keeps the old inertial values around."""
from __future__ import annotations

import math

import mujoco

from .loader import SceneError

HINGE_RANGE = (-math.pi / 2, math.pi / 2)
SLIDE_RANGE = (-0.5, 0.5)
LIMITED = int(mujoco.mjtLimited.mjLIMITED_TRUE)
AUTO = int(mujoco.mjtLimited.mjLIMITED_AUTO)


def _body(spec, name):
    for body in spec.bodies:
        if body.name == name:
            return body
    raise SceneError("unknown_object", f"no body named {name!r}")


def recompute_mass(body):
    body.explicitinertial = False
    for geom in body.geoms:
        geom.density = 1000.0
        geom.mass = float("nan")  # MjSpec's "not set": fall back to density


def enable_collision(body):
    for geom in body.geoms:
        geom.contype = 1
        geom.conaffinity = 1


def _has_limit(joint) -> bool:
    # A joint parsed with a range reports limited=auto (2), not true (1).
    limited = int(joint.limited)
    return limited == LIMITED or (limited == AUTO and float(joint.range[0]) < float(joint.range[1]))


def limit_joints(body):
    for joint in body.joints:
        if _has_limit(joint):
            continue
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint.range = HINGE_RANGE
        elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint.range = SLIDE_RANGE
        else:
            continue
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE


def scale_mm_to_m(body):
    for geom in body.geoms:
        geom.size = geom.size * 0.001
        geom.pos = geom.pos * 0.001


FIXES = {
    "recompute_mass": recompute_mass,
    "enable_collision": enable_collision,
    "limit_joints": limit_joints,
    "scale_mm_to_m": scale_mm_to_m,
}


def apply_fix(xml: str, object_name: str, fix_id: str) -> str:
    try:
        fix = FIXES[fix_id]
    except KeyError:
        raise SceneError("unknown_fix", f"no fix named {fix_id!r}")
    spec = mujoco.MjSpec.from_string(xml)
    fix(_body(spec, object_name))
    return spec.to_xml()
```

- [ ] **Step 3: Run**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_fixes.py -q`
Expected: `7 passed`.

- [ ] **Step 4: Commit**

```bash
git add backend/app/fixes.py backend/tests/test_fixes.py
git commit -F - <<'EOF'
feat(backend): apply suggested fixes through MjSpec

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 6: Geometry export

**Files:**
- Create: `backend/app/geometry.py`
- Test: `backend/tests/test_geometry.py`

**Interfaces:**
- Produces: `geometry.export_geometry(model, data, objects) -> list[dict]`; each dict `{name, movable, geoms: [{type, size, pos, quat (w,x,y,z), rgba, vertices?, faces?}]}`; the first entry is `name == "world"`.

- [ ] **Step 1: Failing tests**

`backend/tests/test_geometry.py`:
```python
from app.geometry import export_geometry
from app.loader import compile_scene
from tests.scenes import wrap


def export(body_xml, extra=""):
    scene = compile_scene(wrap(body_xml, extra=extra))
    return export_geometry(scene.model, scene.data, scene.objects)


def test_world_entry_comes_first_with_the_floor():
    out = export('<body name="b" pos="0 0 1"><geom type="box" size="0.1 0.2 0.3"/></body>')
    assert out[0]["name"] == "world" and out[0]["movable"] is False
    assert out[0]["geoms"][0]["type"] == "plane"


def test_box_pose_and_size():
    out = export('<body name="b" pos="1 2 3"><geom type="box" size="0.1 0.2 0.3" rgba="1 0 0 1"/></body>')
    geom = out[1]["geoms"][0]
    assert geom["type"] == "box"
    assert geom["size"] == [0.1, 0.2, 0.3]
    assert geom["pos"] == [1.0, 2.0, 3.0]
    assert geom["quat"] == [1.0, 0.0, 0.0, 0.0]
    assert geom["rgba"] == [1.0, 0.0, 0.0, 1.0]


def test_inline_mesh_exports_vertices_and_faces():
    asset = '<asset><mesh name="tet" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/></asset>'
    out = export('<body name="m" pos="0 0 1"><geom type="mesh" mesh="tet"/></body>', extra=asset)
    geom = out[1]["geoms"][0]
    assert geom["type"] == "mesh"
    assert len(geom["vertices"]) == 12
    assert len(geom["faces"]) == 12
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_geometry.py -q`
Expected: FAIL, `No module named 'app.geometry'`.

- [ ] **Step 2: Geometry**

`backend/app/geometry.py`:
```python
"""Export compiled geometry so the browser draws exactly what MuJoCo simulates."""
from __future__ import annotations

import mujoco
import numpy as np

GEOM_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def _quat(xmat) -> list:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, xmat)
    return quat.tolist()  # w, x, y, z


def _geom(model, data, g) -> dict:
    kind = int(model.geom_type[g])
    entry = {
        "type": GEOM_TYPES.get(kind, "unknown"),
        "size": model.geom_size[g].tolist(),
        "pos": data.geom_xpos[g].tolist(),
        "quat": _quat(data.geom_xmat[g]),
        "rgba": model.geom_rgba[g].tolist(),
    }
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[g])
        v0, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        f0, nf = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        entry["vertices"] = model.mesh_vert[v0:v0 + nv].ravel().tolist()
        entry["faces"] = model.mesh_face[f0:f0 + nf].ravel().tolist()
    return entry


def export_geometry(model, data, objects) -> list:
    world = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
    out = [{"name": "world", "movable": False, "geoms": [_geom(model, data, g) for g in world]}]
    for obj in objects:
        out.append({
            "name": obj.name,
            "movable": obj.movable,
            "geoms": [_geom(model, data, g) for g in obj.geom_ids],
        })
    return out
```

- [ ] **Step 3: Run**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_geometry.py -q`
Expected: `3 passed`.

- [ ] **Step 4: Commit**

```bash
git add backend/app/geometry.py backend/tests/test_geometry.py
git commit -F - <<'EOF'
feat(backend): export compiled geometry for the viewport

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 7: Run and scene records

**Files:**
- Create: `backend/app/runs.py`
- Test: `backend/tests/test_runs.py`

**Interfaces:**
- Produces: constants `RATE_COMPILE`, `RATE_STATIC_PER_OBJECT`, `RATE_SIM_PER_SECOND`, `MAX_SCENES`, `SCENE_TTL_SECONDS`, `rerun_cost() -> float`.
- Produces: `Event(id, type, data)`; `RunRecord(run_id, scene_id, object_name=None)` with `.emit(type, data)`, `.finish()`, `.done`, `.events_after(last_id) -> list[Event]`, `.wait_for(last_id, timeout) -> list[Event]` (blocking), `.last_id`.
- Produces: `ObjectState(name, movable, status="pending", decision="pending", findings=[])` with `.to_dict()`; `SceneRecord` with `.scene_id .xml .version .objects: dict[str, ObjectState] .geometry .created_at .usable_at .cost .saved .manual_fixes .reruns .fix_runs .current_run_id .lock`, `.charge(amount)`, `.mark_usable_if_done()`, `.metrics() -> dict`, `.snapshot() -> dict`.
- Produces: `idempotency_key(version, object_name, fix_id) -> str`; `Registry(now=time.time)` with `.add_scene(xml, version, objects) -> SceneRecord`, `.scene(id)`, `.add_run(scene_id, object_name=None) -> RunRecord`, `.run(id)`.

- [ ] **Step 1: Failing tests**

`backend/tests/test_runs.py`:
```python
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
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_runs.py -q`
Expected: FAIL, `No module named 'app.runs'`.

- [ ] **Step 2: Records**

`backend/app/runs.py`:
```python
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
```

- [ ] **Step 3: Run**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_runs.py -q`
Expected: `10 passed`.

- [ ] **Step 4: Commit**

```bash
git add backend/app/runs.py backend/tests/test_runs.py
git commit -F - <<'EOF'
feat(backend): scene and run records with event buffer and idempotency

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 8: Pipeline

**Files:**
- Create: `backend/app/pipeline.py`
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: `pipeline.execute(scene: SceneRecord, run: RunRecord, xml: Optional[str] = None, only: Optional[str] = None) -> None`. Full run when `only is None`; object rerun when `xml` (the fixed scene) and `only` are given. Emits the event types named in the spec and always calls `run.finish()`.

- [ ] **Step 1: Failing tests**

`backend/tests/test_pipeline.py`:
```python
import pytest

from app import pipeline
from app.fixes import apply_fix
from app.loader import canonicalise, compile_scene
from app.runs import Registry
from app.samples import load_sample
from tests.scenes import wrap

PLANTED = {
    "table": set(),
    "cabinet": set(),
    "mug_a": {"zero_mass"},
    "mug_b": set(),
    "plate": {"no_collision", "fell_through_floor"},
    "box_small": {"initial_penetration", "default_friction"},
    "cabinet_door": {"unlimited_joint"},
    "tray_mm": {"implausible_scale"},
}


def make_scene(reg, xml):
    compiled = compile_scene(canonicalise(xml))
    return reg.add_scene(compiled.xml, compiled.version, compiled.objects)


def run_full(reg, scene):
    run = reg.add_run(scene.scene_id)
    pipeline.execute(scene, run)
    return run


def types(run):
    return [e.type for e in run.events_after(0)]


@pytest.fixture
def tabletop():
    reg = Registry()
    return reg, make_scene(reg, load_sample("tabletop"))


def test_full_run_finds_exactly_the_planted_issues(tabletop):
    reg, scene = tabletop
    run = run_full(reg, scene)
    assert run.done
    assert {name: {f.rule for f in state.findings} for name, state in scene.objects.items()} == PLANTED
    assert scene.objects["mug_b"].status == "ok"
    assert scene.objects["mug_a"].status == "has_issues"


def test_full_run_event_order_and_cost(tabletop):
    reg, scene = tabletop
    run = run_full(reg, scene)
    seen = types(run)
    assert seen[0] == "stage_started" and seen[-1] == "run_completed"
    assert seen.index("geometry") < seen.index("finding")
    assert [e.data["stage"] for e in run.events_after(0) if e.type == "stage_completed"] == ["load", "static", "simulation"]
    # the stepper may overshoot 2.0 s by one 2 ms step, hence the tolerance
    assert scene.cost == pytest.approx(0.001 + 0.0005 * 8 + 0.01 * 2.0, abs=1e-4)
    assert run.events_after(0)[-1].data["metrics"]["cost"] == pytest.approx(scene.cost, abs=1e-4)


def test_object_rerun_replaces_only_that_object(tabletop):
    reg, scene = tabletop
    run_full(reg, scene)
    scene.objects["mug_a"].decision = "approved"
    scene.objects["plate"].decision = "approved"
    before = scene.version
    xml = apply_fix(scene.xml, "mug_a", "recompute_mass")
    run = reg.add_run(scene.scene_id, "mug_a")
    pipeline.execute(scene, run, xml=xml, only="mug_a")
    assert scene.version != before and scene.xml == xml
    assert scene.objects["mug_a"].findings == []
    assert scene.objects["mug_a"].decision == "pending"
    assert scene.objects["plate"].decision == "approved"
    assert {f.rule for f in scene.objects["plate"].findings} == PLANTED["plate"]
    assert scene.reruns == 1
    assert all(e.data["finding"]["object"] == "mug_a" for e in run.events_after(0) if e.type == "finding")


def test_object_rerun_with_broken_xml_keeps_version(tabletop):
    reg, scene = tabletop
    run_full(reg, scene)
    before = (scene.version, scene.xml)
    run = reg.add_run(scene.scene_id, "mug_a")
    pipeline.execute(scene, run, xml="<mujoco><worldbody><body name='x' pos='0 0 1'><joint type='free'/><inertial pos='0 0 0' mass='0' diaginertia='0 0 0'/><geom type='box' size='1 1 1'/></body></worldbody></mujoco>", only="mug_a")
    assert run.done
    assert types(run)[-1] == "run_failed"
    assert run.events_after(0)[-1].data["code"] == "compile_error"
    assert (scene.version, scene.xml) == before


def test_scene_without_objects_completes():
    reg = Registry()
    scene = make_scene(reg, wrap(""))
    run = run_full(reg, scene)
    assert types(run)[-1] == "run_completed"
    assert scene.metrics()["time_to_usable_s"] is None
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_pipeline.py -q`
Expected: FAIL, `No module named 'app.pipeline'`.

- [ ] **Step 2: Pipeline**

`backend/app/pipeline.py`:
```python
"""Run the three stages for a scene and stream events into a RunRecord.
Runs on a worker thread; it only touches the scene under scene.lock."""
from __future__ import annotations

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


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def execute(scene: SceneRecord, run: RunRecord, xml: Optional[str] = None, only: Optional[str] = None) -> None:
    try:
        _execute(scene, run, xml, only)
    except Exception as e:  # noqa: BLE001 - the client must always see an ending
        run.emit("run_failed", {"stage": "internal", "code": "internal_error", "message": str(e)})
    finally:
        run.finish()


def _execute(scene, run, xml, only):
    started = time.monotonic()
    run.emit("stage_started", {"stage": "load"})
    try:
        compiled = compile_scene(xml if xml is not None else scene.xml)
    except SceneError as e:
        run.emit("run_failed", {"stage": "load", "code": e.code, "message": e.message})
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
```

- [ ] **Step 3: Run**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_pipeline.py -q`
Expected: `5 passed`. If `test_full_run_finds_exactly_the_planted_issues` shows an extra rule on an object, print `scene.objects[name].findings` and compare with the sample comments; the planted set was verified against this exact sample XML before the plan was written, so a difference means a rule, not the sample, drifted.

- [ ] **Step 4: Run the whole backend suite**

Run: `cd backend && ../.venv/Scripts/python -m pytest -q`
Expected: all passed, no warnings about unclosed threads.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline.py backend/tests/test_pipeline.py
git commit -F - <<'EOF'
feat(backend): three-stage pipeline with per-object reruns

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 9: HTTP API, SSE and app factory

**Files:**
- Create: `backend/app/api.py`, `backend/app/main.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: Tasks 1–8.
- Produces: `main.create_app() -> FastAPI`, module-level `main.app`; routes exactly as in the spec's API table; `api.ApiError(status, code, message)`.
- Frontend (Tasks 10–12) relies on: `POST /api/scenes {sample}` → `{scene_id, version}`; `POST /api/scenes/upload` multipart field `file`; `POST /api/scenes/{id}/runs` → `{run_id}`; `GET /api/runs/{id}/events` SSE with named events; `POST .../objects/{name}/fix {fix_id}` → `{run_id, cached}`; `POST .../objects/{name}/review {decision}` → `{object, metrics}`; `GET /api/scenes/{id}` → snapshot; `GET /api/samples` → `[{id, label}]`.

- [ ] **Step 1: Failing tests**

`backend/tests/test_api.py`:
```python
import json

import pytest
from fastapi.testclient import TestClient

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


def test_upload_rejects_non_mjcf(client):
    res = client.post("/api/scenes/upload", files={"file": ("page.xml", b"<html><body>hi</body></html>", "text/xml")})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "not_mjcf"


def test_upload_rejects_include_and_binary(client):
    res = client.post("/api/scenes/upload", files={"file": ("s.xml", b'<mujoco><include file="x.xml"/></mujoco>', "text/xml")})
    assert res.json()["error"]["code"] == "include_not_allowed"
    res = client.post("/api/scenes/upload", files={"file": ("s.xml", b"\xff\xfe\x00", "application/octet-stream")})
    assert res.json()["error"]["code"] == "not_utf8"


def test_upload_accepts_mjcf(client):
    xml = b'<mujoco><worldbody><body name="b" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/></body></worldbody></mujoco>'
    res = client.post("/api/scenes/upload", files={"file": ("s.xml", xml, "text/xml")})
    assert res.status_code == 201
    snap = client.get(f"/api/scenes/{res.json()['scene_id']}").json()
    assert [o["name"] for o in snap["objects"]] == ["b"]


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
```

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_api.py -q`
Expected: FAIL, `No module named 'app.main'`.

- [ ] **Step 2: API**

`backend/app/api.py`:
```python
"""HTTP surface. Handlers are async so they can schedule pipeline work on the
loop; the CPU-bound pipeline itself runs in a worker thread."""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import pipeline
from .fixes import apply_fix
from .loader import MAX_UPLOAD_BYTES, SceneError, canonicalise, compile_scene, scene_version
from .runs import Registry, idempotency_key, rerun_cost
from .samples import SAMPLES, load_sample

router = APIRouter(prefix="/api")
KEEPALIVE_SECONDS = 15.0


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def registry(request: Request) -> Registry:
    return request.app.state.registry


def _scene(reg: Registry, scene_id: str):
    scene = reg.scene(scene_id)
    if scene is None:
        raise ApiError(404, "scene_not_found", f"no scene {scene_id}")
    return scene


def _start(request: Request, scene, run, **kwargs) -> None:
    task = asyncio.create_task(asyncio.to_thread(pipeline.execute, scene, run, **kwargs))
    tasks = request.app.state.tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def _create(reg: Registry, text: str) -> dict:
    try:
        xml = canonicalise(text)
        compiled = compile_scene(xml)
    except SceneError as e:
        raise ApiError(400, e.code, e.message)
    scene = reg.add_scene(compiled.xml, compiled.version, compiled.objects)
    return {"scene_id": scene.scene_id, "version": scene.version}


class CreateScene(BaseModel):
    sample: str


class FixBody(BaseModel):
    fix_id: str


class ReviewBody(BaseModel):
    decision: Literal["approve", "reject"]


@router.get("/health")
async def health():
    return {"ok": True}


@router.get("/samples")
async def samples():
    return [{"id": key, "label": label} for key, label in SAMPLES.items()]


@router.post("/scenes", status_code=201)
async def create_from_sample(body: CreateScene, reg: Registry = Depends(registry)):
    if body.sample not in SAMPLES:
        raise ApiError(404, "sample_not_found", f"no sample {body.sample!r}")
    return _create(reg, load_sample(body.sample))


@router.post("/scenes/upload", status_code=201)
async def create_from_upload(file: UploadFile = File(...), reg: Registry = Depends(registry)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "too_large", "scene must be 1 MB or smaller")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError(400, "not_utf8", "upload must be UTF-8 encoded XML")
    return _create(reg, text)


@router.get("/scenes/{scene_id}")
async def get_scene(scene_id: str, reg: Registry = Depends(registry)):
    scene = _scene(reg, scene_id)
    with scene.lock:
        return scene.snapshot()


@router.post("/scenes/{scene_id}/runs", status_code=202)
async def start_run(scene_id: str, request: Request, reg: Registry = Depends(registry)):
    scene = _scene(reg, scene_id)
    run = reg.add_run(scene_id)
    with scene.lock:
        scene.current_run_id = run.run_id
    _start(request, scene, run)
    return {"run_id": run.run_id}


@router.post("/scenes/{scene_id}/objects/{name}/fix", status_code=202)
async def fix_object(scene_id: str, name: str, body: FixBody, request: Request, reg: Registry = Depends(registry)):
    scene = _scene(reg, scene_id)
    with scene.lock:
        if name not in scene.objects:
            raise ApiError(404, "object_not_found", f"no object {name!r}")
        key = idempotency_key(scene.version, name, body.fix_id)
        cached = scene.fix_runs.get(key)
        if cached is not None:
            scene.saved += rerun_cost()
            return {"run_id": cached, "cached": True}
        try:
            xml = apply_fix(scene.xml, name, body.fix_id)
        except SceneError as e:
            raise ApiError(400 if e.code == "unknown_fix" else 404, e.code, e.message)
        run = reg.add_run(scene_id, name)
        # Asking for the same fix again on the scene this fix produces is the
        # same work, so both the input and the output version point at this run.
        for version in (scene.version, scene_version(xml)):
            scene.fix_runs[idempotency_key(version, name, body.fix_id)] = run.run_id
        scene.manual_fixes += 1
        scene.current_run_id = run.run_id
    _start(request, scene, run, xml=xml, only=name)
    return {"run_id": run.run_id, "cached": False}


@router.post("/scenes/{scene_id}/objects/{name}/review")
async def review_object(scene_id: str, name: str, body: ReviewBody, reg: Registry = Depends(registry)):
    scene = _scene(reg, scene_id)
    with scene.lock:
        state = scene.objects.get(name)
        if state is None:
            raise ApiError(404, "object_not_found", f"no object {name!r}")
        state.decision = "approved" if body.decision == "approve" else "rejected"
        scene.mark_usable_if_done()
        return {"object": state.to_dict(), "metrics": scene.metrics()}


async def _stream(run, last_id: int):
    while True:
        events = await asyncio.to_thread(run.wait_for, last_id, KEEPALIVE_SECONDS)
        for event in events:
            yield f"id: {event.id}\nevent: {event.type}\ndata: {json.dumps(event.data)}\n\n"
            last_id = event.id
        if run.done and last_id >= run.last_id:
            return
        if not events:
            yield ": keepalive\n\n"


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, reg: Registry = Depends(registry)):
    run = reg.run(run_id)
    if run is None:
        raise ApiError(404, "run_not_found", f"no run {run_id}")
    try:
        last_id = int(request.headers.get("last-event-id", "0") or 0)
    except ValueError:
        last_id = 0
    return StreamingResponse(
        _stream(run, last_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

`backend/app/main.py`:
```python
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api import ApiError, router
from .runs import Registry

DEFAULT_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Every open SSE client parks a thread in wait_for and the pipeline runs in
    # the same pool, so the default pool (cpu+4) would stall runs under a
    # handful of viewers.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=64))
    yield


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app() -> FastAPI:
    app = FastAPI(title="Sim-Scene Review", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=_lifespan)
    app.state.registry = Registry()
    app.state.tasks = set()
    app.include_router(router)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return _error(422, "validation_error", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        # unknown paths, 405s and bad path params otherwise answer with Starlette's {"detail": ...}
        return _error(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    dist = Path(os.environ.get("FRONTEND_DIST", str(DEFAULT_DIST)))
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app


app = create_app()
```

- [ ] **Step 3: Run**

Run: `cd backend && ../.venv/Scripts/python -m pytest tests/test_api.py -q`
Expected: `14 passed`. If `test_full_run_streams_in_order` hangs, the pipeline task never ran: confirm the handlers are `async def` (a sync handler runs in a threadpool where `asyncio.create_task` has no loop).

- [ ] **Step 4: Boot the server once by hand**

Run: `cd backend && ../.venv/Scripts/python -m uvicorn app.main:app --port 8000`
Then in another shell: `curl -s http://127.0.0.1:8000/api/health`
Expected: `{"ok":true}`. Stop the server.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api.py backend/app/main.py backend/tests/test_api.py
git commit -F - <<'EOF'
feat(backend): HTTP API with resumable SSE and idempotent fixes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 10: Frontend scaffold, types and reducer

**Files:**
- Create: `frontend/` via Vite (`package.json`, `vite.config.ts`, `tsconfig*.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/styles.css`)
- Create: `frontend/src/types.ts`, `frontend/src/reducer.ts`
- Test: `frontend/src/reducer.test.ts`

**Interfaces:**
- Produces: the types in `types.ts` below; `reducer.initialState()`, `reducer.fromSnapshot(snapshot)`, `reducer.beginRun(state, runId, onlyObject)`, `reducer.reduceEvent(state, event)`, `reducer.select(state, name)`, `reducer.applyReview(state, object, metrics)`.

- [ ] **Step 1: Scaffold**

Run from the repo root:
```bash
npm create vite@latest frontend -- --template react-ts --no-interactive
cd frontend
npm install
npm install three @react-three/fiber @react-three/drei
npm install -D vitest @types/three
```
Expected: `frontend/package.json` exists with `react` 19.2.x, `vite` 8.x and `typescript` 6.x (whatever create-vite ships today; the code in Tasks 10–12 is written against the template's strict `tsconfig.app.json`: `verbatimModuleSyntax`, `erasableSyntaxOnly`, `noUnusedLocals`).

Replace `frontend/vite.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { environment: "node" },
});
```
`vitest/config` re-exports Vite's `defineConfig` with the `test` key typed, so `tsc -b` (which checks this file through `tsconfig.node.json`) stays clean.

In `frontend/package.json` add `"test": "vitest run"` to `scripts`. Delete `src/App.css`, `src/assets/`, and `public/vite.svg`; keep `src/index.css` but empty it (the app's stylesheet is written in Task 11 as `src/styles.css`).

- [ ] **Step 2: Types**

`frontend/src/types.ts`:
```ts
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
```

- [ ] **Step 3: Failing reducer tests**

`frontend/src/reducer.test.ts`:
```ts
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

  it("beginRun resets stages and clears the target object's findings", () => {
    let s = fromSnapshot(snapshot);
    s = reduceEvent(s, { type: "finding", data: { finding } });
    s = beginRun(s, "run_2", "mug");
    expect(s.runId).toBe("run_2");
    expect(s.running).toBe(true);
    expect(s.stages.load.status).toBe("pending");
    expect(s.objects.mug.findings).toEqual([]);
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
```

Run: `cd frontend && npm test`
Expected: FAIL, cannot find module `./reducer`.

- [ ] **Step 4: Reducer**

`frontend/src/reducer.ts`:
```ts
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

/** A new run starts: stages reset; an object rerun clears only that object. */
export function beginRun(state: ViewState, runId: string, onlyObject: string | null): ViewState {
  const objects = { ...state.objects };
  for (const name of onlyObject ? [onlyObject] : state.order) {
    if (objects[name]) objects[name] = { ...objects[name], status: "checking", findings: [] };
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
```

- [ ] **Step 5: Run**

Run: `cd frontend && npm test`
Expected: `9 passed`.

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -F - <<'EOF'
feat(frontend): Vite scaffold, event types and view reducer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```
Check `git status` first: `node_modules/` and `dist/` must not be staged (they are in `.gitignore`).

---

### Task 11: API client, stream hook and the review UI

**Files:**
- Create: `frontend/src/api.ts`, `frontend/src/useRunStream.ts`
- Create: `frontend/src/components/SceneInput.tsx`, `StageProgress.tsx`, `ObjectPanel.tsx`, `MetricsFooter.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/main.tsx`, create `frontend/src/styles.css`

**Interfaces:**
- Consumes: Task 9 routes, Task 10 reducer/types.
- Produces: `api.*` functions; `useRunStream(runId, onEvent)`; `App` with a `viewport` slot that Task 12 fills (`<SceneViewport>`); until then App renders a placeholder `<div className="viewport-placeholder">`.

- [ ] **Step 1: API client**

`frontend/src/api.ts`:
```ts
import type { Metrics, ObjectView, Snapshot } from "./types";

async function parse<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body?.error?.message ?? `${res.status} ${res.statusText}`);
  return body as T;
}

const post = (url: string, body?: unknown) =>
  fetch(url, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  samples: () => fetch("/api/samples").then((r) => parse<{ id: string; label: string }[]>(r)),
  createFromSample: (sample: string) =>
    post("/api/scenes", { sample }).then((r) => parse<{ scene_id: string; version: string }>(r)),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/scenes/upload", { method: "POST", body: form }).then((r) =>
      parse<{ scene_id: string; version: string }>(r),
    );
  },
  snapshot: (sceneId: string) => fetch(`/api/scenes/${sceneId}`).then((r) => parse<Snapshot>(r)),
  startRun: (sceneId: string) => post(`/api/scenes/${sceneId}/runs`).then((r) => parse<{ run_id: string }>(r)),
  fix: (sceneId: string, name: string, fixId: string) =>
    post(`/api/scenes/${sceneId}/objects/${encodeURIComponent(name)}/fix`, { fix_id: fixId }).then((r) =>
      parse<{ run_id: string; cached: boolean }>(r),
    ),
  review: (sceneId: string, name: string, decision: "approve" | "reject") =>
    post(`/api/scenes/${sceneId}/objects/${encodeURIComponent(name)}/review`, { decision }).then((r) =>
      parse<{ object: ObjectView; metrics: Metrics }>(r),
    ),
};
```

- [ ] **Step 2: Stream hook**

`frontend/src/useRunStream.ts`:
```ts
import { useEffect, useRef } from "react";
import { EVENT_TYPES, type RunEvent } from "./types";

/** Subscribes to a run's SSE stream. EventSource reconnects on its own and
 * sends Last-Event-ID, so a dropped connection resumes where it stopped. */
export function useRunStream(runId: string | null, onEvent: (event: RunEvent) => void) {
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(`/api/runs/${runId}/events`);
    for (const type of EVENT_TYPES) {
      source.addEventListener(type, (raw) => {
        const event = { type, data: JSON.parse((raw as MessageEvent).data) } as RunEvent;
        handler.current(event);
        if (type === "run_completed" || type === "run_failed") source.close();
      });
    }
    return () => source.close();
  }, [runId]);
}
```

- [ ] **Step 3: Components**

`frontend/src/components/SceneInput.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api } from "../api";

interface Props {
  busy: boolean;
  onScene: (sceneId: string) => void;
  onError: (message: string) => void;
}

export function SceneInput({ busy, onScene, onError }: Props) {
  const [samples, setSamples] = useState<{ id: string; label: string }[]>([]);

  useEffect(() => {
    api.samples().then(setSamples).catch((e) => onError(e.message));
  }, [onError]);

  const pick = (promise: Promise<{ scene_id: string }>) =>
    promise.then((r) => onScene(r.scene_id)).catch((e) => onError(e.message));

  return (
    <div className="scene-input">
      {samples.map((s) => (
        <button key={s.id} disabled={busy} onClick={() => pick(api.createFromSample(s.id))}>
          Load sample: {s.label}
        </button>
      ))}
      <label className={busy ? "upload disabled" : "upload"}>
        Upload MJCF (.xml, ≤ 1 MB)
        <input
          type="file"
          accept=".xml,text/xml"
          disabled={busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) pick(api.upload(file));
            e.target.value = "";
          }}
        />
      </label>
    </div>
  );
}
```

`frontend/src/components/StageProgress.tsx`:
```tsx
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
```

`frontend/src/components/ObjectPanel.tsx`:
```tsx
import type { ObjectView } from "../types";

interface Props {
  objects: ObjectView[];
  selected: string | null;
  busy: boolean;
  onSelect: (name: string) => void;
  onReview: (name: string, decision: "approve" | "reject") => void;
  onFix: (name: string, fixId: string) => void;
}

export function statusClass(o: ObjectView): string {
  if (o.decision === "approved") return "approved";
  if (o.decision === "rejected") return "rejected";
  if (o.status === "checking") return "checking";
  if (o.findings.some((f) => f.severity === "error")) return "error";
  if (o.findings.length) return "warning";
  return o.status === "ok" ? "ok" : "pending";
}

export function ObjectPanel({ objects, selected, busy, onSelect, onReview, onFix }: Props) {
  const current = objects.find((o) => o.name === selected) ?? null;
  return (
    <aside className="panel">
      <ul className="object-list">
        {objects.map((o) => (
          <li key={o.name} className={`${statusClass(o)} ${o.name === selected ? "selected" : ""}`}>
            <button onClick={() => onSelect(o.name)}>
              <span className="dot" />
              {o.name}
              <span className="count">{o.findings.length ? o.findings.length : ""}</span>
            </button>
          </li>
        ))}
      </ul>
      {current && (
        <div className="details">
          <h2>{current.name}</h2>
          <p className="meta">
            {current.movable ? "movable" : "fixed"} · {current.status} · {current.decision}
          </p>
          {current.findings.length === 0 && current.status === "ok" && <p>No findings.</p>}
          {current.findings.map((f, i) => (
            <div key={i} className={`finding ${f.severity}`}>
              <strong>{f.rule}</strong>
              <p>{f.message}</p>
              {f.fix && (
                {/* stays enabled while running: a second press is exactly what the server deduplicates */}
                <button onClick={() => onFix(current.name, f.fix!.fix_id)}>
                  Apply fix & rerun: {f.fix.label}
                </button>
              )}
            </div>
          ))}
          <div className="actions">
            <button disabled={busy || current.decision === "approved"} onClick={() => onReview(current.name, "approve")}>
              Approve
            </button>
            <button disabled={busy || current.decision === "rejected"} onClick={() => onReview(current.name, "reject")}>
              Reject
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
```

`frontend/src/components/MetricsFooter.tsx`:
```tsx
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
```

- [ ] **Step 4: App**

`frontend/src/App.tsx`:
```tsx
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
```

`frontend/src/main.tsx`:
```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

Set the `<title>` in `frontend/index.html` to `Sim-Scene Review`.

`frontend/src/styles.css`:
```css
:root { color-scheme: light; font-family: system-ui, sans-serif; --ok: #30a46c; --warn: #f5a524; --err: #e5484d; --muted: #8b8d98; --line: #e2e3e8; }
* { box-sizing: border-box; }
body { margin: 0; background: #f7f7f9; color: #1c1d22; }
.app { display: grid; grid-template-rows: auto auto auto 1fr auto; height: 100vh; }
header { display: flex; gap: 12px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line); background: #fff; }
header h1 { font-size: 18px; margin: 0 16px 0 0; }
button { font: inherit; padding: 6px 12px; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
button.secondary { margin-left: auto; }
.scene-input { display: flex; gap: 8px; align-items: center; }
.upload { padding: 6px 12px; border: 1px dashed var(--muted); border-radius: 6px; cursor: pointer; }
.upload.disabled { opacity: .5; }
.upload input { display: none; }
.notice { padding: 8px 16px; background: #fff4e5; border-bottom: 1px solid #f3d9a4; }
.stages { display: flex; gap: 0; margin: 0; padding: 0 16px; list-style: none; background: #fff; border-bottom: 1px solid var(--line); }
.stage { flex: 1; padding: 8px 12px; border-left: 4px solid var(--line); }
.stage.running { border-color: #3e63dd; }
.stage.done { border-color: var(--ok); }
.stage.failed { border-color: var(--err); }
.stage-name { display: block; font-weight: 600; }
.stage-meta { color: var(--muted); font-size: 12px; }
main { display: grid; grid-template-columns: 1fr 360px; min-height: 0; }
.viewport, .viewport-placeholder { position: relative; min-height: 0; background: #e9eaef; display: grid; place-items: center; color: var(--muted); }
.panel { overflow: auto; border-left: 1px solid var(--line); background: #fff; }
.object-list { list-style: none; margin: 0; padding: 8px; }
.object-list button { width: 100%; display: flex; align-items: center; gap: 8px; border: 0; background: none; text-align: left; padding: 6px 8px; border-radius: 6px; }
.object-list li.selected button { background: #eef1fb; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--line); }
li.ok .dot { background: var(--ok); } li.warning .dot { background: var(--warn); } li.error .dot { background: var(--err); }
li.approved .dot { background: var(--ok); box-shadow: 0 0 0 2px #c9eed9; } li.rejected .dot { background: var(--muted); }
li.checking .dot { background: #3e63dd; animation: pulse 1s infinite alternate; }
@keyframes pulse { from { opacity: .3; } to { opacity: 1; } }
.count { margin-left: auto; color: var(--muted); font-size: 12px; }
.details { padding: 12px; border-top: 1px solid var(--line); }
.details h2 { margin: 0 0 4px; font-size: 16px; }
.meta { color: var(--muted); font-size: 12px; margin: 0 0 12px; }
.finding { padding: 8px 10px; margin: 8px 0; border-left: 4px solid var(--warn); background: #fffaf0; border-radius: 4px; }
.finding.error { border-color: var(--err); background: #fff1f1; }
.finding p { margin: 4px 0 8px; font-size: 13px; }
.actions { display: flex; gap: 8px; margin-top: 12px; }
.metrics { display: flex; gap: 24px; padding: 10px 16px; border-top: 1px solid var(--line); background: #fff; }
.metric .label { display: block; font-size: 11px; color: var(--muted); text-transform: uppercase; }
.metric .value { font-weight: 600; }
.rate-card { margin-left: auto; align-self: center; font-size: 11px; color: var(--muted); }
```

- [ ] **Step 5: Type-check and try it against the backend**

Run: `cd frontend && npx tsc -b && npm run build`
Expected: no errors; `frontend/dist/index.html` exists.

Start the backend (`cd backend && ../.venv/Scripts/python -m uvicorn app.main:app --port 8000`) and the dev server (`cd frontend && npm run dev`). Open the dev URL. Click **Load sample**. Expected: the three stages turn blue then green within a few seconds, the object list shows `mug_a`, `plate`, `box_small`, `cabinet_door`, `tray_mm` in red/amber and `mug_b` green; selecting `mug_a` and pressing its fix twice in quick succession reruns only `mug_a`, which turns green, while the second press shows the "reusing run" notice and the footer's saved figure moves. Reload the page: the state comes back from the URL hash. Approve everything: the footer shows a time. Stop both servers.

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -F - <<'EOF'
feat(frontend): review UI driven by the run event stream

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 12: 3D viewport

**Files:**
- Create: `frontend/src/components/SceneViewport.tsx`, `frontend/src/components/geomColor.ts`
- Test: `frontend/src/components/geomColor.test.ts`
- Modify: `frontend/src/App.tsx` (replace the placeholder)

**Interfaces:**
- Consumes: `GeometryObject`, `ObjectView`, `statusClass` from `ObjectPanel`.
- Produces: `<SceneViewport geometry objects selected onSelect />`; `geomColor(object: ObjectView | undefined, rgba: number[], selected: boolean) -> string`.

- [ ] **Step 1: Failing colour test**

`frontend/src/components/geomColor.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { geomColor } from "./geomColor";
import type { ObjectView } from "../types";

const base = (over: Partial<ObjectView>): ObjectView => ({
  name: "o", movable: true, status: "ok", decision: "pending", findings: [], ...over,
});
const finding = (severity: "error" | "warning") => ({ object: "o", rule: "r", severity, message: "", fix: null });

describe("geomColor", () => {
  it("uses the geom's own colour when there is nothing to say", () => {
    expect(geomColor(base({}), [0.2, 0.4, 0.6, 1], false)).toBe("#336699");
  });
  it("paints errors red, warnings amber, approvals green, rejections grey", () => {
    expect(geomColor(base({ findings: [finding("error")] }), [1, 1, 1, 1], false)).toBe("#e5484d");
    expect(geomColor(base({ findings: [finding("warning")] }), [1, 1, 1, 1], false)).toBe("#f5a524");
    expect(geomColor(base({ decision: "approved", findings: [finding("error")] }), [1, 1, 1, 1], false)).toBe("#30a46c");
    expect(geomColor(base({ decision: "rejected" }), [1, 1, 1, 1], false)).toBe("#8b8d98");
  });
  it("world geoms keep their colour", () => {
    expect(geomColor(undefined, [0.85, 0.85, 0.85, 1], false)).toBe("#d9d9d9");
  });
});
```

Run: `cd frontend && npm test`
Expected: FAIL, cannot find module `./geomColor`.

- [ ] **Step 2: Colour helper**

`frontend/src/components/geomColor.ts`:
```ts
import type { ObjectView } from "../types";
import { statusClass } from "./ObjectPanel";

const STATUS_COLORS: Record<string, string> = {
  error: "#e5484d",
  warning: "#f5a524",
  approved: "#30a46c",
  rejected: "#8b8d98",
  checking: "#3e63dd",
};

const hex = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0");

export function geomColor(object: ObjectView | undefined, rgba: number[], selected: boolean): string {
  const own = `#${hex(rgba[0])}${hex(rgba[1])}${hex(rgba[2])}`;
  if (!object) return own;
  const color = STATUS_COLORS[statusClass(object)] ?? own;
  return selected && color === own ? "#3e63dd" : color;
}
```

Run: `cd frontend && npm test` — Expected: all pass (12).

- [ ] **Step 3: Viewport**

`frontend/src/components/SceneViewport.tsx`:
```tsx
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";
import type { GeometryObject, GeomView, ObjectView } from "../types";
import { geomColor } from "./geomColor";

interface Props {
  geometry: GeometryObject[];
  objects: Record<string, ObjectView>;
  selected: string | null;
  onSelect: (name: string) => void;
}

// MuJoCo is Z-up and puts capsule/cylinder axes on local Z; three.js builds
// them on Y, so those two get a quarter turn inside their group.
const AXIS_FIX: [number, number, number] = [Math.PI / 2, 0, 0];

function MeshGeometry({ geom }: { geom: GeomView }) {
  const buffer = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(geom.vertices ?? [], 3));
    g.setIndex(geom.faces ?? []);
    g.computeVertexNormals();
    return g;
  }, [geom.vertices, geom.faces]);
  return <primitive object={buffer} attach="geometry" />;
}

function Shape({ geom }: { geom: GeomView }) {
  const [a, b, c] = geom.size;
  switch (geom.type) {
    case "box": return <boxGeometry args={[2 * a, 2 * b, 2 * c]} />;
    case "sphere": return <sphereGeometry args={[a, 32, 16]} />;
    case "ellipsoid": return <sphereGeometry args={[1, 32, 16]} />;
    case "cylinder": return <cylinderGeometry args={[a, a, 2 * b, 32]} />;
    case "capsule": return <capsuleGeometry args={[a, 2 * b, 4, 16]} />;
    case "plane": return <planeGeometry args={[a || 20, b || 20]} />;
    case "mesh": return <MeshGeometry geom={geom} />;
    default: return null;
  }
}

function Geom({ geom, color, emissive, onClick }: { geom: GeomView; color: string; emissive: boolean; onClick?: () => void }) {
  const quaternion = useMemo(
    () => new THREE.Quaternion(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]),
    [geom.quat],
  );
  const rotated = geom.type === "cylinder" || geom.type === "capsule";
  const scale: [number, number, number] = geom.type === "ellipsoid" ? [geom.size[0], geom.size[1], geom.size[2]] : [1, 1, 1];
  return (
    <group position={geom.pos as [number, number, number]} quaternion={quaternion} onClick={onClick && ((e) => { e.stopPropagation(); onClick(); })}>
      <mesh rotation={rotated ? AXIS_FIX : undefined} scale={scale} receiveShadow castShadow>
        <Shape geom={geom} />
        <meshStandardMaterial color={color} emissive={emissive ? color : "#000000"} emissiveIntensity={emissive ? 0.35 : 0} side={geom.type === "plane" ? THREE.DoubleSide : THREE.FrontSide} />
      </mesh>
    </group>
  );
}

export function SceneViewport({ geometry, objects, selected, onSelect }: Props) {
  return (
    <div className="viewport">
      <Canvas camera={{ position: [2.4, -2.4, 1.9], up: [0, 0, 1], fov: 45, near: 0.01, far: 2000 }} shadows>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, -2, 6]} intensity={1.2} castShadow />
        {geometry.map((obj) =>
          obj.geoms.map((geom, i) => {
            const view = obj.name === "world" ? undefined : objects[obj.name];
            const isSelected = obj.name === selected;
            return (
              <Geom
                key={`${obj.name}:${i}`}
                geom={geom}
                color={geomColor(view, geom.rgba, isSelected)}
                emissive={isSelected}
                onClick={view ? () => onSelect(obj.name) : undefined}
              />
            );
          }),
        )}
        <OrbitControls target={[0, 0, 0.6]} makeDefault />
      </Canvas>
      {geometry.length === 0 && <div className="viewport-empty">Load a scene to see it here</div>}
    </div>
  );
}
```

Add to `styles.css`:
```css
.viewport canvas { display: block; }
.viewport-empty { position: absolute; inset: 0; display: grid; place-items: center; color: var(--muted); pointer-events: none; }
```

In `App.tsx` replace the placeholder `<div className="viewport-placeholder">…</div>` with:
```tsx
<SceneViewport
  geometry={state.geometry}
  objects={state.objects}
  selected={state.selected}
  onSelect={(n) => setState((s) => select(s, n))}
/>
```
and add `import { SceneViewport } from "./components/SceneViewport";`.

- [ ] **Step 4: Check it renders**

Run: `cd frontend && npx tsc -b && npm run build` — Expected: clean.
Start both servers as in Task 11 step 5, load the sample. Expected: a grey floor, a brown table, a red plate, two mugs (one red, one green), a red box half inside the table top (status colour overrides its own orange because it carries an error), a cabinet with a red door; the sample's huge tray is 200 m away so it is out of frame, which is intended. Clicking a mug selects it in the panel; selecting in the panel highlights the geom. Orbit with the mouse. Apply the `tray_mm` fix; it stays out of frame and turns green in the list.

- [ ] **Step 5: Commit**

```bash
git add frontend
git commit -F - <<'EOF'
feat(frontend): React Three Fiber viewport drawing MuJoCo's compiled geometry

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

---

### Task 13: Docker, Render and README

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `render.yaml`
- Modify: `README.md` (already exists with the local-run instructions; replace it wholesale with the version below, which keeps them)

- [ ] **Step 1: Docker**

`Dockerfile`:
```dockerfile
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt ./
# Without this, `import mujoco` tries to load GLFW, which needs libX11 the slim image lacks.
ENV MUJOCO_GL=disable
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import mujoco, fastapi; print(mujoco.__version__)"
COPY backend/ ./
COPY --from=web /web/dist ./frontend_dist
ENV FRONTEND_DIST=/app/frontend_dist
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
```

`.dockerignore`:
```
.git
.venv
node_modules
frontend/node_modules
frontend/dist
**/__pycache__
**/.pytest_cache
docs
```

`render.yaml`:
```yaml
services:
  - type: web
    name: sim-scene-review
    runtime: docker
    plan: free
    healthCheckPath: /api/health
    autoDeploy: true
```

If Docker Desktop is installed: `docker build -t sim-scene-review . && docker run --rm -p 8000:8000 sim-scene-review`, open `http://localhost:8000`, load the sample. Expected: the same flow as Task 12 served from the image. If Docker is not installed, skip; Render builds the image.

- [ ] **Step 2: README**

`README.md`:
````markdown
# Sim-Scene Review

Check a MuJoCo scene before you train or evaluate a robot policy in it.
Upload an MJCF file (or load the sample), watch three stages stream in,
then approve, reject or fix-and-rerun each object.

**Live:** _(URL added after the first deploy)_

## What it checks

| Stage | What happens | Findings |
| -- | -- | -- |
| Load & compile | MuJoCo compiles the scene. Compile errors end the run with the compiler's message. | — |
| Static asset checks | Rules over the compiled model, one object at a time. | `zero_mass`, `no_collision`, `default_friction`, `unlimited_joint`, `implausible_scale` |
| Simulation check | 2 s of simulated time, capped at 10 s wall. | `initial_penetration`, `exploded`, `fell_through_floor`, `unstable` |

Four findings carry a one-click fix (recompute mass, enable collision, add
joint limits, scale millimetres to metres). A fix edits the scene through
MuJoCo's `MjSpec`, the result is compiled fresh, and only that object is
re-checked. There is no inertia rule: MuJoCo's compiler already refuses bad
inertia and zero mass, so those surface at stage 1.

## How it is built

- **Backend** — FastAPI. The scene is compiled once with `mujoco`; the checks
  and the geometry the browser draws read that one compiled model, so what
  you see is what is simulated. Fixes edit the XML through `MjSpec` and the
  result is compiled fresh. Rules are pure functions;
  one that throws becomes a `check_error` finding instead of killing the run.
- **Streaming** — each run keeps its events in memory with increasing ids.
  The browser's `EventSource` reconnects on its own and sends
  `Last-Event-ID`; the server replays from there. Waiting is a blocking
  `Condition.wait_for` in a worker thread, so there is no polling loop.
- **Idempotency** — a fix request is keyed by `scene version + object + fix`,
  recorded for both the version the fix was applied to and the version it
  produced. Pressing the same fix twice, even while the first rerun is still
  going, returns the original run and charges nothing; the footer shows what
  that saved.
- **Frontend** — Vite, React, TypeScript, React Three Fiber. Events fold
  into view state through a pure reducer (`src/reducer.ts`, unit-tested);
  after each run the page re-reads the server snapshot so decisions and the
  scene version are authoritative.
- **One image** — Node builds the frontend; `python:3.12-slim` serves the
  API and the static files from one origin.

## What is measured

- **Time to usable scene** — from scene creation until every object is
  approved. This is the number I would watch first in a real product.
- **Manual fixes** and **reruns** — how much human and compute effort a
  scene needed.
- **Cost** — from a fake rate card (compile $0.001/run, static $0.0005/object,
  simulation $0.01/simulated second) so the idempotency saving is visible.

## Deliberate trade-offs

- State is in memory: 200 scenes, two-hour TTL, gone on restart. A real
  service would persist runs and let the SSE buffer be rebuilt from storage.
- Uploads must be a single MJCF file under 1 MB with no `<include>` and no
  `file=` attributes, so the server never opens a path an upload names.
  Inline mesh vertices still work.
- An object rerun re-checks only that object. Other objects keep their
  earlier findings even though the fix changed the scene; **Run again**
  does the full pass.
- The simulation rules use thresholds (20 m/s, 0.5 m below the floor) rather
  than physics-aware judgement. They catch the gross failures a person would
  otherwise find by eye in the viewer.
- Canonicalisation re-serialises the upload through MuJoCo, which rounds
  every number to 6 significant digits.
- No rate limiting. A scene keeps only its last 20 runs.
- Only MJCF. USD and zip uploads with mesh files are the obvious next inputs.
- No end-to-end browser tests; the backend and the reducer are unit-tested.

## Sample scene

`backend/app/samples/tabletop.xml` plants: a massless mug, a plate with
collision off, a box starting inside the table top (with default friction),
a cabinet door hinge without limits, and a tray modelled in millimetres. The
tray sits 200 m away so its 1000× size does not swallow the table until you
fix it. `mug_b` is the control.

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements-dev.txt   # macOS/Linux: .venv/bin/python
cd backend
../.venv/Scripts/python -m pytest -q
../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
# second shell
cd frontend && npm install && npm test && npm run dev
```

Open the Vite URL; `/api` is proxied to port 8000.

Or run the deployed image locally:

```bash
docker build -t sim-scene-review .
docker run --rm -p 8000:8000 sim-scene-review
```

Then open `http://localhost:8000`.

## Deploy

`render.yaml` describes one free Docker web service on Render. Cold starts
on the free plan take about 30 s. Health check: `GET /api/health`.

## Note on tooling

Built with Claude Code as a pair; every design decision, test and trade-off
above was reviewed and is mine to defend.
````

- [ ] **Step 3: Full test pass**

Run: `cd backend && ../.venv/Scripts/python -m pytest -q` — Expected: all pass.
Run: `cd frontend && npm test && npx tsc -b && npm run build` — Expected: all pass, clean build.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore render.yaml README.md
git commit -F - <<'EOF'
chore: Docker image, Render service and README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014KtsJ752zhNwAEnio4pMEq
EOF
```

- [ ] **Step 5: Publish and deploy (needs the owner's accounts)**

1. Create an empty public repo `sim-scene-review` on the owner's GitHub account, then `git remote add origin <url> && git push -u origin main`.
2. On Render: New → Blueprint → pick the repo; it reads `render.yaml`. Wait for the build.
3. Open the service URL, load the sample, run the five success criteria from the spec. Put the URL into `README.md` under **Live** and commit.
````
