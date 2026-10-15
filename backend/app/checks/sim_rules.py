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


# MuJoCo's default autoreset wipes MjData inside mj_step the moment a qacc goes
# NaN or huge, so a post-step look at qacc sees a fresh scene. The warning
# counters survive the reset and name the dof that blew up.
_INSTABILITY_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)


def _warning_counts(data):
    return [int(data.warning[w].number) for w in _INSTABILITY_WARNINGS]


def _owner(model, body_id: int, by_body: dict):
    while body_id > 0:
        obj = by_body.get(body_id)
        if obj is not None:
            return obj
        body_id = int(model.body_parentid[body_id])
    return None


def _blame(model, data, warning, movable):
    """The movable object whose dof the warning names, or every movable
    object when the index cannot be mapped (BADCTRL names an actuator)."""
    by_body = {o.body_id: o for o in movable}
    info = int(data.warning[warning].lastinfo)
    dof = None
    if warning in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL):
        dof = info
    elif warning == mujoco.mjtWarning.mjWARN_BADQPOS:
        hits = [j for j in range(model.njnt) if int(model.jnt_qposadr[j]) <= info]
        dof = int(model.jnt_dofadr[hits[-1]]) if hits else None
    if dof is not None and 0 <= dof < model.nv:
        obj = _owner(model, int(model.dof_bodyid[dof]), by_body)
        if obj is not None:
            return [obj]
    return list(movable)


def simulate(model, data, objects) -> SimResult:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    findings = initial_penetrations(model, data, objects)
    floor_z = floor_height(model, data)
    movable = [o for o in objects if o.movable]
    exploded, unstable = set(), set()
    started = time.monotonic()
    warnings_before = _warning_counts(data)
    while data.time < SIM_SECONDS and time.monotonic() - started < WALL_SECONDS:
        mujoco.mj_step(model, data)
        warnings_after = _warning_counts(data)
        if warnings_after != warnings_before:
            for warning, before, after in zip(_INSTABILITY_WARNINGS, warnings_before, warnings_after):
                if after > before:
                    unstable.update(o.name for o in _blame(model, data, warning, movable))
            # the state has been reset; nothing stepped after this is meaningful
            break
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
