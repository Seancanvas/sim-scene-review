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
