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
