"""Validate, canonicalise and compile MJCF. Everything downstream works on a
compiled MjModel, so the browser and the checks see exactly one scene."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import mujoco

from .models import ObjectInfo

MAX_UPLOAD_BYTES = 1_000_000
# A compiled scene past these is a denial-of-service, not a review candidate.
MAX_BODIES = 500
MAX_GEOMS = 2000
MAX_MESH_VERTICES = 200_000
MAX_ARENA_BYTES = 64 * 1024 * 1024

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
    check_size(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return CompiledScene(xml, scene_version(xml), model, data, list_objects(model))


def check_size(model: mujoco.MjModel) -> None:
    limits = (
        ("bodies", model.nbody, MAX_BODIES),
        ("geoms", model.ngeom, MAX_GEOMS),
        ("mesh vertices", model.nmeshvert, MAX_MESH_VERTICES),
        ("arena bytes", model.narena, MAX_ARENA_BYTES),
    )
    for what, count, cap in limits:
        if count > cap:
            raise SceneError("scene_too_large", f"scene has {count} {what}; the limit is {cap}")


def list_objects(model: mujoco.MjModel) -> list:
    objects = []
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body_{body}"
        geoms = tuple(int(g) for g in range(model.ngeom) if model.geom_bodyid[g] == body)
        # body_weldid is 0 when nothing between this body and the world can move
        movable = bool(model.body_weldid[body] != 0)
        objects.append(ObjectInfo(name, body, movable, geoms))
    return objects
