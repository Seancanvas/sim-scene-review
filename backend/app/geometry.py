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
