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
