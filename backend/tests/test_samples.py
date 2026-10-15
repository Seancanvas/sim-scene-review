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
