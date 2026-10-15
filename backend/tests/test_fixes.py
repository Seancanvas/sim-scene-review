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
