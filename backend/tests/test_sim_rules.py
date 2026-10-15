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
