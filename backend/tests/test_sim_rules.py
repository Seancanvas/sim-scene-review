import time

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


# Explicit Euler with a 1e9 N/m spring at 10 ms steps blows up on the second
# step; MuJoCo's autoreset then wipes the state, so only its warning counter
# can tell us. Mirrors the autoreset path the rule exists for.
DIVERGING = (
    '<body name="spring" pos="0 0 1">'
    '<joint name="j" type="slide" axis="0 0 1" stiffness="1e9" damping="0"/>'
    '<geom type="sphere" size="0.05"/></body>'
)


def test_unstable_detected_through_mujoco_warning_under_autoreset():
    started = time.monotonic()
    result = run(DIVERGING, extra='<option integrator="Euler" timestep="0.01"/>')
    elapsed = time.monotonic() - started
    assert rules(result) == [("spring", "unstable")]
    assert result.simulated_seconds < 0.5
    assert elapsed < 1.0


def test_unstable_names_the_diverging_object_not_every_movable_one():
    two = DIVERGING + ball(0.05).replace("0 0 0.05", "1 0 0.05")
    result = run(two, extra='<option integrator="Euler" timestep="0.01"/>')
    assert rules(result) == [("spring", "unstable")]


def test_unstable_still_detects_nan_without_autoreset(monkeypatch):
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
