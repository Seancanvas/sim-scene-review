import pytest

from app import pipeline
from app.fixes import apply_fix
from app.loader import canonicalise, compile_scene
from app.runs import Registry
from app.samples import load_sample
from tests.scenes import wrap

PLANTED = {
    "table": set(),
    "cabinet": set(),
    "mug_a": {"zero_mass"},
    "mug_b": set(),
    "plate": {"no_collision", "fell_through_floor"},
    "box_small": {"initial_penetration", "default_friction"},
    "cabinet_door": {"unlimited_joint"},
    "tray_mm": {"implausible_scale"},
}


def make_scene(reg, xml):
    compiled = compile_scene(canonicalise(xml))
    return reg.add_scene(compiled.xml, compiled.version, compiled.objects)


def run_full(reg, scene):
    run = reg.add_run(scene.scene_id)
    pipeline.execute(scene, run)
    return run


def types(run):
    return [e.type for e in run.events_after(0)]


@pytest.fixture
def tabletop():
    reg = Registry()
    return reg, make_scene(reg, load_sample("tabletop"))


def test_full_run_finds_exactly_the_planted_issues(tabletop):
    reg, scene = tabletop
    run = run_full(reg, scene)
    assert run.done
    assert {name: {f.rule for f in state.findings} for name, state in scene.objects.items()} == PLANTED
    assert scene.objects["mug_b"].status == "ok"
    assert scene.objects["mug_a"].status == "has_issues"


def test_full_run_event_order_and_cost(tabletop):
    reg, scene = tabletop
    run = run_full(reg, scene)
    seen = types(run)
    assert seen[0] == "stage_started" and seen[-1] == "run_completed"
    assert seen.index("geometry") < seen.index("finding")
    assert [e.data["stage"] for e in run.events_after(0) if e.type == "stage_completed"] == ["load", "static", "simulation"]
    # the stepper may overshoot 2.0 s by one 2 ms step, hence the tolerance
    assert scene.cost == pytest.approx(0.001 + 0.0005 * 8 + 0.01 * 2.0, abs=1e-4)
    assert run.events_after(0)[-1].data["metrics"]["cost"] == pytest.approx(scene.cost, abs=1e-4)


def test_object_rerun_replaces_only_that_object(tabletop):
    reg, scene = tabletop
    run_full(reg, scene)
    scene.objects["mug_a"].decision = "approved"
    scene.objects["plate"].decision = "approved"
    before = scene.version
    xml = apply_fix(scene.xml, "mug_a", "recompute_mass")
    run = reg.add_run(scene.scene_id, "mug_a")
    pipeline.execute(scene, run, xml=xml, only="mug_a")
    assert scene.version != before and scene.xml == xml
    assert scene.objects["mug_a"].findings == []
    assert scene.objects["mug_a"].decision == "pending"
    assert scene.objects["plate"].decision == "approved"
    assert {f.rule for f in scene.objects["plate"].findings} == PLANTED["plate"]
    assert scene.reruns == 1
    assert all(e.data["finding"]["object"] == "mug_a" for e in run.events_after(0) if e.type == "finding")


def test_object_rerun_with_broken_xml_keeps_version(tabletop):
    reg, scene = tabletop
    run_full(reg, scene)
    before = (scene.version, scene.xml)
    run = reg.add_run(scene.scene_id, "mug_a")
    pipeline.execute(scene, run, xml="<mujoco><worldbody><body name='x' pos='0 0 1'><joint type='free'/><inertial pos='0 0 0' mass='0' diaginertia='0 0 0'/><geom type='box' size='1 1 1'/></body></worldbody></mujoco>", only="mug_a")
    assert run.done
    assert types(run)[-1] == "run_failed"
    assert run.events_after(0)[-1].data["code"] == "compile_error"
    assert (scene.version, scene.xml) == before


def test_scene_without_objects_completes():
    reg = Registry()
    scene = make_scene(reg, wrap(""))
    run = run_full(reg, scene)
    assert types(run)[-1] == "run_completed"
    assert scene.metrics()["time_to_usable_s"] is None
