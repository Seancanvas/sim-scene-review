from app.checks import static_rules
from app.checks.static_rules import (
    default_friction,
    implausible_scale,
    no_collision,
    run_static_rules,
    unlimited_joint,
    zero_mass,
)
from app.loader import compile_scene
from tests.scenes import wrap


def findings_for(body_xml, rule=None):
    scene = compile_scene(wrap(body_xml))
    if rule is None:
        return run_static_rules(scene.model, scene.data, scene.objects)
    return [f for obj in scene.objects for f in rule(scene.model, scene.data, obj)]


def ball(geom="", inertial=""):
    return (
        f'<body name="ball" pos="0 0 1"><joint type="free"/>{inertial}'
        f'<geom type="sphere" size="0.05" {geom}/></body>'
    )


MASSLESS = '<inertial pos="0 0 0" mass="1e-7" diaginertia="1e-10 1e-10 1e-10"/>'


def test_zero_mass_fires_for_massless_movable():
    found = findings_for(ball(inertial=MASSLESS), zero_mass)
    assert [f.rule for f in found] == ["zero_mass"]
    assert found[0].severity == "error"
    assert found[0].fix.fix_id == "recompute_mass"


def test_zero_mass_ignores_static_body():
    static = '<body name="post"><inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/><geom type="sphere" size="0.05"/></body>'
    assert findings_for(static, zero_mass) == []


def test_no_collision_fires_when_all_geoms_disabled():
    found = findings_for(ball(geom='contype="0" conaffinity="0"'), no_collision)
    assert [f.rule for f in found] == ["no_collision"]
    assert found[0].fix.fix_id == "enable_collision"


def test_no_collision_quiet_when_one_geom_collides():
    two = (
        '<body name="b" pos="0 0 1"><joint type="free"/>'
        '<geom type="sphere" size="0.05" contype="0" conaffinity="0"/>'
        '<geom type="sphere" size="0.02" pos="0.05 0 0"/></body>'
    )
    assert findings_for(two, no_collision) == []


def test_default_friction_warns_on_movable_object():
    found = findings_for(ball(), default_friction)
    assert [(f.rule, f.severity) for f in found] == [("default_friction", "warning")]
    assert found[0].fix is None


def test_default_friction_quiet_when_set():
    assert findings_for(ball(geom='friction="0.8 0.005 0.0001"'), default_friction) == []


DOOR = '<body name="door" pos="0 0 1"><joint name="h" type="hinge" axis="0 0 1" {attrs}/><geom type="box" size="0.1 0.01 0.1"/></body>'


def test_unlimited_joint_fires_for_hinge_without_range():
    found = findings_for(DOOR.format(attrs=""), unlimited_joint)
    assert [f.rule for f in found] == ["unlimited_joint"]
    assert "'h'" in found[0].message
    assert found[0].fix.fix_id == "limit_joints"


def test_unlimited_joint_quiet_with_range():
    assert findings_for(DOOR.format(attrs='range="-1 1"'), unlimited_joint) == []


def test_unlimited_joint_ignores_free_joint():
    assert findings_for(ball(), unlimited_joint) == []


def test_scale_fires_with_fix_for_millimetre_box():
    tray = '<body name="tray" pos="0 0 5"><joint type="free"/><geom type="box" size="150 100 5"/></body>'
    found = findings_for(tray, implausible_scale)
    assert [(f.rule, f.severity) for f in found] == [("implausible_scale", "error")]
    assert found[0].fix.fix_id == "scale_mm_to_m"
    assert "millimetres" in found[0].message


def test_scale_warns_without_fix_when_simply_huge():
    huge = '<body name="huge"><geom type="box" size="20000 20000 20000"/></body>'
    found = findings_for(huge, implausible_scale)
    assert [(f.rule, f.severity) for f in found] == [("implausible_scale", "warning")]
    assert found[0].fix is None
    assert "check its units" in found[0].message and "millimetres" not in found[0].message


def test_scale_warns_for_tiny_object():
    # not ball(): that helper already sets size, and a duplicate attribute is invalid XML
    dot = '<body name="dot" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.001"/></body>'
    found = findings_for(dot, implausible_scale)
    assert [(f.rule, f.severity) for f in found] == [("implausible_scale", "warning")]


def test_empty_body_produces_no_findings():
    empty = '<body name="empty" pos="0 0 1"><joint type="free"/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/></body>'
    assert findings_for(empty) == []


def test_broken_rule_becomes_check_error(monkeypatch):
    def explode(model, data, obj):
        raise RuntimeError("boom")

    monkeypatch.setattr(static_rules, "STATIC_RULES", (explode,))
    found = findings_for(ball())
    assert [(f.rule, f.severity) for f in found] == [("check_error", "warning")]
    assert found[0].message == "explode failed: RuntimeError: boom"
