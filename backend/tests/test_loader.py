import pytest

from app import loader
from app.loader import SceneError, canonicalise, compile_scene, scene_version, validate_xml
from tests.scenes import wrap

BALL = '<body name="ball" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/></body>'


def test_rejects_include():
    with pytest.raises(SceneError) as e:
        validate_xml(wrap('<include file="other.xml"/>'))
    assert e.value.code == "include_not_allowed"


def test_rejects_file_attribute():
    with pytest.raises(SceneError) as e:
        validate_xml(wrap(BALL, extra='<asset><mesh name="m" file="m.stl"/></asset>'))
    assert e.value.code == "file_reference_not_allowed"


def test_rejects_oversize():
    with pytest.raises(SceneError) as e:
        validate_xml("<mujoco>" + " " * 1_000_001 + "</mujoco>")
    assert e.value.code == "too_large"


def test_rejects_non_mjcf():
    with pytest.raises(SceneError) as e:
        validate_xml("<html><body>hello</body></html>")
    assert e.value.code == "not_mjcf"


def test_compile_error_is_scene_error():
    massless = (
        '<body name="b" pos="0 0 1"><joint type="free"/>'
        '<inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/>'
        '<geom type="box" size="0.1 0.1 0.1"/></body>'
    )
    with pytest.raises(SceneError) as e:
        compile_scene(wrap(massless))
    assert e.value.code == "compile_error"
    assert "mass" in e.value.message


def boxes(count):
    return "".join(
        f'<body name="b{i}" pos="{i * 0.3} 0 1"><geom type="box" size="0.1 0.1 0.1"/></body>' for i in range(count)
    )


def test_rejects_scene_with_too_many_bodies():
    with pytest.raises(SceneError) as e:
        compile_scene(wrap(boxes(loader.MAX_BODIES)))  # plus the world body makes one too many
    assert e.value.code == "scene_too_large"
    assert "bodies" in e.value.message


def test_accepts_a_small_scene():
    assert len(compile_scene(wrap(boxes(10))).objects) == 10


def test_lists_objects_and_movability():
    xml = wrap(
        '<body name="table"><geom type="box" size="1 1 0.1"/></body>'
        '<body name="cup" pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/>'
        '<body name="handle" pos="0.05 0 0"><geom type="sphere" size="0.01"/></body></body>'
    )
    scene = compile_scene(xml)
    assert [o.name for o in scene.objects] == ["table", "cup", "handle"]
    assert [o.movable for o in scene.objects] == [False, True, True]
    assert all(len(o.geom_ids) == 1 for o in scene.objects)
    assert scene.version == scene_version(xml)


def test_canonicalise_names_unnamed_bodies_and_is_stable():
    xml = canonicalise(wrap('<body pos="0 0 1"><joint type="free"/><geom type="sphere" size="0.05"/></body>'))
    assert 'name="body_1"' in xml
    assert canonicalise(xml) == xml
