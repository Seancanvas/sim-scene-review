"""Tiny MJCF builders for tests. `extra` lands before <worldbody> (assets, options)."""


def wrap(worldbody: str, extra: str = "") -> str:
    return (
        "<mujoco>"
        f"{extra}"
        '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
        f"{worldbody}</worldbody></mujoco>"
    )
