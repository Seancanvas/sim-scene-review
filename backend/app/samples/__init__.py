from __future__ import annotations

from pathlib import Path

SAMPLES = {"tabletop": "Tabletop with planted issues"}


def load_sample(name: str) -> str:
    return Path(__file__).with_name(f"{name}.xml").read_text(encoding="utf-8")
