from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

# "error" blocks approval in spirit; "warning" is advice. The UI colours by this.
Severity = str


@dataclass(frozen=True)
class Fix:
    fix_id: str
    label: str


@dataclass(frozen=True)
class Finding:
    object: str
    rule: str
    severity: Severity
    message: str
    fix: Optional[Fix] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObjectInfo:
    name: str
    body_id: int
    movable: bool
    geom_ids: tuple
