from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

PosePresetIdentifier = NewType("PosePresetIdentifier", str)


@dataclass(slots=True, frozen=True)
class PosePresetDefinition:
    pose_preset_identifier: PosePresetIdentifier
    pose_preset_display_name: str
