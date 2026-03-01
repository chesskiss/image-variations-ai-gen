from __future__ import annotations

from typing import Final

POSE_PRESET_PROMPTS_BY_NAME: Final[dict[str, str]] = {
    "rotate_right": (
        "Create a 5-second motion where the subject rotates to the right into a three-quarter back profile "
        "while preserving identity, outfit, hair, and lighting."
    ),
    "rotate_left": (
        "Create a 5-second motion where the subject rotates to the left into a three-quarter back profile "
        "while preserving identity, outfit, hair, and lighting."
    ),
    "step_forward_head_tilt": (
        "Create a 5-second subtle movement where the subject steps slightly forward and tilts the head "
        "while preserving identity, outfit, and scene consistency."
    ),
}
