from __future__ import annotations

from dataclasses import dataclass

from pose_variations.domain.pose_preset_models import PosePresetDefinition, PosePresetIdentifier


@dataclass(slots=True, frozen=True)
class PromptTemplateDefinition:
    pose_preset_definition: PosePresetDefinition
    internal_prompt_template: str


class PromptTemplateRepository:
    def __init__(self) -> None:
        self._prompt_template_by_identifier: dict[PosePresetIdentifier, PromptTemplateDefinition] = {
            PosePresetIdentifier("rotate_right"): PromptTemplateDefinition(
                pose_preset_definition=PosePresetDefinition(
                    pose_preset_identifier=PosePresetIdentifier("rotate_right"),
                    pose_preset_display_name="Rotate Right",
                ),
                internal_prompt_template=(
                    "Create a 5-second cinematic motion where the subject turns slowly to the right "
                    "to reveal a back-facing three-quarter profile while preserving exact identity, "
                    "clothing, hair, and lighting continuity."
                ),
            ),
            PosePresetIdentifier("rotate_left"): PromptTemplateDefinition(
                pose_preset_definition=PosePresetDefinition(
                    pose_preset_identifier=PosePresetIdentifier("rotate_left"),
                    pose_preset_display_name="Rotate Left",
                ),
                internal_prompt_template=(
                    "Create a 5-second cinematic motion where the subject turns slowly to the left "
                    "to reveal a back-facing three-quarter profile while preserving exact identity, "
                    "clothing, hair, and lighting continuity."
                ),
            ),
            PosePresetIdentifier("step_forward_head_tilt"): PromptTemplateDefinition(
                pose_preset_definition=PosePresetDefinition(
                    pose_preset_identifier=PosePresetIdentifier("step_forward_head_tilt"),
                    pose_preset_display_name="Step Forward + Head Tilt",
                ),
                internal_prompt_template=(
                    "Create a 5-second subtle motion where the subject takes a gentle step forward and "
                    "adds a slight head tilt, maintaining the same outfit, body shape, and facial identity."
                ),
            ),
        }

    def list_pose_presets_for_display(self) -> list[PosePresetDefinition]:
        return [
            prompt_template_definition.pose_preset_definition
            for prompt_template_definition in self._prompt_template_by_identifier.values()
        ]

    def is_valid_pose_preset_identifier(self, pose_preset_identifier: str) -> bool:
        return PosePresetIdentifier(pose_preset_identifier) in self._prompt_template_by_identifier

    def get_prompt_template_for_identifier(self, pose_preset_identifier: PosePresetIdentifier) -> str:
        return self._prompt_template_by_identifier[pose_preset_identifier].internal_prompt_template
