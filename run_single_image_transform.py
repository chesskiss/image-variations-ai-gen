from __future__ import annotations

import argparse
from pathlib import Path

from single_image_transform.config import load_single_image_transform_settings
from single_image_transform.fal_image_transform_client import FalImageTransformClient
from single_image_transform.prompts import POSE_PRESET_PROMPTS_BY_NAME


def parse_command_line_arguments() -> argparse.Namespace:
    command_line_parser = argparse.ArgumentParser(
        description="Run a minimal one-image fal.ai transform flow."
    )
    command_line_parser.add_argument("--image", required=True, help="Local input image path")
    command_line_parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(POSE_PRESET_PROMPTS_BY_NAME.keys()),
        help="Prompt preset name",
    )
    return command_line_parser.parse_args()


def main() -> None:
    command_line_arguments = parse_command_line_arguments()
    settings = load_single_image_transform_settings()

    input_image_file_path = Path(command_line_arguments.image).expanduser().resolve()
    selected_prompt_template = POSE_PRESET_PROMPTS_BY_NAME[command_line_arguments.preset]

    fal_image_transform_client = FalImageTransformClient(settings)
    output_file_path = fal_image_transform_client.transform_local_image(
        input_image_file_path=input_image_file_path,
        prompt_template=selected_prompt_template,
    )

    print(f"Generated asset written to: {output_file_path}")


if __name__ == "__main__":
    main()
