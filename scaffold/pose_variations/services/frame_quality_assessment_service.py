from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
from PIL import Image


@dataclass(slots=True)
class FrameQualityAssessmentResult:
    sharpness_score: float
    perceptual_hash_value: str


class FrameQualityAssessmentService:
    def assess_frame_quality(self, frame_file_path: Path) -> FrameQualityAssessmentResult:
        loaded_image_matrix = cv2.imread(str(frame_file_path))
        if loaded_image_matrix is None:
            raise ValueError(f"Unable to load frame for quality assessment: {frame_file_path}")

        grayscale_image_matrix = cv2.cvtColor(loaded_image_matrix, cv2.COLOR_BGR2GRAY)
        sharpness_score = float(cv2.Laplacian(grayscale_image_matrix, cv2.CV_64F).var())

        with Image.open(frame_file_path) as loaded_pillow_image:
            perceptual_hash_value = str(imagehash.phash(loaded_pillow_image))

        return FrameQualityAssessmentResult(
            sharpness_score=sharpness_score,
            perceptual_hash_value=perceptual_hash_value,
        )

    def calculate_perceptual_hash_distance(
        self,
        first_perceptual_hash_value: str,
        second_perceptual_hash_value: str,
    ) -> int:
        first_perceptual_hash = imagehash.hex_to_hash(first_perceptual_hash_value)
        second_perceptual_hash = imagehash.hex_to_hash(second_perceptual_hash_value)
        return int(first_perceptual_hash - second_perceptual_hash)
