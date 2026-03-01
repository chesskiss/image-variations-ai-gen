from __future__ import annotations

from frame_extraction_stage.domain_models import ExtractedFrameAsset
from frame_judge_stage.domain_models import FrameCandidateForJudging


def map_extracted_frame_assets_to_frame_candidates_for_judging(
    extracted_frame_assets: list[ExtractedFrameAsset],
) -> list[FrameCandidateForJudging]:
    return [
        FrameCandidateForJudging(
            generation_job_identifier=extracted_frame_asset.generation_job_identifier,
            pose_preset_identifier=extracted_frame_asset.pose_preset_identifier,
            frame_sequence_number=extracted_frame_asset.frame_sequence_number,
            timestamp_seconds=extracted_frame_asset.timestamp_seconds,
            local_frame_image_file_path=extracted_frame_asset.local_frame_image_file_path,
            image_width_pixels=extracted_frame_asset.image_width_pixels,
            image_height_pixels=extracted_frame_asset.image_height_pixels,
            file_size_bytes=extracted_frame_asset.file_size_bytes,
            basic_quality_metrics=extracted_frame_asset.basic_quality_metrics,
        )
        for extracted_frame_asset in extracted_frame_assets
    ]
