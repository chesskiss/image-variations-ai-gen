from __future__ import annotations

from frame_extraction_stage.domain_models import ExtractedFrameAsset, GeneratedVideoAsset


def build_generation_job_frame_extraction_status_payload(
    generated_video_asset: GeneratedVideoAsset,
    extracted_frame_assets: list[ExtractedFrameAsset],
) -> dict[str, object]:
    return {
        "generation_job_identifier": str(generated_video_asset.generation_job_identifier),
        "pose_preset_identifier": str(generated_video_asset.pose_preset_identifier),
        "source_video_local_file_path": str(generated_video_asset.local_video_file_path),
        "extracted_frames": [
            {
                "generation_job_identifier": str(extracted_frame_asset.generation_job_identifier),
                "pose_preset_identifier": str(extracted_frame_asset.pose_preset_identifier),
                "frame_sequence_number": extracted_frame_asset.frame_sequence_number,
                "timestamp_seconds": extracted_frame_asset.timestamp_seconds,
                "local_frame_image_file_path": str(
                    extracted_frame_asset.local_frame_image_file_path
                ),
                "image_file_format": extracted_frame_asset.image_file_format,
                "image_width_pixels": extracted_frame_asset.image_width_pixels,
                "image_height_pixels": extracted_frame_asset.image_height_pixels,
                "file_size_bytes": extracted_frame_asset.file_size_bytes,
                "basic_quality_metrics": extracted_frame_asset.basic_quality_metrics,
            }
            for extracted_frame_asset in extracted_frame_assets
        ],
    }
