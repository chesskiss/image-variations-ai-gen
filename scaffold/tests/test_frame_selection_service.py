from __future__ import annotations

from pathlib import Path

import pytest

from pose_variations.domain.asset_models import AssetIdentifier, ExtractedFrameAsset, UploadedImageAsset
from pose_variations.infrastructure.settings import ApplicationSettings
from pose_variations.services.frame_quality_assessment_service import FrameQualityAssessmentService
from pose_variations.services.frame_selection_service import FrameSelectionService


@pytest.mark.asyncio
async def test_frame_selection_prefers_sharp_and_diverse_frames() -> None:
    application_settings = ApplicationSettings(
        enable_openai_similarity_judge=False,
        minimum_diversity_hash_distance=8,
    )
    frame_selection_service = FrameSelectionService(
        application_settings=application_settings,
        frame_quality_assessment_service=FrameQualityAssessmentService(),
        openai_identity_similarity_judge=None,
    )

    uploaded_image_asset = UploadedImageAsset(
        asset_identifier=AssetIdentifier("uploaded-asset"),
        local_file_path=Path("/tmp/uploaded.jpg"),
        public_asset_url="/jobs/example/assets/uploads/uploaded.jpg",
        mime_type="image/jpeg",
        original_file_name="uploaded.jpg",
    )

    candidate_frame_assets = [
        ExtractedFrameAsset(
            asset_identifier=AssetIdentifier("frame-a"),
            generated_video_asset_identifier=AssetIdentifier("video-1"),
            frame_timestamp_seconds=0.0,
            local_file_path=Path("/tmp/frame-a.jpg"),
            public_asset_url="/jobs/example/assets/frames/frame-a.jpg",
            sharpness_score=200.0,
            perceptual_hash_value="ffffffff00000000",
        ),
        ExtractedFrameAsset(
            asset_identifier=AssetIdentifier("frame-b"),
            generated_video_asset_identifier=AssetIdentifier("video-1"),
            frame_timestamp_seconds=1.0,
            local_file_path=Path("/tmp/frame-b.jpg"),
            public_asset_url="/jobs/example/assets/frames/frame-b.jpg",
            sharpness_score=150.0,
            perceptual_hash_value="00000000ffffffff",
        ),
        ExtractedFrameAsset(
            asset_identifier=AssetIdentifier("frame-c"),
            generated_video_asset_identifier=AssetIdentifier("video-1"),
            frame_timestamp_seconds=1.5,
            local_file_path=Path("/tmp/frame-c.jpg"),
            public_asset_url="/jobs/example/assets/frames/frame-c.jpg",
            sharpness_score=120.0,
            perceptual_hash_value="ffffffff00000001",
        ),
    ]

    frame_selection_decision = await frame_selection_service.select_distinct_best_frames(
        uploaded_image_asset=uploaded_image_asset,
        candidate_frame_assets=candidate_frame_assets,
    )

    selected_identifiers = [
        selected_frame.asset_identifier
        for selected_frame in frame_selection_decision.selected_frame_assets
    ]

    assert len(frame_selection_decision.selected_frame_assets) == 2
    assert AssetIdentifier("frame-a") in selected_identifiers
    assert AssetIdentifier("frame-b") in selected_identifiers
