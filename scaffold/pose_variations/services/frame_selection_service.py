from __future__ import annotations

from pose_variations.domain.asset_models import ExtractedFrameAsset, UploadedImageAsset
from pose_variations.domain.scoring_models import (
    FrameDiversityScore,
    FrameSelectionDecision,
    IdentitySimilarityScore,
    RankedFrameCandidate,
)
from pose_variations.infrastructure.settings import ApplicationSettings
from pose_variations.services.frame_quality_assessment_service import FrameQualityAssessmentService
from pose_variations.services.openai_identity_similarity_judge import OpenAiIdentitySimilarityJudge


class FrameSelectionService:
    def __init__(
        self,
        application_settings: ApplicationSettings,
        frame_quality_assessment_service: FrameQualityAssessmentService,
        openai_identity_similarity_judge: OpenAiIdentitySimilarityJudge | None,
    ) -> None:
        self._application_settings = application_settings
        self._frame_quality_assessment_service = frame_quality_assessment_service
        self._openai_identity_similarity_judge = openai_identity_similarity_judge

    async def select_distinct_best_frames(
        self,
        uploaded_image_asset: UploadedImageAsset,
        candidate_frame_assets: list[ExtractedFrameAsset],
    ) -> FrameSelectionDecision:
        if not candidate_frame_assets:
            return FrameSelectionDecision(selected_frame_assets=[], ranked_frame_candidates=[])

        await self._attach_identity_similarity_scores_if_enabled(
            uploaded_image_asset=uploaded_image_asset,
            candidate_frame_assets=candidate_frame_assets,
        )

        eligible_frame_assets = self._filter_candidates_by_identity_threshold(candidate_frame_assets)
        ranked_frame_candidates = self._rank_candidates(eligible_frame_assets)

        selected_frame_assets = self._select_two_diverse_frames(ranked_frame_candidates)
        return FrameSelectionDecision(
            selected_frame_assets=selected_frame_assets,
            ranked_frame_candidates=ranked_frame_candidates,
        )

    async def _attach_identity_similarity_scores_if_enabled(
        self,
        uploaded_image_asset: UploadedImageAsset,
        candidate_frame_assets: list[ExtractedFrameAsset],
    ) -> None:
        if (
            not self._application_settings.enable_openai_similarity_judge
            or self._openai_identity_similarity_judge is None
        ):
            return

        for candidate_frame_asset in candidate_frame_assets:
            identity_similarity_score = await self._openai_identity_similarity_judge.score_identity_similarity(
                original_image_file_path=uploaded_image_asset.local_file_path,
                candidate_image_file_path=candidate_frame_asset.local_file_path,
            )
            candidate_frame_asset.identity_similarity_score = float(identity_similarity_score)

    def _filter_candidates_by_identity_threshold(
        self,
        candidate_frame_assets: list[ExtractedFrameAsset],
    ) -> list[ExtractedFrameAsset]:
        if not self._application_settings.enable_openai_similarity_judge:
            return candidate_frame_assets

        threshold = self._application_settings.openai_identity_similarity_threshold
        threshold_passing_candidates = [
            frame_asset
            for frame_asset in candidate_frame_assets
            if frame_asset.identity_similarity_score is not None
            and frame_asset.identity_similarity_score >= threshold
        ]
        return threshold_passing_candidates or candidate_frame_assets

    def _rank_candidates(self, candidate_frame_assets: list[ExtractedFrameAsset]) -> list[RankedFrameCandidate]:
        if not candidate_frame_assets:
            return []

        normalized_sharpness_values = [max((frame.sharpness_score or 0.0), 0.0) for frame in candidate_frame_assets]
        max_sharpness = max(normalized_sharpness_values) if normalized_sharpness_values else 1.0
        if max_sharpness <= 0.0:
            max_sharpness = 1.0

        ranked_candidates: list[RankedFrameCandidate] = []
        for frame_asset in candidate_frame_assets:
            frame_sharpness_score = max((frame_asset.sharpness_score or 0.0), 0.0)
            normalized_sharpness_score = frame_sharpness_score / max_sharpness

            identity_similarity_score: IdentitySimilarityScore | None = None
            if frame_asset.identity_similarity_score is not None:
                identity_similarity_score = IdentitySimilarityScore(frame_asset.identity_similarity_score)

            identity_component = float(identity_similarity_score) if identity_similarity_score is not None else 0.0
            aggregate_selection_score = normalized_sharpness_score * 0.7 + identity_component * 0.3

            ranked_candidates.append(
                RankedFrameCandidate(
                    extracted_frame_asset=frame_asset,
                    aggregate_selection_score=aggregate_selection_score,
                    sharpness_score=frame_sharpness_score,
                    identity_similarity_score=identity_similarity_score,
                    diversity_score_from_primary_frame=None,
                )
            )

        ranked_candidates.sort(key=lambda ranked_candidate: ranked_candidate.aggregate_selection_score, reverse=True)
        return ranked_candidates

    def _select_two_diverse_frames(
        self,
        ranked_frame_candidates: list[RankedFrameCandidate],
    ) -> list[ExtractedFrameAsset]:
        if not ranked_frame_candidates:
            return []

        primary_frame_candidate = ranked_frame_candidates[0]
        selected_frame_assets = [primary_frame_candidate.extracted_frame_asset]

        primary_perceptual_hash_value = primary_frame_candidate.extracted_frame_asset.perceptual_hash_value
        best_secondary_index: int | None = None
        best_secondary_score = -1.0

        for candidate_index, ranked_candidate in enumerate(ranked_frame_candidates[1:], start=1):
            candidate_perceptual_hash_value = ranked_candidate.extracted_frame_asset.perceptual_hash_value
            diversity_score_value = self._compute_diversity_score(
                primary_perceptual_hash_value,
                candidate_perceptual_hash_value,
            )
            ranked_candidate.diversity_score_from_primary_frame = FrameDiversityScore(diversity_score_value)

            if diversity_score_value >= self._application_settings.minimum_diversity_hash_distance:
                secondary_composite_score = ranked_candidate.aggregate_selection_score + (
                    diversity_score_value / 64.0
                )
                if secondary_composite_score > best_secondary_score:
                    best_secondary_score = secondary_composite_score
                    best_secondary_index = candidate_index

        if best_secondary_index is None and len(ranked_frame_candidates) > 1:
            best_secondary_index = 1

        if best_secondary_index is not None:
            selected_frame_assets.append(ranked_frame_candidates[best_secondary_index].extracted_frame_asset)

        return selected_frame_assets

    def _compute_diversity_score(
        self,
        first_perceptual_hash_value: str | None,
        second_perceptual_hash_value: str | None,
    ) -> float:
        if first_perceptual_hash_value is None or second_perceptual_hash_value is None:
            return 0.0
        return float(
            self._frame_quality_assessment_service.calculate_perceptual_hash_distance(
                first_perceptual_hash_value,
                second_perceptual_hash_value,
            )
        )
