from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from pose_variations.domain.asset_models import ExtractedFrameAsset

IdentitySimilarityScore = NewType("IdentitySimilarityScore", float)
FrameDiversityScore = NewType("FrameDiversityScore", float)


@dataclass(slots=True)
class RankedFrameCandidate:
    extracted_frame_asset: ExtractedFrameAsset
    aggregate_selection_score: float
    sharpness_score: float
    identity_similarity_score: IdentitySimilarityScore | None
    diversity_score_from_primary_frame: FrameDiversityScore | None


@dataclass(slots=True)
class FrameSelectionDecision:
    selected_frame_assets: list[ExtractedFrameAsset]
    ranked_frame_candidates: list[RankedFrameCandidate]
