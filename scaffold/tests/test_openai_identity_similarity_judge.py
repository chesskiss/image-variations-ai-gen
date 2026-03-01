from __future__ import annotations

import pytest

from pose_variations.services.openai_identity_similarity_judge import (
    OpenAiIdentitySimilarityJudge,
    OpenAiIdentitySimilarityJudgeError,
)


def test_parse_identity_similarity_output_with_strict_float() -> None:
    parsed_score = OpenAiIdentitySimilarityJudge.parse_identity_similarity_output("0.873")
    assert float(parsed_score) == pytest.approx(0.873)


def test_parse_identity_similarity_output_with_verbose_text() -> None:
    parsed_score = OpenAiIdentitySimilarityJudge.parse_identity_similarity_output(
        "Similarity score: 0.79"
    )
    assert float(parsed_score) == pytest.approx(0.79)


def test_parse_identity_similarity_output_rejects_missing_float() -> None:
    with pytest.raises(OpenAiIdentitySimilarityJudgeError):
        OpenAiIdentitySimilarityJudge.parse_identity_similarity_output("not available")
