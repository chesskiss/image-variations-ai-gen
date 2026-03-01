from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from frame_extraction_stage.domain_models import (
    ExtractedFrameAsset,
    GeneratedVideoAsset,
    GenerationJobIdentifier as ExtractionGenerationJobIdentifier,
    PosePresetIdentifier as ExtractionPosePresetIdentifier,
)
from frame_extraction_stage.ffmpeg_command_runner import FfmpegCommandRunner
from frame_extraction_stage.settings import load_frame_extraction_stage_settings
from frame_extraction_stage.settings import (
    FrameExtractionStageSettings,
)
from frame_extraction_stage.storage_path_resolver import StoragePathResolver
from frame_extraction_stage.video_frame_extraction_service import (
    VideoFrameExtractionConfiguration,
    VideoFrameExtractionService,
)
from frame_judge_stage.frame_candidate_adapter import (
    map_extracted_frame_assets_to_frame_candidates_for_judging,
)
from frame_judge_stage.frame_selection_service import FrameSelectionService
from frame_judge_stage.openai_vision_frame_judge import OpenAiVisionFrameJudge
from frame_judge_stage.result_exporter import FrameJudgeResultExporter
from frame_judge_stage.rule_based_frame_judge import RuleBasedFrameJudge
from frame_judge_stage.settings import FrameJudgeStageSettings, load_frame_judge_stage_settings
from frame_judge_stage.domain_models import (
    FrameCandidateForJudging,
    FrameJudgeDecision,
    GenerationJobIdentifier as JudgeGenerationJobIdentifier,
    PosePresetIdentifier as JudgePosePresetIdentifier,
)
from single_image_transform.config import (
    SingleImageTransformSettings,
    load_single_image_transform_settings,
)
from single_image_transform.fal_image_transform_client import FalImageTransformClient
from single_image_transform.prompts import POSE_PRESET_PROMPTS_BY_NAME


@dataclass(slots=True, frozen=True)
class PipelineOrchestratorResult:
    generation_job_identifier: str
    pose_preset_identifier: str
    generated_video_file_path: Path
    frames_directory_path: Path
    original_image_file_path: Path
    selected_frame_sequence_numbers: list[int]
    judge_result_output_directory: Path
    orchestrator_summary_json_file_path: Path


class PosePipelineOrchestrator:
    def run_generation_extraction_and_judge_pipeline(
        self,
        input_image_file_path: Path,
        pose_preset_identifier: str,
        output_base_directory: Path,
        generation_job_identifier: str | None = None,
    ) -> PipelineOrchestratorResult:
        if pose_preset_identifier not in POSE_PRESET_PROMPTS_BY_NAME:
            raise ValueError(f"Unsupported pose preset identifier: {pose_preset_identifier}")

        resolved_input_image_file_path = input_image_file_path.expanduser().resolve()
        if (
            not resolved_input_image_file_path.exists()
            or not resolved_input_image_file_path.is_file()
        ):
            raise ValueError(f"Input image file does not exist: {resolved_input_image_file_path}")

        selected_generation_job_identifier = generation_job_identifier or uuid4().hex

        single_image_transform_settings = load_single_image_transform_settings()
        frame_extraction_stage_settings = load_frame_extraction_stage_settings()
        frame_judge_stage_settings = load_frame_judge_stage_settings()

        generated_video_file_path = self._run_generation_stage(
            input_image_file_path=resolved_input_image_file_path,
            pose_preset_identifier=pose_preset_identifier,
            single_image_transform_settings=single_image_transform_settings,
        )

        storage_path_resolver = StoragePathResolver(
            frame_extraction_stage_settings.storage_directory
        )
        generated_video_asset_for_extraction = (
            self._build_generated_video_asset_for_extraction_stage(
                selected_generation_job_identifier=selected_generation_job_identifier,
                pose_preset_identifier=pose_preset_identifier,
                generated_video_file_path=generated_video_file_path,
                storage_path_resolver=storage_path_resolver,
            )
        )
        frames_directory_path = storage_path_resolver.resolve_frames_output_directory(
            generation_job_identifier=generated_video_asset_for_extraction.generation_job_identifier,
            pose_preset_identifier=generated_video_asset_for_extraction.pose_preset_identifier,
        )
        original_image_file_path = self._copy_original_image_into_frames_directory(
            input_image_file_path=resolved_input_image_file_path,
            frames_directory_path=frames_directory_path,
        )

        extracted_frame_assets = self._run_extraction_stage(
            generated_video_asset_for_extraction=generated_video_asset_for_extraction,
            frame_extraction_stage_settings=frame_extraction_stage_settings,
            storage_path_resolver=storage_path_resolver,
        )

        frame_candidates_for_judging = map_extracted_frame_assets_to_frame_candidates_for_judging(
            extracted_frame_assets
        )

        frame_judge_decision, judge_result_output_directory = self._run_judge_stage(
            generation_job_identifier=selected_generation_job_identifier,
            pose_preset_identifier=pose_preset_identifier,
            frame_candidates_for_judging=frame_candidates_for_judging,
            original_image_file_path=original_image_file_path,
            frame_judge_stage_settings=frame_judge_stage_settings,
            output_base_directory=output_base_directory,
        )

        orchestrator_summary_json_file_path = self._write_orchestrator_summary_json(
            output_base_directory=output_base_directory,
            selected_generation_job_identifier=selected_generation_job_identifier,
            pose_preset_identifier=pose_preset_identifier,
            generated_video_file_path=generated_video_file_path,
            frames_directory_path=frames_directory_path,
            original_image_file_path=original_image_file_path,
            frame_judge_decision_payload=asdict(frame_judge_decision),
            judge_result_output_directory=judge_result_output_directory,
        )

        return PipelineOrchestratorResult(
            generation_job_identifier=selected_generation_job_identifier,
            pose_preset_identifier=pose_preset_identifier,
            generated_video_file_path=generated_video_file_path,
            frames_directory_path=frames_directory_path,
            original_image_file_path=original_image_file_path,
            selected_frame_sequence_numbers=frame_judge_decision.selected_frame_sequence_numbers,
            judge_result_output_directory=judge_result_output_directory,
            orchestrator_summary_json_file_path=orchestrator_summary_json_file_path,
        )

    def _run_generation_stage(
        self,
        input_image_file_path: Path,
        pose_preset_identifier: str,
        single_image_transform_settings: SingleImageTransformSettings,
    ) -> Path:
        prompt_template = POSE_PRESET_PROMPTS_BY_NAME[pose_preset_identifier]
        fal_image_transform_client = FalImageTransformClient(single_image_transform_settings)
        return fal_image_transform_client.transform_local_image(
            input_image_file_path=input_image_file_path,
            prompt_template=prompt_template,
        )

    def _build_generated_video_asset_for_extraction_stage(
        self,
        selected_generation_job_identifier: str,
        pose_preset_identifier: str,
        generated_video_file_path: Path,
        storage_path_resolver: StoragePathResolver,
    ) -> GeneratedVideoAsset:
        extraction_generation_job_identifier = ExtractionGenerationJobIdentifier(
            selected_generation_job_identifier
        )
        extraction_pose_preset_identifier = ExtractionPosePresetIdentifier(pose_preset_identifier)

        generation_job_directory_path = storage_path_resolver.resolve_generation_job_directory(
            extraction_generation_job_identifier
        )
        generation_video_directory_path = (
            generation_job_directory_path / pose_preset_identifier / "video"
        )
        generation_video_directory_path.mkdir(parents=True, exist_ok=True)

        video_file_suffix = generated_video_file_path.suffix.lower() or ".mp4"
        copied_generated_video_file_path = (
            generation_video_directory_path / f"generated{video_file_suffix}"
        )
        shutil.copy2(generated_video_file_path, copied_generated_video_file_path)

        return GeneratedVideoAsset(
            generation_job_identifier=extraction_generation_job_identifier,
            pose_preset_identifier=extraction_pose_preset_identifier,
            local_video_file_path=copied_generated_video_file_path,
            video_duration_seconds=None,
            video_file_format=video_file_suffix.replace(".", ""),
        )

    def _copy_original_image_into_frames_directory(
        self,
        input_image_file_path: Path,
        frames_directory_path: Path,
    ) -> Path:
        original_image_suffix = input_image_file_path.suffix.lower() or ".jpg"
        copied_original_image_file_path = frames_directory_path / f"original{original_image_suffix}"
        shutil.copy2(input_image_file_path, copied_original_image_file_path)
        return copied_original_image_file_path

    def _run_extraction_stage(
        self,
        generated_video_asset_for_extraction: GeneratedVideoAsset,
        frame_extraction_stage_settings: FrameExtractionStageSettings,
        storage_path_resolver: StoragePathResolver,
    ) -> list[ExtractedFrameAsset]:
        video_frame_extraction_configuration = VideoFrameExtractionConfiguration(
            ffmpeg_executable_path=frame_extraction_stage_settings.ffmpeg_executable_path,
            extracted_frames_per_second=frame_extraction_stage_settings.extracted_frames_per_second,
            extracted_frame_image_format=frame_extraction_stage_settings.extracted_frame_image_format,
            max_extracted_frames_per_video=frame_extraction_stage_settings.max_extracted_frames_per_video,
        )
        ffmpeg_command_runner = FfmpegCommandRunner(
            ffmpeg_executable_path=frame_extraction_stage_settings.ffmpeg_executable_path
        )
        video_frame_extraction_service = VideoFrameExtractionService(
            video_frame_extraction_configuration=video_frame_extraction_configuration,
            storage_path_resolver=storage_path_resolver,
            ffmpeg_command_runner=ffmpeg_command_runner,
        )
        return video_frame_extraction_service.extract_candidate_frames_from_generated_video(
            generated_video_asset_for_extraction
        )

    def _run_judge_stage(
        self,
        generation_job_identifier: str,
        pose_preset_identifier: str,
        frame_candidates_for_judging: list[FrameCandidateForJudging],
        original_image_file_path: Path,
        frame_judge_stage_settings: FrameJudgeStageSettings,
        output_base_directory: Path,
    ) -> tuple[FrameJudgeDecision, Path]:
        if frame_judge_stage_settings.enable_openai_frame_judge:
            frame_judge = OpenAiVisionFrameJudge(
                openai_api_key=frame_judge_stage_settings.openai_api_key,
                judge_model_name=frame_judge_stage_settings.frame_judge_model_name,
                timeout_seconds=frame_judge_stage_settings.frame_judge_timeout_seconds,
            )
        else:
            frame_judge = RuleBasedFrameJudge(judge_model_name="rule-based-baseline")

        frame_judge_scores = frame_judge.judge_frame_candidates(
            frame_candidates_for_judging,
            original_image_file_path=original_image_file_path,
        )

        frame_selection_service = FrameSelectionService(
            selected_count=frame_judge_stage_settings.frame_judge_selected_count,
            minimum_score_threshold=frame_judge_stage_settings.frame_judge_minimum_score_threshold,
        )
        frame_judge_decision = frame_selection_service.select_best_frames_from_judge_scores(
            generation_job_identifier=JudgeGenerationJobIdentifier(generation_job_identifier),
            pose_preset_identifier=JudgePosePresetIdentifier(pose_preset_identifier),
            frame_candidates_for_judging=frame_candidates_for_judging,
            frame_judge_scores=frame_judge_scores,
        )

        frame_judge_result_exporter = FrameJudgeResultExporter()
        judge_result_output_directory = frame_judge_result_exporter.export_frame_judge_results(
            output_base_directory=output_base_directory,
            frame_candidates_for_judging=frame_candidates_for_judging,
            frame_judge_decision=frame_judge_decision,
        )

        return frame_judge_decision, judge_result_output_directory

    def _write_orchestrator_summary_json(
        self,
        output_base_directory: Path,
        selected_generation_job_identifier: str,
        pose_preset_identifier: str,
        generated_video_file_path: Path,
        frames_directory_path: Path,
        original_image_file_path: Path,
        frame_judge_decision_payload: dict[str, object],
        judge_result_output_directory: Path,
    ) -> Path:
        output_base_directory.mkdir(parents=True, exist_ok=True)
        orchestrator_output_directory = output_base_directory / "pipeline_orchestrator"
        orchestrator_output_directory.mkdir(parents=True, exist_ok=True)
        orchestrator_summary_json_file_path = (
            orchestrator_output_directory
            / f"pipeline_summary_{selected_generation_job_identifier}.json"
        )

        orchestrator_payload = {
            "generation_job_identifier": selected_generation_job_identifier,
            "pose_preset_identifier": pose_preset_identifier,
            "generated_video_file_path": str(generated_video_file_path),
            "frames_directory_path": str(frames_directory_path),
            "original_image_file_path": str(original_image_file_path),
            "judge_result_output_directory": str(judge_result_output_directory),
            "frame_judge_decision": frame_judge_decision_payload,
        }
        orchestrator_summary_json_file_path.write_text(
            json.dumps(orchestrator_payload, indent=2),
            encoding="utf-8",
        )
        return orchestrator_summary_json_file_path
