# PoseVariations

PoseVariations is a FastAPI web application that accepts a single model photo, generates 5-second pose variation videos with fal.ai, extracts candidate frames, and selects at least two distinct best frames while preserving identity and outfit.

## Features

- Single image upload with validation (MIME type, content sniffing, max size).
- Two pose preset selection from server-side prompt templates.
- Background generation pipeline with live job status polling.
- fal.ai image-to-video generation with retry and polling.
- Frame extraction via `ffmpeg` at fixed intervals.
- Frame quality and diversity ranking (sharpness + perceptual hash).
- Optional OpenAI identity similarity judge (`0.0..1.0`).
- Local storage abstraction under `STORAGE_DIRECTORY`.

## Project Structure

```text
pose_variations/
  application/
    api/
      fastapi_application.py
      routes/
        home_routes.py
        job_routes.py
        api_job_routes.py
    background/
      job_processor.py
  domain/
    generation_job_models.py
    pose_preset_models.py
    asset_models.py
    scoring_models.py
  services/
    fal_image_to_video_client.py
    video_downloader.py
    video_frame_extraction_service.py
    frame_quality_assessment_service.py
    frame_selection_service.py
    openai_identity_similarity_judge.py
    prompt_template_repository.py
  infrastructure/
    settings.py
    storage_repository.py
    logging_configuration.py
templates/
static/
tests/
```

## Requirements

- Python 3.11+
- `uv`
- `ffmpeg` available in PATH

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Start the application:

```bash
uv run uvicorn pose_variations.application.api.fastapi_application:application --reload
```

4. Open `http://127.0.0.1:8000`.

## Useful Commands

- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run mypy .`
- Test: `uv run pytest --cov=pose_variations --cov-report=term-missing`
- Pre-commit install: `uv run pre-commit install`

## Environment Variables

See `.env.example`:

- `FAL_API_KEY`
- `OPENAI_API_KEY`
- `ENABLE_OPENAI_SIMILARITY_JUDGE`
- `MAX_UPLOAD_SIZE_BYTES`
- `ALLOWED_UPLOAD_MIME_TYPES`
- `STORAGE_DIRECTORY`
- `FAL_MODEL_ID`
- `FAL_MAX_POLL_ATTEMPTS`
- `FAL_POLL_INTERVAL_SECONDS`
- `OPENAI_SIMILARITY_MODEL`
- `OPENAI_IDENTITY_SIMILARITY_THRESHOLD`
- `FRAME_EXTRACTION_INTERVAL_SECONDS`
- `MINIMUM_SHARPNESS_SCORE`
- `MINIMUM_DIVERSITY_HASH_DISTANCE`

## Security Considerations

- Secrets are loaded from `.env` and never hardcoded.
- `.env` is ignored by git; only `.env.example` is committed.
- Upload validation enforces allowed MIME types, image content verification, and max size.
- Stored assets use random filenames; user-provided names are never trusted.
- Internal prompt templates are server-side only and never returned through API responses.
- API responses contain only safe job metadata and generated asset URLs.
- Logs include job identifiers and status updates without secrets or raw payloads.

## Notes About External Services

- fal.ai endpoints and payload fields may evolve; review `FalImageToVideoClient` if API contracts change.
- OpenAI similarity judging is optional and controlled by `ENABLE_OPENAI_SIMILARITY_JUDGE`.


## Isolated Frame Extraction Stage

Frame extraction is now isolated from video generation logic in:

- `frame_extraction_stage/`

This stage is designed for future judge-model compatibility and returns stable per-frame metadata.

### FFmpeg Requirements

- `FFMPEG_EXECUTABLE_PATH` defaults to `ffmpeg`.
- The extraction service validates FFmpeg availability during service initialization and fails fast if FFmpeg is missing.

### Extraction Environment Variables

- `STORAGE_DIRECTORY=./storage`
- `FFMPEG_EXECUTABLE_PATH=ffmpeg`
- `EXTRACTED_FRAMES_PER_SECOND=2`
- `EXTRACTED_FRAME_IMAGE_FORMAT=jpg` (`jpg` or `png`)
- `MAX_EXTRACTED_FRAMES_PER_VIDEO=24`

### Frame Output Path

Extracted frames are written under:

- `storage/<generation_job_identifier>/<pose_preset_identifier>/frames/frame_0001.jpg`
- `storage/<generation_job_identifier>/<pose_preset_identifier>/frames/frame_0002.jpg`

### Frame Extraction Contract

`VideoFrameExtractionService.extract_candidate_frames_from_generated_video(...)` returns a sorted list of `ExtractedFrameAsset` values containing:

- stable identifiers (`generation_job_identifier`, `pose_preset_identifier`, `frame_sequence_number`)
- `timestamp_seconds`
- local frame path and format
- image dimensions (when available)
- `file_size_bytes`
- `basic_quality_metrics` placeholder dict for future judge/scoring stages

## Isolated Frame Judge Stage

A separate judge module is available in `frame_judge_stage/`.

### Purpose

- Consume extracted frame metadata from `frame_extraction_stage`.
- Score frame candidates.
- Return deterministic top-N selection (top 2 by default) with full ranking for explainability.

### Settings

- `ENABLE_OPENAI_FRAME_JUDGE=false`
- `FRAME_JUDGE_MODEL_NAME=o3`
- `FRAME_JUDGE_SELECTED_COUNT=2`
- `FRAME_JUDGE_MINIMUM_SCORE_THRESHOLD=0.0`
- `FRAME_JUDGE_TIMEOUT_SECONDS=45`

### Testing Judge Stage

```bash
uv run pytest tests/test_frame_judge_contract.py tests/test_frame_selection_service.py -q
```

Optional OpenAI integration test (skipped by default):

```bash
RUN_OPENAI_FRAME_JUDGE_INTEGRATION_TESTS=true uv run pytest tests/integration/test_openai_vision_frame_judge_integration.py -q -rs
```

This integration test expects:

- `OPENAI_API_KEY` in environment
- `tests/fixtures/frame_fixture.jpg` present

### Run Judge Stage and Export Results to `outputs/`

You can run the judge stage directly against an extracted frames directory and write selected frames plus scores to an output folder.

Example:

```bash
uv run python run_frame_judge_stage.py \
  --frames-directory ./storage/<generation_job_identifier>/<pose_preset_identifier>/frames \
  --original-image-file-path ./storage/<generation_job_identifier>/<pose_preset_identifier>/frames/original.jpg \
  --job-id <generation_job_identifier> \
  --preset-id <pose_preset_identifier> \
  --output-directory ./outputs
```

Output structure:

- `outputs/frame_judge_results_<timestamp>/judge_decision.json`
- `outputs/frame_judge_results_<timestamp>/ranked_scores.json`
- `outputs/frame_judge_results_<timestamp>/selected_frames/frame_0001.jpg`

By default this uses the rule-based judge. To use OpenAI judge:

- set `ENABLE_OPENAI_FRAME_JUDGE=true`
- set `OPENAI_API_KEY=...`
- optionally set `FRAME_JUDGE_MODEL_NAME=o3`
- provide an original image using `--original-image-file-path`, or place one of:
  `original.jpg`, `original.jpeg`, `original.png`, `source.jpg`, `source.jpeg`, `source.png`
  inside the frames directory

### Run Full Pipeline Orchestrator

To run generation + extraction + judge in one command:

```bash
uv run python run_pose_pipeline_orchestrator.py \
  --image /absolute/path/to/original_image.jpg \
  --preset rotate_left \
  --output-directory ./outputs/orchestrator
```

This writes:

- extracted frames under `storage/<job>/<preset>/frames/`
- judge outputs under `outputs/orchestrator/frame_judge_results_<timestamp>/`
- orchestrator summary under `outputs/orchestrator/pipeline_orchestrator/pipeline_summary_<job>.json`

This orchestrator is library-first (in-process Python imports). It is API-ready because it emits a stable JSON summary payload that can be returned directly by a future backend endpoint.
