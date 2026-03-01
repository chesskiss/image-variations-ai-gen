# AI Image Variations

AI Image Variations runs a full image pipeline:
- upload one original image,
- generate a pose-variation video,
- extract candidate frames,
- score identity similarity,
- output selected frames + JSON artifacts.

## App Usage (Primary: Docker)

### 1) Configure environment
```bash
cp .env.example .env
```
Set at least:
- `FAL_API_KEY`
- `OPENAI_API_KEY` (only needed when `ENABLE_OPENAI_FRAME_JUDGE=true`)

### 2) Start
```bash
docker compose up --build
```

### 3) Open
`http://127.0.0.1:8000`

### 4) Stop
```bash
docker compose down
```

Persistent data:
- `./storage`
- `./outputs`

## Developer Usage (uv / Local) - Works on MacOS (Windows not guaranteed)

### Requirements
- Python 3.11+
- `uv`
- `ffmpeg` in PATH

### Install dependencies
```bash
uv sync
```

### Run UI bridge locally
```bash
uv run uvicorn ui.app:application --reload
```

## Useful Commands

- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run mypy .`
- Test: `uv run pytest --cov=pose_variations --cov-report=term-missing`
- Pre-commit install: `uv run pre-commit install`

### Run full orchestrator directly
```bash
uv run python run_pose_pipeline_orchestrator.py \
  --image /absolute/path/to/original_image.jpg \
  --preset rotate_left \
  --output-directory ./outputs/orchestrator
```

### Run judge module directly
```bash
uv run python tests/run_modules/run_frame_judge_stage.py \
  --frames-directory ./storage/<generation_job_identifier>/<pose_preset_identifier>/frames \
  --original-image-file-path ./storage/<generation_job_identifier>/<pose_preset_identifier>/frames/original.jpg \
  --job-id <generation_job_identifier> \
  --preset-id <pose_preset_identifier> \
  --output-directory ./outputs
```


### Run tests
```bash
uv run pytest -q
```

## Key Environment Variables

Core:
- `FAL_API_KEY`
- `OPENAI_API_KEY`
- `FAL_MODEL_ID`

Frame extraction:
- `STORAGE_DIRECTORY` (default: `./storage`)
- `FFMPEG_EXECUTABLE_PATH` (default: `ffmpeg`)
- `EXTRACTED_FRAMES_PER_SECOND` (default: `2`)
- `EXTRACTED_FRAME_IMAGE_FORMAT` (default: `jpg`)
- `MAX_EXTRACTED_FRAMES_PER_VIDEO` (default: `24`)

Judge:
- `ENABLE_OPENAI_FRAME_JUDGE` (default: `false`)
- `FRAME_JUDGE_MODEL_NAME` (default: `o3`)
- `FRAME_JUDGE_SELECTED_COUNT` (default: `2`)
- `FRAME_JUDGE_MINIMUM_SCORE_THRESHOLD` (default: `0.0`)
- `FRAME_JUDGE_TIMEOUT_SECONDS` (default: `45`)

## Output Locations

- Extracted frames: `storage/<job>/<preset>/frames/`
- Judge outputs: `outputs/frame_judge_results_<timestamp>/`
- Orchestrator summary: `outputs/orchestrator/pipeline_orchestrator/pipeline_summary_<job>.json`

## Security Notes

- Never commit `.env`.
- Do not log secrets or tokenized URLs.
- Uploaded files are validated and stored under project-controlled directories.
