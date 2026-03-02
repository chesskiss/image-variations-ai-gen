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
- Metrics: `http://127.0.0.1:8000/metrics`
- History: `http://127.0.0.1:8000/history`

### 4) Stop
```bash
docker compose down
```

Persistent data:
- `./storage`
- `./outputs`

### Optional Observability Stack
```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
```
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000` (`admin` / `admin`)

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
- Test: `uv run pytest --cov=ui --cov=single_image_transform --cov=frame_extraction_stage --cov=frame_judge_stage --cov=orchestration_stage --cov-report=term-missing`
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

Cache:
- `ENABLE_RESULT_CACHE` (default: `true`)
- `CACHE_INDEX_DIRECTORY` (default: `./outputs/cache_index`)
- `CACHE_MAX_ENTRIES` (default: `1000`)
- `CACHE_RETENTION_DAYS` (default: `30`)
- `CACHE_KEY_INCLUDE_MODEL_VERSION` (default: `true`)

## API Endpoints

- `GET /api/jobs/{job_id}`: job status + cache metadata
- `GET /api/cache/{cache_key}`: cache lookup debug endpoint
- `GET /api/history?limit=50&state=completed`: history index
- `GET /metrics`: Prometheus metrics

## Output Locations

- Extracted frames: `storage/<job>/<preset>/frames/`
- Judge outputs: `outputs/frame_judge_results_<timestamp>/`
- Orchestrator summary: `outputs/orchestrator/pipeline_orchestrator/pipeline_summary_<job>.json`
- UI job statuses and logs: `outputs/orchestrator_ui/jobs/<job>/`
- Cache index file: `outputs/cache_index/cache_index.json`

## Security Notes

- Never commit `.env`.
- Do not log secrets or tokenized URLs.
- Uploaded files are validated and stored under project-controlled directories.
