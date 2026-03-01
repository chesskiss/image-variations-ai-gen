## Run 
with docker
docker compose up --build

with uvicorn (works on MacOS)
uv run uvicorn scaffold.app:application --reload

visit:
http://127.0.0.1:8000


## Useful Commands

- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run mypy .`
- Test: `uv run pytest --cov=pose_variations --cov-report=term-missing`
- Pre-commit install: `uv run pre-commit install`


## additional cmds to test modules seperately:

judge
uv run python tests/run_modules/run_frame_judge_stage.py --output-directory ./outputs/chosen  --frames-directory outputs/frames/test_video_frame_extraction_fr0/storage/job-123/rotate_right/frames --job-id 1 --preset-id 1 
Frame judge results written to: /Users/arnoldcheskis/Documents/Projects/tmp/interview/Eikona-image/outputs/chosen/frame_judge_results_20260301T211833Z

frame extraction
uv run pytest tests/test_video_frame_extraction_service_integration.py -s --basetemp=outputs/frames      

video gen
uv run python tests/run_modules/run_single_image_transform.py --image "/Users/arnoldcheskis/Documents/Images/green card.JPG"  --preset rotate_left
orchestrator (gen -> extract -> judge)
uv run python run_pose_pipeline_orchestrator.py --image "/Users/arnoldcheskis/Documents/Images/green card.JPG" --preset rotate_left --output-directory ./outputs/orchestrator

Docker (full app)
# build and run
 docker compose up --build

# open
 http://127.0.0.1:8000

# stop
 docker compose down
