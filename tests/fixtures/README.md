# Fixture Files

Add a tiny MP4 fixture named `tiny_sample.mp4` in this directory to run:

- `tests/test_video_frame_extraction_service_integration.py`

The test is marked `slow` and skips automatically when either FFmpeg or the fixture is unavailable.

Also add `frame_fixture.jpg` to run `tests/integration/test_openai_vision_frame_judge_integration.py`.
