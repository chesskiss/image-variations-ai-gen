cmds:


judge
uv run python run_frame_judge_stage.py --output-directory ./outputs/chosen  --frames-directory outputs/frames/test_video_frame_extraction_fr0/storage/job-123/rotate_right/frames --job-id 1 --preset-id 1 
Frame judge results written to: /Users/arnoldcheskis/Documents/Projects/tmp/interview/Eikona-image/outputs/chosen/frame_judge_results_20260301T211833Z

frame extraction
uv run pytest tests/test_video_frame_extraction_service_integration.py -s --basetemp=outputs/frames      

video gen
uv run python run_single_image_transform.py --image "/Users/arnoldcheskis/Documents/Images/green card.JPG"  --preset rotate_left