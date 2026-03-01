from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


class FfmpegCommandRunnerError(RuntimeError):
    """Raised when ffmpeg execution fails."""


@dataclass(slots=True, frozen=True)
class FfmpegCommandExecutionResult:
    command_arguments: list[str]
    return_code: int
    standard_output_text: str
    standard_error_text: str
    timed_out: bool


class FfmpegCommandRunner:
    def __init__(self, ffmpeg_executable_path: str) -> None:
        self._ffmpeg_executable_path = ffmpeg_executable_path

    def verify_ffmpeg_is_available(self) -> None:
        resolved_executable_path = shutil.which(self._ffmpeg_executable_path)
        if resolved_executable_path is None:
            raise FfmpegCommandRunnerError(
                f"FFmpeg executable was not found: {self._ffmpeg_executable_path}"
            )

        version_result = self.run_ffmpeg_command(
            command_arguments=[self._ffmpeg_executable_path, "-version"],
            timeout_seconds=10,
        )
        self.raise_error_when_execution_failed(version_result)

    def run_ffmpeg_command(
        self,
        command_arguments: list[str],
        timeout_seconds: int,
    ) -> FfmpegCommandExecutionResult:
        try:
            completed_process = subprocess.run(
                command_arguments,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return FfmpegCommandExecutionResult(
                command_arguments=command_arguments,
                return_code=completed_process.returncode,
                standard_output_text=completed_process.stdout,
                standard_error_text=completed_process.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as timeout_error:
            return FfmpegCommandExecutionResult(
                command_arguments=command_arguments,
                return_code=-1,
                standard_output_text=timeout_error.stdout or "",
                standard_error_text=timeout_error.stderr or "",
                timed_out=True,
            )

    def raise_error_when_execution_failed(
        self,
        ffmpeg_command_execution_result: FfmpegCommandExecutionResult,
    ) -> None:
        if ffmpeg_command_execution_result.timed_out:
            raise FfmpegCommandRunnerError("FFmpeg command timed out while extracting frames.")

        if ffmpeg_command_execution_result.return_code != 0:
            raise FfmpegCommandRunnerError(
                "FFmpeg command failed with non-zero return code. "
                f"stderr={ffmpeg_command_execution_result.standard_error_text[:600]}"
            )
