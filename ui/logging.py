from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "component": getattr(record, "component", "ui"),
            "request_id": getattr(record, "request_id", None),
            "job_id": getattr(record, "job_id", None),
            "cache_key": getattr(record, "cache_key", None),
        }
        return json.dumps(log_payload, ensure_ascii=True)


def configure_structured_logging() -> logging.Logger:
    logger = logging.getLogger("ui")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
