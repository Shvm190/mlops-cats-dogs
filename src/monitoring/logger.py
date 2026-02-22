"""
src/monitoring/logger.py
========================
Structured JSON logging for the inference service.
All logs include: timestamp, level, service, request_id (when available).
Sensitive data (raw image bytes) is never logged.
"""

import logging
import sys
from typing import Optional
import structlog


def configure_logging(level: str = "INFO", json_logs: bool = True):
    """Configure structlog for structured/JSON output."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger."""
    return structlog.get_logger(name)


# ─── Domain Log Helpers ──────────────────────────────────────────────────────

_logger = get_logger("inference_service")


def log_request(method: str, path: str, status_code: int, duration_ms: float):
    """Log an HTTP request (no sensitive data)."""
    _logger.info(
        "http_request",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
    )


def log_prediction(
    filename: Optional[str],
    label: str,
    confidence: float,
    latency_ms: float,
):
    """Log a prediction result (filename only, no image data)."""
    _logger.info(
        "prediction",
        filename=filename,  # Only filename, not content
        label=label,
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 2),
    )


def log_error(error_type: str, message: str, **kwargs):
    """Log an error event."""
    _logger.error("error", error_type=error_type, message=message, **kwargs)


# Configure on import with sensible defaults
configure_logging(level="INFO", json_logs=False)
