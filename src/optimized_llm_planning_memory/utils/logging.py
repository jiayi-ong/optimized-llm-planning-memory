"""
utils/logging.py
=================
structlog setup with JSON + console sinks.

Usage
-----
    from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger

    configure_logging(level="INFO", json_output=False)  # call once at startup
    logger = get_logger(__name__)
    logger.info("episode_complete", episode_id="abc", reward=0.87)

Design
------
- ``configure_logging()`` installs structlog processors:
    1. Add log level string
    2. Add ISO timestamp
    3. Add caller filename + line
    4. Render as JSON (production) or colourful console (development)
- After calling ``configure_logging()``, use ``get_logger()`` throughout the
  project. Do NOT use ``logging.getLogger()`` directly — structlog's context
  variables (bound with ``bind()``) will not propagate through stdlib loggers.
- ``configure_logging()`` is idempotent: repeated calls are no-ops after the
  first call in a process.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure structlog (and the stdlib root logger) for the project.

    Parameters
    ----------
    level       : Log level string — "DEBUG", "INFO", "WARNING", "ERROR".
    json_output : If True, render as newline-delimited JSON (suitable for
                  production log aggregators). If False, use colourful
                  console rendering for human readability.
    log_file    : Optional path to a log file. When provided, all log
                  records are written to both stdout and this file (tee).
                  The file is opened in append mode; parent directories are
                  created automatically.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    try:
        import structlog

        shared_processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if json_output:
            renderer: Any = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer()

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )

        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            # File sink: plain text (no ANSI colour codes)
            file_renderer = structlog.processors.KeyValueRenderer(
                key_order=["level", "logger", "timestamp", "event"],
                drop_missing=True,
            )
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processor=file_renderer,
                foreign_pre_chain=shared_processors,
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    except ImportError:
        # Fallback to stdlib logging if structlog is not installed
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=handlers,
        )

    _CONFIGURED = True


def get_logger(name: str) -> Any:
    """
    Return a structlog (or stdlib) logger bound to ``name``.

    Parameters
    ----------
    name : Typically ``__name__`` of the calling module.

    Returns
    -------
    A structlog BoundLogger (or stdlib Logger as fallback).
    """
    try:
        import structlog
        return structlog.get_logger(name)
    except ImportError:
        return logging.getLogger(name)
