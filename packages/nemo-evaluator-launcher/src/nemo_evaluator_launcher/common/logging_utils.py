"""Logging configuration module for nv-eval-platform.

This module provides a centralized logging configuration using structlog that outputs
to both stderr and a log file. All modules should import and use the logger from this
module to ensure consistent logging behavior across the application.

LOGGING POLICY:
==============
All logging in this project MUST go through this module. This is enforced by a pre-commit
hook that checks for violations.

DO NOT:
- import structlog directly
- import logging directly
- call structlog.get_logger() directly
- call logging.getLogger() directly

DO:
- from nemo_evaluator_launcher.common.logging_utils import logger
- from nemo_evaluator_launcher.common.logging_utils import get_logger

Examples:
    # Correct
    from nemo_evaluator_launcher.common.logging_utils import logger
    logger.info("User logged in", user_id="12345")

    # Incorrect
    import structlog
    logger = structlog.get_logger()
    logger.info("User logged in")
"""

import logging
import logging.config
import os
import pathlib
import sys
from datetime import datetime

import structlog


def _ensure_log_dir() -> pathlib.Path:
    """Ensure the log directory exists and return its path."""
    log_dir = pathlib.Path.home() / ".nv-eval-platform" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _get_env_log_level() -> str:
    """Get log level from environment variable, translating single letters to full names.

    Translates:
    - D -> DEBUG
    - I -> INFO
    - W -> WARNING
    - E -> ERROR
    - F -> CRITICAL

    Returns:
        Uppercase log level string, defaults to WARNING if not set or invalid.
    """
    env_level = os.getenv("LOG_LEVEL", "WARNING").upper()

    # Translate single letters to full level names
    level_map = {
        "D": "DEBUG",
        "I": "INFO",
        "W": "WARNING",
        "E": "ERROR",
        "F": "CRITICAL",
    }

    return level_map.get(env_level, env_level)


def custom_timestamper(_, __, event_dict):
    """Add ISO UTC timestamp with milliseconds to event_dict['timestamp']."""
    now = datetime.now()
    event_dict["timestamp"] = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    return event_dict


class MainConsoleRenderer:
    """Custom console renderer for [L TIMESTAMP] message with color by level."""

    LEVEL_MAP = {
        "debug": ("D", "\033[90m"),  # grey
        "info": ("I", "\033[32m"),  # green
        "warning": ("W", "\033[33m"),  # yellow
        "warn": ("W", "\033[33m"),  # yellow
        "error": ("E", "\033[31m"),  # red
        "critical": ("F", "\033[41m"),  # red background
        "fatal": ("F", "\033[41m"),  # alias for critical
    }
    RESET = "\033[0m"

    def __init__(self, colors: bool = True):
        self.colors = colors

    def __call__(self, logger, method_name, event_dict):
        timestamp = event_dict.get("timestamp", "")
        message = event_dict.get("event", "")
        level = event_dict.get("level", method_name).lower()
        letter, color = self.LEVEL_MAP.get(level, ("?", ""))
        prefix = f"[{letter} {timestamp}]"
        if self.colors and color:
            prefix = f"{color}{prefix}{self.RESET}"

        # Build the output with message and key-value pairs
        output_parts = [prefix]

        # Make the main message bold
        if self.colors:
            message = f"\033[1m{message}\033[0m"  # bold
        output_parts.append(message)

        # Add key-value pairs (excluding internal structlog keys)
        kv_pairs = []
        for key, value in event_dict.items():
            if key not in ["timestamp", "event", "level"]:
                if self.colors:
                    # Format: magenta key + equals + cyan value
                    kv_pairs.append(f"\033[35m{key}\033[0m=\033[36m{value}\033[0m")
                else:
                    # No colors for plain output
                    kv_pairs.append(f"{key}={value}")

        if kv_pairs:
            kv_text = " ".join(kv_pairs)
            if self.colors:
                kv_text = f"\033[35m{kv_text}{self.RESET}"  # magenta
            output_parts.append(kv_text)

        return " ".join(output_parts)


def _configure_structlog() -> None:
    """Configure structlog for both console and file output."""
    log_dir = _ensure_log_dir()
    log_file = log_dir / "main.log"
    json_log_file = log_dir / "main.log.json"

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                # Formatter for colored console output
                "colored": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        custom_timestamper,
                        *shared_processors,
                        MainConsoleRenderer(colors=True),
                    ],
                },
                # Formatter for plain file output
                "plain": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        custom_timestamper,
                        *shared_processors,
                        MainConsoleRenderer(colors=False),
                    ],
                },
                # Formatter for JSON file output
                "json": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        custom_timestamper,
                        *shared_processors,
                        structlog.processors.JSONRenderer(),
                    ],
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": _get_env_log_level(),
                    "formatter": "colored",
                    "stream": sys.stderr,
                },
                "file": {
                    "class": "logging.handlers.WatchedFileHandler",
                    "level": "DEBUG",
                    "filename": log_file,
                    "formatter": "plain",
                },
                "json_file": {
                    "class": "logging.handlers.WatchedFileHandler",
                    "level": "DEBUG",
                    "filename": json_log_file,
                    "formatter": "json",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file", "json_file"],
                    "level": "DEBUG",
                    "propagate": True,
                },
            },
        }
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    structlog.get_logger().debug("Logger configured", config=structlog.get_config())


# Configure logging on module import
_configure_structlog()


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a configured structlog logger."""
    return structlog.get_logger(name)


# Export the root logger for convenience
logger = get_logger()
