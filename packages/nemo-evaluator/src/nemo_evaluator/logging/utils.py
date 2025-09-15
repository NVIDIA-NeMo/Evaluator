# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging configuration module for nemo-evaluator.

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
- from nemo_evaluator.common.logging_utils import logger
- from nemo_evaluator.common.logging_utils import get_logger

Examples:
    # Correct
    from nemo_evaluator.common.logging_utils import logger
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


def _ensure_log_dir(log_dir: str = None) -> pathlib.Path:
    """Ensure the log directory exists and return its path.

    Args:
        log_dir: Custom log directory path. If None, uses default ~/.nemo-evaluator/logs/

    Returns:
        Path to the log directory
    """
    if log_dir:
        log_dir_path = pathlib.Path(log_dir)
    else:
        log_dir_path = pathlib.Path.home() / ".nemo-evaluator" / "logs"

    log_dir_path.mkdir(parents=True, exist_ok=True)
    return log_dir_path


def _get_env_log_dir() -> str | None:
    """Get log directory from environment variable NEMO_EVALUATOR_LOG_DIR.

    Returns:
        Log directory path if NEMO_EVALUATOR_LOG_DIR is set, None otherwise.
    """
    return os.getenv("NEMO_EVALUATOR_LOG_DIR")


def _get_env_log_level() -> str:
    """Get log level from environment variable, translating single letters to full names.

    Translates:
    - D -> DEBUG
    - I -> INFO
    - W -> WARNING
    - E -> ERROR
    - F -> CRITICAL

    Also accepts full level names in any case (debug, Debug, DEBUG all work).

    Returns:
        Uppercase log level string, defaults to INFO if not set or invalid.
    """
    # Support both LOG_LEVEL (new) and NEMO_EVALUATOR_LOG_LEVEL (legacy) for backward compatibility
    env_level = os.getenv("LOG_LEVEL") or os.getenv("NEMO_EVALUATOR_LOG_LEVEL", "INFO")

    # Define valid log levels (case-insensitive)
    valid_levels = ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"]

    # First check if it's a single letter alias
    level_map = {
        "D": "DEBUG",
        "I": "INFO",
        "W": "WARNING",
        "E": "ERROR",
        "F": "CRITICAL",
    }

    # Check single letter aliases first
    if env_level in level_map:
        return level_map[env_level]

    # Then check if it's a valid level name (case-insensitive)
    env_level_upper = env_level.upper()
    if env_level_upper in valid_levels:
        return env_level_upper

    # If not valid, default to INFO
    print(
        f"Warning: Invalid log level '{env_level}', defaulting to INFO", file=sys.stderr
    )
    return "INFO"


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
            output_parts.append(kv_text)

        return " ".join(output_parts)


def _configure_structlog(log_dir: str = None) -> None:
    """Configure structlog for both console and file output.

    Args:
        log_dir: Custom log directory path. If None, uses default location
    """
    # Use environment variable if no log_dir is provided
    if log_dir is None:
        log_dir = _get_env_log_dir()

    # Only configure file logging if log_dir is specified
    if log_dir:
        log_dir_path = _ensure_log_dir(log_dir)
        log_file = log_dir_path / "main.log"
        json_log_file = log_dir_path / "main.log.json"

        # Configure handlers including file handlers
        handlers_config = {
            "console": {
                "class": "logging.StreamHandler",
                "level": _get_env_log_level(),
                "formatter": "colored",
                "stream": sys.stderr,  # Logs go to stderr, not stdout
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
        }
        logger_handlers = ["console", "file", "json_file"]
    else:
        # Only console logging, no file handlers
        handlers_config = {
            "console": {
                "class": "logging.StreamHandler",
                "level": _get_env_log_level(),
                "formatter": "colored",
                "stream": sys.stderr,  # Logs go to stderr, not stdout
            },
        }
        logger_handlers = ["console"]

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
            "handlers": handlers_config,
            "loggers": {
                "": {
                    "handlers": logger_handlers,
                    "level": _get_env_log_level(),  # Root logger level should match console level
                    "propagate": True,
                },
            },
        }
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Log the configuration details
    console_level = _get_env_log_level()
    if log_dir:
        structlog.get_logger().info(
            "Centralized logging configured",
            console_level=console_level,
            file_level="DEBUG",
            json_file_level="DEBUG",
            log_dir=str(log_dir),
        )
    else:
        structlog.get_logger().info(
            "Centralized logging configured (console only)",
            console_level=console_level,
            log_dir="none (NEMO_EVALUATOR_LOG_DIR not set)",
        )


# Configure logging on module import (with default directory)
_configure_structlog()


def configure_logging(log_dir: str = None) -> None:
    """Configure logging with a custom log directory.

    Args:
        log_dir: Custom log directory path. If None, uses NEMO_EVALUATOR_LOG_DIR environment variable.
                 If neither is set, only console logging is configured.
    """
    _configure_structlog(log_dir)


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a configured structlog logger."""
    return structlog.get_logger(name)


# Export the root logger for convenience
logger = get_logger()
