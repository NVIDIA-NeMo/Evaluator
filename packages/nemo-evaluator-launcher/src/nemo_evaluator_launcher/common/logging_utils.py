# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Logging configuration module for nemo-evaluator-launcher.

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

import json
import logging
import logging.config
import os
import pathlib
import sys
from datetime import datetime
from pprint import pformat
from typing import Any, Dict

import structlog

# If this env var is set, it will override a more standard "LOG_LEVEL". If
# both are unset, default would be used.
_LOG_LEVEL_ENV_VAR = "NEMO_EVALUATOR_LOG_LEVEL"
_DEFAULT_LOG_LEVEL = "WARNING"
_SENSITIVE_KEY_SUBSTRINGS_NORMALIZED = {
    # Keep minimal, broad substrings
    # NOTE: normalized: lowercased, no spaces/_/-
    "authorization",  # covers proxy-authorization, etc.
    "apikey",  # covers api_key, api-key, x-api-key, nvidia_api_key, ...
    "accesskey",  # covers access_key / access-key
    "privatekey",
    "token",  # covers access_token, id_token, refresh_token, *_token
    "secret",  # covers openai_client_secret, aws_secret_access_key, *_secret
    "password",
    "pwd",  # common shorthand
    "passwd",  # common variant
}
_ALLOWLISTED_KEYS_SUBSTRINGS = {
    # NOTE: non-normalized (for allowlisting we want more control)
    "_tokens",  # This likely would allow us to not redact useful stuff like `limit_tokens`, `max_new_tokens`
}


def _mask(val: object) -> str:
    s = str(val)
    if len(s) <= 10:
        return "[REDACTED]"
    return f"{s[:2]}…{s[-2:]}"


def _normalize(name: object) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower()
    # drop spaces, hyphens, underscores
    return s.replace(" ", "").replace("-", "").replace("_", "")


def _is_sensitive_key(key: object) -> bool:
    k_norm = _normalize(key)
    k_non_norm = str(key)
    return any(
        substr in k_norm for substr in _SENSITIVE_KEY_SUBSTRINGS_NORMALIZED
    ) and not any(substr in k_non_norm for substr in _ALLOWLISTED_KEYS_SUBSTRINGS)


def _redact_mapping(m: dict) -> dict:
    red = {}
    for k, v in m.items():
        if _is_sensitive_key(k):
            red[k] = _mask(v)
        elif isinstance(v, dict):
            red[k] = _redact_mapping(v)
        else:
            red[k] = v
    return red


def redact_processor(_: Any, __: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("LOG_DISABLE_REDACTION", "").lower() in {"1", "true", "yes"}:
        return event_dict
    return _redact_mapping(event_dict)


def _ensure_log_dir() -> pathlib.Path:
    """Ensure the log directory exists and return its path."""
    log_dir = pathlib.Path.home() / ".nemo-evaluator" / "logs"
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
    env_level = os.getenv(_LOG_LEVEL_ENV_VAR, os.getenv("LOG_LEVEL"))
    # If empty or unset, default
    if not env_level:
        env_level = _DEFAULT_LOG_LEVEL
    env_level = env_level.upper()

    # Translate single letters to full level names
    level_map = {
        "D": "DEBUG",
        "I": "INFO",
        "W": "WARNING",
        "E": "ERROR",
        "F": "CRITICAL",
    }

    return level_map.get(env_level, env_level)


def custom_timestamper(_: Any, __: Any, event_dict: Dict[str, Any]) -> Dict[str, Any]:
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

    def __call__(
        self, logger: Any, method_name: str, event_dict: Dict[str, Any]
    ) -> str:
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
                # Pretty-format complex values (dict/list) as JSON on new lines
                pretty_value = None
                if isinstance(value, (dict, list)):
                    try:
                        pretty_value = json.dumps(
                            value, ensure_ascii=False, sort_keys=True, indent=2
                        )
                    except Exception:
                        pretty_value = pformat(value, width=100, compact=False)
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    # Fall back to reasonably readable representation for other complex types
                    pretty_value = pformat(value, width=100, compact=False)

                rendered_value = (
                    pretty_value if pretty_value is not None else str(value)
                )

                # If multiline, place value on a new line for readability
                if "\n" in rendered_value:
                    if self.colors:
                        kv_pairs.append(
                            f"\033[35m{key}\033[0m=\n\033[36m{rendered_value}\033[0m"
                        )
                    else:
                        kv_pairs.append(f"{key}=\n{rendered_value}")
                else:
                    if self.colors:
                        # Format: magenta key + equals + cyan value
                        kv_pairs.append(
                            f"\033[35m{key}\033[0m=\033[36m{rendered_value}\033[0m"
                        )
                    else:
                        # No colors for plain output
                        kv_pairs.append(f"{key}={rendered_value}")

        if kv_pairs:
            # If any kv is multiline, join with newlines; otherwise keep single line
            if any("\n" in kv for kv in kv_pairs):
                kv_text = "\n".join(kv_pairs)
            else:
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
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        redact_processor,
        custom_timestamper,
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
                        *shared_processors,
                        MainConsoleRenderer(colors=True),
                    ],
                },
                # Formatter for plain file output
                "plain": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processors": [
                        *shared_processors,
                        MainConsoleRenderer(colors=False),
                    ],
                },
                # Formatter for JSON file output
                "json": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processors": [
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


def get_logger(name: str | None = None) -> Any:
    """Get a configured structlog logger."""
    return structlog.get_logger(name)


# Export the root logger for convenience
logger = get_logger()
