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

"""Core registry system for interceptors and post-evaluation hooks with interface awareness."""

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel

from nemo_evaluator.adapters.types import (
    PostEvalHook,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InterceptorMetadata:
    """Metadata for registered interceptors and post-evaluation hooks"""

    name: str
    description: str
    interceptor_class: Type[
        RequestInterceptor
        | ResponseInterceptor
        | RequestToResponseInterceptor
        | PostEvalHook
    ]
    init_schema: Optional[Type[BaseModel]] = None

    def supports_request_interception(self) -> bool:
        """Check if this interceptor can handle requests"""
        return issubclass(
            self.interceptor_class, (RequestInterceptor, RequestToResponseInterceptor)
        )

    def supports_response_interception(self) -> bool:
        """Check if this interceptor can handle responses"""
        return issubclass(self.interceptor_class, ResponseInterceptor)

    def supports_request_to_response_interception(self) -> bool:
        """Check if this interceptor can handle request-to-response interception"""
        return issubclass(self.interceptor_class, RequestToResponseInterceptor)

    def supports_post_eval_hook(self) -> bool:
        """Check if this is a post-evaluation hook"""
        return issubclass(self.interceptor_class, PostEvalHook)


class InterceptorRegistry:
    """Central registry for all interceptors and post-evaluation hooks with interface awareness. Singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._metadata: dict[str, InterceptorMetadata] = {}
            self._instance_cache: dict[
                str,
                RequestInterceptor
                | ResponseInterceptor
                | RequestToResponseInterceptor
                | PostEvalHook,
            ] = {}
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "InterceptorRegistry":
        """Get the singleton instance of the registry"""
        return cls()

    def register(
        self,
        name: str,
        interceptor_class: Type[
            RequestInterceptor
            | ResponseInterceptor
            | RequestToResponseInterceptor
            | PostEvalHook
        ],
        metadata: InterceptorMetadata,
    ) -> None:
        """Register an interceptor or post-evaluation hook class with metadata"""
        # Validate that the class implements at least one of our interfaces
        if not (
            issubclass(interceptor_class, RequestInterceptor)
            or issubclass(interceptor_class, ResponseInterceptor)
            or issubclass(interceptor_class, RequestToResponseInterceptor)
            or issubclass(interceptor_class, PostEvalHook)
        ):
            raise ValueError(
                f"Class {interceptor_class.__name__} must implement at least one of RequestInterceptor, ResponseInterceptor, RequestToResponseInterceptor, or PostEvalHook"
            )

        metadata.interceptor_class = interceptor_class

        self._metadata[name] = metadata
        logger.debug(
            f"Registered adapter component: {name} ({interceptor_class.__name__})"
        )

    def _get_or_create_instance(
        self, name: str, config: dict[str, Any]
    ) -> (
        RequestInterceptor
        | ResponseInterceptor
        | RequestToResponseInterceptor
        | PostEvalHook
    ):
        """Get or create an interceptor or post-evaluation hook instance with caching"""
        # Use a stable JSON string as cache key to handle unhashable types
        cache_key = f"{name}_{json.dumps(config, sort_keys=True, default=str)}"

        if cache_key not in self._instance_cache:
            metadata = self._metadata[name]
            try:
                # Create Params object and instantiate component
                if metadata.init_schema is not None:
                    params = metadata.init_schema(**config)
                    instance = metadata.interceptor_class(params=params)
                else:
                    # Fallback for components without Params class
                    instance = metadata.interceptor_class(**config)

                self._instance_cache[cache_key] = instance
                logger.debug(f"Created instance of {name} with config: {config}")
            except Exception as e:
                logger.error(
                    f"Failed to create instance of {name} with config {config}: {e}"
                )
                raise

        return self._instance_cache[cache_key]

    def discover_components(
        self, modules: Optional[list[str]] = None, dirs: Optional[list[str]] = None
    ) -> None:
        """Auto-discover interceptors and post-evaluation hooks from modules and directories.

        Args:
            modules: List of module paths to import and discover components from
            dirs: List of directory paths to scan for Python files with components
        """
        self._discover_from_modules(modules)
        self._discover_from_directories(dirs)

    def _discover_from_modules(self, modules: Optional[list[str]]) -> None:
        """Discover components from specified modules."""
        # Always load the default interceptors and reports modules
        modules = modules or []
        default_modules = [
            "nemo_evaluator.adapters.interceptors",
            "nemo_evaluator.adapters.reports",
            "nemo_evaluator_internal.adapters.interceptors",
        ]
        all_modules = default_modules + modules

        for module_name in all_modules:
            try:
                # NOTE(agronskiy): reload needed sometimes after the registry reset to re-execute the
                # adapters logic.
                importlib.reload(importlib.import_module(module_name))
                logger.info(f"Successfully imported module: {module_name}")
            except Exception as e:
                # Silent failure for internal modules
                if "nemo_evaluator_internal" in module_name:
                    pass
                else:
                    logger.warning(f"Failed to import module {module_name}: {e}")

    def _discover_from_directories(self, dirs: Optional[list[str]]) -> None:
        """Discover components from Python files in directories."""
        if not dirs:
            return
        original_path = sys.path.copy()

        try:
            for directory in dirs:
                discovery_path = Path(directory)
                if not discovery_path.exists():
                    logger.warning(f"Discovery directory does not exist: {directory}")
                    continue

                if not discovery_path.is_dir():
                    logger.warning(f"Discovery path is not a directory: {directory}")
                    continue

                # Add parent directory to Python path for imports
                sys.path.insert(0, str(discovery_path.parent))

                for py_file in discovery_path.glob("*.py"):
                    if not self._should_process_file(py_file):
                        continue

                    module_name = f"{discovery_path.name}.{py_file.stem}"
                    try:
                        importlib.import_module(module_name)
                        logger.info(f"Successfully imported module: {module_name}")
                    except Exception as e:
                        logger.warning(f"Failed to import module {module_name}: {e}")

        finally:
            sys.path = original_path

    def _should_process_file(self, py_file: Path) -> bool:
        """Check if a Python file should be processed for component discovery."""
        try:
            file_content = py_file.read_text(encoding="utf-8")
            return "@register_for_adapter" in file_content
        except Exception as e:
            logger.warning(f"Failed to read file {py_file}: {e}")
            return False

    def get_all_components(self) -> dict[str, InterceptorMetadata]:
        """Get all available components (interceptors and post-eval hooks)"""
        return self._metadata

    def get_metadata(self, name: str) -> Optional[InterceptorMetadata]:
        """Get metadata for a specific component"""
        return self._metadata.get(name)

    def get_post_eval_hooks(self) -> dict[str, InterceptorMetadata]:
        """Get all post-evaluation hooks"""
        return {
            name: metadata
            for name, metadata in self._metadata.items()
            if metadata.supports_post_eval_hook()
        }

    def get_interceptors(self) -> dict[str, InterceptorMetadata]:
        """Get all interceptors (excluding post-eval hooks)"""
        return {
            name: metadata
            for name, metadata in self._metadata.items()
            if not metadata.supports_post_eval_hook()
        }

    def is_request_interceptor(self, name: str) -> bool:
        """Check if a component supports request interception"""
        metadata = self._metadata.get(name)
        return metadata.supports_request_interception() if metadata else False

    def is_response_interceptor(self, name: str) -> bool:
        """Check if a component supports response interception"""
        metadata = self._metadata.get(name)
        return metadata.supports_response_interception() if metadata else False

    def is_request_to_response_interceptor(self, name: str) -> bool:
        """Check if a component supports request-to-response interception"""
        metadata = self._metadata.get(name)
        return (
            metadata.supports_request_to_response_interception() if metadata else False
        )

    def is_post_eval_hook(self, name: str) -> bool:
        """Check if a component is a post-evaluation hook"""
        metadata = self._metadata.get(name)
        return metadata.supports_post_eval_hook() if metadata else False

    def clear_cache(self) -> None:
        """Clear the instance cache"""
        self._instance_cache.clear()

    def reset(self) -> None:
        """Reset the registry (for testing)"""
        self._metadata.clear()
        self._instance_cache.clear()
