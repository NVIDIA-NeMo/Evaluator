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

"""Decorators for registering interceptors and post-evaluation hooks with automatic interface detection."""

import inspect
from typing import Type

from pydantic import BaseModel

from nemo_evaluator.adapters.registry import InterceptorMetadata, InterceptorRegistry
from nemo_evaluator.adapters.types import (
    PostEvalHook,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)


def register_for_adapter(
    name: str,
    description: str,
):
    """Decorator to register interceptors or post-evaluation hooks with automatic interface detection.

    Args:
        name: Unique name for the adapter component
        description: Human-readable description of what the component does

    Example:
        from pydantic import BaseModel, Field

        @register_for_adapter(
            name="my_interceptor",
            description="My custom request processing logic"
        )
        class MyInterceptor(RequestInterceptor):
            class Params(BaseModel):
                api_key: str = Field(..., description="API key for authentication")
                timeout: int = Field(default=30, description="Request timeout in seconds")

            def __init__(self, params: Params):
                self.api_key = params.api_key
                self.timeout = params.timeout

            def intercept_request(self, ar: AdapterRequest) -> AdapterRequest:
                # Custom logic here
                return ar

        @register_for_adapter(
            name="my_hook",
            description="My custom post-evaluation hook"
        )
        class MyHook(PostEvalHook):
            class Params(BaseModel):
                output_path: str = Field(..., description="Output path")

            def __init__(self, params: Params):
                self.output_path = params.output_path

            def post_eval_hook(self, context: AdapterGlobalContext) -> None:
                # Custom logic here
                pass
    """

    def decorator(cls):
        # Validate that the class implements at least one interface
        implements_request = issubclass(cls, RequestInterceptor)
        implements_response = issubclass(cls, ResponseInterceptor)
        implements_request_to_response = issubclass(cls, RequestToResponseInterceptor)
        implements_post_eval_hook = issubclass(cls, PostEvalHook)

        if not (
            implements_request
            or implements_response
            or implements_request_to_response
            or implements_post_eval_hook
        ):
            raise ValueError(
                f"Class {cls.__name__} must implement at least one of RequestInterceptor, ResponseInterceptor, RequestToResponseInterceptor, or PostEvalHook"
            )

        # Validate that the class has a Params class
        if not hasattr(cls, "Params"):
            raise ValueError(
                f"Class {cls.__name__} must have a nested Params class that inherits from pydantic.BaseModel"
            )
        params_class = cls.Params
        if not issubclass(params_class, BaseModel):
            raise ValueError(
                f"Class {cls.__name__}.Params must inherit from pydantic.BaseModel, got {type(params_class)}"
            )

        # Validate that __init__ takes only a single Params object
        _validate_init_signature(cls, params_class)

        metadata = InterceptorMetadata(
            name=name,
            description=description,
            interceptor_class=cls,
            init_schema=params_class,
        )

        # Store metadata on the class for discovery
        cls._interceptor_metadata = metadata

        # Register with global registry
        registry = InterceptorRegistry.get_instance()
        registry.register(name, cls, metadata)

        return cls

    return decorator


def _validate_init_signature(cls: Type, params_class: Type[BaseModel]) -> None:
    """Validate that the class __init__ method takes only a single Params object.

    Args:
        cls: The interceptor class to validate
        params_class: The Params class for validation

    Raises:
        ValueError: If the __init__ signature doesn't match the expected format
    """
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())

    # Remove 'self' from the list
    if params and params[0] == "self":
        params = params[1:]

    # Check that there's exactly one parameter named 'params' of the correct type
    if len(params) != 1 or params[0] != "params":
        raise ValueError(
            f"Class {cls.__name__}.__init__ must take exactly one parameter named 'params' of type {params_class.__name__}, got {params}"
        )

    # Check the type annotation
    param = sig.parameters["params"]
    if param.annotation != params_class:
        raise ValueError(
            f"Class {cls.__name__}.__init__ parameter 'params' must be annotated as {params_class.__name__}, got {param.annotation}"
        )
