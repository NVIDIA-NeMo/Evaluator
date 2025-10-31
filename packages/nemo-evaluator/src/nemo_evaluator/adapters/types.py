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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import flask
import requests

# __all__ = ["AdapterGlobalContext", "AdapterRequestContext", "AdapterRequest", "AdapterResponse", "RequestInterceptor", "RequestToResponseInterceptor", "ResponseInterceptor"]


@dataclass
class AdapterGlobalContext:
    """Global context passed to all interceptors containing server-level configuration.

    This context contains information that is available throughout the entire
    request-response cycle and is shared across all interceptors.
    """

    output_dir: str  # Directory for output files
    url: str  # The upstream API URL to forward requests to

    @property
    def metrics_path(self) -> Path:
        """Path to the eval_factory_metrics.json file."""
        return Path(self.output_dir) / "eval_factory_metrics.json"


@dataclass
class AdapterRequestContext:
    """This is passed with the _whole_ chain from request back to response.

    Add here any useful information.
    """

    cache_hit: bool = False  # Whether there was a cache hit
    cache_key: str | None = None  # Cache key
    request_id: str | None = None  # Request ID for identification


@dataclass
class AdapterRequest:
    r: flask.Request
    rctx: AdapterRequestContext


@dataclass
class AdapterResponse:
    r: requests.Response
    rctx: AdapterRequestContext
    latency_ms: float | None = None


class FatalErrorException(Exception):
    """Exception raised when an interceptor encounters a fatal error that should kill the process."""

    pass


class RequestInterceptor(ABC):
    """Interface for providing interception of requests."""

    @abstractmethod
    def intercept_request(
        self,
        req: AdapterRequest,
        context: AdapterGlobalContext,
    ) -> AdapterRequest:
        """Function that will be called by `AdapterServer` on the way upstream.

        This interceptor can modify the request but must return an AdapterRequest
        to continue the chain upstream.

        Args:
            req: The adapter request to intercept
            context: Global context containing server-level configuration

        Ex.: This is used for request preprocessing, logging, etc.
        """
        pass


class RequestToResponseInterceptor(ABC):
    """Interface for interceptors that can either continue the request chain or return a response."""

    @abstractmethod
    def intercept_request(
        self,
        req: AdapterRequest,
        context: AdapterGlobalContext,
    ) -> AdapterRequest | AdapterResponse:
        """Function that will be called by `AdapterServer` on the way upstream.

        If the return type is `AdapterRequest`, the chain will continue upstream.
        If the return type is `AdapterResponse`, the `AdapterServer` will consider the
        chain finished and start the reverse chain of responses.

        Args:
            req: The adapter request to intercept
            context: Global context containing server-level configuration

        Ex.: the latter case is e.g. how caching interceptor works. For cache miss
        it will continue the chain, passing request unchanged. For cache hit,
        it will go for the response.
        """
        pass


class ResponseInterceptor(ABC):
    """Interface for providing interception of responses."""

    @abstractmethod
    def intercept_response(
        self,
        resp: AdapterResponse,
        context: AdapterGlobalContext,
    ) -> AdapterResponse:
        """Function that will be called by `AdapterServer` on the way downstream.

        Args:
            resp: The adapter response to intercept
            context: Global context containing server-level configuration
        """
        pass


class PostEvalHook(ABC):
    """Interface for post-evaluation hooks that run after the evaluation completes.

    Post-evaluation hooks are executed after the main evaluation has finished,
    allowing for cleanup, report generation, or other post-processing tasks.
    """

    @abstractmethod
    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Function that will be called by the evaluation system after evaluation completes.

        Args:
            context: Global context containing server-level configuration and evaluation results

        Ex.: This is used for report generation, cleanup, metrics collection, etc.
        """
        pass
