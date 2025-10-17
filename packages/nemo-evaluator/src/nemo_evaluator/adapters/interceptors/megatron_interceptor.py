import json
import os
import threading
import time
from collections import deque
from typing import Any, List, Optional, Tuple, Union, final

import requests
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestToResponseInterceptor,
)
from pydantic import BaseModel


@register_for_adapter(
    name="megatron",
    description="Handles Megatron API requests with automatic batching for efficiency",
)
@final
class MegatronEndpointInterceptor(RequestToResponseInterceptor):
    """Adapter for Megatron API endpoint with automatic request batching and chat template support."""

    class Params(BaseModel):
        """Configuration parameters for Megatron endpoint interceptor."""

        tokenizer: Any  # Tokenizer instance with apply_chat_template method
        batch_size: int = 8  # Maximum batch size
        batch_timeout_ms: float = (
            100.0  # Maximum time to wait for batch in milliseconds
        )
        template_kwargs: dict = {}  # Additional kwargs to pass to apply_chat_template

    _tokenizer: Any
    _batch_size: int
    _batch_timeout_ms: float
    _template_kwargs: dict
    _request_queue: deque
    _queue_lock: threading.Lock
    _batch_event: threading.Event

    def __init__(self, params: Params):
        """
        Initialize the Megatron adapter with batching support.

        Args:
            params: Configuration parameters including tokenizer and batch settings
        """
        self._tokenizer = params.tokenizer
        self._batch_size = params.batch_size
        self._batch_timeout_ms = params.batch_timeout_ms
        self._template_kwargs = params.template_kwargs
        self._request_queue = deque()
        self._queue_lock = threading.Lock()
        self._batch_event = threading.Event()

    def _process_batch(
        self,
        batch: List[Tuple[dict, int, AdapterGlobalContext]],
        context: AdapterGlobalContext,
    ) -> List[str]:
        """
        Process a batch of requests by applying chat template and calling Megatron API.

        Args:
            batch: List of (messages, tokens_to_generate, context) tuples
            context: Global context

        Returns:
            List of response texts
        """
        # Extract all messages and apply chat template in one call
        all_messages = [item[0] for item in batch]
        tokens_to_generate = batch[0][1]  # Assume all requests have same token count

        # Apply chat template to ALL messages at once for maximum efficiency
        # Merge template_kwargs with add_generation_prompt
        template_args = {"add_generation_prompt": True, **self._template_kwargs}
        prompts = self._tokenizer.apply_chat_template(all_messages, **template_args)

        # Ensure prompts is a list
        if not isinstance(prompts, list):
            prompts = [prompts]

        # Construct Megatron API request
        megatron_data = {"prompts": prompts, "tokens_to_generate": tokens_to_generate}

        # Prepare headers
        headers = {"Content-Type": "application/json"}

        # Construct the URL - append /api to the base URL
        base_url = context.url.rstrip("/")
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            base_url = "http://" + base_url
        megatron_url = base_url + "/api"

        # Make the PUT request
        response = requests.put(
            url=megatron_url,
            data=json.dumps(megatron_data),
            headers=headers,
            allow_redirects=False,
        )

        if response.status_code != 200:
            raise Exception(f"Megatron API error: {response.status_code}")

        response_json = response.json()
        return response_json.get("text", [])

    @final
    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> Union[AdapterRequest, AdapterResponse]:
        """
        Handle the Megatron API request with automatic batching.

        Accumulates incoming requests and batches them for efficiency.
        - Applies chat template to all batched messages at once
        - Makes single PUT request with all prompts
        - Returns individual responses

        Args:
            ar: The adapter request to intercept
            context: Global context containing server-level configuration

        Returns:
            AdapterResponse with the result for this specific request
        """
        try:
            # Extract messages and tokens_to_generate from the incoming request
            request_json = ar.r.json

            if not isinstance(request_json, dict) or "messages" not in request_json:
                raise ValueError("Request must contain 'messages' field")

            messages = request_json["messages"]
            tokens_to_generate = request_json.get("max_tokens", 100)

            # Create a response holder for this request
            response_holder = {"response": None, "event": threading.Event()}

            # Add to queue
            with self._queue_lock:
                self._request_queue.append(
                    (messages, tokens_to_generate, context, response_holder)
                )
                queue_size = len(self._request_queue)

            # If we reached batch size, process immediately
            if queue_size >= self._batch_size:
                self._process_queue(force=True)
            else:
                # Start a timer to process after timeout
                threading.Timer(
                    self._batch_timeout_ms / 1000.0,
                    lambda: self._process_queue(force=False),
                ).start()

            # Wait for response
            response_holder["event"].wait()

            return response_holder["response"]

        except Exception as e:
            print(f"Error in intercept_request: {e}")
            raise

    def _process_queue(self, force: bool = False):
        """
        Process the current queue of requests as a batch.

        Args:
            force: If True, process regardless of queue size
        """
        with self._queue_lock:
            if len(self._request_queue) == 0:
                return

            if not force and len(self._request_queue) < self._batch_size:
                return

            # Take all items from queue
            batch = list(self._request_queue)
            self._request_queue.clear()

        try:
            # Get context from first item
            context = batch[0][2]

            # Process batch
            batch_data = [(item[0], item[1], item[2]) for item in batch]
            results = self._process_batch(batch_data, context)

            # Return results to each request
            for i, item in enumerate(batch):
                response_holder = item[3]

                # Create mock response with the result
                mock_response = requests.Response()
                mock_response.status_code = 200
                mock_response._content = json.dumps({"text": [results[i]]}).encode(
                    "utf-8"
                )
                mock_response.headers["Content-Type"] = "application/json"

                response_holder["response"] = AdapterResponse(
                    r=mock_response,
                    rctx=(
                        batch[i][2]
                        if hasattr(batch[i], "__len__") and len(batch[i]) > 2
                        else None
                    ),
                )
                response_holder["event"].set()

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Return error to all waiting requests
            for item in batch:
                response_holder = item[3]
                mock_response = requests.Response()
                mock_response.status_code = 500
                mock_response._content = json.dumps({"error": str(e)}).encode("utf-8")
                response_holder["response"] = AdapterResponse(
                    r=mock_response,
                    rctx=None,
                )
                response_holder["event"].set()
