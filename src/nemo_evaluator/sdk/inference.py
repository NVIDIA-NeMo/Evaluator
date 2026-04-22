import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from contextvars import ContextVar
from typing import Any, Dict, Optional, Protocol
from urllib.parse import parse_qsl, urlparse, urlunparse

import openai
from openai import AsyncOpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, PrivateAttr

from nemo_evaluator.sdk.constants import PLACEHOLDER_INFERENCE_API_KEY
from nemo_evaluator.sdk.enums import ModelFormat
from nemo_evaluator.sdk.resilience.api import run_with_resilience
from nemo_evaluator.sdk.resilience.classifier import endpoint_identity
from nemo_evaluator.sdk.resilience.scheduler import ResilienceCancelledError
from nemo_evaluator.sdk.values import Model, ReasoningParams
from nemo_evaluator.sdk.values.models import filter_auth_headers
from nemo_evaluator.sdk.values.params import InferenceParams

# Local cache for OpenAI clients
_openai_clients = {}

# We use a context variable to store the requests log for the current request.
requests_log_var = ContextVar("requests_log")

# We use a context variable for the name of the logger to use
logger_var = ContextVar("logger_name")


def get_logger() -> logging.Logger:
    return logging.getLogger(logger_var.get(__name__))


def merge_default_headers(model: Model, default_headers: dict | None) -> dict[str, str] | None:
    """Merge model-level and per-call default headers for one inference request."""
    if model.default_headers is None and default_headers is None:
        return None

    return {
        **(model.default_headers or {}),
        **(default_headers or {}),
    }


def redact_request_for_logging(request_body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the request payload that is safe to persist in request logs."""
    redacted_request = dict(request_body)
    if "extra_headers" in redacted_request:
        filtered_headers = filter_auth_headers(redacted_request["extra_headers"])
        if filtered_headers:
            redacted_request["extra_headers"] = filtered_headers
        else:
            redacted_request.pop("extra_headers")
    return redacted_request


class ClientInferenceError(RuntimeError):
    def __init__(self, e: openai.APIStatusError, context: str | None = None):
        error_detail = getattr(e.response, "text", str(e))
        # Build the base message
        message = f"Unable to complete inference because a {e.status_code} error occurred."
        # Add context if provided
        if context:
            message = f"{message} {context}"
        # Add details only if there's actual content
        if error_detail and error_detail.strip():
            message = f"{message} Details: {error_detail}"
        super().__init__(message)


class InferenceFn(Protocol):
    """Callable protocol for inference function dependency injection."""

    def __call__(
        self,
        model: Model,
        request: dict,
        max_retries: int | None,
        api_key: str | None = None,
        *,
        default_headers: dict | None = None,
        timeout: float | None = None,
    ) -> Awaitable[dict]: ...


class InferenceHookParams(Protocol):
    """Fields that new_hooks() actually reads from its params argument."""

    @property
    def system_prompt(self) -> str | None: ...

    @property
    def inference(self) -> InferenceParams | None: ...

    @property
    def structured_output(self) -> dict | None: ...

    @property
    def reasoning(self) -> ReasoningParams | None: ...


class InferenceMetricBase(BaseModel):
    """Reusable inference transport state for V2 metrics."""

    _inference_fn: InferenceFn | None = PrivateAttr(default=None)

    @property
    def inference_fn(self) -> InferenceFn:
        """Return the effective inference function for this metric."""
        return self._inference_fn or make_inference_request

    def set_inference_fn(self, inference_fn: InferenceFn) -> None:
        """Inject the inference function to use for this metric."""
        self._inference_fn = inference_fn


class PreprocessRequest(ABC):
    """Interface for preprocessing inference request."""

    @abstractmethod
    def preprocess(self, request: Dict, id: Optional[str] = None) -> Dict:
        pass


class PostprocessResponse(ABC):
    """Interface for postprocessing inference response."""

    @abstractmethod
    def postprocess(self, response: Dict, id: Optional[str] = None) -> Dict:
        pass


class LogHook(PreprocessRequest, PostprocessResponse):
    """
    Log the inference request and response
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def preprocess(self, request: Dict, id: Optional[str] = None) -> Dict:
        if id:
            self.logger.debug("Request %s: %s", id, request)
        else:
            self.logger.debug("Request: %s", request)
        return request

    def postprocess(self, response: Dict, id: Optional[str] = None) -> Dict:
        if id:
            self.logger.debug("Response %s: %s", id, response)
        else:
            self.logger.debug("Response: %s", response)
        return response


class AddInferenceParameter(PreprocessRequest):
    def __init__(self, params: Dict[str, Any]):
        if not params:
            raise ValueError("params cannot be empty")
        self.params = params

    def preprocess(self, request: Dict, id: Optional[str] = None) -> Dict:
        return deep_merge(request, self.params)


class InjectSystemMessage(PreprocessRequest):
    def __init__(self, system_message: str, logger: logging.Logger | None = None):
        if not system_message:
            raise ValueError("system_message cannot be empty")
        self.system_message = system_message
        self.logger = logger or logging.getLogger(__name__)

    def preprocess(self, request: Dict, id: Optional[str] = None) -> Dict:
        """
        Prepend system message into the payload for existing message or insert as a new system message.
        """
        if request.get("messages"):
            msg = request["messages"][0]
            if msg.get("role") == "system":
                # Prefix the first message with the custom message
                request["messages"][0]["content"] = f"{self.system_message} {msg['content']}"
            else:
                # Add new system message
                request["messages"].insert(0, {"role": "system", "content": self.system_message})
        elif request.get("prompt"):
            request["prompt"] = f"{self.system_message} {request['prompt']}"
        else:
            prefix = "Request:"
            if id:
                prefix = f"Request {id}:"
            self.logger.warning(
                f"{prefix} Custom system message was not added to request due to unexpected format: missing prompt or messages"
            )
        return request


class TransformReasoningOutput(PostprocessResponse):
    """
    TransformReasoningOutput postprocess hook is primarily targeted for handling reasoning with
    Nemotron models which include reasoning context within the model output. Reasoning context is
    denoted with token <think>context</think> and is removed from the output and moved to a new
    response field `reasoning_content`.
    """

    def __init__(self, end_reasoning_token: Optional[str] = None):
        self.end_reasoning_token = end_reasoning_token

    def postprocess(self, response: Dict, id: Optional[str] = None) -> Dict:
        """
        Move reasoning tokens from output to a new field
        """
        for i, choice in enumerate(response.get("choices", [])):
            msg = choice.get("message")
            if msg and msg.get("role") == "assistant":
                content = msg.get("content")
                if not isinstance(content, str):
                    # Content can be None with function calling
                    continue

                split_content = content.rsplit(self.end_reasoning_token, 1)
                if len(split_content) == 2:
                    # Add last token back after split
                    response["choices"][i]["message"]["reasoning_content"] = split_content[0] + self.end_reasoning_token
                    response["choices"][i]["message"]["content"] = split_content[1]
            elif choice.get("text"):
                split_content = choice["text"].rsplit(self.end_reasoning_token, 1)
                if len(split_content) == 2:
                    choice["reasoning_content"] = split_content[0] + self.end_reasoning_token
                    choice["text"] = split_content[1]
        return response


def new_hooks(
    params: InferenceHookParams | None,
    model_format: ModelFormat | None = ModelFormat.NVIDIA_NIM,
    logger: logging.Logger | None = None,
) -> tuple[list[PreprocessRequest], list[PostprocessResponse]]:
    """Build the standard online generation hooks used by SDK and service flows."""
    from nemo_evaluator.sdk.structured_output import InferenceStructuredOutput, default_structured_output_mode

    log_hook = LogHook(logger)
    preprocess_hooks: list[PreprocessRequest] = []
    postprocess_hooks: list[PostprocessResponse] = [log_hook]

    # System prompt injection - must be first to prepend system message
    if params and params.system_prompt:
        preprocess_hooks.append(InjectSystemMessage(params.system_prompt, logger))

    if params and params.inference:
        preprocess_hooks.append(AddInferenceParameter(params.inference.model_dump(mode="json", exclude_none=True)))

    if params and params.structured_output:
        preprocess_hooks.append(
            InferenceStructuredOutput(
                default_structured_output_mode(model_format or "nim"),
                params.structured_output,
            )
        )

    preprocess_hooks.append(log_hook)

    if params and params.reasoning and params.reasoning.end_token:
        postprocess_hooks.append(TransformReasoningOutput(params.reasoning.end_token))

    return preprocess_hooks, postprocess_hooks


async def make_inference_request(
    model: Model,
    request: dict,
    max_retries: int | None = 3,
    api_key: str | None = None,
    *,
    default_headers: dict | None = None,
    timeout: float | None = None,
) -> dict:
    """
    Helper to run inference on a model with a given prompt.

    Only OpenAI format is supported (nim and openai formats).

    Args:
        model: The Model to run inference on.
        request: The request to run. Can be a completion or a chat request.
        max_retries: Maximum number of retries for the request.
        api_key: Optional explicit API key. If provided, overrides the placeholder.
                 If not provided, uses placeholder (caller must resolve api_key_secret).
        timeout: Optional request timeout in seconds. If None, client default behavior is used.

    Returns:
        The result of the inference.
    """
    log = get_logger()

    model_id = model.name
    extra_headers = merge_default_headers(model, default_headers)

    # Explicit api_key parameter has highest priority, then model.api_key, then placeholder
    api_key = api_key or model.api_key or PLACEHOLDER_INFERENCE_API_KEY

    # Make sure the base_url does not end in /completions or /chat/completions
    base_url = model.url
    parsed_url = urlparse(model.url)
    extra_query = dict(parse_qsl(parsed_url.query))
    for suffix in ["/chat/completions", "/completions"]:
        if parsed_url.path.endswith(suffix):
            base_url = urlunparse(parsed_url._replace(path=parsed_url.path[: -1 * len(suffix)], query=""))
            parsed_url = urlparse(base_url)

    cache_key = f"{base_url}-{api_key}"

    # Run inference against the API endpoint using the OpenAI client
    # We cache the models in `_openai_clients` per endpoint URL + api_key.
    # This is to avoid creating a new client for each request.
    if cache_key not in _openai_clients:
        _openai_clients[cache_key] = AsyncOpenAI(
            base_url=base_url,
            # Sometimes, a fake key is still required for the OpenAI client to work.
            api_key=api_key,
        )

    client: AsyncOpenAI = _openai_clients[cache_key]

    # To distinguish between completions and chat completions, we look at the request body.
    # TODO: add typing for the request.
    max_attempts = max(1, (max_retries if max_retries is not None else 0) + 1)
    endpoint_key = endpoint_identity(str(base_url), model_id=model_id, auth_identity=api_key)
    try:
        log.info("Making request to %s: %s", base_url, {"model": model_id, **request})

        requests_log = requests_log_var.get([])
        request_body = {"model": model_id, **request}
        if extra_query:
            request_body["extra_query"] = extra_query
        if extra_headers:
            request_body["extra_headers"] = extra_headers

        options = {"max_retries": 0}
        if timeout:
            options["timeout"] = timeout
        client_with_options = client.with_options(**options)
        fn = (
            client_with_options.chat.completions.create
            if "messages" in request
            else client_with_options.completions.create
        )
        completion: ChatCompletion | Completion = await run_with_resilience(
            endpoint_key, fn, max_attempts=max_attempts, **request_body
        )
        logged_request_body = redact_request_for_logging(request_body)
        requests_log.append({"request": logged_request_body, "response": completion.model_dump()})
        return completion.model_dump()

    except openai.APIConnectionError as e:
        log.warning(f"Error connecting to inference server at {base_url}, cause: {e.__cause__}")
        raise RuntimeError(f"Error connecting to inference server at {base_url}") from e
    except openai.RateLimitError as e:
        log.warning(f"Rate limit exceeded when issuing inference requests for {model_id}")
        raise RuntimeError(f"Rate limit exceeded when issuing inference requests for {model_id}") from e
    except openai.BadRequestError as e:
        if "guided_json is unsupported" in str(e):
            raise ClientInferenceError(e, "Verify whether the model version supports structured outputs.")
        raise ClientInferenceError(e, f"base_url: {base_url}, model_id: {model_id}")
    except openai.APIStatusError as e:
        exception = ClientInferenceError(e, f"base_url: {base_url}, model_id: {model_id}")
        log.warning(exception)
        raise exception
    except ResilienceCancelledError:
        # Preserve cancellation semantics for callers coordinating task/group shutdown.
        raise
    except Exception as e:
        # TODO: it maybe is sharing too much information to expose this error if it
        # ends up propagating back to the user
        # RRA: Better to err on sharing too much information than too little.
        log.exception(f"Unexpected error making completion request to {model_id}")
        raise RuntimeError(f"Unexpected error making completion request to {model_id}") from e


def preprocess_request(request: dict, hooks: list[PreprocessRequest], id: Optional[str] = None) -> Dict:
    """
    Applies preprocessing hooks to request. Hooks are applied in order.
    """
    for hook in hooks:
        request = hook.preprocess(request, id=id)
    return request


def process_output(response: dict, hooks: list[PostprocessResponse], id: Optional[str] = None) -> str:
    """
    Applies postprocessing hooks to response before extracting the text from the full LM response.

    Args:
        response (dict): The full LLM response.
        id (str): Optional identifier of the response
        hooks (List[PostprocessResponse]): Optional hooks to apply for postprocessing of the response. Hooks are applied in order.
    Returns:
        str: The text extracted from the response based on endpoint type.
    """
    for hook in hooks:
        response = hook.postprocess(response, id=id)

    if not ("choices" in response and len(response["choices"]) > 0):
        raise ValueError("Invalid response format: No choices found in the response.")

    if ("message" in response["choices"][0]) and ("content" in response["choices"][0]["message"]):
        # Return text from chat-completion response
        return response["choices"][0]["message"]["content"]
    elif "text" in response["choices"][0]:
        # Return text from completion response
        return response["choices"][0]["text"]
    else:
        # If neither field is present, raise an error
        raise ValueError(f"Invalid response format: No text found in the response {response}.")


def deep_merge(request: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge provider request params into an existing request.

    Nested dictionaries are merged recursively so provider-specific payloads such as
    ``extra_body.nvext`` can combine user-specified options with evaluator-added fields.

    Example:
    request = {"extra_body": {"nvext": {"max_thinking_tokens": 256}}}
    params = {"extra_body": {"nvext": {"guided_json": {...}}}}

    A plain ``request.update(params)`` would replace the entire ``extra_body`` payload and drop
    ``max_thinking_tokens``. This helper preserves both keys under ``extra_body.nvext``.

    Non-dict values are overwritten by the newer params.
    """
    merged = request.copy()

    for key, value in params.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged
