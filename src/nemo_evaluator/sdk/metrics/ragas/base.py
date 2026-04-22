from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
from collections.abc import Awaitable, Callable
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import httpx
import nemo_evaluator.sdk.constants as constants
from nemo_evaluator.sdk.enums import MetricType
from nemo_evaluator.sdk.inference import get_logger, requests_log_var

# Lazy imports for RAGAS - these are getter functions that defer the expensive
# RAGAS/langchain imports (~20-30s) until first use, improving startup time.
from nemo_evaluator.sdk.metrics.ragas.imports import (
    get_evaluate_function,
    get_evaluation_dataset_class,
    get_langchain_embeddings_wrapper_class,
    get_langchain_llm_wrapper_class,
    get_run_config_class,
)

# Type-only imports for static analysis (not imported at runtime)
if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
from nemo_evaluator.sdk.templates import render_request
from nemo_evaluator.sdk.values import (
    MetricBase,
    MetricResult,
    MetricScore,
    Model,
    SecretRef,
)
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

# RAGAS configuration constants
RAGAS_MAX_WAIT = 600  # 10 minutes in seconds
RAGAS_LOG_TENACITY = True
RAGAS_SEED = 42
DEFAULT_JUDGE_TIMEOUT = 120  # 2 minutes in seconds
DEFAULT_JUDGE_MAX_RETRIES = 3
DEFAULT_JUDGE_MAX_WORKER = 1
log = logging.getLogger(__name__)


# Lazy loaders for langchain classes (cached to avoid repeated imports)
@cache
def _get_langchain_chat_openai():
    """Lazy load ChatOpenAI from langchain."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI


@cache
def _get_nvidia_embeddings():
    """Lazy load NVIDIAEmbeddings from langchain."""
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    return NVIDIAEmbeddings


@cache
def _get_base_callback_handler():
    """Lazy load BaseCallbackHandler from langchain_core."""
    from langchain_core.callbacks import BaseCallbackHandler

    return BaseCallbackHandler


@cache
def _get_output_parser_exception_type() -> type[BaseException] | None:
    """Lazy load OutputParserException from langchain_core when available."""
    try:
        from langchain_core.exceptions import OutputParserException
    except Exception:
        return None
    return OutputParserException


class BaseRAGASMetric(MetricBase):
    """Base class for all RAGAS metrics in v2.

    Generic over the params type to provide proper type inference in subclasses.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Note: Subclasses must define 'type: Literal[MetricType.XXX] = MetricType.XXX'
    input_template: dict[str, Any] | None = Field(
        default=None,
        description="Optional Jinja template for rendering the input payload for RAGAS evaluation.",
    )

    metric_threshold: float | None = None
    metric_threshold_score: str | None = None
    _llm_model: dict | None = None
    _inference_params: dict | None = None
    _embed_params: dict | None = None
    _secrets: dict[str, SecretRef] = PrivateAttr(default_factory=dict)
    _log: logging.Logger = logging.getLogger(__name__)

    def score_names(self) -> list[str]:
        if isinstance(self.type, MetricType):
            return [self.type.value]
        return []

    def __init__(self, logger: logging.Logger | None = None, **data):
        super().__init__(**data)
        if logger:
            self._log = logger

        self._inference_params = {}
        inference = getattr(self, "inference", None)
        if isinstance(inference, BaseModel):
            self._inference_params = inference.model_dump(mode="json", exclude_none=True)

        judge_model = getattr(self, "judge_model", None)
        if judge_model is not None:
            assert isinstance(judge_model, Model)

            # Determine initial API key:
            # - If api_key_secret is configured, resolve secret from env
            # - If no api_key_secret, use placeholder immediately (no secret resolution needed)
            if judge_model.api_key_secret:
                assert judge_model.api_key_env is not None
                self._secrets[judge_model.api_key_env] = judge_model.api_key_secret
                initial_api_key = judge_model.api_key
            else:
                initial_api_key = constants.PLACEHOLDER_INFERENCE_API_KEY

            self._llm_model = {
                "model": judge_model.name,
                "base_url": judge_model.url.replace("/completions", "").replace("/chat", ""),
                "api_key": initial_api_key,
            }

        embeddings_model = getattr(self, "embeddings_model", None)
        if embeddings_model is not None:
            assert isinstance(embeddings_model, Model)

            # Determine initial API key:
            # - If api_key_secret is configured, resolve secret from env
            # - If no api_key_secret, use placeholder immediately (no secret resolution needed)
            if embeddings_model.api_key_secret:
                assert embeddings_model.api_key_env is not None
                self._secrets[embeddings_model.api_key_env] = embeddings_model.api_key_secret
                initial_api_key = embeddings_model.api_key
            else:
                initial_api_key = constants.PLACEHOLDER_INFERENCE_API_KEY

            self._embed_params = {
                "model": embeddings_model.name,
                "base_url": embeddings_model.url.replace("/embeddings", ""),
                "api_key": initial_api_key,
                "truncate": self._inference_params.get("truncate", "NONE"),
            }

    async def resolve_secrets(self, secret_resolver: Callable[[str], Awaitable[str | None]]) -> None:
        """Resolve API key secrets if configured. Must be called before using the metric.

        This follows the same pattern as LLMJudgeMetric.resolve_secrets().

        Args:
            secret_resolver: Async callback to resolve secret names to values.
        """
        # Resolve judge API key (only if api_key_secret is configured)
        judge_model = getattr(self, "judge_model", None)
        if judge_model is not None:
            assert isinstance(judge_model, Model)
            if judge_model.api_key_secret:
                secret_name = judge_model.api_key_secret.root
                api_key = await secret_resolver(secret_name)
                if not api_key:
                    raise ValueError(f"Missing secret '{secret_name}' for API key authentication with LLM judge.")
                # Update the model config with resolved API key
                if self._llm_model:
                    self._llm_model["api_key"] = api_key

        # Resolve embeddings API key (only if api_key_secret is configured)
        embeddings_model = getattr(self, "embeddings_model", None)
        if embeddings_model is not None:
            assert isinstance(embeddings_model, Model)
            if embeddings_model.api_key_secret:
                secret_name = embeddings_model.api_key_secret.root
                api_key = await secret_resolver(secret_name)
                if not api_key:
                    raise ValueError(
                        f"Missing secret '{secret_name}' for API key authentication with embeddings model."
                    )
                # Update the model config with resolved API key
                if self._embed_params:
                    self._embed_params["api_key"] = api_key

    def secrets(self) -> dict[str, SecretRef]:
        """Return mapping of env var names to secret names.

        This is used by the framework to know which secrets need to be injected.
        """
        return self._secrets

    def _ignore_inference_failure(self) -> bool:
        """Return whether this metric should ignore inference-call failures."""
        return getattr(self, "ignore_inference_failure", False)

    def _nan_scores_for_metrics(self, metrics: list) -> dict[str, float]:
        """Build a NaN score mapping for the active RAGAS metric objects."""
        metric_names: list[str] = []
        for metric in metrics:
            metric_name = getattr(metric, "name", None)
            if isinstance(metric_name, str) and metric_name:
                metric_names.append(metric_name)

        if not metric_names:
            metric_names = self.score_names()

        return {metric_name: float("nan") for metric_name in metric_names}

    def _get_llm_judge(self, client: httpx.AsyncClient | None = None) -> LangchainLLMWrapper | None:
        """Get the LLM judge instance based on configuration."""
        if not self._llm_model:
            return None

        chat_params: dict[str, Any] = {**self._llm_model}
        if self._inference_params:
            chat_params.update(self._inference_params)

        # Filter out None values
        chat_params = {k: v for k, v in chat_params.items() if v is not None}

        # Lazy load ChatOpenAI and LangchainLLMWrapper
        ChatOpenAI = _get_langchain_chat_openai()
        LangchainLLMWrapper = get_langchain_llm_wrapper_class()

        llm_judge = ChatOpenAI(**chat_params, http_async_client=client)
        return LangchainLLMWrapper(llm_judge)

    def _get_embeddings_client(self):
        """Get the RAGAS embeddings client."""
        if not self._embed_params:
            return None

        # Lazy load NVIDIAEmbeddings and LangchainEmbeddingsWrapper
        NVIDIAEmbeddings = _get_nvidia_embeddings()
        LangchainEmbeddingsWrapper = get_langchain_embeddings_wrapper_class()

        embeddings = NVIDIAEmbeddings(**self._embed_params)
        return LangchainEmbeddingsWrapper(embeddings)

    def _get_run_config(self):
        """Get the RAGAS run configuration."""
        inference_params = {}
        inference = getattr(self, "inference", None)
        if isinstance(inference, BaseModel):
            inference_params = inference.model_dump(exclude_none=True)

        # Lazy load RunConfig
        RunConfig = get_run_config_class()

        return RunConfig(
            timeout=inference_params.get("request_timeout", DEFAULT_JUDGE_TIMEOUT),
            max_retries=inference_params.get("max_retries", DEFAULT_JUDGE_MAX_RETRIES),
            max_workers=inference_params.get("max_workers", DEFAULT_JUDGE_MAX_WORKER),
            max_wait=RAGAS_MAX_WAIT,
            log_tenacity=RAGAS_LOG_TENACITY,
            seed=RAGAS_SEED,
        )

    def _run_evaluate(self, dataset: EvaluationDataset, metrics: list) -> dict[str, float]:
        """Run evaluation with the given dataset and metrics."""
        run_config = self._get_run_config()
        ChatModelCallBackHandler = _get_chat_model_callback_handler_class()
        callback_cls = cast(Any, ChatModelCallBackHandler)
        cb = callback_cls(self._log)

        # Lazy load evaluate function
        evaluate = get_evaluate_function()

        # The evaluate function has a decorator that confuses type checkers.
        # Call it and cast the result to bypass decorator type issues.
        evaluate_fn = cast(Any, evaluate)
        parse_exceptions: tuple[type[BaseException], ...] = (json.JSONDecodeError, ValidationError)
        output_parser_exception = _get_output_parser_exception_type()
        if output_parser_exception is not None:
            parse_exceptions = (*parse_exceptions, output_parser_exception)

        try:
            results = evaluate_fn(
                dataset=dataset,
                metrics=metrics,
                run_config=run_config,
                callbacks=[cb],
                raise_exceptions=True,
            )
        except parse_exceptions as error:
            self._log.warning(
                "RAGAS evaluate failed with parse/output error; returning NaN score",
                extra={"error": str(error), "metric_type": self.type},
            )
            return self._nan_scores_for_metrics(metrics)
        except (httpx.HTTPError, TimeoutError) as error:
            if self._ignore_inference_failure():
                self._log.warning(
                    "RAGAS judge inference failed and is ignored by metric policy; returning NaN score",
                    extra={"error": str(error), "metric_type": self.type},
                )
                return self._nan_scores_for_metrics(metrics)
            raise
        except Exception:
            raise

        scores: dict[str, float] = {}
        for metric_dict in results.scores:
            for metric_name, metric_value in metric_dict.items():
                scores[metric_name] = metric_value

        if not scores:
            self._log.warning(
                "RAGAS evaluation returned no scores; returning NaN score",
                extra={"metric_type": self.type},
            )
            return self._nan_scores_for_metrics(metrics)

        invalid_score_names = _invalid_score_names(scores)
        if invalid_score_names:
            self._log.warning(
                "RAGAS evaluation produced invalid scores; returning NaN score",
                extra={
                    "metric_type": self.type,
                    "invalid_score_names": sorted(invalid_score_names),
                },
            )
            return self._nan_scores_for_metrics(metrics)

        return scores

    def _create_evaluation_dataset(self, item: dict, sample: dict) -> EvaluationDataset:
        """Create an EvaluationDataset from the given item and sample."""
        # Lazy load EvaluationDataset class (use different name to avoid shadowing type annotation)
        EvaluationDatasetCls = get_evaluation_dataset_class()

        template = self.input_template
        response = sample.get("output_text") or sample.get("response")
        payload = {}

        # if template is provided, add response to the payload if it's not already present
        # otherwise, use the item and add response if it's not already present. For Online evaluation,
        # model response supersedes the response in the item.
        if template:
            payload = render_request(template, context={**item, **sample, "item": item, "sample": sample})
            if response and "response" not in payload:
                payload["response"] = response
        else:
            payload = item.copy()
            if response:
                payload["response"] = response

        return EvaluationDatasetCls.from_list([payload])

    def metric(self, item: dict, sample: dict, trace=None) -> float | bool:
        """Compute a single raw score for the metric."""
        if not (self.metric_threshold and self.metric_threshold_score):
            # Access type from the class, not instance
            metric_type = self.type
            type_value = getattr(metric_type, "value", metric_type) or "unknown"
            raise ValueError(
                f"metric_threshold and metric_threshold_score are required to compute metric for {type_value}"
            )
        data = self._create_evaluation_dataset(item, sample)
        llm_judge = self._get_llm_judge()
        scores = self._metric(data, llm_judge)
        score = scores[self.metric_threshold_score]

        if trace:
            return score >= self.metric_threshold

        return score

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute the scores for the metric."""
        return await _run_function_in_plain_loop(self.compute_scores_async, item, sample)

    async def compute_scores_async(self, item: dict, sample: dict) -> MetricResult:
        """Compute the scores for the metric asynchronously."""
        async with httpx.AsyncClient() as client:
            data = self._create_evaluation_dataset(item, sample)
            llm_judge = self._get_llm_judge(client)
            scores = self._metric(data, llm_judge)
            return MetricResult(
                scores=[MetricScore(name=metric_name, value=score_value) for metric_name, score_value in scores.items()]
            )

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        """
        Compute raw scores for the metric. This method must be implemented by subclasses.

        Args:
            data: The evaluation dataset to compute metrics on
            llm_judge: The LLM judge to use for metrics that require it. If None, the metric
                      doesn't use an LLM judge.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _metric method")


async def _run_function_in_plain_loop(fn, *args, **kwargs):
    """
    Run any function inside a dedicated thread with a plain asyncio DefaultEventLoopPolicy (not uvloop).
    Args:
        fn: The function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    """

    def _call():
        # Make a *fresh* loop for this thread and own its lifecycle
        policy = asyncio.DefaultEventLoopPolicy()
        loop = policy.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run_and_cleanup():
            try:
                # Run the provided function
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result

                # Cleanup any pending tasks
                tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
                if tasks:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Cleanup async generators
                await loop.shutdown_asyncgens()

                return result
            except Exception as e:
                log.error(f"Error during function execution: {str(e)}")
                raise

        try:
            return loop.run_until_complete(_run_and_cleanup())
        finally:
            try:
                # Close the loop only if it's still running
                if not loop.is_closed():
                    loop.close()
            except Exception as e:
                log.warning(f"Error while closing event loop: {str(e)}")

    return await asyncio.to_thread(_call)


@cache
def _get_chat_model_callback_handler_class() -> type[BaseCallbackHandler]:
    """Create callback handler class that inherits from BaseCallbackHandler.

    Uses a factory pattern to defer the import of BaseCallbackHandler until first use.
    """
    BaseCallbackHandler = _get_base_callback_handler()

    class ChatModelCallBackHandler(BaseCallbackHandler):
        """A callback handler that logs chat model interactions using thread-safe context variables."""

        def __init__(self, log_list):
            super().__init__()
            self._logger = get_logger()
            # Get the thread-local request log from context
            self.request_log = requests_log_var.get([])
            # Store current request data between callbacks
            self._current_request = None

        def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
            """Stores request data temporarily until completion or error."""
            # Create a new request entry but don't add it to the log yet
            self._current_request = {"request": messages}

        def on_llm_end(self, response, **kwargs) -> None:
            """Creates a complete log entry with both request and response data."""
            if self._current_request is None:
                self._logger.warning("Received response callback without a matching request")
                return

            # Create complete log entry
            log_entry = {
                **self._current_request,
                "response": response,
            }

            # Add the complete entry to the log
            self.request_log.append(log_entry)
            self._current_request = None

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            """Logs error information along with the original request."""
            if self._current_request is None:
                self._logger.warning("Received error callback without a matching request")
                return

            # Create error log entry
            log_entry = {
                **self._current_request,
                "error": str(error),
                "error_type": error.__class__.__name__,
            }

            # Add the error entry to the log
            self.request_log.append(log_entry)
            self._current_request = None

    return ChatModelCallBackHandler


def _invalid_score_names(scores: dict[str, float]) -> list[str]:
    invalid: list[str] = []
    for metric_name, score in scores.items():
        if isinstance(score, bool):
            invalid.append(metric_name)
            continue
        if not isinstance(score, (int, float)):
            invalid.append(metric_name)
            continue
        if not math.isfinite(float(score)):
            invalid.append(metric_name)
    return invalid
