"""Shared execution parameter types for evaluator SDK and service runtimes."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_evaluator.sdk.values.models import ReasoningParams


# Copy from nmp_common.inference.schemas.InferenceParams to avoid nmp.common dependency
class InferenceParams(BaseModel):
    """
    Parameters for model inference. Extra fields can be supplied for additional options applied to the inference request directly. Fields not supported by the model may cause inference errors during evaluation.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, description="Model identifier")
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Float value between 0 and 1. temp of 0 indicates greedy decoding, "
        "where the token with highest prob is chosen. Temperature can't be set to 0.0 currently",
    )
    max_tokens: int | None = Field(default=None, ge=1, description="Max tokens to generate")
    max_completion_tokens: int | None = Field(default=None, ge=1, description="Max tokens to generate")
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Float value between 0 and 1; limits to the top tokens within a certain "
        "probability. top_p=0 means the model will only consider the single most likely "
        "token for the next prediction",
    )
    stop: list[str] | None = Field(default=None)

    @model_validator(mode="after")
    def check_max_tokens(self) -> Self:
        if self.max_tokens and self.max_completion_tokens:
            raise ValueError(
                "max_tokens and max_completion_tokens cannot both be configured. "
                "Choose the appropriate tokens parameter for the model."
            )
        return self


class EvaluationJobParams(BaseModel):
    """Runtime execution parameters for evaluation jobs."""

    model_config = ConfigDict(extra="forbid")

    inference: InferenceParams | None = Field(
        default=None,
        description="Custom settings that control the model's text generation behavior.",
    )
    ignore_inference_failure: bool = Field(
        default=False,
        description="Whether inference failures should be converted into NaN evaluation results.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Initial instructions that define the model's role and behavior for the conversation.",
    )
    reasoning: ReasoningParams | None = Field(
        default=None,
        description="Custom settings that control the model's reasoning behavior.",
    )
    structured_output: dict | None = Field(
        default=None,
        description="JSON schema to apply structured output for the model.",
    )
    parallelism: int = Field(
        default=8,
        description="Maximum number of concurrent generation requests.",
    )
    request_timeout: int | None = Field(
        default=None,
        description="Timeout, in seconds, for requests made to the model.",
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests.")
    limit_samples: int | None = Field(
        default=None,
        description="Optional maximum number of evaluation samples to process.",
    )

    @field_validator("inference", mode="before")
    @classmethod
    def coerce_inference_params(cls, value: object) -> object:
        """Normalize equivalent Pydantic inference models before SDK validation runs."""
        # Compatibility shim for service-side inference models. This converts other
        # Pydantic models into plain data and lets SDK InferenceParams validation
        # decide whether the payload is compatible.
        if isinstance(value, BaseModel) and not isinstance(value, InferenceParams):
            return value.model_dump(mode="python", exclude_none=True)
        return value
