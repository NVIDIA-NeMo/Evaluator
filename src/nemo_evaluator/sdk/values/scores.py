"""Score configuration types for metric definitions."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal, Self

import jsonschema
from jsonschema import validators
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

from nemo_evaluator.sdk.values.results import MetricScore, RubricScoreStat, ScoreStats

_logger = logging.getLogger(__name__)


class JSONScoreParser(BaseModel):
    """Parse a score from JSON structured content."""

    type: Literal["json"] = "json"
    json_path: str = Field(
        description="The JSON path to parse the score from the judge response when using structured output."
    )


class RegexScoreParser(BaseModel):
    """Parse a score from content in any format using regular expression."""

    type: Literal["regex"] = "regex"
    pattern: str = Field(description="The regular expression to parse the score from the judge response.")
    method: Literal["search", "match"] = Field(
        default="match",
        description="The regex method to use: 'search' looks anywhere in the string, 'match' starts from the beginning.",
    )

    @model_validator(mode="after")
    def valid_regex(self) -> Self:
        if self.type != "regex":
            # model_validator will run when resolving parser JSONScoreParser | RegexScoreParser
            # if it's RegexScoreParser, continue with validation.
            return self

        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"invalid regex for score parser: {e}")
        return self


class _Score(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        pattern=r"^[a-z0-9_]+$",
        description="The name of the score. Only lowercase letters, numbers, and underscores allowed.",
    )
    description: str | None = Field(default=None, description="Human-readable description of the score.")
    parser: JSONScoreParser | RegexScoreParser = Field(
        default_factory=lambda data: JSONScoreParser(json_path=data["name"]),
        description="The method to parse the score. When used with llm-judge metric, and no parser is set, JSONScoreParser is the default parser inferred from the score parameters.",
    )


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(
        description='The label to use for the level of the rubric grading criteria. (e.g., "helpful", "not_helpful", "positive")'
    )
    description: str | None = Field(
        default=None,
        description="Describe the semantic meaning of each criteria for the given rubric. If no judge template is set, the input description for labels are included in the generated judge prompt.",
    )
    value: float | int = Field(
        description="The score value to assign for the criteria used for aggregation and ranking."
    )


class RubricScore(_Score):
    """Score definition for a rubric with optional parser. If no parser is set, JSONScoreParser is the default parser inferred from the score parameters"""

    model_config = ConfigDict(extra="forbid")
    rubric: list[Rubric] = Field(min_length=2, description="The rubric for the score.")


class RangeScore(_Score):
    """Score definition for a range of values with optional parser. If no parser is set, JSONScoreParser is the default parser inferred from the score parameters"""

    model_config = ConfigDict(extra="forbid")
    minimum: float | int = Field(description="Minimum value for the score range. Must be less than maximum.")
    maximum: float | int = Field(description="Maximum value for the score range. Must be greater than minimum.")

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.minimum >= self.maximum:
            raise ValueError(f"minimum must be less than maximum: {self.minimum} < {self.maximum}")
        return self


class RemoteScore(_Score):
    """Score configuration for remote metrics.

    Unlike RangeScore, minimum and maximum are optional (default to None = no bounds).
    This avoids JSON serialization issues with infinity values.
    """

    minimum: float | int | None = Field(
        default=None,
        description="Minimum value for the score range. Defaults to None (no lower bound).",
    )
    maximum: float | int | None = Field(
        default=None,
        description="Maximum value for the score range. Defaults to None (no upper bound).",
    )
    parser: JSONScoreParser = Field(
        default_factory=lambda data: JSONScoreParser(json_path=data["name"]),
        description="The method to parse the score. Only JSON parsing is supported for remote metrics.",
    )

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        """Validate that minimum < maximum when both are configured."""
        if self.minimum is not None and self.maximum is not None:
            if self.minimum >= self.maximum:
                raise ValueError(f"minimum must be less than maximum: {self.minimum} >= {self.maximum}")
        return self


def score_discriminator(data: dict[str, Any] | RubricScore | RangeScore) -> Literal["rubric", "range"]:
    if "rubric" in data or isinstance(data, RubricScore):
        return "rubric"
    return "range"


Score = Annotated[
    (Annotated[RubricScore, Tag("rubric")] | Annotated[RangeScore, Tag("range")]), Discriminator(score_discriminator)
]


class ScoreParser(ABC):
    """Parse model output text into a normalized ``MetricScore``."""

    class Params(BaseModel):
        pass

    rubric_mapping: dict[str, Rubric] | None = None

    def __init__(self, score: Score):
        self.score = score
        if isinstance(score, RubricScore):
            self.rubric_mapping = {rubric.label: rubric for rubric in score.rubric}

    @abstractmethod
    def parse(self, text: str | None) -> MetricScore:
        """Parse the provided text and extract a single score."""
        raise NotImplementedError

    def _get_rubric_score(self, label: str) -> MetricScore:
        """Map rubric label to score value and build rubric distribution stats."""
        assert isinstance(self.score, RubricScore)
        assert self.rubric_mapping is not None
        rubric_distribution = [
            RubricScoreStat(
                label=rubric.label, description=rubric.description, value=rubric.value, count=int(label == rubric.label)
            )
            for rubric in self.score.rubric
        ]
        rubric = self.rubric_mapping.get(label)
        score_value = float("nan") if rubric is None else rubric.value
        return MetricScore(
            name=self.score.name, value=score_value, stats=ScoreStats(rubric_distribution=rubric_distribution)
        )


class ScoreParserRegex(ScoreParser):
    """Parse score values from free-form text using a regex capture group."""

    parser_type: Literal["regex"] = "regex"
    pattern: re.Pattern
    method: Literal["search", "match"]

    def __init__(self, score: Score):
        super().__init__(score)
        if not isinstance(score.parser, RegexScoreParser):
            raise ValueError(f"incompatible score parser to initialize ScoreParserRegex: {type(score.parser)}")
        try:
            self.pattern = re.compile(score.parser.pattern)
        except re.error as e:
            raise ValueError(f"invalid regex pattern for score parser with LLM-as-a-Judge: {score.parser.pattern} {e}")
        self.method = score.parser.method

    def parse(self, text: str | None) -> MetricScore:
        if not text:
            return MetricScore(name=self.score.name, value=float("nan"))

        match = self.pattern.search(text) if self.method == "search" else self.pattern.match(text)
        if match is None:
            return MetricScore(name=self.score.name, value=float("nan"))

        groups = match.groups()
        if len(groups) == 0:
            return MetricScore(name=self.score.name, value=float("nan"))

        try:
            if self.rubric_mapping:
                return self._get_rubric_score(groups[0])
            return MetricScore(name=self.score.name, value=float(groups[0]))
        except ValueError:
            # This is expected when models drift from requested output format.
            _logger.info("Failed to parse score from text: %s.", text)
            return MetricScore(name=self.score.name, value=float("nan"))


class ScoreParserJSON(ScoreParser):
    """Parse score values from JSON output using ``json_path`` key lookup."""

    parser_type: Literal["json"] = "json"
    json_path: str

    # Optional when structured output is defined.
    structured_output: dict | None = None
    json_schema: dict | None = None
    json_validator: jsonschema.Validator | None = None

    def __init__(self, score: Score, structured_output: dict | None = None):
        super().__init__(score)
        if not isinstance(score.parser, JSONScoreParser):
            raise ValueError(f"incompatible score parser to initialize ScoreParserJSON: {type(score.parser)}")

        self.json_path = score.parser.json_path

        if structured_output:
            self._validate_structured_output(score, structured_output)
            self.structured_output = structured_output
            self.json_schema = structured_output["schema"]

    def _validate_structured_output(self, score: Score, structured_output: dict) -> None:
        json_schema = structured_output.get("schema")
        if not json_schema:
            raise ValueError("missing schema for structured output")

        # Validate schema itself and verify parser expectations for json_path.
        validator = validators.validator_for(json_schema)
        validator.check_schema(json_schema)

        schema_type = json_schema.get("type", "")
        if schema_type != "object":
            raise ValueError(f"schema must be type 'object' for JSON score parser: {schema_type}")

        assert score.parser is not None
        assert isinstance(score.parser, JSONScoreParser)
        assert score.parser.json_path is not None

        schema_defined_json_path = json_schema.get("properties", {}).get(score.parser.json_path)
        if not schema_defined_json_path:
            raise ValueError(
                f"schema must have {score.parser.json_path} defined as an object property for JSON score parser: {json_schema}"
            )
        json_type = schema_defined_json_path.get("type")
        if isinstance(score, RubricScore):
            if json_type != "string":
                raise ValueError(
                    f"expected string type in schema for property '{score.parser.json_path}' when used with score rubric and JSON score parser: {json_type}"
                )
        elif json_type not in ["number", "integer", "boolean"]:
            raise ValueError(
                f"schema property {score.parser.json_path} must be type number, integer, or boolean for JSON score parser: {schema_defined_json_path}"
            )

    def parse(self, text: str | None) -> MetricScore:
        if not isinstance(text, str):
            return MetricScore(name=self.score.name, value=float("nan"))
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return MetricScore(name=self.score.name, value=float("nan"))

        score = obj.get(self.json_path)
        if score is None:
            return MetricScore(name=self.score.name, value=float("nan"))

        if self.rubric_mapping:
            return self._get_rubric_score(score)

        if isinstance(score, str):
            return MetricScore(name=self.score.name, value=float("nan"))

        if isinstance(score, bool):
            score = 1.0 if score else 0.0

        return MetricScore(name=self.score.name, value=score)
