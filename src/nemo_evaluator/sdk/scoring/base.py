"""Optional metric capabilities and shared helpers for evaluator SDK runtime."""

# Migrated from: services/evaluator/src/nmp/evaluator/app/metrics/base.py

import re
import string
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from nemo_evaluator.scoring.types import ScorerInput
from nemo_evaluator.sdk.values import SecretRef
from nemo_evaluator.sdk.values.results import MetricResult

SecretResolver = Callable[[str], Awaitable[str | None]]


class NELScorerMixin:
    """Makes a SDK metric callable as a NEL scorer: ``__call__(ScorerInput) -> dict``.

    NEL's scoring pipeline expects scorers to be callables with signature
    ``(ScorerInput) -> dict``. SDK metrics natively use a different shape
    (``metric(item, sample) -> float | bool``). This mixin bridges the two so
    a SDK metric can be used wherever NEL accepts a scorer, without an
    external adapter.

    Concrete metrics inheriting this mixin must implement ``type`` and
    ``metric(item, sample, trace=None)`` (both already required by the
    :class:`Metric` protocol).
    """

    def __call__(self, scorer_input: ScorerInput) -> dict[str, Any]:
        item: dict[str, Any] = {"reference": scorer_input.target, **(scorer_input.metadata or {})}
        sample: dict[str, Any] = {"output_text": scorer_input.response, "response": scorer_input.response}
        score = self.metric(item, sample)  # type: ignore[attr-defined]
        score_value = float(score) if isinstance(score, (bool, int, float)) else 0.0
        return {
            "score": score_value,
            "metric_type": getattr(self, "type", self.__class__.__name__),
        }


@runtime_checkable
class Metric(Protocol):
    """Structural contract for SDK runtime metrics used by generic evaluator code.

    This protocol describes what execution and orchestration code may rely on
    when working with a metric instance. In particular, ``type`` is treated as
    the public string metric identifier, even though concrete metric classes may
    implement it with:

    - a built-in ``MetricType`` member
    - a plain string such as ``"my-custom-metric"``
    - a custom ``str``-backed enum

    Generic consumers must therefore treat ``type`` as a string identifier and
    must not depend on enum-only APIs such as ``.value``. Callers that need to
    normalize supported runtime shapes should use ``metric_type_name(...)``.
    """

    @property
    def type(self) -> str:
        """Return the public metric key/type identifier.

        Examples:
            Built-in runtime metrics may expose ``MetricType.BLEU``.

            Custom metrics may expose a plain string such as
            ``"my-custom-metric"``.

            Custom metrics may also expose a custom ``str``-backed enum such as
            ``CustomMetricType.FOO`` where ``CustomMetricType(str, Enum)``.
        """
        ...

    def metric(self, item: dict, sample: dict, trace: Any = None) -> float | bool:
        """Compute one raw score value for an item/sample pair."""
        ...

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute structured score output for one item/sample pair."""
        ...

    def score_names(self) -> list[str]:
        """Return canonical score names emitted by this metric."""
        ...


@runtime_checkable
class CorpusMetric(Protocol):
    """Protocol for metrics that also emit corpus-level scores."""

    async def compute_corpus_scores(self, items: list[dict], samples: list[dict]) -> MetricResult | None:
        """Compute corpus-level scores across all evaluated rows.

        Args:
            items: Original dataset rows.
            samples: Sample payloads paired to ``items``.

        Returns:
            Optional corpus-level metric result.
        """
        ...


@runtime_checkable
class MetricWithSecrets(Protocol):
    """Protocol for metrics that require secrets (e.g., API keys)."""

    def secrets(self) -> dict[str, SecretRef]:
        """
        Returns a dictionary of environment variables to the secret reference.
        Used by the job flow to set up environment variables.
        """
        ...

    async def resolve_secrets(self, secret_resolver: SecretResolver) -> None:
        """
        Resolve secrets using the provided resolver function.
        Called before the metric is used for evaluation.
        """
        ...


@runtime_checkable
class MetricWithPreflight(Protocol):
    """Protocol for metrics that need one-time setup before parallel evaluation starts."""

    async def preflight(self) -> None:
        """Run one-time preflight (e.g., capability detection) before processing rows."""
        ...


# TODO: migrate the rest of the protocols from services/evaluator/src/nmp/evaluator/app/metrics/base.py


def normalize_text(s: str) -> str:
    """Normalize free-form text for token/equality-based metric comparisons."""
    if not s:
        return ""
    # lower case
    s = s.lower()
    # remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # collapse whitespace
    s = " ".join(s.split())
    return s
