"""Shared helpers for working with runtime metric identifiers."""

from nemo_evaluator.sdk.metrics.base import Metric


def metric_type_name(metric: Metric) -> str:
    """Resolve a stable public type name for one runtime metric.

    Args:
        metric: Metric object used during execution or optimization.

    Returns:
        ``metric.type.value`` for enum-like types when available, otherwise the
        plain-string metric type, otherwise the metric class name.

    This helper exists for generic call sites that operate on the runtime
    ``Metric`` protocol and must support all documented ``Metric.type`` shapes without depending on
    enum-only APIs:

    - built-in ``MetricType`` members
    - plain string custom metric types
    - custom ``str``-backed enums

    Examples:
        Built-in metrics still commonly expose ``MetricType`` members, so a
        BLEU runtime metric resolves to ``"bleu"`` via ``metric.type.value``.

        Custom metrics may expose ``type`` as a plain string such as
        ``"my-custom-metric"``, which is returned directly.

        Custom metrics may also use their own ``str``-backed enum, for example
        ``class CustomMetricType(str, Enum): FOO = "foo"``; those also resolve
        through ``metric.type.value`` and return ``"foo"``.
    """
    metric_type = getattr(metric, "type", None)
    metric_value = getattr(metric_type, "value", None)
    if isinstance(metric_value, str):
        return metric_value
    if isinstance(metric_type, str):
        return metric_type
    return metric.__class__.__name__
