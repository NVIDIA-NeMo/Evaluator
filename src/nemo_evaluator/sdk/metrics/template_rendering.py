"""Free-function helpers for rendering metric templates against row data."""

import re
from collections.abc import Mapping
from typing import Any

from jinja2 import UndefinedError
from nemo_evaluator.sdk.templates import render_template
from pydantic import BaseModel

TemplateValue = str | dict[Any, Any] | list[Any]
_DICT_ATTRIBUTE_ERROR_RE = re.compile(r"^'dict object' has no attribute '(?P<name>[^']+)'$")
_UNDEFINED_NAME_ERROR_RE = re.compile(r"^'(?P<name>[^']+)' is undefined$")


def build_template_context(item: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Build the template context shared by item and sample rendering."""
    return {**item, **sample, "item": item, "sample": sample}


def template_metric_repr(metric: BaseModel | object) -> str:
    """Return a compact repr used in row-level template rendering errors."""
    class_name = metric.__class__.__name__
    if not isinstance(metric, BaseModel):
        return class_name

    public_fields = metric.model_dump(
        exclude={"type", "description", "labels", "supported_job_types"},
        exclude_none=False,
    )
    if not isinstance(public_fields, Mapping):
        return class_name

    args = ", ".join(f"{name}={value!r}" for name, value in public_fields.items())
    return f"{class_name}({args})"


def extract_missing_template_key(template: TemplateValue, exc: UndefinedError) -> str | None:
    """Infer the most specific missing key path from a Jinja undefined error."""
    message = exc.message or str(exc)
    missing_leaf: str | None = None
    if match := _DICT_ATTRIBUTE_ERROR_RE.fullmatch(message):
        missing_leaf = match.group("name")
    elif match := _UNDEFINED_NAME_ERROR_RE.fullmatch(message):
        missing_leaf = match.group("name")

    return missing_leaf


def render_template_or_raise(
    *,
    template_name: str,
    template: TemplateValue,
    context: dict[str, Any],
    item: dict[str, Any],
    sample: dict[str, Any],
    metric_repr: str,
    item_keys_label: str = "item",
    sample_keys_label: str = "sample",
) -> object:
    """Render one template and raise a specific validation error on missing keys."""
    try:
        return render_template(template, context)
    except UndefinedError as exc:
        missing_key = extract_missing_template_key(template, exc)
        base_message = (
            f"{metric_repr} could not render its '{template_name}' template for this row.\n"
            f"Available {item_keys_label} keys={sorted(item.keys())}. \n"
            f"Available {sample_keys_label} keys={sorted(sample.keys())}.\n"
        )
        if missing_key is not None:
            detail = f"Dataset item has missing_key='{missing_key}' but the '{template_name}' template references it.\n"
        else:
            detail = f"jinja_error={str(exc)!r}.\n"
        raise ValueError(
            base_message + detail + "Ensure that the dataset provides the fields referenced by the templates."
        ) from exc


def render_default_output_text_candidate_or_raise(*, sample: dict[str, Any], metric_name: str) -> object:
    """Return the default output-text candidate or raise a clear guidance error."""
    prediction = sample.get("output_text")
    if prediction is None:
        raise ValueError(
            f"{metric_name} has missing `candidate` field.\n"
            f"For offline evaluation, `candidate=...` field is required when constructing {metric_name}.\n"
            "For online evaluation, this usually means the evaluated model produced no output."
        )
    return prediction


def render_reference_and_candidate(
    *,
    metric_repr: str,
    metric_name: str,
    reference_template: str,
    candidate_template: str | None,
    item: dict[str, Any],
    sample: dict[str, Any],
) -> tuple[str, str]:
    """Render reference and candidate templates, returning validated strings."""
    context = build_template_context(item, sample)
    ground_truth = render_template_or_raise(
        template_name="reference",
        template=reference_template,
        context=context,
        item=item,
        sample=sample,
        metric_repr=metric_repr,
        item_keys_label="reference",
        sample_keys_label="candidate",
    )
    if candidate_template:
        prediction = render_template_or_raise(
            template_name="candidate",
            template=candidate_template,
            context=context,
            item=item,
            sample=sample,
            metric_repr=metric_repr,
            item_keys_label="reference",
            sample_keys_label="candidate",
        )
    else:
        prediction = render_default_output_text_candidate_or_raise(sample=sample, metric_name=metric_name)

    if not isinstance(ground_truth, str):
        raise TypeError("The reference must be a string.")
    if not isinstance(prediction, str):
        raise TypeError("The candidate must be a string.")
    return ground_truth, prediction
