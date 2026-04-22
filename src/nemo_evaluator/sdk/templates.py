"""Template rendering helpers for evaluator SDK runtime."""

# Migrated from: services/evaluator/src/nmp/evaluator/app/templates.py

import json
from typing import Any

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

env = SandboxedEnvironment(undefined=StrictUndefined)

# Preserve native Python values for bare expressions like `{{ item.score }}`.
# Full template rendering always returns strings, which breaks structured payloads.
_EXPR_PREFIX = "{{"
_EXPR_SUFFIX = "}}"


def _is_single_expression_template(template: str) -> bool:
    """Check whether a template is exactly one Jinja expression.

    Args:
        template: Raw template string.

    Returns:
        ``True`` when the string is a single ``{{ ... }}`` expression with no
        statement blocks; otherwise ``False``.
    """
    stripped = template.strip()
    return (
        stripped.startswith(_EXPR_PREFIX)
        and stripped.endswith(_EXPR_SUFFIX)
        and stripped.count(_EXPR_PREFIX) == 1
        and stripped.count(_EXPR_SUFFIX) == 1
        and "{%" not in stripped
        and "%}" not in stripped
    )


def _identifier_kwargs(context: dict) -> dict:
    """Filter context keys that can be passed to ``compile_expression`` kwargs.

    Args:
        context: Template rendering context.

    Returns:
        Dictionary containing only string keys that are valid Python identifiers.
    """
    # compile_expression() only accepts keyword arguments, so keys like
    # `foo-bar` need to stay accessible through `item`/`sample` instead.
    return {k: v for k, v in context.items() if isinstance(k, str) and k.isidentifier()}


def render_template(template: str | dict | list, context: dict) -> Any:
    """Render strings, dicts, or lists using sandboxed Jinja evaluation.

    For bare expression templates (for example ``{{ item.payload }}``), the
    function uses ``compile_expression`` to preserve native value types instead
    of forcing string output.

    Args:
        template: Template payload to render.
        context: Variables available to the Jinja runtime.

    Returns:
        Rendered value preserving dict/list structure and native expression types.

    Raises:
        jinja2.UndefinedError: If the template references a missing variable.
        json.JSONDecodeError: If ``tojson`` output is not valid JSON.
    """
    if isinstance(template, dict):
        return {k: render_template(v, context) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(v, context) for v in template]
    if isinstance(template, str):
        if _is_single_expression_template(template):
            # Use the expression compiler here so `{{ some_dict }}` returns a dict
            # instead of a stringified representation.
            expr = template.strip()[len(_EXPR_PREFIX) : -len(_EXPR_SUFFIX)].strip()
            compiled = env.compile_expression(expr, undefined_to_none=False)
            result = compiled(**_identifier_kwargs(context))
            if isinstance(result, StrictUndefined):
                str(result)
            if "tojson" in template and isinstance(result, str):
                return json.loads(result)
            return result

        rendered_str = env.from_string(template).render(context)
        if "tojson" in template:
            return json.loads(rendered_str)
        return rendered_str
    return template


def render_request(template: str | dict, context: dict) -> dict:
    """Render a request payload and normalize string output into prompt dicts.

    Args:
        template: String or dictionary request template.
        context: Variables available to the Jinja runtime.

    Returns:
        Dictionary request payload. String templates are wrapped as
        ``{"prompt": rendered_text}``.
    """
    request = render_template(template, context=context)
    if isinstance(request, str):
        request = {"prompt": request}
    if isinstance(request, list):
        raise TypeError("Request template must not produce a list output.")
    return request
