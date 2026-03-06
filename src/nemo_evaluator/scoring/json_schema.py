"""JSON schema validation scoring for structured output evaluation.

Validates model responses against expected JSON schemas. Useful for
function-calling benchmarks (BFCL), structured output tasks, and
API response validation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def extract_json(text: str) -> Any | None:
    """Extract JSON from model output, handling markdown code blocks."""
    block = _JSON_BLOCK_RE.search(text)
    if block:
        text = block.group(1).strip()

    text = text.strip()
    for start, end in [("{", "}"), ("[", "]")]:
        si = text.find(start)
        ei = text.rfind(end)
        if si != -1 and ei != -1 and ei > si:
            try:
                return json.loads(text[si:ei + 1])
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_json_schema(response: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Validate model response against a JSON schema.

    Returns a dict with:
    - valid: bool
    - extracted: the parsed JSON (or None)
    - errors: list of validation error messages
    - score: 1.0 if valid, 0.0 otherwise
    """
    extracted = extract_json(response)
    if extracted is None:
        return {
            "valid": False,
            "extracted": None,
            "errors": ["Could not extract valid JSON from response"],
            "score": 0.0,
        }

    errors = _validate(extracted, schema)
    return {
        "valid": len(errors) == 0,
        "extracted": extracted,
        "errors": errors,
        "score": 1.0 if not errors else 0.0,
    }


def _validate(data: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Basic JSON schema validation (type, required, properties, items)."""
    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type:
        type_map = {
            "object": dict, "array": list, "string": str,
            "number": (int, float), "integer": int, "boolean": bool,
            "null": type(None),
        }
        expected = type_map.get(expected_type)
        if expected and not isinstance(data, expected):
            errors.append(f"{path}: expected {expected_type}, got {type(data).__name__}")
            return errors

    if expected_type == "object" and isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}: missing required field '{req}'")

        props = schema.get("properties", {})
        for key, prop_schema in props.items():
            if key in data:
                errors.extend(_validate(data[key], prop_schema, f"{path}.{key}"))

    if expected_type == "array" and isinstance(data, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                errors.extend(_validate(item, items_schema, f"{path}[{i}]"))

        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(data) < min_items:
            errors.append(f"{path}: expected at least {min_items} items, got {len(data)}")
        if max_items is not None and len(data) > max_items:
            errors.append(f"{path}: expected at most {max_items} items, got {len(data)}")

    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: value {data!r} not in enum {schema['enum']}")

    return errors
