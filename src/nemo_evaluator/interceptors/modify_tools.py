"""Interceptor that modifies tool schemas in LLM requests.

Supports stripping properties from and adding properties to the
``parameters.properties`` dict of every tool schema.  Useful for removing
framework-injected fields that confuse certain models or for injecting
extra parameters the model should fill in.

Configuration (via YAML)::

    interceptors:
      - modify_tools:
          strip_properties: [security_risk]
          add_properties:
            priority:
              type: string
              enum: [low, medium, high]
              description: "Task priority"
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


def _apply_modifications(
    tools: list[dict[str, Any]],
    strip: frozenset[str],
    add: dict[str, dict[str, Any]],
) -> int:
    """Modify tool parameter schemas in-place.  Returns count of changes."""
    count = 0
    for tool in tools:
        fn = tool.get("function") or tool
        params = fn.get("parameters") or {}
        props = params.get("properties")
        if not isinstance(props, dict):
            continue

        req: list[str] | None = params.get("required")

        for field in strip:
            if field in props:
                del props[field]
                count += 1
            if isinstance(req, list) and field in req:
                req.remove(field)

        for field, schema in add.items():
            if field not in props:
                props[field] = copy.deepcopy(schema)
                count += 1
    return count


class Interceptor(CustomLogger):
    """Adds / removes properties from tool schemas on every LLM request.

    Constructor kwargs (from YAML config):

    * ``strip_properties`` — list of property names to remove
    * ``add_properties`` — dict mapping property name to its JSON Schema
    """

    def __init__(
        self,
        *,
        strip_properties: list[str] | None = None,
        add_properties: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self._strip = frozenset(strip_properties or [])
        self._add: dict[str, dict[str, Any]] = add_properties or {}
        self._logged_once = False

        if not self._strip and not self._add:
            logger.warning("modify_tools: no strip_properties or add_properties configured — interceptor is a no-op")

    def _log_change(self, n: int, model: str) -> None:
        if n and not self._logged_once:
            logger.info(
                "modify_tools: applied %d change(s) to tool schemas "
                "(model=%s, strip=%s, add=%s)",
                n, model, sorted(self._strip), sorted(self._add),
            )
            self._logged_once = True

    # -- proxy pre-call hook (modifies request before routing) -------------

    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        tools = data.get("tools")
        if tools:
            n = _apply_modifications(tools, self._strip, self._add)
            self._log_change(n, data.get("model", "?"))
        return data

    # -- litellm pre-api-call hook (fallback, mutates in place) ------------

    async def async_log_pre_api_call(
        self,
        model: str,
        messages: Any,
        kwargs: dict[str, Any],
    ) -> None:
        tools = kwargs.get("tools")
        if tools:
            n = _apply_modifications(tools, self._strip, self._add)
            self._log_change(n, model)
