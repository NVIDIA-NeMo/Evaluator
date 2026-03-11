"""Shared Gym protocol response parsing."""
from __future__ import annotations

from typing import Any


def extract_assistant_text(response: Any) -> str:
    """Extract plain text from various response formats.

    Handles: plain strings, NeMoGymResponse (output[].content[].text),
    and OpenAI ChatCompletion (choices[].message.content).
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        texts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                if isinstance(content, list):
                    texts.extend(c["text"] for c in content if isinstance(c, dict) and "text" in c)
                elif isinstance(content, str):
                    texts.append(content)
        if texts:
            return "\n".join(texts).strip()
        for ch in response.get("choices", []):
            msg = ch.get("message", {}).get("content")
            if msg:
                return msg
    return str(response)
