"""Hook capability mixin for V2 SDK metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import nemo_evaluator.sdk.inference as inference
from pydantic import BaseModel, PrivateAttr


class HooksBase(BaseModel):
    """Reusable hook state and helpers for V2 metrics."""

    _preprocess_hooks: list[inference.PreprocessRequest] = PrivateAttr(default_factory=list)
    _postprocess_hooks: list[inference.PostprocessResponse] = PrivateAttr(default_factory=list)

    def with_hooks(
        self,
        *,
        preprocess: Sequence[inference.PreprocessRequest] | None = None,
        postprocess: Sequence[inference.PostprocessResponse] | None = None,
    ) -> Self:
        """Attach preprocess and postprocess hooks to this metric instance."""
        self._preprocess_hooks = list(preprocess or [])
        self._postprocess_hooks = list(postprocess or [])
        return self

    def _apply_preprocess_hooks(self, request: dict, *, id: str | None = None) -> dict:
        """Apply preprocess hooks in order or return the request unchanged."""
        return inference.preprocess_request(request, hooks=self._preprocess_hooks, id=id)

    def _apply_postprocess_hooks(self, response: dict, *, id: str | None = None) -> dict:
        """Apply postprocess hooks in order or return the response unchanged."""
        processed = response
        for hook in self._postprocess_hooks:
            processed = hook.postprocess(processed, id=id)
        return processed
