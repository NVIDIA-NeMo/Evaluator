"""Tests for the async-from-sync bridge."""
import asyncio
import pytest
from nemo_evaluator.runner.async_bridge import run_async


async def _double(x: int) -> int:
    await asyncio.sleep(0.01)
    return x * 2


class TestRunAsync:
    def test_from_sync_context(self):
        result = run_async(_double(21))
        assert result == 42

    def test_multiple_calls(self):
        results = [run_async(_double(i)) for i in range(5)]
        assert results == [0, 2, 4, 6, 8]

    def test_exception_propagates(self):
        async def _fail():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_async(_fail())
