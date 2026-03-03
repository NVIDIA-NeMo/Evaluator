from types import SimpleNamespace

import pytest

from nemo_evaluator.api.api_dataclasses import Evaluation
from nemo_evaluator.api.capabilities import BENCHMARK_CAPABILITIES
from nemo_evaluator.core.input import verify_capabilities
from nemo_evaluator.core.utils import MisconfigurationError


@pytest.mark.parametrize("capability", BENCHMARK_CAPABILITIES.keys())
def test_capability_is_valid(capability):
    assert "model" not in BENCHMARK_CAPABILITIES[capability].payload


@pytest.mark.parametrize("capability", BENCHMARK_CAPABILITIES.keys())
def test_verify_capabilities(capability, monkeypatch):
    url = "https://dummy-url.com"
    model_id = "dummy-model-id"

    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            text="Dummy response.",
        ),
    )

    passed = verify_capabilities(
        Evaluation(
            command="dummy-command",
            framework_name="dummy-framework",
            pkg_name="dummy_pkg",
            config={"required_capabilities": [capability]},
            target={"api_endpoint": {"url": url, "model_id": model_id}},
        )
    )
    assert passed


@pytest.mark.parametrize("capability", BENCHMARK_CAPABILITIES.keys())
def test_verify_capabilities_not_supported(capability, monkeypatch):
    url = "https://dummy-url.com"
    model_id = "dummy-model-id"

    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=500,
            text="Internal server error.",
        ),
    )

    with pytest.raises(MisconfigurationError):
        verify_capabilities(
            Evaluation(
                command="dummy-command",
                framework_name="dummy-framework",
                pkg_name="dummy_pkg",
                config={"required_capabilities": [capability]},
                target={"api_endpoint": {"url": url, "model_id": model_id}},
            )
        )
