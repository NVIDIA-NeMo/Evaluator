from jinja2 import UndefinedError
from nemo_evaluator.sdk.scoring.template_rendering import extract_missing_template_key, template_metric_repr


class _NoModelDumpMetric:
    """Metric-like object without a Pydantic model_dump implementation."""


class _NonMappingModelDumpMetric:
    """Metric-like object whose model_dump returns a non-mapping value."""

    def model_dump(self, **_: object) -> str:
        return "not-a-mapping"


class TestTemplateRendering:
    def test_template_metric_repr_without_model_dump_falls_back_to_class_name(self):
        metric = _NoModelDumpMetric()
        assert template_metric_repr(metric) == "_NoModelDumpMetric"

    def test_template_metric_repr_with_non_mapping_model_dump_falls_back_to_class_name(self):
        metric = _NonMappingModelDumpMetric()
        assert template_metric_repr(metric) == "_NonMappingModelDumpMetric"

    def test_extract_missing_template_key_detects_bracket_item_access(self):
        error = UndefinedError("'dict object' has no attribute 'answer'")
        assert extract_missing_template_key("{{ item['answer'] }}", error) == "answer"

    def test_extract_missing_template_key_detects_bracket_sample_access(self):
        error = UndefinedError("'dict object' has no attribute 'output_text'")
        assert extract_missing_template_key('{{ sample["output_text"] }}', error) == "output_text"
