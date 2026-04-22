from nemo_evaluator.sdk.metrics.exact_match import ExactMatchMetric


class TestExactMatchMetric:
    def test_score_names(self):
        metric = ExactMatchMetric(reference="{{item.reference}}")
        assert metric.score_names() == ["exact-match"]

    def test_metric_returns_bool_when_trace_is_enabled(self):
        metric = ExactMatchMetric(reference="{{item.reference}}")
        assert metric.metric({"reference": "The Cat"}, {"output_text": "the cat"}, trace=True) is True
