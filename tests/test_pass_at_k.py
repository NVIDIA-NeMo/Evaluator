import pytest

from nemo_evaluator.metrics.pass_at_k import aggregate_pass_at_k, pass_at_k


class TestPassAtK:
    def test_all_correct(self):
        assert pass_at_k(n=5, c=5, k=1) == 1.0

    def test_none_correct(self):
        assert pass_at_k(n=5, c=0, k=1) == 0.0

    def test_one_of_five_k1(self):
        result = pass_at_k(n=5, c=1, k=1)
        assert abs(result - 0.2) < 1e-9  # 1 - C(4,1)/C(5,1) = 1/5

    def test_one_of_five_k5(self):
        assert pass_at_k(n=5, c=1, k=5) == 1.0

    def test_rejects_k_greater_than_n(self):
        with pytest.raises(ValueError, match="must be >= k"):
            pass_at_k(n=3, c=2, k=5)

    def test_rejects_c_greater_than_n(self):
        with pytest.raises(ValueError, match="must be in"):
            pass_at_k(n=3, c=5, k=1)

    def test_monotone_in_k(self):
        scores = [pass_at_k(n=10, c=3, k=k) for k in [1, 2, 4, 8, 10]]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]


class TestAggregatePassAtK:
    def test_single_problem(self):
        result = aggregate_pass_at_k([(10, 5)], k=1)
        assert result == pass_at_k(10, 5, 1)

    def test_multiple_problems_averages(self):
        problems = [(10, 5), (10, 3), (10, 0)]
        result = aggregate_pass_at_k(problems, k=1)
        expected = sum(pass_at_k(n, c, 1) for n, c in problems) / 3
        assert abs(result - expected) < 1e-10

    def test_empty(self):
        assert aggregate_pass_at_k([], k=1) == 0.0
