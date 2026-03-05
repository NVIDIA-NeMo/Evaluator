from nemo_evaluator.scoring import exact_match, extract_answer, math_equal


class TestExtractAnswer:
    def test_boxed_latex(self):
        assert extract_answer("So \\boxed{42}") == "42"

    def test_boxed_fraction(self):
        assert extract_answer("Step 1: ... Step 2: ... \\boxed{7/3}") == "7/3"

    def test_answer_is_pattern(self):
        assert extract_answer("Therefore, the answer is 42.") == "42"

    def test_pure_numeric_last_line(self):
        assert extract_answer("Some reasoning\n42") == "42"

    def test_non_numeric_line_falls_back_to_last_line(self):
        assert extract_answer("Step 1: 20\nFinal: 42") == "Final: 42"

    def test_empty_input(self):
        assert extract_answer("") == ""

    def test_boxed_takes_priority_over_answer_is(self):
        assert extract_answer("The answer is 99. \\boxed{42}") == "42"


class TestMathEqual:
    def test_integer_vs_float(self):
        assert math_equal("42", "42.0") is True

    def test_different_values(self):
        assert math_equal("42", "43") is False

    def test_whitespace_ignored(self):
        assert math_equal(" 42 ", "42") is True

    def test_close_floats_match(self):
        assert math_equal("0.333333", "0.333333") is True

    def test_non_close_floats_differ(self):
        assert math_equal("0.33", "0.34") is False


class TestExactMatch:
    def test_case_insensitive(self):
        assert exact_match("PARIS", "paris") is True

    def test_whitespace_normalized(self):
        assert exact_match("  Paris  ", "Paris") is True

    def test_punctuation_stripped(self):
        assert exact_match("Paris.", "Paris") is True

    def test_different_strings(self):
        assert exact_match("London", "Paris") is False

    def test_articles_stripped(self):
        assert exact_match("the Eiffel Tower", "Eiffel Tower") is True
