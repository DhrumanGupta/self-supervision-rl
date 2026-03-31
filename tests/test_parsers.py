from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.parsers import (
    extract_final_answer,
    extract_last_boxed_answer,
    has_valid_think_format,
    math_answers_equal,
    normalize_math_answer,
)


class ExtractLastBoxedAnswerTests(unittest.TestCase):
    def test_extracts_simple_boxed_answer(self) -> None:
        self.assertEqual(extract_last_boxed_answer(r"\boxed{x}"), "x")

    def test_extracts_nested_latex_expression(self) -> None:
        text = r"Yes. So final answer. \[ \boxed{\frac{h^2}{m}} \]"
        self.assertEqual(extract_last_boxed_answer(text), r"\frac{h^2}{m}")

    def test_extracts_boxed_answer_with_display_math_across_lines(self) -> None:
        text = r"""Yes. So final answer.
\[
some context before \boxed{\frac{h^2}{m}} and after
\]
"""
        self.assertEqual(extract_last_boxed_answer(text), r"\frac{h^2}{m}")

    def test_returns_last_valid_boxed_answer(self) -> None:
        text = r"\boxed{1} then finally \boxed{\sqrt{a^2+b^2}}"
        self.assertEqual(extract_last_boxed_answer(text), r"\sqrt{a^2+b^2}")

    def test_does_not_salvage_nested_boxed_answer_inside_unclosed_box(self) -> None:
        text = r"\boxed{\frac{1}{2} then later \boxed{7}"
        self.assertEqual(extract_last_boxed_answer(text), "")

    def test_returns_empty_string_for_unclosed_boxed_answer(self) -> None:
        self.assertEqual(extract_last_boxed_answer(r"\boxed{\frac{1}{2}"), "")


class ExtractFinalAnswerTests(unittest.TestCase):
    def test_extracts_boxed_answer_after_think_block(self) -> None:
        completion = r"<think>work here</think> Yes. So final answer. \[ \boxed{\frac{h^2}{m}} \]"
        self.assertEqual(extract_final_answer(completion), r"\frac{h^2}{m}")

    def test_returns_empty_string_when_no_boxed_answer_exists(self) -> None:
        completion = r"<think>work here</think> Final answer: \frac{h^2}{m}"
        self.assertEqual(extract_final_answer(completion), "")


class ThinkFormatTests(unittest.TestCase):
    def test_accepts_reasoning_and_boxed_answer_after_close_tag(self) -> None:
        self.assertTrue(
            has_valid_think_format(
                "question <think>",
                r"work here</think> \[ \boxed{\frac{h^2}{m}} \]",
            )
        )

    def test_rejects_empty_reasoning_before_close_tag(self) -> None:
        self.assertFalse(
            has_valid_think_format(
                "question <think>",
                r"</think> \[ \boxed{\frac{h^2}{m}} \]",
            )
        )

    def test_rejects_missing_boxed_answer_after_close_tag(self) -> None:
        self.assertFalse(
            has_valid_think_format(
                "question <think>",
                r"work here</think> Final answer: \frac{h^2}{m}",
            )
        )

    def test_rejects_multiple_close_tags(self) -> None:
        self.assertFalse(
            has_valid_think_format(
                "question <think>",
                r"work here</think> \[ \boxed{1} \]</think>",
            )
        )


class NormalizeMathAnswerTests(unittest.TestCase):
    def test_normalizes_deepseek_style_fraction_shorthand(self) -> None:
        self.assertEqual(normalize_math_answer(r"\frac12"), r"\frac{1}{2}")

    def test_normalizes_display_wrappers_and_formatting_commands(self) -> None:
        text = r"\[ \left( \mathbf{0.0} \right) \]"
        self.assertEqual(normalize_math_answer(text), "(0)")


class MathAnswersEqualTests(unittest.TestCase):
    def test_matches_function_style_assignment_to_bare_answer(self) -> None:
        self.assertTrue(math_answers_equal("f(x) = 0", "0"))

    def test_matches_short_assignment_to_bare_answer(self) -> None:
        self.assertTrue(math_answers_equal("x = \\frac{1}{2}", r"\frac{1}{2}"))

    def test_does_not_match_real_equation_to_rhs(self) -> None:
        self.assertFalse(math_answers_equal("x + 1 = 2", "2"))

    def test_matches_numeric_equivalence(self) -> None:
        self.assertTrue(math_answers_equal("2.000", "2"))

    def test_matches_percentage_flexibility(self) -> None:
        self.assertTrue(math_answers_equal("0.5", "50%"))

    def test_matches_symbolic_fraction_equivalence(self) -> None:
        self.assertTrue(math_answers_equal("0.5", r"\frac{1}{2}"))

    def test_matches_tuple_elementwise(self) -> None:
        self.assertTrue(math_answers_equal("(1, 0.5)", r"(1, \frac{1}{2})"))

    def test_rejects_tuple_arity_mismatch(self) -> None:
        self.assertFalse(math_answers_equal("(1, 2)", "(1, 2, 3)"))

    def test_matches_matrix_elementwise(self) -> None:
        predicted = r"\begin{pmatrix} 1 & 0.5 \\ 3 & 4 \end{pmatrix}"
        gold = r"\begin{pmatrix} 1 & \frac{1}{2} \\ 3 & 4.0 \end{pmatrix}"
        self.assertTrue(math_answers_equal(predicted, gold))

    def test_matches_symbolically_equivalent_equations(self) -> None:
        self.assertTrue(math_answers_equal("x=1", "x-1=0"))


if __name__ == "__main__":
    unittest.main()
