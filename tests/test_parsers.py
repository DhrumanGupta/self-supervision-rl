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
    match_answers_hybrid,
    normalize_exact_answer,
    parse_confidence_label,
    parse_correctness_label,
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


class VerifierLabelParserTests(unittest.TestCase):
    def test_parses_correctness_label(self) -> None:
        self.assertEqual(parse_correctness_label("CORRECTNESS: YES"), 1.0)

    def test_parses_confidence_label(self) -> None:
        self.assertEqual(parse_confidence_label("CONFIDENCE: LOW"), 0.0)


class AnswerMatchingTests(unittest.TestCase):
    def test_normalize_exact_answer_canonicalizes_simple_latex_variants(self) -> None:
        self.assertEqual(
            normalize_exact_answer(r" \left( \dfrac{1}{2} \right) "),
            r"(\frac{1}{2})",
        )

    def test_match_answers_hybrid_accepts_boolean_literals(self) -> None:
        self.assertEqual(match_answers_hybrid("Yes", "YES"), (1.0, False))

    def test_match_answers_hybrid_accepts_semantic_fraction_equivalence(self) -> None:
        self.assertEqual(match_answers_hybrid(r"\frac{1}{2}", "0.5"), (1.0, False))

    def test_match_answers_hybrid_accepts_equation_style_assignment(self) -> None:
        self.assertEqual(match_answers_hybrid("2", "x+1=2"), (1.0, False))

    def test_match_answers_hybrid_falls_back_to_exact_for_unparseable_symbolic_forms(
        self,
    ) -> None:
        self.assertEqual(
            match_answers_hybrid(r"O(n^{\log_2 6})", r"O(n^{\log_2 6})"), (1.0, False)
        )

    def test_match_answers_hybrid_rejects_actual_mismatch(self) -> None:
        self.assertEqual(
            match_answers_hybrid(r"\frac{1}{2}", r"\frac{2}{3}"), (0.0, False)
        )


if __name__ == "__main__":
    unittest.main()
