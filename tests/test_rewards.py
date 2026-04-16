from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.rewards import RewardWeights, self_reward_function


class SelfRewardFunctionTests(unittest.TestCase):
    def test_missing_think_close_tag_zeroes_positive_reward_but_keeps_length_penalty(
        self,
    ) -> None:
        completion = r"\[ \boxed{0} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=["0"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.1,
                verifier=0.2,
                length_penalty=0.5,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [-0.5 * len(completion)])

    def test_exact_match_accepts_function_style_assignment(self) -> None:
        completion = r"work here</think> \[ \boxed{f(x)=0} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=["0"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.0,
                verifier=0.0,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [1.0])

    def test_exact_match_accepts_symbolic_fraction_equivalence(self) -> None:
        completion = r"work here</think> \[ \boxed{0.5} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=[r"\frac{1}{2}"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.0,
                verifier=0.0,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [1.0])

    def test_exact_match_accepts_fraction_style_variants(self) -> None:
        completion = r"work here</think> \[ \boxed{\frac34} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=[r"\dfrac34"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.0,
                verifier=0.0,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [1.0])

    def test_exact_match_accepts_literal_yes_no_answers(self) -> None:
        completion = r"work here</think> \[ \boxed{YES} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=["Yes"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.0,
                verifier=0.0,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [1.0])

    def test_exact_match_accepts_numeric_variants_via_semantic_match(self) -> None:
        completion = r"work here</think> \[ \boxed{+5} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=["5"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.0,
                verifier=0.0,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [1.0])

    def test_exact_match_uses_trl_math_verify_behavior_for_equations(self) -> None:
        completion = r"work here</think> \[ \boxed{x+1=2} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=["2"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.0,
                verifier=0.0,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [1.0])

    def test_unparseable_symbolic_gold_falls_back_to_exact_match_and_does_not_skip(
        self,
    ) -> None:
        completion = r"work here</think> \[ \boxed{O(n^{\log_2 6})} \]"
        extra_logs = {}
        metric_logs = {}

        def log_extra(name, values):
            extra_logs[name] = list(values)

        def log_metric(name, value):
            metric_logs[name] = value

        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=[r"O(n^{\log_2 6})"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.1,
                verifier=0.2,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
            log_extra=log_extra,
            log_metric=log_metric,
        )

        self.assertEqual(rewards, [1.1])
        self.assertEqual(extra_logs["exact_match"], [1.0])
        self.assertEqual(extra_logs["exact_match_skipped"], [0.0])
        self.assertEqual(metric_logs["self_reward/exact_match"], 1.0)

    def test_missing_boxed_answer_returns_zero_exact_match_instead_of_skip(
        self,
    ) -> None:
        completion = r"work here</think> Final answer: 1"
        extra_logs = {}

        def log_extra(name, values):
            extra_logs[name] = list(values)

        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=["1"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                formatting=0.1,
                verifier=0.2,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
            log_extra=log_extra,
        )

        self.assertEqual(rewards, [0.0])
        self.assertEqual(extra_logs["exact_match"], [0.0])
        self.assertEqual(extra_logs["exact_match_skipped"], [0.0])


if __name__ == "__main__":
    unittest.main()
