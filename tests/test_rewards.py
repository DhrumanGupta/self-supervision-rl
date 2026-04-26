from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.rewards import RewardWeights, self_reward_function


class SelfRewardFunctionTests(unittest.TestCase):
    def test_missing_think_close_tag_gets_negative_reward_and_length_penalty(
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
                verifier=0.2,
                length_penalty=0.5,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [-1.0 - (0.5 * len(completion))])

    def test_unparseable_symbolic_gold_falls_back_to_exact_match_and_does_not_skip(
        self,
    ) -> None:
        completion = r"We reason carefully here</think> \[ \boxed{O(n^{\log_2 6})} \]"
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
                verifier=0.2,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
            log_extra=log_extra,
            log_metric=log_metric,
        )

        self.assertEqual(rewards, [1.0])
        self.assertEqual(extra_logs["exact_match"], [1.0])
        self.assertEqual(extra_logs["exact_match_skipped"], [0.0])
        self.assertEqual(metric_logs["self_reward/exact_match"], 1.0)

    def test_missing_boxed_answer_gets_negative_reward_instead_of_skip(
        self,
    ) -> None:
        completion = r"We reason carefully here</think> Final answer: 1"
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
                verifier=0.2,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
            log_extra=log_extra,
        )

        self.assertEqual(rewards, [-1.0])
        self.assertEqual(extra_logs["exact_match"], [0.0])
        self.assertEqual(extra_logs["exact_match_skipped"], [0.0])

    def test_wrong_but_well_formatted_answer_gets_negative_reward_and_length_penalty(
        self,
    ) -> None:
        completion = r"We reason carefully here</think> \[ \boxed{2} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=["1"],
            rendered_prompt_text=["question <think>"],
            first_completion_text=[completion],
            reward_weights=RewardWeights(
                exact_match=1.0,
                verifier=0.2,
                length_penalty=0.5,
                enable_verifier_reward=False,
            ),
        )
        self.assertEqual(rewards, [-1.0 - (0.5 * len(completion))])

    def test_trailing_tag_invalidates_formatting_and_still_gets_negative_reward(self) -> None:
        completion = r"We reason carefully here</think> \[ \boxed{1} \]</tool_response>"
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
                verifier=0.2,
                length_penalty=0.0,
                enable_verifier_reward=False,
            ),
            log_extra=log_extra,
        )

        self.assertEqual(rewards, [-1.0])
        self.assertEqual(extra_logs["predicted_answer"], [""])
        self.assertEqual(extra_logs["formatting_score"], [0.0])


if __name__ == "__main__":
    unittest.main()
