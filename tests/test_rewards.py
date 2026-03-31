from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.rewards import RewardWeights, self_reward_function


class SelfRewardFunctionTests(unittest.TestCase):
    def test_exact_match_accepts_function_style_assignment(self) -> None:
        completion = r"<think>work here</think> \[ \boxed{f(x)=0} \]"
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

    def test_exact_match_rejects_expression_equation_rhs_only(self) -> None:
        completion = r"<think>work here</think> \[ \boxed{x+1=2} \]"
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
        self.assertEqual(rewards, [0.0])

    def test_exact_match_accepts_symbolic_fraction_equivalence(self) -> None:
        completion = r"<think>work here</think> \[ \boxed{0.5} \]"
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

    def test_exact_match_accepts_tuple_equivalence(self) -> None:
        completion = r"<think>work here</think> \[ \boxed{(1,0.5)} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=[r"(1,\frac{1}{2})"],
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

    def test_exact_match_accepts_matrix_equivalence(self) -> None:
        completion = r"<think>work here</think> \[ \boxed{\begin{pmatrix}1 & 0.5 \\ 3 & 4\end{pmatrix}} \]"
        rewards = self_reward_function(
            prompts=[[{"role": "user", "content": "question"}]],
            completions=[completion],
            answer=[r"\begin{pmatrix}1 & \frac{1}{2} \\ 3 & 4.0\end{pmatrix}"],
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


if __name__ == "__main__":
    unittest.main()
