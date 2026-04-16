from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.rollout import (
    _extract_completion_ids,
    self_reward_rollout,
)


class ExtractCompletionIdsTests(unittest.TestCase):
    def test_trims_completion_ids_at_eos_before_batch_padding(self) -> None:
        input_ids = torch.tensor([[11, 12, 13], [21, 22, 23]])
        generated_ids = torch.tensor(
            [
                [11, 12, 13, 101, 102, 99, 0, 0],
                [21, 22, 23, 201, 99, 0, 0, 0],
            ]
        )

        completion_ids = _extract_completion_ids(
            generated_ids,
            input_ids,
            eos_token_id=99,
            pad_token_id=0,
        )

        self.assertEqual(completion_ids, [[101, 102], [201]])

    def test_trims_completion_ids_at_pad_when_no_eos_is_present(self) -> None:
        input_ids = torch.tensor([[11, 12], [21, 22]])
        generated_ids = torch.tensor(
            [
                [11, 12, 101, 102, 0, 0],
                [21, 22, 201, 202, 203, 0],
            ]
        )

        completion_ids = _extract_completion_ids(
            generated_ids,
            input_ids,
            eos_token_id=99,
            pad_token_id=0,
        )

        self.assertEqual(completion_ids, [[101, 102], [201, 202, 203]])


class RolloutProfilingTests(unittest.TestCase):
    def test_logs_rollout_timings_with_self_eval(self) -> None:
        trainer = _FakeTrainer(enable_verifier_reward=True)

        with patch(
            "environments.self_supervision.rollout.time.perf_counter",
            side_effect=[0.0, 1.0, 3.0, 5.0, 9.0, 12.0],
        ):
            self_reward_rollout(
                [[{"role": "user", "content": "What is 2+2?"}]], trainer
            )

        self.assertEqual(
            trainer.logged_metrics["profiling/rollout/main_generate_s"], [2.0]
        )
        self.assertEqual(
            trainer.logged_metrics["profiling/rollout/self_eval_generate_s"],
            [4.0],
        )
        self.assertEqual(trainer.logged_metrics["profiling/rollout/total_s"], [12.0])

    def test_logs_rollout_timings_without_self_eval(self) -> None:
        trainer = _FakeTrainer(enable_verifier_reward=False)

        with patch(
            "environments.self_supervision.rollout.time.perf_counter",
            side_effect=[0.0, 1.0, 2.5, 4.0],
        ):
            self_reward_rollout(
                [[{"role": "user", "content": "What is 2+2?"}]], trainer
            )

        self.assertEqual(
            trainer.logged_metrics["profiling/rollout/main_generate_s"], [1.5]
        )
        self.assertNotIn(
            "profiling/rollout/self_eval_generate_s", trainer.logged_metrics
        )
        self.assertEqual(trainer.logged_metrics["profiling/rollout/total_s"], [4.0])


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        return messages[-1]["content"]

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        del texts, return_tensors, padding, truncation, max_length
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return [f"decoded-{ids[0]}" if ids else "" for ids in token_ids]


class _FakeModel:
    def __init__(self) -> None:
        self.training = True
        self._parameter = torch.nn.Parameter(torch.zeros(1))
        self._generate_call_count = 0

    def parameters(self):
        yield self._parameter

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def generate(self, **kwargs):
        del kwargs
        self._generate_call_count += 1
        completion_token = 10 if self._generate_call_count == 1 else 20
        return torch.tensor([[11, 12, completion_token, 99]], dtype=torch.long)


class _FakeTrainer:
    def __init__(self, *, enable_verifier_reward: bool) -> None:
        self.processing_class = _FakeTokenizer()
        self.model = _FakeModel()
        self.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        self.args = SimpleNamespace(max_prompt_length=128)
        self.enable_verifier_reward = enable_verifier_reward
        self.max_completion_length = 32
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 50
        self.repetition_penalty = 1.0
        self.logged_metrics: dict[str, list[float]] = {}

    def _log_metric(self, name: str, value: float) -> None:
        self.logged_metrics.setdefault(name, []).append(value)


if __name__ == "__main__":
    unittest.main()
