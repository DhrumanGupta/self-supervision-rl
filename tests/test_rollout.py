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

        self.assertEqual(completion_ids, [[101, 102, 99], [201, 99]])

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
        self.assertEqual(trainer.vllm_generation.calls[0]["num_generations"], 4)
        self.assertEqual(trainer.vllm_generation.calls[1]["num_generations"], 1)

    def test_flattens_sampled_logprobs_from_vllm(self) -> None:
        trainer = _FakeTrainer(enable_verifier_reward=False)

        result = self_reward_rollout(
            [[{"role": "user", "content": "What is 2+2?"}]], trainer
        )

        self.assertEqual(result["logprobs"], [[-0.1, -0.2]])

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

    def test_uses_eval_generation_count_when_model_is_not_training(self) -> None:
        trainer = _FakeTrainer(enable_verifier_reward=False, model_training=False)

        self_reward_rollout([[{"role": "user", "content": "What is 2+2?"}]], trainer)

        self.assertEqual(trainer.vllm_generation.calls[0]["num_generations"], 2)


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
    def __init__(self, *, training: bool) -> None:
        self.training = training


class _FakeVLLMGeneration:
    def __init__(self) -> None:
        self._call_count = 0
        self.calls: list[dict[str, object]] = []

    def generate(self, *, prompts, images, num_generations, profiler=None):
        del images, profiler
        self._call_count += 1
        self.calls.append(
            {
                "prompts": prompts,
                "num_generations": num_generations,
            }
        )
        if self._call_count == 1:
            return prompts, [[10, 99]], [[[-0.1], [-0.2]]], [[[10], [99]]]
        return prompts, [[20, 99]], [[[-0.3], [-0.4]]], [[[20], [99]]]


class _FakeTrainer:
    def __init__(
        self, *, enable_verifier_reward: bool, model_training: bool = True
    ) -> None:
        self.processing_class = _FakeTokenizer()
        self.model = _FakeModel(training=model_training)
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.args = SimpleNamespace(max_prompt_length=128)
        self.enable_verifier_reward = enable_verifier_reward
        self.num_generations = 4
        self.num_generations_eval = 2
        self.max_completion_length = 32
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 50
        self.repetition_penalty = 1.0
        self.vllm_generation = _FakeVLLMGeneration()
        self.logged_metrics: dict[str, list[float]] = {}

    def _log_metric(self, name: str, value: float) -> None:
        self.logged_metrics.setdefault(name, []).append(value)


if __name__ == "__main__":
    unittest.main()
