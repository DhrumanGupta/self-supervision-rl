from __future__ import annotations

import sys
import tempfile
import unittest
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.trainer import (  # noqa: E402
    ProfilingCallback,
    SelfSupervisionGRPOTrainer,
)


class ProfilingCallbackTests(unittest.TestCase):
    def test_records_optimizer_and_total_update_timings(self) -> None:
        trainer = SimpleNamespace(_metrics={"train": defaultdict(list)})

        def record_step_metric(
            name: str, value: float, *, mode: str | None = None
        ) -> None:
            trainer._metrics[mode][name].append(value)

        trainer.record_step_metric = record_step_metric
        callback = ProfilingCallback(trainer)

        with patch(
            "environments.self_supervision.trainer.time.perf_counter",
            side_effect=[1.0, 3.0, 5.5, 9.0],
        ):
            callback.on_step_begin(None, None, None)
            callback.on_pre_optimizer_step(None, None, None)
            callback.on_optimizer_step(None, None, None)
            callback.on_step_end(None, None, None)

        self.assertEqual(
            trainer._metrics["train"]["profiling/optimizer/step_s"],
            [2.5],
        )
        self.assertEqual(
            trainer._metrics["train"]["profiling/update/total_s"],
            [8.0],
        )


class SelfSupervisionTrainerLogTests(unittest.TestCase):
    def test_keeps_parquet_logging_without_console_completion_prints(self) -> None:
        trainer = object.__new__(SelfSupervisionGRPOTrainer)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer._logs = {
            "images": deque(),
            "prompt": deque(["prompt"]),
            "completion": deque(["completion"]),
            "rewards": defaultdict(deque),
            "advantages": deque([0.5]),
            "extra": defaultdict(deque),
        }
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.log_completions = True
        trainer.completion_logging_steps = 1
        trainer.log_unique_prompts = False
        trainer.model = SimpleNamespace(training=True)
        trainer.state = SimpleNamespace(epoch=None, global_step=7, log_history=[])
        trainer.control = object()
        trainer.callback_handler = SimpleNamespace(
            on_log=lambda args, state, control, logs: control
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.args = SimpleNamespace(
                include_num_input_tokens_seen="no",
                output_dir=tmpdir,
                report_to=[],
            )
            with patch(
                "environments.self_supervision.trainer.trl_grpo_trainer.pd.DataFrame.to_parquet",
                autospec=True,
            ) as to_parquet:
                trainer.log({"loss": 1.23})

        self.assertEqual(trainer.state.log_history[-1]["loss"], 1.23)
        to_parquet.assert_called_once()
        df = to_parquet.call_args.args[0]
        parquet_path = to_parquet.call_args.args[1]
        self.assertNotIn("step", df.columns)
        self.assertTrue(parquet_path.endswith("completions/train/completions_00007.parquet"))

    def test_skips_completion_logging_until_cadence_boundary(self) -> None:
        trainer = object.__new__(SelfSupervisionGRPOTrainer)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer._logs = {
            "images": deque(),
            "prompt": deque(["prompt"]),
            "completion": deque(["completion"]),
            "rewards": defaultdict(deque),
            "advantages": deque([0.5]),
            "extra": defaultdict(deque),
        }
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.log_completions = True
        trainer.completion_logging_steps = 5
        trainer.log_unique_prompts = False
        trainer.model = SimpleNamespace(training=True)
        trainer.state = SimpleNamespace(epoch=None, global_step=7, log_history=[])
        trainer.control = object()
        trainer.callback_handler = SimpleNamespace(
            on_log=lambda args, state, control, logs: control
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.args = SimpleNamespace(
                include_num_input_tokens_seen="no",
                output_dir=tmpdir,
                report_to=[],
            )
            with patch(
                "environments.self_supervision.trainer.trl_grpo_trainer.pd.DataFrame.to_parquet"
            ) as to_parquet:
                trainer.log({"loss": 1.23})

        self.assertEqual(trainer.state.log_history[-1]["loss"], 1.23)
        to_parquet.assert_not_called()

    def test_filters_trl_sampling_and_duplicate_reward_metrics(self) -> None:
        trainer = object.__new__(SelfSupervisionGRPOTrainer)
        trainer._metrics = {
            "train": defaultdict(
                list,
                {
                    "reward": [1.0],
                    "reward_std": [0.25],
                    "rewards/self_reward_function/mean": [1.0],
                    "rewards/self_reward_function/std": [0.25],
                    "sampling/sampling_logp_difference/mean": [0.1],
                    "sampling/importance_sampling_ratio/mean": [1.1],
                },
            ),
            "eval": defaultdict(list),
        }
        trainer._logs = {
            "images": deque(),
            "prompt": deque(),
            "completion": deque(),
            "rewards": defaultdict(deque),
            "advantages": deque(),
            "extra": defaultdict(deque),
        }
        trainer.accelerator = SimpleNamespace(is_main_process=False)
        trainer.log_completions = False
        trainer.log_unique_prompts = False
        trainer.model = SimpleNamespace(training=True)
        trainer.state = SimpleNamespace(epoch=None, global_step=7, log_history=[])
        trainer.control = object()
        trainer.callback_handler = SimpleNamespace(
            on_log=lambda args, state, control, logs: control
        )
        trainer.args = SimpleNamespace(
            include_num_input_tokens_seen="no",
            output_dir="unused",
            report_to=[],
        )

        trainer.log({"loss": 1.23})

        logged = trainer.state.log_history[-1]
        self.assertEqual(logged["reward"], 1.0)
        self.assertEqual(logged["reward_std"], 0.25)
        self.assertNotIn("rewards/self_reward_function/mean", logged)
        self.assertNotIn("rewards/self_reward_function/std", logged)
        self.assertNotIn("sampling/sampling_logp_difference/mean", logged)
        self.assertNotIn("sampling/importance_sampling_ratio/mean", logged)


class SelfSupervisionTrainerComputeLossTests(unittest.TestCase):
    def test_uses_liger_path_during_training(self) -> None:
        trainer = object.__new__(SelfSupervisionGRPOTrainer)
        trainer.use_liger_kernel = True
        trainer.model = SimpleNamespace(training=True)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: "unwrapped")
        trainer.compute_liger_loss = object()
        trainer._forward_redirection = lambda *args: ("liger", args)
        trainer._compute_loss = lambda model, inputs: ("plain", model, inputs)

        result = trainer.compute_loss("wrapped", {"x": 1})

        self.assertEqual(result[0], "liger")
        self.assertEqual(result[1][0], "wrapped")
        self.assertEqual(result[1][1], "unwrapped")
        self.assertEqual(result[1][2], trainer.compute_liger_loss)

    def test_uses_plain_loss_during_eval_even_if_liger_enabled(self) -> None:
        trainer = object.__new__(SelfSupervisionGRPOTrainer)
        trainer.use_liger_kernel = True
        trainer.model = SimpleNamespace(training=False)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: "unwrapped")
        trainer.compute_liger_loss = object()
        trainer._forward_redirection = lambda *args: ("liger", args)
        trainer._compute_loss = lambda model, inputs: ("plain", model, inputs)

        result = trainer.compute_loss("wrapped", {"x": 1})

        self.assertEqual(result, ("plain", "wrapped", {"x": 1}))


if __name__ == "__main__":
    unittest.main()
