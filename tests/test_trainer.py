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
                "environments.self_supervision.trainer.trl_grpo_trainer.pd.DataFrame.to_parquet"
            ) as to_parquet:
                trainer.log({"loss": 1.23})

        self.assertEqual(trainer.state.log_history[-1]["loss"], 1.23)
        to_parquet.assert_called_once()
        parquet_path = to_parquet.call_args.args[0]
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


if __name__ == "__main__":
    unittest.main()
