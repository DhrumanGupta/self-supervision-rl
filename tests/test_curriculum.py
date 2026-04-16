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

from environments.self_supervision.curriculum import (  # noqa: E402
    CurriculumConfig,
    CurriculumController,
    CurriculumGRPOTrainer,
    CurriculumRepeatSampler,
    CurriculumStageConfig,
)


class CurriculumControllerTests(unittest.TestCase):
    def test_does_not_promote_before_min_stage_steps(self) -> None:
        controller = CurriculumController(CurriculumConfig.default_deepmath())

        promoted = controller.record_frontier_eval(global_step=300, exact_match=0.9)

        self.assertFalse(promoted)
        self.assertEqual(controller.current_stage_index, 0)
        self.assertEqual(controller.consecutive_frontier_successes, 0)

    def test_promotes_after_two_consecutive_qualifying_evals(self) -> None:
        controller = CurriculumController(CurriculumConfig.default_deepmath())

        self.assertFalse(
            controller.record_frontier_eval(global_step=400, exact_match=0.30)
        )
        self.assertTrue(
            controller.record_frontier_eval(global_step=500, exact_match=0.30)
        )
        self.assertEqual(controller.current_stage_index, 1)
        self.assertEqual(controller.stage_start_step, 500)
        self.assertEqual(controller.consecutive_frontier_successes, 0)

    def test_failed_eval_resets_success_streak(self) -> None:
        controller = CurriculumController(CurriculumConfig.default_deepmath())

        self.assertFalse(
            controller.record_frontier_eval(global_step=400, exact_match=0.30)
        )
        self.assertFalse(
            controller.record_frontier_eval(global_step=500, exact_match=0.10)
        )
        self.assertEqual(controller.consecutive_frontier_successes, 0)

    def test_terminal_stage_never_promotes(self) -> None:
        config = CurriculumConfig.default_deepmath()
        controller = CurriculumController(
            config, current_stage_index=len(config.stages) - 1
        )

        promoted = controller.record_frontier_eval(global_step=5000, exact_match=1.0)

        self.assertFalse(promoted)
        self.assertEqual(controller.current_stage_index, len(config.stages) - 1)

    def test_state_roundtrip(self) -> None:
        controller = CurriculumController(
            CurriculumConfig.default_deepmath(),
            current_stage_index=2,
            stage_start_step=700,
            consecutive_frontier_successes=1,
        )
        restored = CurriculumController(CurriculumConfig.default_deepmath())

        restored.load_state_dict(controller.state_dict())

        self.assertEqual(restored.current_stage_index, 2)
        self.assertEqual(restored.stage_start_step, 700)
        self.assertEqual(restored.consecutive_frontier_successes, 1)


class CurriculumRepeatSamplerTests(unittest.TestCase):
    def test_preserves_repeat_sampler_pattern(self) -> None:
        config = CurriculumConfig(
            stages=[
                CurriculumStageConfig(
                    name="stage_0",
                    sampling_weights={"B0": 1.0},
                    frontier_band="B0",
                    min_stage_steps=0,
                    promotion_threshold=0.0,
                )
            ]
        )
        controller = CurriculumController(config)
        sampler = CurriculumRepeatSampler(
            data_source=list(range(4)),
            band_to_indices={"B0": [0, 1, 2, 3]},
            controller=controller,
            mini_repeat_count=2,
            batch_size=2,
            repeat_count=2,
            shuffle=False,
            seed=0,
        )

        sampled = list(sampler)

        self.assertEqual(sampled[:8], [0, 0, 1, 1, 0, 0, 1, 1])
        self.assertEqual(len(sampled), len(sampler))

    def test_uses_updated_stage_for_later_chunks(self) -> None:
        config = CurriculumConfig(
            stages=[
                CurriculumStageConfig(
                    name="stage_0",
                    sampling_weights={"B0": 1.0},
                    frontier_band="B0",
                    min_stage_steps=0,
                    promotion_threshold=0.0,
                ),
                CurriculumStageConfig(
                    name="stage_1",
                    sampling_weights={"B1": 1.0},
                    frontier_band="B1",
                    min_stage_steps=0,
                    promotion_threshold=0.0,
                ),
            ]
        )
        controller = CurriculumController(config)
        sampler = CurriculumRepeatSampler(
            data_source=list(range(4)),
            band_to_indices={"B0": [0, 1], "B1": [2, 3]},
            controller=controller,
            mini_repeat_count=1,
            batch_size=2,
            repeat_count=1,
            shuffle=False,
            seed=0,
        )

        iterator = iter(sampler)
        first_chunk = [next(iterator), next(iterator)]
        controller.current_stage_index = 1
        second_chunk = [next(iterator), next(iterator)]

        self.assertEqual(first_chunk, [0, 1])
        self.assertEqual(second_chunk, [2, 3])


class CurriculumTrainerStateTests(unittest.TestCase):
    def test_saves_and_loads_curriculum_state_file(self) -> None:
        trainer = object.__new__(CurriculumGRPOTrainer)
        trainer.curriculum_controller = CurriculumController(
            CurriculumConfig.default_deepmath(),
            current_stage_index=2,
            stage_start_step=800,
            consecutive_frontier_successes=1,
        )

        restored_trainer = object.__new__(CurriculumGRPOTrainer)
        restored_trainer.curriculum_controller = CurriculumController(
            CurriculumConfig.default_deepmath()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer._save_curriculum_state(tmpdir)
            restored_trainer._load_curriculum_state(tmpdir)

        self.assertEqual(restored_trainer.curriculum_controller.current_stage_index, 2)
        self.assertEqual(restored_trainer.curriculum_controller.stage_start_step, 800)
        self.assertEqual(
            restored_trainer.curriculum_controller.consecutive_frontier_successes,
            1,
        )

    def test_logs_extended_curriculum_metrics(self) -> None:
        trainer = object.__new__(CurriculumGRPOTrainer)
        trainer.curriculum_controller = CurriculumController(
            CurriculumConfig.default_deepmath()
        )
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer._logs = {
            "images": deque(),
            "prompt": deque(),
            "completion": deque(),
            "rewards": defaultdict(deque),
            "advantages": deque(),
            "extra": defaultdict(deque),
        }
        trainer.accelerator = SimpleNamespace(is_main_process=False)
        trainer.log_completions = True
        trainer.log_unique_prompts = False
        trainer.model = SimpleNamespace(training=True)
        trainer.state = SimpleNamespace(epoch=None, global_step=450, log_history=[])
        trainer.control = object()
        trainer.callback_handler = SimpleNamespace(
            on_log=lambda args, state, control, logs: control
        )
        trainer.args = SimpleNamespace(
            include_num_input_tokens_seen="no",
            output_dir="unused",
            report_to=[],
        )

        with patch(
            "environments.self_supervision.trainer.trl_grpo_trainer.pd.DataFrame.to_parquet"
        ):
            trainer.log({"loss": 1.0})

        logged = trainer.state.log_history[-1]
        self.assertEqual(logged["curriculum/stage"], 0)
        self.assertEqual(logged["curriculum/frontier_band_index"], 0)
        self.assertEqual(logged["curriculum/frontier_threshold"], 0.25)
        self.assertEqual(logged["curriculum/mix_b0"], 1.0)
        self.assertEqual(logged["curriculum/mix_b1"], 0.0)
        self.assertEqual(logged["curriculum/mix_b2"], 0.0)
        self.assertEqual(logged["curriculum/mix_b3"], 0.0)


if __name__ == "__main__":
    unittest.main()
