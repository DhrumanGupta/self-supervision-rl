from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.train_grpo_self_reward import (
    DEEP_MATH_DATASET_NAME,
    parse_args,
    should_enable_curriculum,
    validate_supported_model_name,
)


class ValidateSupportedModelNameTests(unittest.TestCase):
    def test_accepts_supported_qwen35_base_model(self) -> None:
        validate_supported_model_name("Qwen/Qwen3.5-9B-Base")

    def test_rejects_unsupported_model_family(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Qwen/Qwen3\.5-\*-Base"):
            validate_supported_model_name("meta-llama/Llama-3.1-8B")

    def test_rejects_non_base_qwen35_model(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Qwen/Qwen3\.5-\*-Base"):
            validate_supported_model_name("Qwen/Qwen3.5-9B-Instruct")


class ParseArgsTests(unittest.TestCase):
    def test_accepts_num_generations_eval_override(self) -> None:
        argv = [
            "train_grpo_self_reward.py",
            "--model_name",
            "Qwen/Qwen3.5-9B-Base",
            "--output_dir",
            "outputs/test",
            "--num_generations_eval",
            "4",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.num_generations_eval, 4)

    def test_enables_curriculum_by_default_for_deepmath(self) -> None:
        argv = [
            "train_grpo_self_reward.py",
            "--model_name",
            "Qwen/Qwen3.5-9B-Base",
            "--dataset_name",
            DEEP_MATH_DATASET_NAME,
            "--output_dir",
            "outputs/test",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(should_enable_curriculum(args))

    def test_disable_curriculum_flag_overrides_deepmath_default(self) -> None:
        argv = [
            "train_grpo_self_reward.py",
            "--model_name",
            "Qwen/Qwen3.5-9B-Base",
            "--dataset_name",
            DEEP_MATH_DATASET_NAME,
            "--output_dir",
            "outputs/test",
            "--disable_curriculum",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertFalse(should_enable_curriculum(args))


if __name__ == "__main__":
    unittest.main()
