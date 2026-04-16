from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.dataset import (  # noqa: E402
    DEEP_MATH_B0,
    DEEP_MATH_B1,
    DEEP_MATH_B2,
    DEEP_MATH_B3,
    build_fixed_band_eval_subsets,
    build_train_eval_datasets,
    difficulty_to_band,
    get_row_difficulty,
    normalize_dataset,
)


class NormalizeDatasetTests(unittest.TestCase):
    def test_preserves_difficulty_and_topic_in_info(self) -> None:
        dataset = Dataset.from_list(
            [
                {
                    "question": "What is 2+2?",
                    "final_answer": "4",
                    "difficulty": 4.5,
                    "topic": "Arithmetic",
                }
            ]
        )

        normalized = normalize_dataset(
            dataset,
            dataset_name="zwhe99/DeepMath-103K",
            split="train",
            prompt_key="prompt",
            question_key="question",
            answer_key="final_answer",
            num_examples=-1,
            seed=42,
        )

        row = normalized[0]
        self.assertEqual(row["answer"], "4")
        self.assertEqual(row["prompt"][-1]["content"], "What is 2+2?")
        self.assertEqual(row["info"]["difficulty"], 4.5)
        self.assertEqual(row["info"]["topic"], "Arithmetic")


class BuildTrainEvalDatasetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_dataset = Dataset.from_list(
            [
                {
                    "question": f"Question {index}",
                    "final_answer": str(index),
                    "difficulty": float(index % 5),
                    "topic": f"Topic {index % 3}",
                }
                for index in range(20)
            ]
        )

    def test_falls_back_to_seeded_train_test_split_when_eval_missing(self) -> None:
        with (
            patch(
                "environments.self_supervision.dataset._get_available_splits",
                return_value={"train"},
            ),
            patch(
                "environments.self_supervision.dataset._load_split_dataset",
                return_value=self.source_dataset,
            ),
        ):
            train_dataset, eval_dataset = build_train_eval_datasets(
                dataset_name="zwhe99/DeepMath-103K",
                train_split="train",
                eval_split="test",
                question_key="question",
                answer_key="final_answer",
                train_examples=-1,
                eval_examples=-1,
                seed=7,
            )

        self.assertEqual(len(train_dataset), 19)
        self.assertEqual(len(eval_dataset), 1)
        self.assertEqual(train_dataset[0]["info"]["split"], "train")
        self.assertEqual(eval_dataset[0]["info"]["split"], "test")

    def test_seeded_split_is_deterministic(self) -> None:
        with (
            patch(
                "environments.self_supervision.dataset._get_available_splits",
                return_value={"train"},
            ),
            patch(
                "environments.self_supervision.dataset._load_split_dataset",
                return_value=self.source_dataset,
            ),
        ):
            train_a, eval_a = build_train_eval_datasets(
                dataset_name="zwhe99/DeepMath-103K",
                question_key="question",
                answer_key="final_answer",
                eval_examples=-1,
                seed=11,
            )
            train_b, eval_b = build_train_eval_datasets(
                dataset_name="zwhe99/DeepMath-103K",
                question_key="question",
                answer_key="final_answer",
                eval_examples=-1,
                seed=11,
            )

        self.assertEqual(train_a["answer"], train_b["answer"])
        self.assertEqual(eval_a["answer"], eval_b["answer"])

    def test_different_seeds_change_split_membership(self) -> None:
        with (
            patch(
                "environments.self_supervision.dataset._get_available_splits",
                return_value={"train"},
            ),
            patch(
                "environments.self_supervision.dataset._load_split_dataset",
                return_value=self.source_dataset,
            ),
        ):
            _, eval_a = build_train_eval_datasets(
                dataset_name="zwhe99/DeepMath-103K",
                question_key="question",
                answer_key="final_answer",
                eval_examples=-1,
                seed=3,
            )
            _, eval_b = build_train_eval_datasets(
                dataset_name="zwhe99/DeepMath-103K",
                question_key="question",
                answer_key="final_answer",
                eval_examples=-1,
                seed=17,
            )

        self.assertNotEqual(eval_a["answer"], eval_b["answer"])

    def test_example_limits_apply_after_split(self) -> None:
        with (
            patch(
                "environments.self_supervision.dataset._get_available_splits",
                return_value={"train"},
            ),
            patch(
                "environments.self_supervision.dataset._load_split_dataset",
                return_value=self.source_dataset,
            ),
        ):
            train_dataset, eval_dataset = build_train_eval_datasets(
                dataset_name="zwhe99/DeepMath-103K",
                question_key="question",
                answer_key="final_answer",
                train_examples=5,
                eval_examples=1,
                seed=13,
            )

        self.assertEqual(len(train_dataset), 5)
        self.assertEqual(len(eval_dataset), 1)


class DifficultyBandTests(unittest.TestCase):
    def test_assigns_expected_boundary_bands(self) -> None:
        self.assertEqual(difficulty_to_band(5.5), DEEP_MATH_B0)
        self.assertEqual(difficulty_to_band(5.5001), DEEP_MATH_B1)
        self.assertEqual(difficulty_to_band(6.5), DEEP_MATH_B1)
        self.assertEqual(difficulty_to_band(6.5001), DEEP_MATH_B2)
        self.assertEqual(difficulty_to_band(7.5), DEEP_MATH_B2)
        self.assertEqual(difficulty_to_band(7.5001), DEEP_MATH_B3)

    def test_requires_difficulty_in_info(self) -> None:
        with self.assertRaisesRegex(ValueError, r"difficulty"):
            get_row_difficulty({"info": {}})


class FixedBandEvalSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Dataset.from_list(
            [
                {"answer": f"a{index}", "info": {"difficulty": difficulty}}
                for index, difficulty in enumerate(
                    [4.0, 5.0, 6.0, 6.2, 7.0, 7.3, 8.0, 8.5],
                    start=1,
                )
            ]
        )

    def test_is_deterministic_for_same_seed(self) -> None:
        subsets_a = build_fixed_band_eval_subsets(
            self.dataset, per_band_limit=1, seed=11
        )
        subsets_b = build_fixed_band_eval_subsets(
            self.dataset, per_band_limit=1, seed=11
        )

        self.assertEqual(
            subsets_a[DEEP_MATH_B0]["answer"], subsets_b[DEEP_MATH_B0]["answer"]
        )
        self.assertEqual(
            subsets_a[DEEP_MATH_B1]["answer"], subsets_b[DEEP_MATH_B1]["answer"]
        )
        self.assertEqual(
            subsets_a[DEEP_MATH_B2]["answer"], subsets_b[DEEP_MATH_B2]["answer"]
        )
        self.assertEqual(
            subsets_a[DEEP_MATH_B3]["answer"], subsets_b[DEEP_MATH_B3]["answer"]
        )

    def test_changes_membership_for_different_seed(self) -> None:
        subsets_a = build_fixed_band_eval_subsets(
            self.dataset, per_band_limit=1, seed=3
        )
        subsets_b = build_fixed_band_eval_subsets(
            self.dataset, per_band_limit=1, seed=17
        )

        self.assertNotEqual(
            subsets_a[DEEP_MATH_B0]["answer"], subsets_b[DEEP_MATH_B0]["answer"]
        )

    def test_applies_per_band_limit(self) -> None:
        subsets = build_fixed_band_eval_subsets(self.dataset, per_band_limit=1, seed=5)

        self.assertEqual(len(subsets[DEEP_MATH_B0]), 1)
        self.assertEqual(len(subsets[DEEP_MATH_B1]), 1)
        self.assertEqual(len(subsets[DEEP_MATH_B2]), 1)
        self.assertEqual(len(subsets[DEEP_MATH_B3]), 1)


if __name__ == "__main__":
    unittest.main()
