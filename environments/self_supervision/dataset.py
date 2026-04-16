from __future__ import annotations

import logging
import random
from typing import Any

from datasets import Dataset, get_dataset_split_names, load_dataset

from environments.self_supervision.prompts import build_main_prompt_messages


logger = logging.getLogger(__name__)

DEEP_MATH_B0 = "B0"
DEEP_MATH_B1 = "B1"
DEEP_MATH_B2 = "B2"
DEEP_MATH_B3 = "B3"
DEEP_MATH_BAND_ORDER = [DEEP_MATH_B0, DEEP_MATH_B1, DEEP_MATH_B2, DEEP_MATH_B3]


def _normalize_prompt(
    row: dict[str, Any], prompt_key: str, question_key: str
) -> list[dict[str, str]]:
    if prompt_key in row and row[prompt_key] is not None:
        prompt = row[prompt_key]
        if isinstance(prompt, list):
            return build_main_prompt_messages(prompt)
        return build_main_prompt_messages([{"role": "user", "content": str(prompt)}])

    if question_key in row and row[question_key] is not None:
        return build_main_prompt_messages(
            [{"role": "user", "content": str(row[question_key])}]
        )

    for fallback_key in ("problem", "question", "prompt"):
        if fallback_key in row and row[fallback_key] is not None:
            return build_main_prompt_messages(
                [{"role": "user", "content": str(row[fallback_key])}]
            )

    raise KeyError(f"No prompt field found in row keys: {sorted(row.keys())}")


def _normalize_answer(row: dict[str, Any], answer_key: str) -> str:
    for key in (answer_key, "answer", "final_answer", "solution"):
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    raise KeyError(f"No answer field found in row keys: {sorted(row.keys())}")


def _normalize_row(
    row: dict[str, Any],
    index: int,
    *,
    dataset_name: str,
    split: str,
    prompt_key: str,
    question_key: str,
    answer_key: str,
) -> dict[str, Any]:
    info = {
        "source_dataset": dataset_name,
        "split": split,
    }
    if "difficulty" in row and row["difficulty"] is not None:
        info["difficulty"] = float(row["difficulty"])
    if "topic" in row and row["topic"] is not None:
        info["topic"] = str(row["topic"])

    return {
        "example_id": index,
        "prompt": _normalize_prompt(
            row, prompt_key=prompt_key, question_key=question_key
        ),
        "answer": _normalize_answer(row, answer_key=answer_key),
        "info": info,
    }


def get_row_difficulty(row: dict[str, Any]) -> float:
    info = row.get("info")
    if not isinstance(info, dict) or info.get("difficulty") is None:
        raise ValueError(
            "Curriculum requires every example to include info['difficulty']."
        )
    return float(info["difficulty"])


def difficulty_to_band(difficulty: float) -> str:
    if difficulty <= 5.5:
        return DEEP_MATH_B0
    if difficulty <= 6.5:
        return DEEP_MATH_B1
    if difficulty <= 7.5:
        return DEEP_MATH_B2
    return DEEP_MATH_B3


def build_band_indices(dataset: Dataset) -> dict[str, list[int]]:
    band_to_indices = {band: [] for band in DEEP_MATH_BAND_ORDER}
    for index in range(len(dataset)):
        band = difficulty_to_band(get_row_difficulty(dataset[index]))
        band_to_indices[band].append(index)
    return band_to_indices


def build_fixed_band_eval_subsets(
    dataset: Dataset,
    *,
    per_band_limit: int,
    seed: int,
) -> dict[str, Dataset]:
    band_to_indices = build_band_indices(dataset)
    band_to_datasets = {}

    for band_index, band in enumerate(DEEP_MATH_BAND_ORDER):
        indices = list(band_to_indices[band])
        rng = random.Random(seed + band_index)
        rng.shuffle(indices)
        if per_band_limit >= 0:
            indices = indices[:per_band_limit]
        band_to_datasets[band] = dataset.select(indices)

    return band_to_datasets


def validate_curriculum_band_coverage(
    *,
    train_band_indices: dict[str, list[int]],
    eval_band_datasets: dict[str, Dataset],
    required_train_bands: list[str],
    required_eval_bands: list[str],
) -> None:
    for band in required_train_bands:
        if not train_band_indices.get(band):
            raise ValueError(
                f"Curriculum requires at least one training example in band {band}."
            )

    for band in required_eval_bands:
        eval_dataset = eval_band_datasets.get(band)
        if eval_dataset is None or len(eval_dataset) == 0:
            raise ValueError(
                f"Curriculum requires at least one evaluation example in band {band}."
            )


def build_dataset(
    dataset_name: str,
    split: str,
    *,
    dataset_config: str | None = None,
    prompt_key: str = "prompt",
    question_key: str = "problem",
    answer_key: str = "answer",
    num_examples: int = -1,
    seed: int = 42,
) -> Dataset:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    return normalize_dataset(
        dataset,
        dataset_name=dataset_name,
        split=split,
        prompt_key=prompt_key,
        question_key=question_key,
        answer_key=answer_key,
        num_examples=num_examples,
        seed=seed,
    )


def normalize_dataset(
    dataset: Dataset,
    *,
    dataset_name: str,
    split: str,
    prompt_key: str,
    question_key: str,
    answer_key: str,
    num_examples: int,
    seed: int,
) -> Dataset:
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)
    if num_examples >= 0 and hasattr(dataset, "select"):
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    remove_columns = (
        list(dataset.column_names) if hasattr(dataset, "column_names") else None
    )

    def normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
        return _normalize_row(
            row,
            index,
            dataset_name=dataset_name,
            split=split,
            prompt_key=prompt_key,
            question_key=question_key,
            answer_key=answer_key,
        )

    map_kwargs = {
        "with_indices": True,
        "desc": f"Normalizing {dataset_name}[{split}]",
    }
    if remove_columns is not None:
        map_kwargs["remove_columns"] = remove_columns
    return dataset.map(normalize_row, **map_kwargs)


def _load_split_dataset(
    dataset_name: str,
    split: str,
    *,
    dataset_config: str | None = None,
) -> Dataset:
    return load_dataset(dataset_name, dataset_config, split=split)


def _get_available_splits(
    dataset_name: str, *, dataset_config: str | None = None
) -> set[str]:
    return set(get_dataset_split_names(dataset_name, dataset_config))


def build_train_eval_datasets(
    *,
    dataset_name: str,
    dataset_config: str | None = None,
    train_split: str = "train",
    eval_split: str = "test",
    prompt_key: str = "prompt",
    question_key: str = "problem",
    answer_key: str = "answer",
    train_examples: int = -1,
    eval_examples: int = 64,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    available_splits = _get_available_splits(
        dataset_name,
        dataset_config=dataset_config,
    )

    if eval_split in available_splits:
        train_dataset = build_dataset(
            dataset_name,
            train_split,
            dataset_config=dataset_config,
            prompt_key=prompt_key,
            question_key=question_key,
            answer_key=answer_key,
            num_examples=train_examples,
            seed=seed,
        )

        try:
            eval_dataset = build_dataset(
                dataset_name,
                eval_split,
                dataset_config=dataset_config,
                prompt_key=prompt_key,
                question_key=question_key,
                answer_key=answer_key,
                num_examples=eval_examples,
                seed=seed + 1,
            )
        except (FileNotFoundError, ValueError) as error:
            logger.warning(
                "Skipping eval dataset build for %s[%s]: %s: %s",
                dataset_name,
                eval_split,
                type(error).__name__,
                error,
            )
            eval_dataset = None

        return train_dataset, eval_dataset

    if train_split not in available_splits:
        raise ValueError(
            f"Requested train split {train_split!r} not found in {dataset_name}. "
            f"Available splits: {sorted(available_splits)}"
        )

    logger.info(
        "Dataset %s does not provide split %s; creating deterministic 95/5 split from %s with seed %s.",
        dataset_name,
        eval_split,
        train_split,
        seed,
    )
    source_dataset = _load_split_dataset(
        dataset_name,
        train_split,
        dataset_config=dataset_config,
    )
    split_datasets = source_dataset.train_test_split(test_size=0.05, seed=seed)

    train_dataset = normalize_dataset(
        split_datasets["train"],
        dataset_name=dataset_name,
        split=train_split,
        prompt_key=prompt_key,
        question_key=question_key,
        answer_key=answer_key,
        num_examples=train_examples,
        seed=seed,
    )
    eval_dataset = normalize_dataset(
        split_datasets["test"],
        dataset_name=dataset_name,
        split=eval_split,
        prompt_key=prompt_key,
        question_key=question_key,
        answer_key=answer_key,
        num_examples=eval_examples,
        seed=seed + 1,
    )

    return train_dataset, eval_dataset
