from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, load_dataset

from environments.self_supervision.prompts import build_main_prompt_messages


logger = logging.getLogger(__name__)


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
    return {
        "example_id": index,
        "prompt": _normalize_prompt(
            row, prompt_key=prompt_key, question_key=question_key
        ),
        "answer": _normalize_answer(row, answer_key=answer_key),
        "info": {
            "source_dataset": dataset_name,
            "split": split,
        },
    }


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
