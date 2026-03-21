from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset


TOY_ROWS = [
    {
        "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
        "answer": "4",
        "info": {"task_type": "math", "source": "toy"},
    },
    {
        "prompt": [{"role": "user", "content": "What is 7 * 8?"}],
        "answer": "56",
        "info": {"task_type": "math", "source": "toy"},
    },
    {
        "prompt": [{"role": "user", "content": "Solve for x: 3x = 21"}],
        "answer": "7",
        "info": {"task_type": "math", "source": "toy"},
    },
]


def _normalize_prompt(
    row: dict[str, Any], prompt_key: str, question_key: str
) -> list[dict[str, str]]:
    if prompt_key in row and row[prompt_key] is not None:
        prompt = row[prompt_key]
        if isinstance(prompt, list):
            return prompt
        return [{"role": "user", "content": str(prompt)}]

    if question_key in row and row[question_key] is not None:
        return [{"role": "user", "content": str(row[question_key])}]

    for fallback_key in ("problem", "question", "prompt"):
        if fallback_key in row and row[fallback_key] is not None:
            return [{"role": "user", "content": str(row[fallback_key])}]

    raise KeyError(f"No prompt field found in row keys: {sorted(row.keys())}")


def _normalize_answer(row: dict[str, Any], answer_key: str) -> str:
    for key in (answer_key, "answer", "final_answer", "solution"):
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    raise KeyError(f"No answer field found in row keys: {sorted(row.keys())}")


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
    if dataset_name == "toy":
        rows = [dict(row) for row in TOY_ROWS]
        if num_examples >= 0:
            rows = rows[:num_examples]
        return Dataset.from_list(rows)

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)
    if num_examples >= 0 and hasattr(dataset, "select"):
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    rows = []
    for index, row in enumerate(dataset):
        rows.append(
            {
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
        )
    return Dataset.from_list(rows)


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
    except Exception:
        eval_dataset = None

    return train_dataset, eval_dataset
