from __future__ import annotations

import logging
from dataclasses import dataclass

from environments.self_supervision.parsers import (
    extract_final_answer,
    has_valid_think_format,
    math_answers_equal,
    parse_confidence_label,
    parse_correctness_label,
)

logger = logging.getLogger(__name__)


def _format_binary_label(value: float, positive: str, negative: str) -> str:
    return positive if value == 1.0 else negative


@dataclass(frozen=True)
class RewardWeights:
    exact_match: float = 1.0
    formatting: float = 0.1
    verifier: float = 0.2
    length_penalty: float = 0.0001
    enable_verifier_reward: bool = True


def _validate_batch_lengths(**batch_values) -> None:
    lengths = {name: len(values) for name, values in batch_values.items()}
    if len(set(lengths.values())) == 1:
        return

    logger.error("Reward batch length mismatch: %s", lengths)
    raise ValueError(f"Reward batch length mismatch: {lengths}")


def self_reward_function(
    prompts,
    completions,
    answer,
    completion_ids=None,
    first_completion_text=None,
    rendered_prompt_text=None,
    self_eval_text=None,
    reward_weights: RewardWeights | None = None,
    log_extra=None,
    log_metric=None,
    **kwargs,
):
    weights = reward_weights or RewardWeights()
    verifier_enabled = weights.enable_verifier_reward
    rewards = []
    exact_scores = []
    formatting_scores = []
    verifier_scores = []
    predicted_answers = []
    correctness_labels = []
    confidence_labels = []
    completion_lengths = []

    if first_completion_text is None:
        if completions and isinstance(completions[0], list):
            first_completion_text = [
                completion[-1].get("content", "") if completion else ""
                for completion in completions
            ]
        else:
            first_completion_text = [str(completion) for completion in completions]

    if self_eval_text is None:
        self_eval_text = [""] * len(first_completion_text)

    if rendered_prompt_text is None:
        rendered_prompt_text = [
            "\n".join(message.get("content", "") for message in prompt_messages)
            for prompt_messages in prompts
        ]

    if completion_ids is None:
        completion_ids = [None] * len(first_completion_text)

    _validate_batch_lengths(
        rendered_prompt_text=rendered_prompt_text,
        first_completion_text=first_completion_text,
        self_eval_text=self_eval_text,
        answer=answer,
        completion_ids=completion_ids,
    )

    for prompt_text, completion_text, probe_text, gold, token_ids in zip(
        rendered_prompt_text,
        first_completion_text,
        self_eval_text,
        answer,
        completion_ids,
        strict=True,
    ):
        formatting_score = (
            1.0 if has_valid_think_format(prompt_text, completion_text or "") else 0.0
        )
        predicted_answer = extract_final_answer(completion_text)
        exact_match = (
            1.0 if math_answers_equal(predicted_answer, str(gold).strip()) else 0.0
        )
        said_correct = 0.0
        confidence = 0.0

        if verifier_enabled:
            said_correct = parse_correctness_label(probe_text)
            self_consistency = 1.0 if said_correct == exact_match else 0.0

            confidence = parse_confidence_label(probe_text)
            if exact_match == 1.0 and said_correct == 1.0:
                confidence_score = 1.0 if confidence == 1.0 else 0.5
            elif exact_match == 0.0 and said_correct == 0.0:
                confidence_score = 1.0 if confidence == 0.0 else 0.5
            else:
                confidence_score = 0.0
        else:
            self_consistency = 0.0
            confidence_score = 0.0

        verifier_score = 0.5 * (self_consistency + confidence_score)
        predicted_answers.append(predicted_answer)
        correctness_labels.append(_format_binary_label(said_correct, "YES", "NO"))
        confidence_labels.append(_format_binary_label(confidence, "HIGH", "LOW"))

        completion_length = (
            len(token_ids) if token_ids is not None else len(completion_text or "")
        )
        total = (
            weights.exact_match * exact_match
            + weights.formatting * formatting_score
            + weights.verifier * verifier_score
            - weights.length_penalty * completion_length
        )
        rewards.append(float(total))
        exact_scores.append(exact_match)
        formatting_scores.append(formatting_score)
        verifier_scores.append(verifier_score)
        completion_lengths.append(completion_length)

    if log_extra:
        log_extra("gold_answer", [str(item) for item in answer])
        log_extra("rendered_prompt_text", list(rendered_prompt_text))
        log_extra("first_completion_text", list(first_completion_text))
        log_extra("self_eval_text", list(self_eval_text))
        log_extra("predicted_answer", predicted_answers)
        log_extra("completion_length", completion_lengths)
        log_extra("exact_match", exact_scores)
        log_extra("formatting_score", formatting_scores)
        log_extra("verifier_score", verifier_scores)
        log_extra("verifier_correctness_label", correctness_labels)
        log_extra("verifier_confidence_label", confidence_labels)

    if log_metric and rewards:
        log_metric("self_reward/verifier_enabled", float(verifier_enabled))
        log_metric("self_reward/exact_match", sum(exact_scores) / len(exact_scores))
        log_metric(
            "self_reward/formatting", sum(formatting_scores) / len(formatting_scores)
        )
        log_metric("self_reward/verifier", sum(verifier_scores) / len(verifier_scores))
        log_metric("self_reward/total", sum(rewards) / len(rewards))

    return rewards
