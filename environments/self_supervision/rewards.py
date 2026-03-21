from __future__ import annotations

from dataclasses import dataclass

from parsers import (
    extract_final_answer,
    parse_confidence_label,
    parse_correctness_label,
)


@dataclass(frozen=True)
class RewardWeights:
    exact_match: float = 1.0
    self_consistency: float = 0.25
    confidence: float = 0.15
    length_penalty: float = 0.0001
    enable_verifier_reward: bool = True


def self_reward_function(
    prompts,
    completions,
    answer,
    first_completion_text,
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
    consistency_scores = []
    confidence_scores = []

    if self_eval_text is None:
        self_eval_text = [""] * len(first_completion_text)

    for completion_text, probe_text, gold in zip(
        first_completion_text, self_eval_text, answer, strict=False
    ):
        predicted_answer = extract_final_answer(completion_text)
        exact_match = 1.0 if predicted_answer == str(gold).strip() else 0.0

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

        total = (
            weights.exact_match * exact_match
            + weights.self_consistency * self_consistency
            + weights.confidence * confidence_score
            - weights.length_penalty * len(completion_text or "")
        )
        rewards.append(float(total))
        exact_scores.append(exact_match)
        consistency_scores.append(self_consistency)
        confidence_scores.append(confidence_score)

    if log_extra:
        log_extra("gold_answer", [str(item) for item in answer])
        log_extra("first_completion_text", list(first_completion_text))
        if verifier_enabled:
            log_extra("self_eval_text", list(self_eval_text))

    if log_metric and rewards:
        log_metric("self_reward/verifier_enabled", float(verifier_enabled))
        log_metric("self_reward/exact_match", sum(exact_scores) / len(exact_scores))
        log_metric(
            "self_reward/self_consistency",
            sum(consistency_scores) / len(consistency_scores),
        )
        log_metric(
            "self_reward/confidence", sum(confidence_scores) / len(confidence_scores)
        )
        log_metric("self_reward/total", sum(rewards) / len(rewards))

    return rewards
