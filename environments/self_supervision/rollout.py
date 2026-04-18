from __future__ import annotations

import time
from typing import Any

import torch

from environments.self_supervision.prompts import build_self_eval_messages


def _render_qwen_prompt(
    tokenizer, messages: list[dict[str, str]], *, enable_thinking: bool
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _tokenize_texts(
    tokenizer, texts: list[str], device: torch.device, max_length: int | None = None
) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in batch.items()}


def _trim_completion_ids(
    token_ids: list[int], *, eos_token_id: int | None, pad_token_id: int | None
) -> list[int]:
    trimmed_ids = []
    for token_id in token_ids:
        if token_id == eos_token_id:
            trimmed_ids.append(token_id)
            break
        if token_id == pad_token_id:
            break
        trimmed_ids.append(token_id)
    return trimmed_ids


def _extract_completion_ids(
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> list[list[int]]:
    prompt_length = input_ids.size(1)
    completion_ids = []
    for row in generated_ids:
        completion_ids.append(
            _trim_completion_ids(
                row[prompt_length:].tolist(),
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        )
    return completion_ids


def _decode_sequences(tokenizer, token_ids: list[list[int]]) -> list[str]:
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def _flatten_sampled_logprobs(
    logprobs: list[list[list[float | None]]] | None,
) -> list[list[float | None]] | None:
    if logprobs is None:
        return None
    return [[token_logprobs[0] for token_logprobs in sequence] for sequence in logprobs]


def _build_prompt_ids(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> list[list[int]]:
    prompt_ids = []
    for ids, mask in zip(input_ids, attention_mask, strict=False):
        prompt_ids.append(ids[mask.bool()].tolist())
    return prompt_ids


def _log_timing_metric(trainer, name: str, duration_seconds: float) -> None:
    if hasattr(trainer, "_log_metric"):
        trainer._log_metric(name, float(duration_seconds))


@torch.no_grad()
def self_reward_rollout(prompts: list[list[dict[str, str]]], trainer) -> dict[str, Any]:
    rollout_start_time = time.perf_counter()
    tokenizer = (
        trainer.processing_class.tokenizer
        if hasattr(trainer.processing_class, "tokenizer")
        else trainer.processing_class
    )
    device = (
        trainer.accelerator.device
        if hasattr(trainer, "accelerator")
        else torch.device("cpu")
    )
    max_prompt_length = getattr(
        getattr(trainer, "args", None), "max_prompt_length", None
    )
    enable_verifier_reward = getattr(trainer, "enable_verifier_reward", True)
    mode = "train" if getattr(trainer.model, "training", True) else "eval"
    num_generations = (
        trainer.num_generations if mode == "train" else trainer.num_generations_eval
    )

    try:
        rendered_prompts = [
            _render_qwen_prompt(tokenizer, prompt, enable_thinking=True)
            for prompt in prompts
        ]
        first_inputs = _tokenize_texts(
            tokenizer, rendered_prompts, device, max_length=max_prompt_length
        )
        prompt_ids = _build_prompt_ids(
            first_inputs["input_ids"], first_inputs["attention_mask"]
        )
        first_generate_start_time = time.perf_counter()
        prompt_ids, completion_ids, logprobs, _ = trainer.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=num_generations,
        )
        _log_timing_metric(
            trainer,
            "profiling/rollout/main_generate_s",
            time.perf_counter() - first_generate_start_time,
        )
        logprobs = _flatten_sampled_logprobs(logprobs)
        first_completion_text = _decode_sequences(tokenizer, completion_ids)

        if not enable_verifier_reward:
            return {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": logprobs,
                "rendered_prompt_text": rendered_prompts,
                "first_completion_text": first_completion_text,
            }

        self_eval_prompts = [
            build_self_eval_messages(prompt_messages=prompt, answer_text=completion)
            for prompt, completion in zip(prompts, first_completion_text, strict=False)
        ]
        rendered_self_eval_prompts = [
            _render_qwen_prompt(tokenizer, prompt, enable_thinking=False)
            for prompt in self_eval_prompts
        ]
        self_eval_inputs = _tokenize_texts(
            tokenizer,
            rendered_self_eval_prompts,
            device,
            max_length=max_prompt_length,
        )
        self_eval_prompt_ids = _build_prompt_ids(
            self_eval_inputs["input_ids"], self_eval_inputs["attention_mask"]
        )
        self_eval_generate_start_time = time.perf_counter()
        _, self_eval_completion_ids, _, _ = trainer.vllm_generation.generate(
            prompts=self_eval_prompt_ids,
            images=None,
            num_generations=1,
        )
        _log_timing_metric(
            trainer,
            "profiling/rollout/self_eval_generate_s",
            time.perf_counter() - self_eval_generate_start_time,
        )
        self_eval_text = _decode_sequences(tokenizer, self_eval_completion_ids)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "rendered_prompt_text": rendered_prompts,
            "first_completion_text": first_completion_text,
            "self_eval_text": self_eval_text,
            "self_eval_prompt": self_eval_prompts,
        }
    finally:
        _log_timing_metric(
            trainer,
            "profiling/rollout/total_s",
            time.perf_counter() - rollout_start_time,
        )
