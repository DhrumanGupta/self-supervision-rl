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
        if token_id == eos_token_id or token_id == pad_token_id:
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
    model = (
        trainer.accelerator.unwrap_model(trainer.model)
        if hasattr(trainer, "accelerator")
        else trainer.model
    )
    device = next(model.parameters()).device
    max_prompt_length = getattr(
        getattr(trainer, "args", None), "max_prompt_length", None
    )
    enable_verifier_reward = getattr(trainer, "enable_verifier_reward", True)
    was_training = model.training
    model.eval()

    try:
        rendered_prompts = [
            _render_qwen_prompt(tokenizer, prompt, enable_thinking=True)
            for prompt in prompts
        ]
        first_inputs = _tokenize_texts(
            tokenizer, rendered_prompts, device, max_length=max_prompt_length
        )
        first_generate_start_time = time.perf_counter()
        first_generated = model.generate(
            **first_inputs,
            max_new_tokens=trainer.max_completion_length,
            do_sample=True,
            temperature=trainer.temperature,
            top_p=trainer.top_p,
            top_k=trainer.top_k,
            repetition_penalty=trainer.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        _log_timing_metric(
            trainer,
            "profiling/rollout/main_generate_s",
            time.perf_counter() - first_generate_start_time,
        )

        prompt_ids = []
        for ids, mask in zip(
            first_inputs["input_ids"], first_inputs["attention_mask"], strict=False
        ):
            prompt_ids.append(ids[mask.bool()].tolist())

        completion_ids = _extract_completion_ids(
            first_generated,
            first_inputs["input_ids"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        logprobs = None
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
        self_eval_generate_start_time = time.perf_counter()
        self_eval_generated = model.generate(
            **self_eval_inputs,
            max_new_tokens=trainer.max_completion_length,
            do_sample=True,
            temperature=trainer.temperature,
            top_p=trainer.top_p,
            top_k=trainer.top_k,
            repetition_penalty=trainer.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        _log_timing_metric(
            trainer,
            "profiling/rollout/self_eval_generate_s",
            time.perf_counter() - self_eval_generate_start_time,
        )

        self_eval_completion_ids = _extract_completion_ids(
            self_eval_generated,
            self_eval_inputs["input_ids"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
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
        if was_training:
            model.train()
