from __future__ import annotations

from typing import Any

import torch

from prompts import build_self_eval_messages


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


def _extract_completion_ids(
    generated_ids: torch.Tensor, attention_mask: torch.Tensor
) -> list[list[int]]:
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    completion_ids = []
    for row, prompt_length in zip(generated_ids, prompt_lengths, strict=False):
        completion_ids.append(row[int(prompt_length) :].tolist())
    return completion_ids


def _decode_sequences(tokenizer, token_ids: list[list[int]]) -> list[str]:
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def _repeat_prompts(
    prompts: list[list[dict[str, str]]], num_generations: int
) -> list[list[dict[str, str]]]:
    repeated = []
    for prompt in prompts:
        for _ in range(num_generations):
            repeated.append(prompt)
    return repeated


@torch.no_grad()
def self_reward_rollout(prompts: list[list[dict[str, str]]], trainer) -> dict[str, Any]:
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
    num_generations = int(getattr(trainer, "num_generations", 0) or 0)
    if num_generations <= 0:
        num_generations = int(
            getattr(getattr(trainer, "args", None), "num_generations", 1)
        )
    was_training = model.training
    model.eval()

    try:
        rollout_prompts = _repeat_prompts(prompts, num_generations)
        rendered_prompts = [
            _render_qwen_prompt(tokenizer, prompt, enable_thinking=True)
            for prompt in rollout_prompts
        ]
        first_inputs = _tokenize_texts(
            tokenizer, rendered_prompts, device, max_length=max_prompt_length
        )
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

        prompt_ids = []
        for ids, mask in zip(
            first_inputs["input_ids"], first_inputs["attention_mask"], strict=False
        ):
            prompt_ids.append(ids[mask.bool()].tolist())

        completion_ids = _extract_completion_ids(
            first_generated, first_inputs["attention_mask"]
        )
        first_completion_text = _decode_sequences(tokenizer, completion_ids)

        if not enable_verifier_reward:
            return {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": None,
                "first_completion_text": first_completion_text,
            }

        self_eval_prompts = [
            build_self_eval_messages(prompt_messages=prompt, answer_text=completion)
            for prompt, completion in zip(
                rollout_prompts, first_completion_text, strict=False
            )
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
        self_eval_generated = model.generate(
            **self_eval_inputs,
            max_new_tokens=min(128, trainer.max_completion_length),
            do_sample=True,
            temperature=max(0.2, trainer.temperature),
            top_p=trainer.top_p,
            top_k=trainer.top_k,
            repetition_penalty=trainer.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        self_eval_completion_ids = _extract_completion_ids(
            self_eval_generated, self_eval_inputs["attention_mask"]
        )
        self_eval_text = _decode_sequences(tokenizer, self_eval_completion_ids)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": None,
            "first_completion_text": first_completion_text,
            "self_eval_text": self_eval_text,
            "self_eval_prompt": self_eval_prompts,
        }
    finally:
        if was_training:
            model.train()
