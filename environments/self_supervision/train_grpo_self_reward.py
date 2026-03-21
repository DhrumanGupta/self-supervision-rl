from __future__ import annotations

import argparse

from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from dataset import build_train_eval_datasets
from rewards import RewardWeights, self_reward_function
from rollout import self_reward_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GRPO with auxiliary same-policy self-rewarding."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="toy")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--question_key", type=str, default="problem")
    parser.add_argument("--answer_key", type=str, default="answer")
    parser.add_argument("--train_examples", type=int, default=-1)
    parser.add_argument("--eval_examples", type=int, default=64)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--enable_verifier_reward",
        action="store_true",
        default=False,
        help="Enable the auxiliary same-model verifier reward pass.",
    )
    parser.add_argument("--exact_match_weight", type=float, default=1.0)
    parser.add_argument("--formatting_weight", type=float, default=0.1)
    parser.add_argument("--verifier_weight", type=float, default=0.2)
    parser.add_argument("--length_penalty_weight", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = build_train_eval_datasets(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        prompt_key=args.prompt_key,
        question_key=args.question_key,
        answer_key=args.answer_key,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        seed=args.seed,
    )

    reward_weights = RewardWeights(
        exact_match=args.exact_match_weight,
        formatting=args.formatting_weight,
        verifier=args.verifier_weight,
        length_penalty=args.length_penalty_weight,
        enable_verifier_reward=args.enable_verifier_reward,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        bf16=args.use_bf16,
        remove_unused_columns=False,
        logging_steps=1,
        save_steps=50,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=50 if eval_dataset is not None else None,
        report_to="none",
        log_completions=True,
    )
    training_args.max_prompt_length = args.max_prompt_length

    peft_config = None
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    def reward_wrapper(**kwargs):
        return self_reward_function(reward_weights=reward_weights, **kwargs)

    trainer = GRPOTrainer(
        model=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_wrapper,
        rollout_func=self_reward_rollout,
        peft_config=peft_config,
    )
    trainer.enable_verifier_reward = args.enable_verifier_reward
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
