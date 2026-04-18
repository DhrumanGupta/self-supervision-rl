from __future__ import annotations

import argparse
import re

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from environments.self_supervision.curriculum import (
    CurriculumCallback,
    CurriculumConfig,
    CurriculumController,
    CurriculumGRPOTrainer,
)
from environments.self_supervision.dataset import (
    DEEP_MATH_BAND_ORDER,
    build_band_indices,
    build_fixed_band_eval_subsets,
    build_train_eval_datasets,
    validate_curriculum_band_coverage,
)
from environments.self_supervision.rewards import RewardWeights, self_reward_function
from environments.self_supervision.rollout import self_reward_rollout
from environments.self_supervision.trainer import (
    ProfilingCallback,
    SelfSupervisionGRPOTrainer,
)


SUPPORTED_MODEL_NAME_PATTERN = re.compile(r"^Qwen/Qwen3\.5-[^/]+-Base$")
DEEP_MATH_DATASET_NAME = "zwhe99/DeepMath-103K"


def validate_supported_model_name(model_name: str) -> None:
    if SUPPORTED_MODEL_NAME_PATTERN.fullmatch(model_name):
        return

    raise ValueError(
        "Unsupported model_name "
        f"{model_name!r}. This training entrypoint currently supports only "
        "Qwen/Qwen3.5-*-Base models because the custom rollout depends on "
        "Qwen's apply_chat_template(..., enable_thinking=...) behavior."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GRPO with auxiliary same-policy self-rewarding."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Supported: Qwen/Qwen3.5-*-Base",
    )
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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory to resume training from.",
    )
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--num_generations_eval", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--disable_curriculum",
        action="store_true",
        help="Disable the default difficulty curriculum for DeepMath runs.",
    )
    parser.add_argument(
        "--curriculum_eval_examples_per_band",
        type=int,
        default=512,
        help="Fixed number of evaluation examples per difficulty band.",
    )
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
    parser.add_argument("--verifier_weight", type=float, default=0.2)
    parser.add_argument("--length_penalty_weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def should_enable_curriculum(args: argparse.Namespace) -> bool:
    return args.dataset_name == DEEP_MATH_DATASET_NAME and not args.disable_curriculum


def main() -> None:
    args = parse_args()
    validate_supported_model_name(args.model_name)
    use_curriculum = should_enable_curriculum(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")

    eval_examples = -1 if use_curriculum else args.eval_examples
    train_dataset, eval_dataset = build_train_eval_datasets(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        prompt_key=args.prompt_key,
        question_key=args.question_key,
        answer_key=args.answer_key,
        train_examples=args.train_examples,
        eval_examples=eval_examples,
        seed=args.seed,
    )

    curriculum_controller = None
    curriculum_eval_datasets = None
    if use_curriculum:
        if eval_dataset is None:
            raise ValueError("DeepMath curriculum requires an evaluation dataset.")

        curriculum_config = CurriculumConfig.default_deepmath(
            eval_examples_per_band=args.curriculum_eval_examples_per_band
        )
        curriculum_controller = CurriculumController(curriculum_config)
        train_band_indices = build_band_indices(train_dataset)
        curriculum_eval_datasets = build_fixed_band_eval_subsets(
            eval_dataset,
            per_band_limit=curriculum_config.eval_examples_per_band,
            seed=args.seed,
        )
        required_train_bands = [
            band
            for band in DEEP_MATH_BAND_ORDER
            if any(
                stage.sampling_weights.get(band, 0.0) > 0.0
                for stage in curriculum_config.stages
            )
        ]
        required_eval_bands = [
            band
            for band in DEEP_MATH_BAND_ORDER
            if any(stage.frontier_band == band for stage in curriculum_config.stages)
        ]
        validate_curriculum_band_coverage(
            train_band_indices=train_band_indices,
            eval_band_datasets=curriculum_eval_datasets,
            required_train_bands=required_train_bands,
            required_eval_bands=required_eval_bands,
        )
        eval_dataset = curriculum_eval_datasets[
            curriculum_controller.current_frontier_band
        ]

    reward_weights = RewardWeights(
        exact_match=args.exact_match_weight,
        verifier=args.verifier_weight,
        length_penalty=args.length_penalty_weight,
        enable_verifier_reward=args.enable_verifier_reward,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
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
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        report_to=args.report_to,
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

    def configured_reward_func(**kwargs):
        return self_reward_function(reward_weights=reward_weights, **kwargs)

    configured_reward_func.__name__ = self_reward_function.__name__

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "reward_funcs": configured_reward_func,
        "rollout_func": self_reward_rollout,
        "peft_config": peft_config,
    }
    if use_curriculum:
        trainer = CurriculumGRPOTrainer(
            curriculum_controller=curriculum_controller,
            curriculum_train_band_indices=train_band_indices,
            curriculum_eval_datasets=curriculum_eval_datasets,
            **trainer_kwargs,
        )
        trainer.add_callback(CurriculumCallback(trainer))
    else:
        trainer = SelfSupervisionGRPOTrainer(**trainer_kwargs)

    trainer.add_callback(ProfilingCallback(trainer))

    trainer.enable_verifier_reward = args.enable_verifier_reward
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
