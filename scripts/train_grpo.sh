#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

accelerate launch environments/self_supervision/train_grpo_self_reward.py \
  --model_name Qwen/Qwen3.5-2B-Base \
  --dataset_name trl-lib/DeepMath-103K \
  --question_key problem \
  --answer_key answer \
  --output_dir outputs/qwen35-2b-base-deepmath-grpo-peft \
  --use_peft \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 100 \
  --max_prompt_length 1024 \
  --max_completion_length 256 \
  --learning_rate 1e-5 \
  --exact_match_weight 1.0 \
  --formatting_weight 0.1
