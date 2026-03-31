#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

export WANDB_PROJECT=self-supervision-rl
export WANDB_ENTITY=berlm-ashoka-university
export WANDB_NAME=8k-context-qwen35-2b-grpo-deepmath

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# Count length of visible devices
NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "--------------------------------"

accelerate launch --num_processes $NUM_PROCESSES --num_machines 1 --mixed_precision bf16 -m environments.self_supervision.train_grpo_self_reward \
  --model_name Qwen/Qwen3.5-2B-Base \
  --dataset_name trl-lib/DeepMath-103K \
  --question_key problem \
  --answer_key answer \
  --output_dir outputs/4k-context-qwen35-2b-base-deepmath-grpo-peft \
  --report_to wandb \
  --use_peft \
  --use_bf16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_generations 6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 1000 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --learning_rate 1e-5 \
  --exact_match_weight 1.0 \
  --formatting_weight 0.1 \
  --length_penalty_weight 1e-4 \
  --save_steps 1
