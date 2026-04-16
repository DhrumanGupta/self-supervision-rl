#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

RESUME_ARGS=()
if [[ $# -gt 0 ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "$1")
fi

export WANDB_PROJECT=self-supervision-rl
export WANDB_ENTITY=berlm-ashoka-university
export WANDB_NAME=4k-context-qwen35-9b-base-deepmath

# Launch with run.sh, e.g.:
#   CUDA_VISIBLE_DEVICES=2,3 ./run.sh logs/test.log ./scripts/train_grpo.sh

CUDA_VISIBLE_DEVICES=1,2,3,4

# Count length of visible devices
NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "--------------------------------"

accelerate launch --num_processes $NUM_PROCESSES --num_machines 1 --mixed_precision bf16 -m environments.self_supervision.train_grpo_self_reward \
  --model_name Qwen/Qwen3.5-9B-Base \
  --dataset_name zwhe99/DeepMath-103K \
  --question_key question \
  --answer_key final_answer \
  --output_dir outputs/${WANDB_NAME} \
  --report_to wandb \
  --use_peft \
  --use_bf16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_generations 8 \
  --num_generations_eval 4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_steps 1000 \
  --eval_steps 100 \
  --eval_examples -1 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --learning_rate 1e-5 \
  --exact_match_weight 1.0 \
  --formatting_weight 0.2 \
  --length_penalty_weight 1e-4 \
  --save_steps 50 \
  --curriculum_eval_examples_per_band 128 \
  "${RESUME_ARGS[@]}"
