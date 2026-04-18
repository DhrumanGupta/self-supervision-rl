#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

RESUME_ARGS=()
if [[ $# -gt 0 ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "$1")
fi

export WANDB_PROJECT=self-supervision-rl
export WANDB_ENTITY=berlm-ashoka-university
export WANDB_NAME=4k-context-qwen35-9b-base-deepmath-vllm

export MODEL_NAME="Qwen/Qwen3.5-9B-Base"

export VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-0,1}
export TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-2,3,4,5,6}
export SELF_SUPERVISION_VLLM_SERVER_PORT=${SELF_SUPERVISION_VLLM_SERVER_PORT:-8000}
export SELF_SUPERVISION_VLLM_GROUP_PORT=${SELF_SUPERVISION_VLLM_GROUP_PORT:-51216}
export SELF_SUPERVISION_VLLM_MAX_MODEL_LEN=${SELF_SUPERVISION_VLLM_MAX_MODEL_LEN:-5120}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

OUTPUT_DIR="outputs/${WANDB_NAME}"
VLLM_LOG_PATH="${OUTPUT_DIR}/logs/vllm_server.log"

mkdir -p "${OUTPUT_DIR}/logs"

# Launch with run.sh, e.g.:
#   ./run.sh logs/test.log ./scripts/train_grpo.sh

NUM_PROCESSES=$(printf '%s\n' "$TRAIN_CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

cleanup() {
  if [[ -n "${VLLM_SERVER_PID:-}" ]] && kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "VLLM_CUDA_VISIBLE_DEVICES: $VLLM_CUDA_VISIBLE_DEVICES"
echo "TRAIN_CUDA_VISIBLE_DEVICES: $TRAIN_CUDA_VISIBLE_DEVICES"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "VLLM_GROUP_PORT: $SELF_SUPERVISION_VLLM_GROUP_PORT"
echo "VLLM_MAX_MODEL_LEN: $SELF_SUPERVISION_VLLM_MAX_MODEL_LEN"
echo "VLLM_WORKER_MULTIPROC_METHOD: $VLLM_WORKER_MULTIPROC_METHOD"
echo "VLLM_LOG_PATH: $VLLM_LOG_PATH"
echo "--------------------------------"

CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" \
  trl vllm-serve \
  --model "$MODEL_NAME" \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --host 127.0.0.1 \
  --port "$SELF_SUPERVISION_VLLM_SERVER_PORT" \
  --max-model-len "$SELF_SUPERVISION_VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  > "$VLLM_LOG_PATH" 2>&1 &

VLLM_SERVER_PID=$!
echo "Started vLLM server with PID: $VLLM_SERVER_PID"

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${SELF_SUPERVISION_VLLM_SERVER_PORT}/health/" >/dev/null; then
    echo "vLLM server is healthy"
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${SELF_SUPERVISION_VLLM_SERVER_PORT}/health/" >/dev/null; then
  echo "Timed out waiting for vLLM server health on port ${SELF_SUPERVISION_VLLM_SERVER_PORT}"
  exit 1
fi

CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
accelerate launch --num_processes "$NUM_PROCESSES" --num_machines 1 --mixed_precision bf16 -m environments.self_supervision.train_grpo_self_reward \
  --model_name "$MODEL_NAME" \
  --dataset_name zwhe99/DeepMath-103K \
  --question_key question \
  --answer_key final_answer \
  --output_dir "$OUTPUT_DIR" \
  --report_to wandb \
  --use_peft \
  --use_bf16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_generations 12 \
  --num_generations_eval 2 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 3 \
  --max_steps 100000 \
  --eval_steps 200 \
  --eval_examples -1 \
  --temperature 0.6 \
  --top_p 0.95 \
  --learning_rate 5e-6 \
  --exact_match_weight 1.0 \
  --length_penalty_weight 1e-5 \
  --save_steps 50 \
  --curriculum_eval_examples_per_band 64 \
  "${RESUME_ARGS[@]}"
