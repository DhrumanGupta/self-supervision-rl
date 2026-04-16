#!/usr/bin/env bash

set -euo pipefail

# Run like ./run.sh <log-path> <command> [args...]
# Will run with nohup and write stdout/stderr to the log path.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <log-path> <command> [args...]"
  exit 1
fi

log_path="$1"
shift

source .venv/bin/activate

mkdir -p "$(dirname "$log_path")"

nohup "$@" > "$log_path" 2>&1 < /dev/null &

runner_pid=$!
echo "Running $* with nohup, logging to $log_path"
echo "Runner PID: $runner_pid"
echo "Command: $*"
echo "Path: $log_path"
