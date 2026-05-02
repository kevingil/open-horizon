#!/usr/bin/env bash
# Launch SGLang as the policy server for the RL stack.
#
# Usage:
#   scripts/serve_sglang.sh [extra sglang args...]
#
# Environment overrides:
#   SGLANG_MODEL          model path or HF id (default: Qwen/Qwen2.5-7B-Instruct)
#   SGLANG_PORT           HTTP port (default: 30000)
#   SGLANG_HOST           bind host (default: 127.0.0.1)
#   SGLANG_MAX_LORAS      --max-loras-per-batch (default: 4)
#   SGLANG_LORA_PATHS     space-separated id=path pairs to hot-mount LoRAs.
#                         Example: "adapter-aaa=./artifacts/adapters/adapter-aaa"
#
# After it boots, point the stack at it:
#   RL_POLICY_BACKEND=openai \
#   RL_LLM_BASE_URL=http://127.0.0.1:30000/v1 \
#   RL_LLM_API_KEY=not-needed \
#   RL_LLM_MODEL=sglang:Qwen/Qwen2.5-7B-Instruct \
#       make dev
set -euo pipefail

MODEL="${SGLANG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${SGLANG_PORT:-30000}"
HOST="${SGLANG_HOST:-127.0.0.1}"
MAX_LORAS="${SGLANG_MAX_LORAS:-4}"

args=(
  --model-path "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --enable-lora
  --max-loras-per-batch "$MAX_LORAS"
)

if [[ -n "${SGLANG_LORA_PATHS:-}" ]]; then
  # shellcheck disable=SC2206
  lora_pairs=($SGLANG_LORA_PATHS)
  args+=(--lora-paths "${lora_pairs[@]}")
fi

exec python -m sglang.launch_server "${args[@]}" "$@"
