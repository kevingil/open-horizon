#!/usr/bin/env bash
# End-to-end smoke test: mock policy + horizon serve + Vite dashboard,
# driven through the browser with Playwright. Produces screenshots under
# $SMOKE_OUT (default ./artifacts/smoke).
#
#   scripts/smoke/run.sh
#
# Requires: cargo, node, and a global or local playwright install with
# Chromium available (PLAYWRIGHT_BROWSERS_PATH honoured).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${SMOKE_OUT:-$ROOT/artifacts/smoke}"
ART="$OUT/state"
API_PORT="${SMOKE_API_PORT:-8000}"
MOCK_PORT="${SMOKE_MOCK_PORT:-8090}"
UI_PORT="${SMOKE_UI_PORT:-5173}"
mkdir -p "$OUT" "$ART"
rm -f "$ART"/runs.db*

cargo build -q --release -p horizon-server
BIN="$ROOT/target/release/horizon"

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

# Refuse to run against a stale server from an earlier invocation.
if curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
  echo "port $API_PORT already serving; stop it or set SMOKE_API_PORT" >&2
  exit 1
fi

"$BIN" mock-policy --bind "127.0.0.1:$MOCK_PORT" --latency-ms "${SMOKE_LATENCY_MS:-500}" > "$OUT/mock-policy.log" 2>&1 &
pids+=($!)

(
  cd "$ROOT"
  RL_STORE_BACKEND=sqlite RL_ARTIFACTS_DIR="$ART" RL_ADAPTERS_DIR="$ART/adapters" \
  RL_ENV_BACKEND=repo RL_ENV_SANDBOX=none RL_WORKSPACE_ROOT="$ROOT" \
  RL_LLM_BASE_URL="http://127.0.0.1:$MOCK_PORT/v1" RL_LLM_API_KEY=not-needed RL_LLM_MODEL=gpt-4.1-mini \
  RL_MAX_PARALLEL_ROLLOUTS=3 RL_DAILY_BUDGET_USD=2.0 RL_TRAIN_STEP_DELAY_S=0.25 \
  RL_PORT="$API_PORT" RL_LOG_LEVEL=info \
  exec "$BIN" serve > "$OUT/horizon.log" 2>&1
) &
pids+=($!)

(cd "$ROOT/frontend" && VITE_API_BASE="http://127.0.0.1:$API_PORT" exec node node_modules/vite/bin/vite.js --port "$UI_PORT" --strictPort > "$OUT/vite.log" 2>&1) &
pids+=($!)

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null && curl -sf "http://127.0.0.1:$UI_PORT" >/dev/null && break
  sleep 0.5
done

NODE_PATH="${NODE_PATH:-$(npm root -g)}" \
SMOKE_API="http://127.0.0.1:$API_PORT" SMOKE_UI="http://127.0.0.1:$UI_PORT" SMOKE_OUT="$OUT" \
  node "$ROOT/scripts/smoke/drive.mjs"
