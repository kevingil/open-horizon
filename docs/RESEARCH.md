# Research engineering guide

This is the deeper-than-the-main-README doc for people training models
against this stack. The main `README.md` covers the Mac dev loop,
observability, and the rollout-only flow; everything that needs a real
GPU lives here.

If you're just here to drive rollouts, read `README.md`.
If you're here to train, calibrate, or deploy distributed inference, read on.

## Table of contents

- [Mental model](#mental-model)
- [Hardware floor by phase](#hardware-floor-by-phase)
- [Profiles: per-run inference targets](#profiles-per-run-inference-targets)
- [Training data: TurnTrainingRecord + the recorder](#training-data-turntrainingrecord--the-recorder)
- [Training recipe: prime-rl + verifiers + SGLang](#training-recipe-prime-rl--verifiers--sglang)
- [Distributed SGLang](#distributed-sglang)
- [Adapter lifecycle](#adapter-lifecycle)
- [Eval harness](#eval-harness)
- [Test discipline](#test-discipline)

---

## Mental model

Three concerns, three layers, no overlap:

| Layer | Job | What we own |
|---|---|---|
| **Inference** | Serve the policy via OpenAI-compat | `scripts/serve_sglang.sh`, optional admin sidecar |
| **Rollout loop** | Drive the agent through N turns | `verifiers.run_rollout` (prod) or `run_repo_rollout` (repo path) |
| **Training** | Update the policy from rollouts | `prime-rl` shelled out from `PrimeRLTrainer` |

What we **don't** own: a custom serving stack, a custom rollout loop
for long-horizon agentic envs, or a custom GRPO implementation. Those
all live in upstream projects (SGLang, verifiers, prime-rl). This repo
contributes the orchestration around them: per-run profiles, a
training recorder, an event bus, an artifact store, an adapter
registry, and an observability dashboard.

---

## Hardware floor by phase

| Phase | What's running | Floor |
|---|---|---|
| Mac dev loop | `make dev` against a static or remote policy | macOS, 16 GB RAM |
| Single-GPU rollout via SGLang | SGLang + verifiers env | 1× H100/H200 (80 GB) for Qwen3-8B; 1× A100 40 GB for Qwen3-0.6B |
| Single-GPU GRPO (in-tree, dev-loop reference) | `RL_TRAINER_BACKEND=grpo` | 1× A100 40 GB; throughput-limited |
| Multi-GPU prime-rl training | `RL_TRAINER_BACKEND=prime-rl` | 8× H100/H200, NVLink, CUDA 12.4, flash-attn 2.7+ |
| Multi-node prime-rl | prime-rl with FSDP across nodes | NCCL across nodes, 200+ Gbps interconnect |
| Distributed SGLang DP | N replicas behind a router | 1+ GPUs per replica; sticky-by-conversation LB recommended |

Mac dev never reaches for anything heavy. `[envs]` adds verifiers;
`[train]` adds Torch + transformers + peft for the in-tree GRPO and
the `TokenizedTrainingRecorder`; `[prime-rl]` adds the prime-rl CLI.
Default install stays Mac-runnable with none of these.

---

## Profiles: per-run inference targets

Profiles solve the "I want to A/B this task across providers without
recompiling the app" problem. Define them once in settings, pick one
per rollout via `RolloutRequest.policy_profile`. Unknown profile names
get rejected as 422 at request time, never mid-rollout.

Configure via `RL_POLICY_PROFILES` (JSON) plus `RL_DEFAULT_POLICY_PROFILE`:

```bash
export RL_POLICY_PROFILES='{
  "default": {
    "base_url": "http://127.0.0.1:30000/v1",
    "model": "sglang:Qwen/Qwen3-8B",
    "routes_to": "verifiers",
    "env_id": "vf-math"
  },
  "anthropic-haiku": {
    "base_url": "https://api.anthropic.com/v1",
    "model": "claude-haiku-4-5",
    "api_key_env": "ANTHROPIC_API_KEY",
    "routes_to": "repo"
  },
  "oai-mini": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-5.4-mini",
    "api_key_env": "OPENAI_API_KEY",
    "routes_to": "repo"
  }
}'
export RL_DEFAULT_POLICY_PROFILE=default
```

`api_key_env` reads the named environment variable at runtime; literal
`api_key` is supported but discouraged. `routes_to` decides whether
the rollout runs through verifiers or the in-house repo loop.

`GET /api/profiles` returns the configured profiles with API keys
redacted, so a frontend can build a profile picker.

---

## Training data: TurnTrainingRecord + the recorder

Long-horizon rollouts produce a lot of token data per turn. Storing
that in `TrajectoryStep` would bloat `RunDetail` payloads and force
the dashboard to drag bytes nobody asked for. We store training
metadata in its own table, joined to steps by `(run_id, step_index)`,
and only fetch it on demand.

### The shape

```python
class TurnTrainingRecord:
    id: str
    run_id: str
    step_index: int
    prompt_ids: list[int]      # tokenizer ids
    completion_ids: list[int]
    attention_mask: list[int]  # 1s over real tokens
    loss_mask: list[int]       # 1 over completion span only
    sampling_args: dict        # max_tokens etc. used at this turn
    model_name: str            # which model produced this turn
    token_count: int           # cheap aggregate without loading bytes
    created_at: datetime
```

The trajectory step gains a single `has_training_metadata: bool` flag
so consumers can tell whether a row is fetchable without paying the
bytes cost.

### Wiring

```python
from infrastructure.training.recorder import TokenizedTrainingRecorder
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
coordinator.training_recorder = TokenizedTrainingRecorder(tokenizer=tokenizer)
```

Default is `None`, which means no training metadata gets captured.
Wire the recorder only when you intend to train against the run.

### Querying

```bash
# Lightweight index: step indexes, token counts, no payload bytes
GET /api/runs/run-abc123/turns

# Full payload for one turn
GET /api/runs/run-abc123/turns/4/training
```

Trainer dataset builders should use these endpoints rather than
loading `RunDetail` and reaching across the trajectory.

### Storage backends

`InMemoryArtifactStore` keeps records in a `dict[(run_id, step_index)]`.
`SqliteArtifactStore` writes to a dedicated `turn_training` table with
token id arrays packed as 4-byte signed ints (`array.array.tobytes()`).
The `ArtifactStore` ABC's defaults are no-ops so backends that don't
support training data can opt out cleanly.

Postgres is on the roadmap (own follow-up phase) once SQLite hurts.

### Pruning

We don't. The plan is to keep all historical training data; SQLite can
absorb it until volume warrants Postgres. If you need to prune, do it
out-of-band (`DELETE FROM turn_training WHERE created_at < ...`).

---

## Training recipe: prime-rl + verifiers + SGLang

The production loop. End-to-end:

1. **Serve** a base model on SGLang. RadixAttention prefix caching
   makes multi-turn rollouts cheap.
2. **Roll out** via the verifiers backend. Each rollout writes
   trajectory + reward to the artifact store. With the recorder
   wired, each turn also writes a `TurnTrainingRecord`.
3. **Kick off training** (`POST /api/training-runs`) selecting the
   sample rollouts. `PrimeRLTrainer` pulls TurnTrainingRecord rows,
   writes a JSONL dataset + TOML config, and shells out to
   `prime-rl train`.
4. **Stream metrics** back via the bus. The trainer parses
   `step=N loss=F` lines from prime-rl stdout.
5. **Register** the produced adapter at `RL_ADAPTERS_DIR/<id>`.
6. **Hot-load** into SGLang via the admin sidecar; the new adapter is
   immediately addressable as `sglang:<adapter_id>`.
7. **Eval** the adapter via `rl-eval --adapter <id>` (or `POST
   /api/adapters/<id>/eval`), which routes a fixed task set through
   the new model id.
8. **Iterate**: the next training run can use this adapter as the
   parent and continue from there.

### Setup

```bash
pip install -e '.[envs,train,prime-rl]'

# Terminal 1: serving
SGLANG_MODEL=Qwen/Qwen3-8B \
SGLANG_TP=4 \
    scripts/serve_sglang.sh

# Terminal 2: app + dashboard
RL_ENV_BACKEND=verifiers \
RL_VERIFIERS_ENV_ID=vf-math \
RL_LLM_BASE_URL=http://127.0.0.1:30000/v1 \
RL_LLM_MODEL=sglang:Qwen/Qwen3-8B \
RL_TRAINER_BACKEND=prime-rl \
RL_PRIME_RL_BASE_MODEL=Qwen/Qwen3-8B \
RL_SGLANG_ADMIN_URL=http://127.0.0.1:30000 \
RL_SGLANG_AUTOLOAD_LORA=true \
    make dev
```

### Custom prime-rl config

If the default flat-key TOML the trainer emits isn't what your
prime-rl version wants, supply a template:

```toml
# ./configs/prime-rl-template.toml
[policy]
model = "{{base_model}}"

[data]
path = "{{dataset_path}}"

[trainer]
output_dir = "{{output_dir}}"
resume_adapter = "{{parent_adapter_path}}"
```

Then:

```bash
export RL_PRIME_RL_CONFIG_TEMPLATE=./configs/prime-rl-template.toml
```

The trainer substitutes `{{base_model}}`, `{{dataset_path}}`,
`{{output_dir}}`, `{{parent_adapter_id}}`, `{{parent_adapter_path}}`,
and `{{training_run_id}}` per run.

### Why no verl

Verifiers' env-owns-the-loop assumption clashes with verl's
trainer-owns-the-loop architecture. Bridging requires real adapter
work for algorithms (PPO, RLOO, REINFORCE++, DPO, SFT) we don't need
for our long-horizon agentic task shape. GRPO via prime-rl is the
right algorithm for multi-turn rubric-scored rollouts; we revisit
verl only if a future task creates real demand for the algorithms
it would unlock.

---

## Distributed SGLang

Three deployment shapes, picked by your throughput / latency budget:

### Single node, multi-GPU

Tensor parallel sharding across GPUs on one box:

```bash
SGLANG_MODEL=Qwen/Qwen3-32B \
SGLANG_TP=4 \
    scripts/serve_sglang.sh
```

Add `SGLANG_PP=2` for pipeline parallel on top of TP if the model
doesn't fit even sharded. No code change on our side.

### Multi-node

SGLang's own NCCL-based init:

```bash
# head node
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-72B \
    --tp 8 --pp 2 \
    --dist-init-addr 10.0.0.1:50000 \
    --node-rank 0 --nnodes 2

# worker node
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-72B \
    --tp 8 --pp 2 \
    --dist-init-addr 10.0.0.1:50000 \
    --node-rank 1 --nnodes 2
```

We just point `RL_LLM_BASE_URL` at the head node. The topology is
opaque from our coordinator's POV.

### Data-parallel replicas behind a router

For throughput. RadixAttention prefix caching is per-replica, so a
sticky-by-conversation-id load balancer is the right pattern. SGLang
ships `sglang_router` for this — drop-in OpenAI-compat in front of N
SGLang nodes, with prefix-cache locality across replicas. Worth a
note here, not a fork.

---

## Adapter lifecycle

```
training run completes
        │
        ▼
AdapterRegistry.register(adapter_id, path=RL_ADAPTERS_DIR/<id>)
        │
        ▼
AdapterPublished event on the bus
        │
        ▼
SGLang LoRA hot-reload sidecar (when configured) POSTs to
        /load_lora_adapter
        │
        ▼
RL_LLM_MODEL=sglang:<adapter_id> now routes to it
        │
        ▼
rl-eval --adapter <id> runs the eval task set through the new model
        │
        ▼
EvalReport persisted; the next training run can use this as parent
```

The hot-reload sidecar is best-effort: failures log and move on, the
adapter remains on disk so the next SGLang launch picks it up via
`--lora-paths`.

---

## Eval harness

`rl-eval --adapter <id>` (or `POST /api/adapters/<id>/eval`) routes a
fixed task set through whatever model `RL_LLM_MODEL` resolves to. To
eval a freshly trained adapter:

```bash
RL_LLM_MODEL=sglang:adapter-aaa rl-eval --adapter adapter-aaa
```

Or supply a custom task set via the API body. The harness just
re-uses the rollout coordinator under the hood, so all the same
budget / cancel / observability machinery applies.

---

## Test discipline

The repo's hard rule: **no paid inference in tests, ever.** Smoke
tests opt in via `RL_*_SMOKE` env vars and must live under
`tests/smoke/`. Everything under `tests/unit/` and
`tests/integration/` runs offline against `FakeOpenAI` or stubs.

The conftest enforces this with an autouse fixture that monkeypatches
`openai.OpenAI` to refuse construction unless `base_url` is
`localhost` / `127.0.0.1` / `stub://` or the test lives in
`tests/smoke/`. If you see `AssertionError: Test ... attempted to
construct openai.OpenAI(base_url=...)`, you've leaked a paid call into
a non-smoke test. Use `FakeOpenAI` from `tests/_fakes/openai_compat`
or pre-populate `coordinator._client_cache` directly.

CI never sets `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `RL_LLM_API_KEY`
in the test environment.

---

## Recap

- Inference: SGLang only, OpenAI-compat at the seam, distributed via
  SGLang's own knobs.
- Rollouts: verifiers (long-horizon agentic) or repo loop (sandboxed
  shell against a snapshot).
- Training: prime-rl, GRPO, fed by `TurnTrainingRecord` rows from the
  artifact store.
- Profiles: named per-run inference targets, validated up front.
- Recorder: optional per-turn token-level capture; default off.
- Adapter lifecycle: hot-reload sidecar bridges training to eval.

If something here is out of date, the source is the truth - this doc
gets refreshed when the underlying integration drifts.
