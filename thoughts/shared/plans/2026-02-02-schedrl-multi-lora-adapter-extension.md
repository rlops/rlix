# SchedRL Multi‑LoRA Adapter Extension (vLLM-first, ROLL + SkyRL-train + NeMo-RL) Implementation Plan

**Date**: 2026-02-02

## Overview

Extend the shared multi-pipeline protocol (`design_doc/multi-pipeline-adaptation-plan.md`) so **each RL pipeline** can run in either:

1) **Full fine-tune mode** (single evolving base checkpoint, current protocol), or
2) **Multi‑LoRA mode** where the **base model is fixed** and the pipeline trains **multiple LoRA adapters concurrently**, and rollout supports **S-LoRA-style mixed-adapter batching**: a single inference batch may include prompts targeting different adapters, with adapter selection done per request.

### ROLL-first adapter identity (canonical `adapter_id`)
This extension is ROLL-first: ROLL already has per-domain/per-env labels (`tag` in agentic envs and `domain` in async sampling).

Standardize on a single canonical protocol field name:
- `adapter_id` is the **only** protocol-level key for “which LoRA adapter to apply”.
- ROLL maps `adapter_id := env_config["tag"]` (agentic) and `adapter_id := domain` (async) at the coordinator boundary, and treats `adapter_id` as the source of truth thereafter (request IDs, caching, progress metrics, optimizer state).

Core requirement: LoRA weights are **trained + synchronized at adapter granularity**, while shrink/expand time-sharing remains safe and uses SchedRL’s existing primitives (admission control, abort+ACK, offload, expand, selective sync).

This plan focuses on a **vLLM-first** shape because:
- ROLL already plumbs `lora_request` into vLLM generation (`third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:172`).
- SkyRL-train already has LoRA disk-load hooks using vLLM `add_lora` (`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:313`).
- NeMo-RL documents LoRA as “base model frozen” (expected semantics for adapter mode) (`third_party/nemo-rl/docs/guides/sft.md:168`).

---

## Current State Analysis

### What the protocol assumes today (single checkpoint axis)
The current protocol models “weights” as a single monotonic `checkpoint_version` chosen by the coordinator, with:
- `active_checkpoint_version` as the rollout target version.
- a trainer-side CPU checkpoint cache service (“bucket list”) as the source of truth for expand/resume.
- shrink/expand orchestration that assumes a single weight version to sync/activate.

This is sufficient for full fine-tune, but insufficient for **multi-LoRA**, because:
- multiple adapters can update at different times (multi-dimensional versioning),
- rollout needs to select **adapter identity** per request/batch,
- expand-from-zero needs to ensure “base + required adapters” are available before opening admission.

### Reference implementation hooks already exist
- **ROLL**: vLLM strategy builds `LoRARequest` and passes `lora_request=...` into generation (`third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:172`, `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:326`).
- **SkyRL-train**: vLLM engine loads LoRA from disk via `add_lora` (`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:313`) and uses `sleep(level=1)` when LoRA is enabled (`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:298`).
  - **SchedRL note**: for time-sharing shrink, SchedRL still requires **full GPU release** (weights+KV), so shrink must use deep sleep (`level=2`) even in LoRA mode; “level=1” can remain valid only for internal non-time-sharing pauses.
  - **Mixed-adapter batching note**: vLLM supports passing a per-prompt `lora_request` list (one `LoRARequest` per prompt). This is the mechanism used for S-LoRA-style mixed-adapter batches.
  - **Adapter update note (embedded API)**: in our current embedded integration surfaces (ROLL/SkyRL), we have `add_lora(...)` and `list_loras()`, but no explicit `remove_lora(...)`/`reload_lora(...)` surfaced. Therefore, we should not assume we can safely overwrite/replace adapter X “in place” while other requests are executing unless we validate it in the exact vLLM build used by the framework.

### ROLL reference: `tag` / domain concept maps naturally to adapter identity
ROLL already carries a per-environment/per-sample “domain” concept that is close to “adapter selection”:
- Agentic env managers use `env_config["tag"]` to select templates and per-tag settings (`third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:76`), and also use the tag for rate limiting (`third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:64`).
- Async generation scheduler tracks a per-item `domain` and produces per-domain metrics (`third_party/ROLL/roll/distributed/scheduler/async_generate_scheduler.py:454`, `third_party/ROLL/roll/distributed/scheduler/async_generate_scheduler.py:547`).

**Recommendation (ROLL-first)**:
- Standardize on a single canonical protocol field name: `adapter_id`.
- Define `adapter_id := env_config["tag"]` (agentic) or `adapter_id := domain` (async_generate_scheduler) in `MULTI_LORA`.
- Treat `tag`/`domain` as **source fields** that are mapped/aliased to `adapter_id` at the coordinator boundary; `adapter_id` is the source of truth thereafter.
- Treat “tag/domain” as the *routing key* for both:
  - which LoRA adapter to apply at inference time, and
  - which adapter’s optimizer state/version to update at training time.

This keeps the mental model consistent: “tag/domain” becomes the stable adapter identity across rollout, caching, progress reporting, and training updates.

---

## Desired End State

### Protocol supports both modes without forking the scheduler
The scheduler continues to reason at **pipeline granularity** (one engine group per pipeline; isolation assumption unchanged), but the protocol gains a first-class notion of:
- **Base model** artifact (full weights, slow to move),
- **Adapter** artifacts (small, frequent updates),
- A combined **Active Model Spec** that defines what rollouts should use.

### Correctness and safety invariants (carried over)
For shrink/expand and migration:
- Shrink ordering remains: **Close Admission → Abort(P) → Wait ACK → Offload/Stop(P) → Release GPUs**.
- If abort ACK does not arrive by timeout, **fail fast** (crash pipeline) as in the existing protocol.
- No “resume partial turn” across shrink; shrink uses `REQUEST_RETRY` (abort + re-issue).

### Multi-LoRA-specific requirements
- **One adapter per trajectory**, and rollout may be **mixed-adapter batched** (a batch can contain multiple adapters; each request carries its own `adapter_id`).
- Adapter synchronization happens at **adapter granularity** (update adapter X without requiring a full base sync).
- Expand-from-zero can warm only the adapters that have queued work (avoid preloading all adapters).

---

## Key Design Decisions (Options + Recommendations)

This section “answers the open questions” for multi-LoRA by choosing defaults that are compatible with ROLL/SkyRL/NeMo-RL and keep scheduler complexity low.

### Decision A — Base weights behavior in LoRA mode
**Option A1 (recommended)**: Base model **frozen** for the entire run; only adapters update.
- Reasons: matches common LoRA semantics (NeMo-RL docs), makes cache + versioning tractable, avoids “two axes” (base+adapter) changing concurrently.
- Confirmed requirement for this plan: in `MULTI_LORA`, the pipeline uses a **single shared frozen base** (never updated) and trains multiple adapters concurrently (shared trainer; per-adapter optimizer state).

**Option A2**: Base also updates (full FT + adapters simultaneously).
- Reasons to avoid initially: scheduler would need to coordinate base sync boundaries while adapters are also updating; the “active spec” becomes a true multi-dimensional vector clock and increases failure modes.

### Decision B — Adapter versioning model
**Option B1 (recommended)**: Each adapter has its own monotonic version, `adapter_version[adapter_id]`.
- Base version is fixed in LoRA mode (or changes rarely in future).
- Reasons: enables independent adapter updates and small artifact caching.

**Option B2**: Single global step for all adapters.
- Reasons to avoid: forces lock-step updates and wastes work if adapters progress at different rates.

### Decision C — When does a new adapter version become active for rollouts?
Because rollouts may be mixed-adapter batched, cutovers must be **scoped to the adapter being updated** (not a global stop-the-world boundary).

**Option C1 (recommended default for correctness)**: adapter-scoped `QUIESCE-by-abort` (ROLL-aligned).
- Close admission for adapter X (do not schedule X into mixed batches), abort in-flight requests for adapter X, wait abort ACK, then activate adapter X@v and reopen X admission.
- Other adapters may continue generating (and continue to appear in mixed batches) while X is paused, **if** the embedded inference API is safe to mutate adapter state without a global stop (see fallback below; default is a brief global `QUIESCE-by-abort` during `add_lora(...)` if unvalidated).
- Reasons: aligns with ROLL’s default safety boundary (`QUIESCE-by-abort`) and avoids waiting for natural completion.

**Option C2**: `INFLIGHT` for adapters (finish old trajectories on old adapter; new ones use new adapter).
- Requires tagging samples with `(adapter_id, adapter_version)` (recommended anyway) and accepting mixed-version data.
- Reasons to choose: better throughput if aborting is expensive or too disruptive.

**Option C3**: multi-version residency for the same adapter (keep X@v_old and X@v_new both loaded) and select `(adapter_id, adapter_version)` per request.
- Only valid if the inference engine supports it (or if `adapter_id` is versioned, e.g., `adapter_id = f"{name}@{version}"` mapping to distinct loaded LoRA handles).
- Reasons to choose: avoids pausing adapter X during updates, at the cost of higher memory pressure and more complex GC.

**Implementation fallback rule (embedded API, recommended)**:
- Default to **C1** at the coordinator level (stop scheduling adapter X; abort X in-flight; wait abort ACK).
- The actual “activate X@v_new” step may still require a **brief global control critical section** on each engine (because `add_lora(...)` mutates shared engine state). If the framework’s embedded API is not proven safe to call concurrently with generation, fall back to a short global `QUIESCE-by-abort` of the whole engine group for the duration of the `add_lora(...)` call, then resume mixed-adapter generation immediately.
- This keeps correctness while allowing us to later optimize toward “pure adapter-scoped update” if/when validated.

### Decision D — Retry semantics after shrink-triggered abort
**Option D1 (recommended)**: Abort+retry produces a fresh completion and is attributed to whatever `(adapter_id, active_adapter_version)` is active at retry time.
- Reasons: avoids having to pin old adapter versions just to satisfy retries; aligns with “no mid-turn resume”.

**Option D2**: Strict snapshot retry (retry must use the exact adapter version snapshot).
- Requires pinning old adapter versions until all in-flight/retry windows close; more cache/GC complexity.

---

## Protocol Extensions (What Must Change)

### 1) New concepts in the shared protocol
Add the following protocol-level objects (names illustrative; final naming should match `schedrl/protocol/types.py` once implemented):

- `ModelMode = {FULL_FT, MULTI_LORA}`
- `BaseSpec = {base_version, base_uri?}` (for FULL_FT, `base_version` is the usual checkpoint version; for MULTI_LORA it is constant for the run)
- `AdapterId` (stable identifier; recommended string)
- `AdapterSpec = {adapter_id, adapter_version, adapter_uri?}`
- `ActiveModelSpec = {mode, base: BaseSpec, adapters: dict[AdapterId, AdapterSpec]}`

**Compatibility rule**:
- In `FULL_FT`, `ActiveModelSpec.adapters = {}` and the existing single-axis `checkpoint_version` semantics remain.

### 2) Extend the cache contract (“checkpoint cache” → “artifact cache”)
Generalize the trainer-side cache service to manage **artifacts**, not just full checkpoints:
- Base weights cache: same as today (CPU bucket list / staged snapshots).
- Adapter cache: per `(adapter_id, adapter_version)` artifacts (likely file paths or in-memory blobs, depending on framework).

Base-frozen implication (important):
- In `MULTI_LORA`, do **not** repeatedly “sync/update the base” on every adapter update. The base artifact is immutable for the run.
- For time-sharing shrink, rollout workers must fully release GPU memory; this implies the **base weights and all adapters are dropped** on the shrunk subset (and possibly the whole generation cluster if shrinking to zero). On expand/resume, the base is re-loaded from the trainer cache, and adapters are re-loaded as needed.

GC rules (Phase 1, recommended):
- Keep: current active adapter versions.
- Keep: newest `K` versions per adapter (configurable; default small like 2–4).
- Do **not** guarantee strict snapshot retries (per Decision D1), so old versions can GC aggressively.

### 3) Extend scheduler↔adapter interface to express “what to warm on expand”
The scheduler does not need to micromanage adapters, but expand-from-zero needs a way to avoid opening admission before adapters needed for queued work are available.

**Option 1 (recommended)**: extend `expand_workers(...)` with an optional warmup payload:
- `expand_workers(worker_indices, base_version, warmup_adapters=[AdapterId], action_id, activation_epoch)`
- Coordinator loads base (per `base_version`) and then ensures `warmup_adapters` are loaded/ready on those workers before ACK.

Warmup policy (recommended):
- Warm **only** adapters that have queued work “now” (or an explicit `warmup_adapters` set). Do not preload all adapters by default.
- In `MULTI_LORA`, avoid any “base resync” beyond the one-time base load required after expand/resume.
- Warmup failure semantics (Phase 1, recommended):
  - `expand_workers(...)` is **atomic** with respect to warmup: return `success=true` only if **all** requested `warmup_adapters` are successfully loaded/ready on **all** requested `worker_indices`.
  - If any adapter warmup fails on any worker, return `success=false` with an error listing the failed `(worker_index, adapter_id)` pairs. Do not open admission on those workers for workloads that may require those adapters.

**Option 2**: add a separate call:
- `prepare_adapters(worker_indices, adapters=[AdapterSpec], action_id, activation_epoch)`
- This splits “resource expand” from “adapter warmup” but adds an extra RPC/action ordering surface.

### 4) Request identity must include adapter identity (for abort + attribution)
Deterministic request IDs should incorporate adapter identity so we can:
- debug mixed adapter workloads,
- target aborts correctly,
- tag samples/trajectories with model spec.

Recommended convention (string):
- `request_id = f\"{trajectory_id}:{turn_id}:{attempt}:{adapter_id}\"`

ROLL mapping (recommended):
- use `adapter_id = env_config["tag"]` (agentic) or `domain` (async_generate_scheduler) and include it in the request id.

### 5) Version tagging for produced data
Tag each completed trajectory with:
- `base_checkpoint_version` (or model hash),
- `adapter_id`,
- `adapter_version`.

This keeps the training side honest under Options C2/D1 (mixed/in-flight updates, retry on latest).

### 6) Progress reporting in multi-LoRA mode (aggregation + per-adapter percent)
SchedRL’s `report_progress(...)` has a single `percent_completed` scalar, but multi-LoRA naturally has “per-adapter readiness”.

**Option 1 (recommended for mixed-adapter batching)**: aggregate queued/inflight, and report per-adapter completion percent via `metrics`.
- Aggregation (required fields):
  - `queued_trajectories = sum_a queued_trajectories[a]`
  - `inflight_trajectories = sum_a inflight_trajectories[a]`
  - `oldest_unfinished_creation_ts = min_a oldest_unfinished_creation_ts[a]` over all unfinished work
- Pipeline-level `percent_completed` (scalar required by the protocol; recommended definition):
  - define `target_trajectories[a]` for the next readiness window (configuration; could be uniform across adapters)
  - define `collected_trajectories[a]` as “complete and ready-to-train for adapter a”
  - Validation (fail fast): require `sum_a target_trajectories[a] > 0` for every readiness window; otherwise crash the pipeline with a clear error (invalid configuration / empty adapter set).
  - `percent_completed = min(1.0, sum_a collected_trajectories[a] / sum_a target_trajectories[a])`
- Per-adapter (extra metrics):
  - `metrics["percent_completed_by_adapter"] = {adapter_id: pct}`
  - (optional) `metrics["queued_by_adapter"]`, `metrics["inflight_by_adapter"]`

ROLL mapping (recommended):
- Use the existing “domain/tag” label as `adapter_id` for the per-adapter metrics so the same key appears in:
  - rollout routing,
  - reward/quality reporting (already emitted as `scheduler/{domain}/...` today),
  - multi-LoRA progress readiness.

**Option 2**: single-target adapter progress (`target_adapter_id`) drives the scalar `percent_completed`.
- Good fit for “one adapter-at-a-time” collection/training, but ambiguous if the pipeline is collecting for many adapters concurrently in mixed batches.

---

## Shrink/Expand + Abort/Resume Semantics (LoRA-aware)

### Shrink (time-sharing preemption)
Unchanged from the shared protocol, with two LoRA-specific clarifications:
1) Shrink must release **all** GPU memory (base + adapters + KV). Any “cheap sleep” that preserves weights is not valid for time-sharing shrink.
2) If requests (for any adapters) are running on a worker in `P`, we abort and retry (no mid-turn resume). If we are mid-update for adapter X, we must not schedule X into mixed batches until X update completes (Decision C1).

### Resume strategy for shrink/expand (Option A only: drop-on-shrink, sync-on-expand)
Use the simplest contract:
- Any workers in the shrunk subset drop everything (base + all adapters + KV).
- On expand/resume, newly activated workers load base (from trainer-side bucket cache) and load/warm needed adapters before opening admission.
- Shrink-to-zero is the “remaining set is empty” special case: all rollout workers drop everything; the next expand loads base+adapters on the newly activated set (which is the full active set).

### Expand (resume or grow)
Expand must guarantee:
- base model weights are present for the active base spec, and
- for LoRA mode: adapters needed for the next scheduled mixed batches are loaded (or loadable before admission opens).

Recommended operational approach:
- Coordinator maintains per-adapter queues and builds mixed batches by drawing from multiple queues.
- Scheduler can optionally pass `warmup_adapters` to reduce “first batch penalty” after resume (see `expand_workers(..., warmup_adapters=...)` in Protocol Extensions).
- Mixed-batch fairness (Phase 1, recommended):
  - Use a simple no-starvation rule in the mixed-batch builder: each `adapter_id` with non-empty queue gets at least 1 prompt admitted per scheduling tick (up to batch capacity), then fill remaining slots proportional to backlog (or round-robin).
  - If an adapter has consistently low volume, this guarantees eventual service without requiring the central scheduler to be adapter-aware.

### Adapter activation (“activate LoRA”) without resizing
Adapter updates do not require scheduler involvement unless you want scheduling policy to depend on them.

Recommended coordinator behavior (Decision C1):
1) Stop starting new trajectories for adapter X (adapter-scoped admission close).
2) Abort in-flight requests/trajectories for adapter X and wait abort ACK.
3) Load/activate adapter X@v_new on any active workers that may serve adapter X (default: all currently active rollout workers, unless the coordinator tracks a smaller “workers with X resident” set).
4) Reopen adapter X admission **only after** all targeted active workers report “adapter X is ready at v_new” (avoid mixed X@old and X@new serving simultaneously).

If a shrink arrives in the middle of steps 2–3, the same fail-fast rules apply:
- do not proceed if abort ACK cannot be established.
- If shrink/worker failure interrupts step 3, keep adapter X admission closed and retry activation for the remaining active worker set (and any newly expanded workers will load X on resume via warmup).

---

## Framework Mapping (Reference-first: ROLL, then SkyRL-train, then NeMo-RL)

### ROLL (reference target)
Why it’s a good reference:
- already passes `lora_request` into vLLM and has a clear abort path for `REQUEST_RETRY`.

Plan deltas for multi-LoRA:
- Replace “pick first LoRA id from `list_loras()`” logic with “select adapter_id per request in a mixed batch”.
  - Current placeholder behavior exists in `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:172`.
- Ensure the coordinator can build mixed batches by drawing from multiple per-adapter queues, and provide a per-request `lora_request` list (one entry per prompt) matching each prompt’s adapter_id.
- Ensure adapter activation can pause only adapter X admission while other adapters continue generating (Option C1).
- Ensure shrink-triggered abort targets only the affected requests/workers and that retry re-issues the same `(trajectory_id, turn_id)` with incremented `attempt`.

### SkyRL-train
Why it’s useful:
- it already has a LoRA load path via vLLM `add_lora` (`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:313`).

Plan deltas for multi-LoRA:
- make LoRA load deterministic: adapter_id should map to a known LoRA in the engine (instead of generating random int ids).
- ensure time-sharing shrink uses deep sleep (`level=2`) even when LoRA is enabled (SchedRL invariant).
- align request_id construction with SchedRL deterministic IDs for abort+retry.

### NeMo-RL
Why it’s useful:
- it’s a clean reference for LoRA semantics (base frozen) and for a “cheap wake/sleep + selective sync” story.

Plan deltas for multi-LoRA:
- treat adapter artifacts as first-class cached items (parallel to base checkpoint cache).
- if/when RL LoRA rollouts are enabled, mirror the same adapter-batch boundaries for activation.

---

## What We’re NOT Doing (Explicitly Out of Scope)

- Mixed versions of the **same** adapter within a single engine without an explicit multi-version residency mechanism (Option C3).
- Composing multiple LoRAs simultaneously for a single trajectory (adapter stacking).
- Mid-turn suspend/resume (no “resume running turn” across shrink; only abort+retry).
- Sharing a single rollout engine group across multiple pipelines (isolation assumption stands).
- SGLang-first design; this plan is vLLM-first.

---

## Implementation Phases

## Phase 1: Protocol Types + Contracts (mode + model spec + cache)

### Overview
Define the protocol additions so frameworks can implement the same adapter surface for both full FT and multi-LoRA.

### Changes Required
1) Extend protocol schema to add `ModelMode` + `ActiveModelSpec` and adapter artifact definitions (see “Protocol Extensions”).
2) Extend cache contract from “checkpoint-only” to “base + adapter artifacts” with clear GC rules.
3) Decide and document the `expand_workers` warmup mechanism (Option 1 recommended).

### Success Criteria
#### Automated Verification
- [ ] N/A (doc/protocol-only phase)

#### Manual Verification
- [ ] Protocol doc has a single, unambiguous definition of “active model” in both modes.
- [ ] Shrink/expand ordering remains identical to the base protocol and is LoRA-safe.

---

## Phase 2: ROLL Reference Wiring (mixed-adapter batching + adapter activation)

### Overview
Implement multi-LoRA semantics in the ROLL reference path first, because it already matches `QUIESCE-by-abort` + `REQUEST_RETRY` patterns.

### Changes Required
1) Maintain per-adapter queues in the coordinator (one queue per adapter_id), plus a **mixed-batch builder** that draws prompts from multiple adapters for each scheduling tick.
2) Ensure vLLM calls use a per-prompt `lora_request` list (one `LoRARequest` per prompt) matching each prompt’s adapter_id.
3) Implement adapter activation (`adapter_id -> adapter_version`) and ensure it happens only at safe boundaries (Decision C1 recommended).
4) Ensure deterministic request IDs include adapter_id, and aborted work retries with incremented attempt (Decision D1).

### Success Criteria
#### Automated Verification
- [ ] ROLL suite passes: `cd third_party/ROLL && make test`

#### Manual Verification
- [ ] Run a multi-adapter rollout where a single inference batch contains prompts from multiple adapters, and produced trajectories are correctly tagged per `(adapter_id, adapter_version)`.
- [ ] Trigger shrink during active mixed-adapter generation; aborted requests retry and complete on remaining workers with no side effects duplicated.

---

## Phase 3: SkyRL-train Wiring (adapter identity + deep-shrink correctness)

### Overview
Align SkyRL’s existing LoRA load hooks with SchedRL’s adapter identity + shrink/expand requirements.

### Changes Required
1) Make adapter IDs stable and map them to vLLM LoRA IDs deterministically (avoid random `time_ns` ids for “the adapter identity”).
2) Ensure SchedRL shrink uses deep release semantics (`level=2`) even if LoRA is enabled (time-sharing invariant).
3) Add deterministic request IDs and adapter-aware retry routing (consistent with Phase 2).

### Success Criteria
#### Automated Verification
- [ ] SkyRL-train test or smoke run command per existing docs (no new tests added in this phase)

#### Manual Verification
- [ ] While mixed-adapter generation continues, update adapter X and verify: requests for X are temporarily not scheduled (or are retried) until X is activated, and then resume using the new adapter version; other adapters continue uninterrupted.
- [ ] Shrink mid-flight fully releases GPU memory and training continues after expand.

---

## Phase 4: NeMo-RL Wiring (adapter artifacts + activation boundaries)

### Overview
Adopt the same adapter artifact + activation semantics for NeMo-RL, reusing its selective/cheap wake/sleep patterns where applicable.

### Changes Required
1) Extend the “artifact cache” notion to include adapter artifacts and GC (parallel to base cache).
2) Implement adapter-scoped activation (Decision C1) in the rollout path when mixed-adapter multi-LoRA rollouts are enabled.

### Success Criteria
#### Automated Verification
- [ ] NeMo-RL tests pass: `cd third_party/nemo-rl && uv run --group test pytest -q`

#### Manual Verification
- [ ] Adapter updates can be staged/activated without requiring full base resync.

---

## Testing Strategy (End-to-End)

### Unit / Component
- ROLL: `cd third_party/ROLL && make test`
- NeMo-RL: `cd third_party/nemo-rl && uv run --group test pytest -q`

### Manual (multi-pipeline safety)
1) Run two pipelines concurrently under SchedRL:
   - one in full-FT mode,
   - one in multi-LoRA mode with multiple adapters.
2) Force shrink/expand cycles during active rollouts.
3) Verify:
   - no admission on inactive workers,
   - abort ACK gating is respected,
   - trajectory tags correctly record `(base_version, adapter_id, adapter_version)`.

---

## Performance Considerations

- Adapter updates should be much cheaper than base sync; avoid turning adapter updates into “full sync events”.
- Don’t preload all adapters on expand; warm only adapters with queued work.
- Keep shrink strict: full GPU release is non-negotiable for time-sharing, even if it drops adapter caches.

---

## Risks & Mitigations (ROLL-first)

### Risk 1: Cache service coupling (checkpoint cache → artifact cache)
**Issue**: existing “trainer CPU bucket list” code may be shaped around monolithic checkpoints. A generic refactor into an “artifact cache” could be larger than expected.

**Mitigation (Phase 1, recommended)**:
- Do not attempt a full “one cache to rule them all” refactor initially.
- Define two minimal, concrete surfaces:
  - `BaseBucketCache.get_base(base_version) -> bucket_iterable` (existing bucketized base copy; base frozen in `MULTI_LORA`)
  - `AdapterArtifactCache.get_adapter(adapter_id, adapter_version) -> artifact_ref` (file path or in-memory handle)
- Add a small pin/unpin (or refcount) contract to avoid GC races during expand/resume (scheduler dispatch must not observe 404s for the target base/adapters).

### Risk 2: vLLM `add_lora` concurrency (embedded API)
**Issue**: `add_lora(...)` mutates shared engine state and may not be safe to call concurrently with ongoing generation in the exact vLLM build used by ROLL/SkyRL.

**Mitigation (Phase 2, recommended)**:
- Treat adapter updates as “adapter-scoped gating + brief global critical section”:
  1) stop scheduling adapter X into new mixed batches,
  2) abort in-flight X requests and wait abort ACK (or fall back to abort-all if mapping is unavailable),
  3) perform `add_lora(...)` under a short global `QUIESCE-by-abort` for the engine group if concurrency safety is not validated,
  4) resume mixed-adapter generation immediately.
- Before optimizing to “pure adapter-scoped update while other adapters continue”, validate behavior on the project’s vLLM build with a minimal reproduction (no new test files required).

### Risk 3: Abort granularity in mixed-adapter batches
**Issue**: “abort adapter X” requires mapping to concrete in-flight request IDs when batches contain multiple adapters. If this mapping is missing, the safest fallback is abort-all on the targeted workers.

**Mitigation (v1-safe default)**:
- For **shrink**: abort is worker-subset scoped; abort all in-flight work on workers in `P` and retry elsewhere (SchedRL core path).
- For **adapter update**: keep “adapter X admission gating” in the coordinator. If request-id mapping is not available, use the same short global quiesce around `add_lora(...)` rather than trying to surgically abort X.

---

## References

- Shared protocol: `design_doc/multi-pipeline-adaptation-plan.md`
- Dual-mode scheduler plan: `thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`
- ROLL adaptation plan: `thoughts/shared/plans/2026-01-28-roll-schedrl-adaptation.md`
- SkyRL-train adaptation plan: `thoughts/shared/plans/2026-01-28-skyrl-train-adaptation-plan.md`
- NeMo-RL adaptation plan: `thoughts/shared/plans/2026-01-28-nemo-rl-schedrl-adaptation.md`
