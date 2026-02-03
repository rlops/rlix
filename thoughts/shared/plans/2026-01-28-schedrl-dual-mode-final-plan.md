# SchedRL Dual-Mode Final Plan (Library Mode + Service Mode)

**Date**: 2026-01-28

**Source plans consolidated into this file:**
- `thoughts/shared/plans/2026-01-28-schedrl-common-component-implementation-plan.md`
- `thoughts/shared/plans/2026-01-28-schedrl-common-core.md`
- `thoughts/shared/plans/2026-01-28-schedrl-dual-mode-consolidated-plan.md`
- `thoughts/shared/plans/2026-01-28-schedrl-dual-mode-implementation-plan.md`

---

## 0) Objective (Two Modes, One Integration)

Goal: implement SchedRL as a shared time-sharing library that also supports concurrent multi-framework arbitration.

Two operational modes, with the SAME framework-side integration:

1. **Library/Plugin Mode (time-sharing)**
   - A framework can run SchedRL embedded inside its own Ray job.
   - Useful for single-framework development and incremental adoption.
2. **Service Mode (concurrent multi-framework)**
   - A standalone, long-lived scheduler actor runs in the Ray cluster.
   - Multiple frameworks connect to it and share GPUs concurrently.

Key constraint: extra adaptation per framework is minimal and does not fork by mode.
- “Adapt once” per framework: implement a small Adapter surface + registration + progress reporting.
- After that, running the same framework standalone (Library Mode) or concurrently (Service Mode) is a deployment choice, not a code change.

---

## 1) Non-Negotiables (Shared Invariants)

1. Push model on Ray Actors (same Ray cluster).
   - Scheduler directly calls pipeline coordinator Adapter RPCs (no polling loop, no HTTP/gRPC).
2. Fail-fast on safety-critical timeouts.
   - If a safety-critical ACK does not arrive by timeout, the pipeline crashes (per protocol).
3. No new third-party dependencies for the common component.
   - Shared protocol/types use stdlib + Ray usage that already exists in the workspace.
4. Time-sharing requires full GPU release on shrink.
   - `shrink_workers(...)` must fully release GPU memory for the worker subset so other pipelines can use it.

---

## 2) “Write Once, Run Both Ways”: Discovery + Naming

All mode differences are confined to scheduler entrypoint/lifetime; framework integration is identical.

Ray namespace + actor naming conventions:
- Service Mode Namespace: `schedrl` (fixed).
- Library Mode Namespace: Defaults to `schedrl_{uuid}` (unique per job) to ensure isolation in shared clusters, unless explicitly overridden.
- Scheduler Actor Name: `schedrl:scheduler`
- Adapter Actor Name (per pipeline): `schedrl:adapter:{pipeline_id}`

Client discovery rule (idempotent):
- Determine candidate namespaces in order:
  1) Service Mode namespace (`schedrl`) if Service Mode is enabled for this deployment.
  2) The current job namespace (Library Mode; typically `schedrl_{uuid}`).
- For each candidate namespace: try `ray.get_actor("schedrl:scheduler", namespace=...)`.
- If found: connect and use it.
- If not found in any candidate namespace:
  - If `create_if_missing=True`: create one in the current job namespace (Library Mode path).
  - If `create_if_missing=False`: fail fast.

Service Mode startup:
- Start the scheduler explicitly (detached actor) before launching framework pipelines.
- Pipelines always do “get first, then create” and will find the persistent scheduler.
  - In Service Mode deployments, pipelines SHOULD use `connect(create_if_missing=False)` so they fail fast if the scheduler is not running (prevents accidental creation of a job-scoped scheduler).

Library Mode startup:
- If no scheduler exists, the first pipeline starts it inside its job.
- Scheduler lifetime is scoped to the job unless explicitly detached by the entrypoint.

Concurrency edge case (get-or-create race):
- If multiple drivers start concurrently and all attempt “get then create”, the client MUST handle the create race by catching the “already exists” failure and re-`get_actor(...)` (so `connect()` is effectively idempotent under concurrency).

---

## 3) Shared Package Layout (Repo-Root `schedrl/`)

Create a shared Python package at repo root:

```text
schedrl/
  __init__.py
  protocol/                  # pure data + invariants
    __init__.py
    types.py                 # IDs, enums, dataclasses (incl. ResourceRequest/ResourceRelease)
    actions.py               # Shrink/Expand/CheckpointSync schemas
    validation.py            # invariant checks (fail fast)
  client/                    # framework-facing SDK
    __init__.py
    adapter.py               # Adapter ABC: ONLY thing frameworks implement
    client.py                # connect/get_or_create + register/report helpers
  scheduler/                 # central orchestration
    __init__.py
    scheduler.py             # SchedRLScheduler (Ray Actor)
    state.py                 # ClusterState, PipelineState, allocations
    queues.py                # request queue + coalescing (progress overwrite, sync intents)
    policy.py                # FIFO baseline + gap-ratio policy (later phase)
    planner.py               # plan -> ordered actions compiler
    executor.py              # safe RPC execution (timeouts + retries + fail-fast)
    health.py                # per-action deadlines + pipeline health
    run.py                   # service-mode entrypoint (detached scheduler)
```

Importability expectation:
- `python -c "import schedrl"` should work when running from repo root.
- If running from a subproject directory, ensure repo root is on `PYTHONPATH` in that launch path.
- If using import-time patching (`sitecustomize.py`) for frameworks we do not control, the patch directory MUST be on `PYTHONPATH` for the driver and all Ray workers (via Ray `runtime_env`).

---

## 4) Minimal Framework Adaptation Contract (Same in Both Modes)

Framework-side integration is limited to:

1. Provide an Adapter actor implementing the shared surface.
   - The Adapter is owned by the framework coordinator and maps SchedRL actions to existing internal APIs.
2. Register with the scheduler and continuously report progress.
   - Registration: `pipeline_id`, framework name, policies, `dp_max`, `tp_size`, and any role metadata needed by scheduling policy.
   - Progress heartbeats: queued/inflight/percent_completed in SchedRL-standard units (trajectory units).
     - `percent_completed` is allowed to exceed `1.0` for reporting, but scheduling triggers that depend on “2% bands” MUST clamp it to `[0.0, 1.0]` (to avoid unbounded band increases).
   - Re-registration idempotency:
     - Duplicate `register()` calls for the same `pipeline_id` while the Adapter actor is still live SHOULD be rejected or treated as a no-op.
     - A re-register that points to a different Adapter actor handle MUST replace the prior handle (and trigger State Reset on Registration semantics).
3. Respect action semantics (idempotency + supersession).
   - Every scheduler→adapter call includes `action_id` and `activation_epoch`.
   - Adapter must be idempotent by `action_id` and ignore superseded intents via `activation_epoch`.
4. **State Reset on Registration**:
   - If the scheduler restarts (re-registration), it has no record of prior allocations.
   - Therefore, upon (re)registration, the Adapter MUST assume it has 0 allocations (S_actual={}).
   - If it holds resources from a previous session, it MUST release/kill them immediately to prevent zombie resource conflict with the fresh scheduler state.

### 4.1) Minimal Patching Strategy (Per Framework)

Constraint: third-party framework code is not always in our control. Prefer the smallest integration surface:

- **First choice**: use existing extension points and coordinator-owned Adapter code (no monkeypatching).
- **Fallback**: import-time patching via `sitecustomize.py` shipped to Ray workers via `runtime_env` (driver + all workers).

Order of attack for adaptation work:
1) ROLL → 2) NeMo-RL → 3) SkyRL-train → 4) Miles

#### 4.1.1) ROLL (we control source code)

Approach: implement required hooks directly in ROLL, and make runtime environment propagation explicit (so SchedRL code and any optional shims are importable in workers).

- Update ROLL Ray initialization to optionally export `PYTHONPATH` into `runtime_env["env_vars"]` (toggle-based; default off).
  - Current runtime_env is platform env only: `third_party/ROLL/roll/distributed/scheduler/initialize.py`.
- Implement the ROLL-specific requirements from `thoughts/shared/plans/2026-01-28-roll-schedrl-adaptation.md` as direct code changes (deterministic request ids, targeted abort+ACK, subset expand/shrink, subset-scoped sync/progress).

#### 4.1.2) NeMo-RL (third-party; no direct edits assumed)

Approach: use `sitecustomize.py` patch shims and rely on NeMo-RL’s existing pattern of passing through environment variables into Ray `runtime_env`.

Operational requirement:
- Ensure the driver `PYTHONPATH` includes the patch directory before NeMo-RL initializes Ray.
- Ensure Ray workers inherit that `PYTHONPATH` (NeMo-RL commonly uses `runtime_env={"env_vars": dict(os.environ)}` in its Ray init paths).
- If NeMo-RL async GRPO is enabled, enforce at startup: `grpo.async_grpo.in_flight_weight_updates=true` (fail fast otherwise).

#### 4.1.3) SkyRL-train (third-party; no direct edits assumed)

Approach: use `sitecustomize.py` patch shims and rely on SkyRL’s built-in capability to export `PYTHONPATH` into Ray runtime env.

Operational requirement:
- Set `SKYRL_PYTHONPATH_EXPORT=true` so the driver `PYTHONPATH` is propagated to Ray workers.

#### 4.1.4) Miles (third-party; no direct edits assumed)

Approach: use `sitecustomize.py` patch shims and validate runtime env propagation because Miles frequently sets per-actor `runtime_env={"env_vars": ...}`.

Operational requirement:
- Ensure Ray job runtime_env includes `PYTHONPATH` (patch directory), and confirm per-actor runtime_env does not drop it.
- Validate by running a small Ray worker check (see §4.2).

### 4.2) Runtime Env / `sitecustomize.py` Verification (Mode-Agnostic)

If using `sitecustomize.py`, add a lightweight sanity check to confirm patch load in workers:
- `sitecustomize.py` sets a marker (e.g., env var `SCHEDRL_SITECUSTOMIZE_LOADED=1` or a log line).
- A tiny Ray task/actor prints/returns that marker to confirm the worker process loaded the patch directory via `PYTHONPATH`.

---

## 5) Adapter RPC Surface (Keep Small; Put Complexity Behind It)

Scheduler → Adapter (required, minimal):
- `close_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `open_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `shrink_workers(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `expand_workers(worker_indices, checkpoint_version, action_id, activation_epoch) -> ActionResponse`

`ActionResponse` schema (minimum, Phase 1):
- `success: bool`
- `error: Optional[str]` (use `"Superseded"` for supersession ACKs; otherwise a short machine-readable reason)

Required semantics (time-sharing compatible):
- `close_admission`: stop admitting NEW work for the specified subset.
- `shrink_workers`: complete the shrink sequence for the subset and ACK only when done.
  - shrink ordering is strict: close admission → abort/drain → wait for ACK that inflight==0 → offload/stop → return success.
  - MUST fully release GPU memory for the subset.
    - For vLLM-based rollout engines, this means using vLLM Sleep Mode deep sleep (`sleep(level=2)`) so both model weights and KV cache are released. It is acceptable to keep CUDA runtime / CUDA graph allocations for now.
- `expand_workers`: bring workers online for the subset and ACK only when ready.
  - onload/restore → sync checkpoint (if required by policy) → open admission → return success.

Supersession / “ignore” semantics (avoid deadlocks):
- Adapters MUST NOT silently drop a call with a superseded `activation_epoch`.
- If an action is superseded, the Adapter MUST return an ACK immediately with `success=False` and `error="Superseded"` (no-op).
  - The Scheduler MUST handle this explicit error by DISCARDING the state update (e.g., do not free resources if a shrink was skipped).

Timeout ownership (Phase 1):
- The Scheduler owns safety-critical deadlines and enforces them by timing out the scheduler→adapter Ray RPC.
- Adapter implementations may have internal timeouts, but the Adapter RPC signature MUST NOT require `timeout_s` parameters to satisfy the protocol.

Scheduler → Pipeline (optional later surface; keep out of Phase 1 unless already required):
- explicit abort/drain handles, onload/offload granularity, checkpoint sync RPC splits

---

## 6) Scheduler Responsibilities (Shared Between Modes)

### 6.1) Canonical Scheduler API + Ownership (Phase 1)

To remove ambiguity across the source plans, Phase 1 uses:
- `request_gpus(request: ResourceRequest)` where `ResourceRequest` includes `pipeline_id` (no separate `pipeline_id` argument).
- `release_gpus(release: ResourceRelease)` where `ResourceRelease` includes `pipeline_id` and `worker_indices` (bundle indices) being voluntarily returned (no `allocation_id` in Phase 1).

Invariant (double-free prevention):
- When processing `release_gpus(worker_indices)`, the scheduler MUST ignore any indices that are not in state ACTIVE.

Ownership / monotonicity:
- `activation_epoch` is scheduler-owned, monotonic per pipeline.
  - Clients/Adapters NEVER GENERATE epochs. Client requests (like `request_checkpoint_sync`) must NOT include an epoch; the scheduler assigns it upon ingestion.
  - Used for supersession (“ignore older epochs”).

State model (in-memory, Phase 1):
- **Global Physical Constraint**: The scheduler tracks `TotalPhysicalGPUs` vs `AllocatedPhysicalGPUs`.
  - Allocating a bundle in *any* pool decrements the global counter by `tp_size`. This prevents oversubscription if multiple pools (with different `tp_size`) share the same physical cluster.
- GPU pool state (free vs allocated) is also tracked per `cluster_id` (logical grouping).
  - Phase 1 constraint: within a given `cluster_id`, `tp_size` is fixed (homogeneous bundle size). Pipelines with a different `tp_size` MUST use a different `cluster_id`.
- Per-pipeline state:
  - registration metadata + policies (update/migration/expand rebalance)
  - desired vs actual subset allocation (`S_desired` / `S_actual`)
  - per-worker lifecycle state (ACTIVE/DRAINING/OFFLOADED/FAILED)
  - `activation_epoch` (monotonic supersession token)
  - outstanding actions with deadlines + retry count
  - progress state (queued/inflight/percent_completed) for fairness and anti-thrashing

Main loop (event-driven):
1. Ingest events: register, request/release, report_progress, action ACKs
2. Plan: compute shrink/expand/sync actions based on policy and current state
3. Execute actions with ordering:
   - Shrinks first (reclaim resources)
   - Expands second (consume reclaimed resources)
4. Update state only on ACK (or fail-fast on safety-critical timeout)

Pipeline death / health (Phase 1):
- If an Adapter actor dies (`RayActorError` on RPC), the scheduler MUST mark the pipeline unhealthy.
  - **Zombie Worker Safety**: The scheduler CANNOT assume physical resources are free unless fate-sharing is guaranteed (e.g., same Ray Job).
  - If unsure, the scheduler MUST quarantine the bundles for a lease timeout (or attempt active cleanup) before reclaiming them to the free pool.
- If progress heartbeats stop for longer than a configured timeout, the scheduler SHOULD treat the pipeline as unhealthy (stop allocating to it until it re-registers).

---

## 7) Two-Mode “Seamless Concurrency” Guarantee (What Makes It Automatic)

To ensure “adapt once → run standalone or concurrent”:

Framework code MUST:
- Always use the same discovery flow: `connect()` always tries “get existing scheduler actor” first.
  - Whether “create if missing” is allowed is a deployment configuration (e.g., `create_if_missing=True` for Library Mode, `False` for Service Mode).
- Always register Adapter actor with the same naming + namespace.
- Never encode mode-specific behavior in the Adapter implementation.

Deployment selects the mode:
- Library Mode: nothing else is started; first pipeline creates scheduler if missing.
- Service Mode: operator starts detached scheduler first; all pipelines connect to it.

---

## 8) Execution Phases (Shared Deliverables First; Wiring Later)

Phase 1: Protocol + Adapter surface (unblocks all frameworks)
- Implement `schedrl/protocol/{types,actions,validation}.py`
- Protocol MUST support both `FULL_FT` and `MULTI_LORA` via an `ActiveModelSpec`-style model descriptor (base + optional adapters). Reference: `thoughts/shared/plans/2026-02-02-schedrl-multi-lora-adapter-extension.md`.
- Implement `schedrl/client/adapter.py` (ABC) + minimal `ActionResponse` schema
- Implement `schedrl/client/client.py` (`connect`/`get_or_create` + register + report helpers)

Phase 2: Scheduler skeleton + baseline policy
- Implement `schedrl/scheduler/{state,scheduler,queues,executor,run}.py`
- FIFO policy first (priority + timestamp) to validate end-to-end RPC and state transitions

Phase 3: Full shrink/expand orchestration + safety rules
- Enforce strict action ordering + timeouts
- Track `activation_epoch` and `action_id` idempotency rules end-to-end

Phase 4: Dual-mode hardening (discovery + lifecycle)
- Ensure scheduler actor creation matches the mode expectations (detached vs job-scoped)
- Confirm multiple pipelines can connect and operate concurrently in the same Ray cluster

Phase 5: Framework wiring (minimal patching strategy; in this order)
- ROLL: implement direct code changes + runtime_env `PYTHONPATH` export (toggle-based), then wire Adapter + progress reporting
- NeMo-RL: add `sitecustomize.py` shims + verify worker `PYTHONPATH` propagation, then wire Adapter + progress reporting
- SkyRL-train: add `sitecustomize.py` shims + enable `SKYRL_PYTHONPATH_EXPORT=true`, then wire Adapter + progress reporting
- Miles: add `sitecustomize.py` shims + verify `PYTHONPATH` survives per-actor runtime_env, then wire Adapter + progress reporting
- References:
  - `thoughts/shared/plans/2026-01-28-roll-schedrl-adaptation.md`
  - `thoughts/shared/plans/2026-01-28-nemo-rl-schedrl-adaptation.md`
  - `thoughts/shared/plans/2026-01-28-miles-schedrl-adaptation.md`
  - `thoughts/shared/plans/2026-01-28-skyrl-train-adaptation-plan.md`

---

## 9) Success Criteria (Mode-Agnostic)

Common core correctness:
- A scheduler actor can be created or discovered by name + namespace.
- A pipeline can register, request/release, report progress, and receive shrink/expand calls.
- Shrink ordering is enforced and shrink fully releases GPU memory for the specified subset.
- Safety-critical timeouts fail fast with clear error messages.
- If a pipeline adapter dies, the scheduler reclaims its bundles and stops scheduling it until re-registration.

Library Mode (single framework):
- One pipeline can start without any pre-existing scheduler and still run with dynamic allocation.

Service Mode (multi-framework concurrency):
- Scheduler is started once (detached) and remains available.
- Two independent pipelines from different frameworks connect and can be time-shared concurrently.
