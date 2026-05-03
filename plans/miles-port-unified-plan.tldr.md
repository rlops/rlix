# TLDR Plan

Feature: **MILES → RLix port (fullasync GRPO under partial overlap)**
Source: [miles-port-unified-plan.md](miles-port-unified-plan.md)

## 0. Audit Dashboard

- **Goal**: make MILES fullasync GRPO schedulable by RLix; partial overlap (`actor_train ⊂ actor_infer`) + subset sleep/wake + CPU bucket cache + selective sync.
- **Top-level decision (D1)**: 3 new RLix-side actors (`MilesPipeline`, `MilesCoordinator`, `MilesModelUpdateService`) + `MilesPlacementProvider` adapter; reuse `ReloadableProcessGroup`; manual-init `MilesCoordinator` (no `super`).
- **Main behavior change**: subset abort-drain-sleep + atomic sync unit (transport + finalize + version publish) + turn-level redispatch on preempt + dual-mask receivers; **M11.2 router 0-active suspend (block-with-notify + HTTP 503 sentinel + client EnginePreemptedError translation)**.
- **Highest-risk decision**: **D3.d** atomic sync unit (single `_cache_lock` over full transport; v=-1 short-circuit; `_cache_ready_step` inside `_resize_sync_lock`).
- **Highest-risk assumption**: **A5** scheduler runs `resize_infer(add)` Phase 5 RPC BEFORE signaling pending GEN waiter (Phase 6).
- **Highest-risk constraint**: **C12** M11.1 forces `model_update_transport == "cpu_serialize"` (vast.ai container); **C20** router suspend three-piece contract (router + http_utils._post + http_utils.post) is the M11.2 load-bearer.
- **Most important decision to audit first**: **D3.d** + interaction with **A5** + **C13** (base-version equivalence at v=-1); then **D4.b** router admission (now extended for 0-active suspend).
- **Likely touched module families**: MILES side: `ray/ + backends/{sglang_utils,megatron_utils}/ + router/ + utils/{http_utils,arguments}/ + rollout/ + examples/fully_async/`; 4 new files in `rlix/pipeline/`; new entry `examples/rlix/run_miles_rlix.py`; 4 vendored SGLang patches under `external/sglang/3rdparty/cpu_serialize/`. See Appendix B.
- **Must-not-change**: standalone `train_async.py`; existing `RolloutManager.update_weights_from_distributed/tensor`; existing `_send_to_colocated_engine` (standalone cuda_ipc); MILES tool-error group-recycle (orthogonal to RLix preempt); `ReloadableProcessGroup`.
- **User audit focus**: (1) F4 sequence (D3.f) + HF-format cache; (2) D3.d atomic unit + D5.c v=-1 short-circuit; (3) F10 fail-fast on M11.1 (C12 cpu_serialize, C6 contiguous mapping, C10 offload_train, C8 EP=1, C11 no async_save, **C20 MILES_ROUTER_DISPATCH_WAIT_S > 0**); (4) milestone scoping M11.1 vs M11.2/4/5/6; (5) **router 0-active suspend three-piece contract** (router `_use_url` async + `_post` sentinel detection + `post()` `RayTaskError` unwrap with guarded import).

## 1. Problem Context

- **Current system behavior**: MILES `train_async.py` runs fullasync GRPO standalone — owns its placement group, its rollout/training loop, its weight-update path (`RolloutManager.update_weights_from_distributed/tensor`). One MILES pipeline = one process = exclusive GPUs.
- **Problem / gap**: cannot run multiple MILES pipelines (or MILES + ROLL) on the same GPUs, because (a) no time-share scheduler arbitrates ownership across pipelines and (b) MILES weight-update path requires weights to live on the GPU at refit time, conflicting with releasing GPU back to a peer pipeline.
- **Why now**: ROLL/RLix already implements partial-overlap scheduling (`actor_train ⊂ actor_infer`, scheduler-driven shrink/expand) for vLLM-backed pipelines. Bringing MILES under the same scheduler unlocks GPU-density goals without re-platforming MILES.
- **System-visible impact if nothing is done**: MILES cannot co-tenant; throughput per GPU is bounded by single-pipeline utilization; no path to multi-tenant RL fleets.
- **Existing architecture context**: MILES owns SGLang inference workers + Megatron training actors, glued by `RolloutManager` and a custom `Miles Router`. RLix already provides `Orchestrator`, `Scheduler`, `PipelineCoordinator`, `ModelUpdateService`, `RollResourceManagerProxy`. Port = bridge from MILES side into these RLix abstractions.
- **Non-obvious background**: (a) SGLang `release_memory_occupation` keeps TP NCCL alive (vLLM-equivalent); (b) MILES has `ReloadableProcessGroup` already, so destroy/reload Megatron NCCL is free (NeMo had to add this); (c) MILES uses an out-of-process Miles Router (not in-process collective_rpc like vLLM), so admission control needs HTTP endpoints — and 0-active-engine suspension must cross the HTTP boundary via sentinel header + client translation, not via in-process raise. Reference squash baseline: `41615af9` for staleness control + `9a0036447` for legacy custom-generate signature compatibility.

## 2. Assumptions

| ID | Assumption | Why it matters | Evidence / check | If it fails |
| -- | ---------- | -------------- | ---------------- | ----------- |
| A1 | First-build contiguous `infer_device_mapping` makes scheduler `dp_rank == MILES engine_index` an identity. | Lets `MilesPlacementProvider` skip the `scheduler_dp_rank → engine_index` lookup table. | C6 (F10 startup assert that mapping is sorted-contiguous). | Non-contiguous mapping mis-routes shrink/expand; needs explicit lookup table (deferred to follow-up). |
| A2 | SGLang `release_memory_occupation` keeps TP NCCL communicator alive (vLLM-equivalent). | If destroyed, SGLang's own release path breaks (uses `barrier(tp_cpu_group)` internally). | Gate 2 (regression confirmation, not new risk). | Cannot fall back to destroy-and-re-init; ~50–200 MB NCCL buffer residue absorbed by `miles_post_sleep_vram_threshold_gb`. |
| A3 | Ray runtime auto-derefs top-level `ObjectRef` at remote-method boundary; wrapper receives `bytes`, not `ObjectRef`. | `update_weights_from_cpu_bucket(payload_bytes, ...)` signature relies on this. | `ray.put → ref → engine.method.remote(ref)` round-trip test. | `ray.get(payload_bytes)` raises `TypeError`; only surfaces at first sync. |
| A4 | `/dev/shm` is RAM-backed and `mmap`/`read` is effectively zero-copy. | tmpfs file path is the cpu_serialize transport between Ray wrapper and SGLang HTTP child. | Startup `shutil.disk_usage("/dev/shm")` check at F10. | Hot-path latency spikes; OOM if `/dev/shm` < bucket size + 256MB margin. |
| A5 | Scheduler executes `coordinator.resize_infer(add)` RPC in Phase 5 BEFORE signaling pending GEN waiter in Phase 6. | Init Step 6.6 must pre-register manager + service resources + base `_cache_ready_step=-1` BEFORE Step 7's GEN request, otherwise `_expand_workers` reads `None` and raises. | Read scheduler.py:1178-1187 + planner.py:418-425; Gate 4 (e) timestamp test verifies `T1 < T2 < T3`. | `_expand_workers` raises mid-init; Phase 2 cleanup. **Highest-risk assumption.** |
| A6 | Cache owner Megatron rank (`pp0+dp0+tp0+cp0` = `_is_distributed_src_rank`) is unique and stable across pipeline lifetime. | Init Step 6.5 collects exactly one owner; service caches it forever. | C15 (init Step 6.5 `collect_cache_owner_roles` returns exactly one True). | Multiple owners → duplicate writes; zero owners → empty cache. |
| A7 | SGLang `/v1/loads` returns `num_running_reqs + num_waiting_reqs` correctly per slot. | Drain hot path polls this; `/server_info` lacks the field. | Unit test: dispatch N requests, poll until 0. | False idle → release while requests in-flight → SGLang `assert is_fully_idle()` fires. |
| A8 | Bounded request-level mis-attribution during active-refresh is acceptable (in-flight may finish on old weights while engine reports new version). | Justifies "no drain-then-sync" design; aligns with NeMo `in_flight_weight_updates=True`. | Gate 3 quantifies count vs `in-flight × decode-step` bound. | Trajectory `weight_versions` becomes unreliable for `--max-weight-staleness`. |
| A9 | MILES tool-error / agentic group-recycle is orthogonal to RLix scheduler preempt. | Lets the port keep group-recycle untouched; turn-level redispatch is the *only* RLix preempt path. | Code grep: `Sample.Status.ABORTED` from tool error vs `EnginePreemptedError` from router metadata. | Mixing the two paths makes preempt look like silent retries; bugs hide. |
| A10 | MILES `ReloadableProcessGroup.destroy_process_groups + reload_process_groups` is correct and leak-free. | Lets the port skip NeMo-style `nccl_offload.py` (~90 lines) + Gate 2.5 fallback rule. | Gate 2.5 destroy/reload cycle ≥3 steps with VRAM monitoring. | NCCL group leak across train↔infer toggling; OOM after N steps. |
| A11 | M11.2 0-active-engine pipeline state is a legit runtime status, not an exception. | Justifies router suspend (block-with-notify) instead of fail-fast crash. The fully_async outer loop's background generation tasks must pause silently through the pipeline's own train cycle or peer-pipeline contention window. | C20 (router unit test + Gate 4 (f) confirm suspend + sentinel translation). | Router raises `RuntimeError` under empty active set → fully_async sees generic exception → silent group-recycle attribution → bugs hide. |

## 3. Scope & Constraints

### 3.1 Out of Scope

PRODUCT BOUNDARY axis — describes capabilities the BUILD does NOT include in this milestone. Where an entry has a corresponding §3.2 runtime guardrail, this is intentional overlap (different framings for different readers — auditor here, implementer in §3.2).

- **PD disaggregation** — not supported in any milestone.
- **vLLM backend** — only SGLang.
- **MoE / EP** (`expert_model_parallel_size > 1`, `moe_router_topk > 0`) — F4 cache covers dense Megatron only.
- **`sglang_data_parallel_size > 1`** — adds a DP axis the port does not handle.
- **Selective P2P weight transfer** — broadcast / tensor subset sync only; P2P is a follow-up.
- **Multi-LoRA / DPO / SFT** — LoRA RLix-mode rollout to **M11.4**; DPO/SFT entirely out.
- **Request-level deterministic migration** (ROLL `RequestScheduler`) — replaced by turn-level redispatch (NeMo F3 form).
- **Non-contiguous / custom-ordered `infer_device_mapping`** — first-build forces sorted contiguous; non-contiguous needs `scheduler_dp_rank → engine_index` lookup (follow-up).
- **`RadixTreeMiddleware` in RLix mode + `partial_rollout + radix_tree`** — disabled at startup; transparency / prefix-cache pollution are follow-up.
- **Cross-node single rollout (SGLang) engine** (intra-engine sharding spans nodes) — `WorkerPlacement` is per-node; multi-node DP with each engine node-local IS supported in M11.1.
- **Reading `cache_owner` internal state cross-process from the service** — owner is reported at init, not queried in sync hot path.
- **Branching RLix vs standalone inside `train_async.py`** — dual-entry: `train_async.py` standalone, `examples/rlix/run_miles_rlix.py` RLix.
- **Per-request id tracking inside `RolloutManager`** — replaced by SGLang `/v1/loads` idle bool.
- **Subclassing `PipelineCoordinator`** for `MilesCoordinator` — manual init only.
- **`args.async_save` in RLix train_step** — F10 fail-fast (M7 segfault).
- **`verify_model` debug validation** in receiver API — per-bucket barrier + warmup allreduce replace it.
- **Driver top-level `ray.shutdown()` + global `try/except`** — failures raise; user runs `ray stop` (M11.1 simplification).
- **Receiver crash tolerance / port pool / TIME_WAIT cooldown** — **M11.5** production hardening.
- **Multi-pipeline orchestrator-driven cleanup, admission_epoch race defense, graceful actor drain** — **M11.2** follow-ups (NOT Gate 4 pass criteria).
- **cuda_ipc colocate transport in M11.1** — vast.ai container blocks `--ipc=host` / `CAP_SYS_PTRACE`; cuda_ipc adapter ships in **M11.6**.
- **Router-side request retry / queue / spillover** — empty active set is handled by suspend (C20), not by retry/queue. Production retry/queue is M11.5 follow-up.

### 3.2 Hard Constraints

RUNTIME GUARDRAIL axis — each row is a checkable predicate the implementer can lift directly into an `assert` or test. §3.1 entries that describe the same boundary in product-named form (e.g. "MoE / EP not supported") are intentional overlap with the corresponding checkable form here (e.g. C8 `expert_model_parallel_size == 1 and moe_router_topk == 0`).

| ID | Constraint | Enforced by | Source |
| -- | ---------- | ----------- | ------ |
| C1 | `train_devices.issubset(infer_devices)` (partial overlap topology) | F10 startup assert | `assert train_devices.issubset(infer_devices)` |
| C2 | `infer_engine_count >= 2` | F10 | `assert infer_engine_count >= 2` |
| C3 | `len(infer_devices - train_devices) >= rollout_num_gpus_per_engine` (declared-topology precondition; bounds single-pipeline M11.1 own-train overlap-subset shrink so it cannot reach 0 active engines; **does NOT hold at runtime in M11.2** — runtime 0-active is handled by C20 router suspend, not by this constraint) | F10 | `plans/miles-port-unified-plan.md:2862` |
| C4 | `args.sglang_data_parallel_size == 1` | F10 | `assert args.sglang_data_parallel_size == 1` |
| C5 | `args.rollout_num_gpus % args.rollout_num_gpus_per_engine == 0` | F10 | divisibility |
| C6 | First-build `infer_device_mapping` sorted-contiguous (per engine) | F10 | derived; relaxes when non-contiguous adapter ships |
| C7 | `args.rollout_num_gpus_per_engine <= num_gpus_per_node` (rollout engine node-local) | F10 (M3) | per-engine total GPU |
| C8 | `args.expert_model_parallel_size == 1 and getattr(args, 'moe_router_topk', 0) == 0` (no MoE/EP) | F10 | `assert args.expert_model_parallel_size == 1` |
| C9 | `not has_pd_disaggregation` | F10 | `SglangConfig` property |
| C10 | `args.offload_train == True` | F10 | `actor.sleep()` asserts internally |
| C11 | `not getattr(args, 'async_save', False)` (M7 segfault) | F10 | `assert not getattr(args, 'async_save', False)` |
| C12 | RLix mode `args.model_update_transport == "cpu_serialize"` (M11.1 only; M11.6 relaxes) | F10 | conditional on `DO_TIME_SHARING` |
| C13 | M11.1 `args.load in {None, args.hf_checkpoint}` AND `args.ref_load in {None, args.hf_checkpoint}` (Fix #14) | F10 | conditional on `DO_TIME_SHARING` |
| C14 | `set(allocated) == set(declared)` for `actor_train` allocation (no partial) | `_request_cluster_gpus` post-check | `assert set(allocated) == set(declared)` |
| C15 | `collect_cache_owner_roles` returns exactly one `True` (cache owner unique = `pp0+dp0+tp0+cp0`) | init Step 6.5 | exactly one True |
| C16 | `master_port != 0` for NCCL TCP rendezvous | service `setup_collective_group` | `get_free_port()` + SharedStorage claim |
| C17 | `'RadixTreeMiddleware' not in args.miles_router_middleware_paths` (RLix mode) | F10 | conditional on `DO_TIME_SHARING` |
| C18 | `not getattr(args, 'rollout_force_stream', False)` (router metadata injection requires JSON) | F10 | `assert not getattr(args, 'rollout_force_stream', False)` |
| C19 | `single_updateable_model_and_server(args)` (single updateable model + single SGLang server group) | F10 | `single_updateable_model_and_server(args)` |
| C20 | Router `_use_url` is `async def` and blocks-with-notify on empty active set (`asyncio.Condition`) with bounded timeout `MILES_ROUTER_DISPATCH_WAIT_S` (default 60s); on timeout returns HTTP 503 + `X-Miles-Preempt: empty_active_set_timeout` header; client `miles.utils.http_utils._post` translates that header to `EnginePreemptedError` BEFORE its retry loop (`EnginePreemptedError` must be imported **outside the `try` path**: module-top preferred if cycle-free — verified for `miles.rollout.base_types`; otherwise function-top fallback); `miles.utils.http_utils.post()` unwraps `RayTaskError` (imported under module-top `try/except ImportError: RayTaskError = ()` guard for Ray-absent envs) and re-raises `EnginePreemptedError` BEFORE generic `except Exception` fallback | Router unit test + client unit test + Gate 4 (f) | F10 startup assert `MILES_ROUTER_DISPATCH_WAIT_S > 0`; full mechanism spec in §Feature 3 §7+§8+§9 of source plan |
| C21 | `async_generation_enabled` (partial overlap requires fullasync; sync generation path lacks the in-flight refresh window F5+F6 depends on) | F10 | `assert async_generation_enabled, "partial overlap requires fullasync"` ([source L3096](miles-port-unified-plan.md#L3096)) |
| C22 | `len(train_devices) % megatron_parallelism_product == 0` where `megatron_parallelism_product = tp_size * pp_size * cp_size * dp_size * ep_size` (Megatron init divisibility on train side; non-divisible config raises mid-init AFTER F10 → harder to debug) | F10 | `assert len(train_devices) % megatron_parallelism_product == 0` ([source L3272](miles-port-unified-plan.md#L3272)) |
| C23 | `len(infer_devices) % args.rollout_num_gpus_per_engine == 0` (engine-mapping divisibility on the actual `infer_device_mapping`; **distinct from C5** which checks `args.rollout_num_gpus` — config vs declared-mapping mismatch fails this independently) | F10 | `assert len(infer_devices) % args.rollout_num_gpus_per_engine == 0` ([source L3276](miles-port-unified-plan.md#L3276)) |

### 3.3 Acceptance Criteria

Outcome-level (Tier 1) commitments to the human at delivery time. `Derives from` cites only `Goal | An | Cn` (allow-list). `Mn` is forbidden here — milestone is schedule (§3.4), not requirement.

| ID | Acceptance Criterion (outcome) | Derives from | Verified by | Milestone |
| -- | ------------------------------ | ------------ | ----------- | --------- |
| AC1 | MILES fullasync GRPO runs end-to-end under RLix scheduler with a single pipeline (4 GPU, tp=2 partial overlap) | `Goal`, `A8` | E3 | M11.1 |
| AC2 | Trajectory `weight_versions` is consumable by `--max-weight-staleness 2` (per-decision version label is preserved end-to-end) | `A8` | E3 | M11.1 |
| AC3 | Subset sleep/wake of overlap engines releases GPU VRAM while non-overlap engines keep serving (post-sleep VRAM under `args.miles_post_sleep_vram_threshold_gb`) | `Goal`, `A2`, `A7` | E1 | M11.1 |
| AC4 | tp>1 mixed receiver mask sync (cpu_serialize colocate + NCCL broadcast non-colocate in same bucket) succeeds without leak under M11.1 transport (`cpu_serialize` only, `C12`) | `C5`, `C12`, `C16` | E2 | M11.1 |
| AC5 | Megatron NCCL destroy/reload cycle stable across ≥3 train→sync→train iterations (no VRAM leak, no NCCL group leak) | `A10` | E9 | M11.1 |
| AC6 | Multi-turn preempt resumes from the aborted turn (preserving completed turns); exhausting `MAX = total_engine_count` raises `EnginePreemptedError` (no silent group-recycle) | `A9` | E4 | M11.1 |
| AC7 | F10 startup fail-fasts reject misconfigurations: non-contiguous mapping, async_save, RadixTreeMiddleware, streaming, EP/MoE>1, sglang_data_parallel_size>1, transport mismatch, resume args, **non-fullasync generation, Megatron parallelism non-divisible by train_devices, infer_device_mapping non-divisible by rollout_num_gpus_per_engine** | `C1`,`C2`,`C3`,`C4`,`C5`,`C6`,`C7`,`C8`,`C9`,`C10`,`C11`,`C12`,`C13`,`C17`,`C18`,`C19`,`C21`,`C22`,`C23` | E7, E8 | M11.1 |
| AC8 | Standalone `train_async.py` continues to run unchanged when `RLIX_CONTROL_PLANE` is unset (no behavioral regression on the standalone path) | `Goal` | E10 | M11.1 |
| AC9 | Two MILES pipelines admitted under one orchestrator share GPUs via `RollResourceManagerProxy`; both schedulable on the same physical GPUs | `Goal`, `A1` | E11 | M11.2 |
| AC10 | Pipeline B init under contention gets a partial GENERATION allocation: shell engines stay shell (no Ray actor, no GPU), active engines serve correctly, router worker list contains only active subset | `C14` | E12 | M11.2 |
| AC11 | Expand-before-first-after_training uses base v=-1 short-circuit (no transport, no finalize, only `manager.set_weight_version(-1)`); engine reports v=-1 base label equivalent to `args.hf_checkpoint` weights | `C13` | E5 | M11.2 |
| AC12 | Donor-shrink-before-recipient-expand ordering enforced: `T1 < T2 < T3` (A.shrink returns < B.expand starts < B.request returns) — Pipeline B never spawns Ray/SGLang actor on GPUs A still holds | `A5` | E13 | M11.2 |
| AC13 | In M11.2 partial-allocation + own-train cycle, the router **suspends** dispatch when its active set is empty (background generation requests block until expand/enable, not crash). On bounded timeout, router emits HTTP 503 + `X-Miles-Preempt` sentinel and the client translates to `EnginePreemptedError` so the existing turn-redispatch + fatal-sentinel chain handles it (does NOT consume the 60-retry HTTP budget) | `A11`, `C20` | E14 | M11.2 |

### 3.4 Milestones

Single canonical table — staged delivery, organizing AC delivery into sign-off boundaries.

| Milestone | Scope | Key deliverables | Delivers AC | Gates |
| --------- | ----- | ---------------- | ----------- | ----- |
| **M11.1** | First build, vast.ai single-pipeline | F1–F12 happy path; cpu_serialize colocate (only); NCCL broadcast non-colocate; multi-node DP capable; turn-level redispatch; F10 fail-fasts (C6/C10/C11/C12/C13/C20 enforced) | `AC1 ∧ AC2 ∧ AC3 ∧ AC4 ∧ AC5 ∧ AC6 ∧ AC7 ∧ AC8` | Gate 1, Gate 2, Gate 2.5, Gate 3 |
| **M11.2** | Dual-pipeline happy path | Shell partial GENERATION allocation; RolloutManager shell→loading→active lazy ctor; coordinator base v=-1 short-circuit (Fix #13); F10 startup fail-fast on resume (Fix #14); donor-shrink-before-recipient ordering; **router 0-active suspend (C20 — `_use_url` async + sentinel + client translation)** | `AC9 ∧ AC10 ∧ AC11 ∧ AC12 ∧ AC13` | Gate 4 (c)+(d)+(e)+(f) happy path; admission_epoch race / graceful drain / multi-pipeline cleanup are M11.2-tagged but **NOT** Gate 4 pass criteria |
| ~~M11.3~~ | (skipped) | Cross-node rollout engine not supported; multi-node DP capability rolled into M11.1 | — | — |
| **M11.4** | (future) LoRA + multi-stream aggregation | LoRA cpu_serialize / cuda_ipc adapter under RLix; multi-stream impl in F9 reporter (M11.1 hook signature already preserves `mode/adapter_id` nullable) | (future — AC not yet declared) | LoRA-specific gates (out of scope here) |
| **M11.5** | (future) Production hardening | Ingress 503 + 5xx synthesis (preempt sentinel — reuses C20 `X-Miles-Preempt` header) + cleanup daemon + NCCL port hardening (port cooldown queue, receiver crash tolerance, conditional port leak, periodic port GC) | (future — AC not yet declared) | Production gates (out of scope here) |
| **M11.6** | (future) cuda_ipc colocate transport | New cuda_ipc adapter (CPU cache → per-bucket H2D staging → IPC handle serialize, ~50–80 lines, NOT reusing `_send_to_colocated_engine`); startup smoke-test capability check | (future — AC not yet declared) | cuda_ipc gate added to Gate 2.5 cycle |

### 3.5 Strategy Comparison

| Alternative | When chosen | Trade-off | Status |
| ----------- | ----------- | --------- | ------ |
| **`cpu_serialize`** | Colocate receiver on M11.1 vast.ai | wrapper auto-deref bytes → tmpfs `/dev/shm/miles_cpu_bucket_{uuid}.pt` → SGLang HTTP route; ~1 extra copy through plasma + tmpfs (memcpy-bound); requires no special container caps | **M11.1 default + only colocate transport** |
| **`cuda_ipc`** | Colocate receiver on M11.6 production | Zero-copy GPU IPC handle (baseline performance); requires `--ipc=host` or `CAP_SYS_PTRACE`; **must use new adapter** (CPU cache → per-bucket H2D staging → IPC serialize) NOT reuse standalone `_send_to_colocated_engine` (live `dist.gather_object` conflicts with F4 destroy-NCCL ordering) | M11.6 only; standalone cuda_ipc path unchanged |
| **NCCL broadcast** | Non-colocate receiver — **all milestones, partial-overlap requirement** | TCP rendezvous + warmup allreduce + per-bucket H2D staging on cache_owner GPU + dynamic group destroy with `is_group_exist` no-op guard for colocate-only ranks | Always required |

Receiver-side dual-mask: `cpu_serialize_local_ranks` (M11.1 colocate) + `broadcast_local_ranks` (always) + `ipc_local_ranks` (M11.6); each receiver method skips when its rank not in mask.

## 4. Critical Views

### 4.1 Architecture Integration View

*Audit question: where does the new control plane attach to existing MILES, and what stays unchanged?*

```mermaid
flowchart LR
  subgraph Driver[Driver entry]
    Std[train_async.py standalone unchanged]
    Rlix[examples/rlix/run_miles_rlix.py NEW]
  end

  subgraph RLixSide[RLix side NEW actors]
    Orch[Orchestrator]
    Sched[Scheduler]
    Coord[MilesCoordinator manual init no super]
    Pipe[MilesPipeline initialize_pipeline sync def]
    Service[MilesModelUpdateService atomic sync unit]
    Hooks[MilesRLixHooks progress wire]
    Proxy[RollResourceManagerProxy shared PG]
  end

  subgraph MilesSide[MILES side modified]
    Provider[MilesPlacementProvider NEW adapter]
    RM[RolloutManager EngineInfo 5-state]
    SE[SGLangEngine update_weights_from_cpu_bucket is_idle abort_all]
    Router[Miles Router admission plus async _use_url block-with-notify]
    HttpUtil[http_utils _post sentinel detect plus post RayTaskError unwrap]
    MT[multi_turn.generate snapshot restore retry]
    FA[fully_async_rollout RLixHooks plus fatal sentinel]
    TG[RayTrainGroup worker_placements plus build_cpu_bucket_cache fan-out]
    Cache[cpu_bucket_cache.py NEW]
    Patch[SGLang fork patches cpu_serialize_*.patch]
  end

  Std --> RM
  Rlix --> Orch
  Rlix --> Coord
  Coord --> Pipe
  Pipe --> Provider
  Provider --> Proxy
  Pipe --> TG
  Pipe --> RM
  Coord --> Service
  Sched --> Coord
  Service --> TG
  Service --> RM
  RM --> SE
  SE --> Router
  MT --> HttpUtil
  HttpUtil --> Router
  FA --> MT
  FA --> Hooks
  Hooks --> Coord
  TG --> Cache
  SE --> Patch

  classDef stable fill:#eee,stroke:#888
  classDef new fill:#cfc,stroke:#363
  classDef mod fill:#ffd,stroke:#960
  class Std stable
  class Rlix,Orch,Sched,Coord,Pipe,Service,Hooks,Proxy,Provider,Cache,Patch new
  class RM,SE,Router,HttpUtil,MT,FA,TG mod
```

Standalone `train_async.py` unchanged. New RLix entry `run_miles_rlix.py` drives the orchestrator chain. `MilesCoordinator` does NOT subclass `PipelineCoordinator`. `MilesPlacementProvider` is a real adapter from ROLL `List[List[Dict]]` to MILES per-worker `WorkerPlacement`. SGLang vendored fork patches add a single new HTTP admin route + in-process `torch.load` path. **`http_utils` is the load-bearing seam for the M11.2 router 0-active suspend** — `_post` translates the router's `X-Miles-Preempt` 503 sentinel into `EnginePreemptedError` so the existing `multi_turn` redispatch + `fully_async` fatal-sentinel chain handles it cleanly.

### 4.2 Runtime / Data Path View

*Audit question: how does the steady-state train step plus scheduler-driven expand sequence work, and what holds the atomic invariant?*

```mermaid
sequenceDiagram
  participant Sched as RLix Scheduler
  participant Coord as MilesCoordinator
  participant Pipe as MilesPipeline
  participant Train as RayTrainGroup cache owner
  participant Svc as MilesModelUpdateService
  participant E0 as Engine 0 overlap slept
  participant E1 as Engine 1 non-overlap active

  Note over E0,E1: Initial state E0 slept and E1 serving v2

  Sched->>Pipe: request_cluster_gpus actor_train ACTOR_TRAINING step3
  Pipe->>Train: onload then reload NCCL
  Note over Train,E1: training on overlap GPUs while E1 keeps serving v2

  Train->>Train: train_step 3
  Train->>Train: build_cpu_bucket_cache 3 HF gather single owner
  Train->>Train: offload then destroy NCCL via ReloadableProcessGroup

  Pipe->>Coord: sync_base_weights_to_active 3
  Note over Coord: under _resize_sync_lock and set _cache_ready_step to 3 atomic with sync
  Coord->>Svc: sync_selected_workers sid E1 v3
  Note over Svc: atomic unit a transport b finalize c set_weight_version
  Svc->>E1: per bucket cpu_serialize tmpfs OR NCCL broadcast staging
  Svc->>E1: finalize_weight_update
  Svc->>E1: set_weight_version 3
  Svc-->>Coord: ok
  Coord-->>Pipe: return v3
  Pipe->>Sched: notify_release_cluster_gpus actor_train

  Sched->>Coord: resize_infer add E0
  Note over Coord: under _resize_sync_lock with guard _cache_ready_step is not None
  Coord->>E0: manager.expand_engines E0 shell to loading or offloaded to loading
  Coord->>Svc: sync_selected_workers sid E0 v3
  Svc->>E0: same atomic unit a plus b plus c
  Coord->>E0: manager.activate_routing E0
  Note over E0,E1: Both at v3
```

Atomic unit covers (a) per-bucket transport + (b) per-engine `finalize_weight_update` fan-out + (c) `manager.set_weight_version`, all within one `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S`. Pipeline never directly calls finalize or `set_weight_version`. `_cache_ready_step` updated inside `_resize_sync_lock` BEFORE the service call returns. Per-bucket payload carries NO `weight_version` (avoids version inversion). v=-1 short-circuit (Fix #13/#15) skips (a)+(b), runs only (c).

Error paths: receiver hang → 150s timeout → coordinator → pipeline crash (fail-fast); turn redispatch exhausted → `EnginePreemptedError` → fully_async fatal sentinel → pipeline crash (NOT silent group recycle); router metadata missing in RLix mode → `RLixRouterMetadataError` → fatal sentinel; **M11.2 router 0-active dispatch → `_use_url` blocks-with-notify until expand/enable; on `MILES_ROUTER_DISPATCH_WAIT_S` timeout → router 503 + `X-Miles-Preempt` → client `_post` raises `EnginePreemptedError` → turn redispatch / fatal sentinel** (NEVER generic RuntimeError, NEVER `httpx.HTTPStatusError` retry-loop burn).

### 4.3 Physical Topology View

*Audit question: which logical roles share which physical GPUs in the M11.1 dev gate (4 GPU, tp=2)?*

ASCII layout — Mermaid cannot express overlap rectangles cleanly:

```text
Node 0 (4 GPU dev gate)

GPU lane:           GPU 0       GPU 1       GPU 2       GPU 3
                    -----       -----       -----       -----
actor_train      [============]                                       (overlap subset)
actor_infer      [================================================]   (full set)
SGLang Engine 0  [============]                                       (TP=2, overlap GPUs)
SGLang Engine 1                              [============]            (TP=2, non-overlap)

Behavior:
- During train: Engine 0 sleep_partial (release weights+kv+graph); Engine 1 keeps serving.
- After train + sync: Engine 0 wake_up; selective sync to E0; activate_routing.
- Active set bootstrap: Coordinator._active_engine_indices = {1} initially; expand adds {0}.
```

Multi-node DP supported in M11.1 (each engine still node-local):

```text
Node 0:  train [0,1] + infer engine0 [0,1] + infer engine1 [2,3]
Node 1:  infer engine2 [0,1] + infer engine3 [2,3]   (multi-node DP, each engine node-local)
```

Cross-node single rollout engine FORBIDDEN in M11.1 — `WorkerPlacement.placement_group` is per-node; a single SGLang engine occupying GPUs [0,1] on node0 + [0,1] on node1 violates **C7**.

**M11.2 partial-allocation runtime (where C20 fires)**: Pipeline B may receive only [0,1] of declared [0,1,2,3] → engine 0 active on [0,1], engine 1 stays in `shell` state (no Ray actor). When B's own train cycle then sleeps engine 0, B's runtime active engine count = 0. C3 declared-topology precondition still holds; C20 router suspend handles the runtime gap.

## 5. Decision Map

*Audit question: which assumptions, constraints, and acceptance criteria derive each decision?*

```mermaid
flowchart TD
  Ctx[Problem Context partial overlap multi-tenant gap]
  A[Assumptions A1..A11]
  C[Constraints C1..C20]
  AC[Acceptance AC1..AC13]

  D0[D0 MILES schedulable under RLix partial overlap]

  D1A[D1 Inject 3 RLix actors via real adapter no super]
  D1B[D1 Shared PG via MilesPlacementProvider real adapter]
  D1C[D1 Milestoning M11.1 vast.ai single-pipeline first build]

  D2A[D2 F1-F3 partial sleep wake plus router admission]
  D2B[D2 F4-F6 cache plus selective sync atomic unit]
  D2C[D2 F7-F12 RLix glue namespace lifecycle flag topology]

  D3A[D3 EngineInfo 5-state plus admission close abort drain]
  D3B[D3 multi-turn snapshot restore redispatch on preempt]
  D3C[D3 two-path refresh active in-flight plus expand sync]
  D3D[D3 atomic sync unit transport plus finalize plus set_weight_version]
  D3E[D3 Init bootstrap pre-register before scheduler resize_infer fires]
  D3F[D3 F4 sequence build then offload then destroy NCCL then sync]
  D3G[D3 Router 0-active suspend block-with-notify plus HTTP sentinel]

  D4A[D4 EngineInfo state and routing lock and preempted set]
  D4B[D4 Router admission and metadata injection plus async _use_url]
  D4C[D4 CPU bucket cache slot and cache_ready_step single owner]
  D4D[D4 WorkerPlacement node-local multi-node compatible]
  D4E[D4 SyncSessionPlan dataclass dual mask receivers]
  D4F[D4 X-Miles-Preempt sentinel header plus _post translation contract]

  D5A[D5 cpu_serialize tmpfs path M11.1 only]
  D5B[D5 Dynamic NCCL group TCP rendezvous plus warmup allreduce]
  D5C[D5 v equals minus one short circuit no transport no finalize]
  D5D[D5 SGLang fork patches cpu_serialize routes]
  D5E[D5 SharedStorage master_port claim forbid zero]
  D5F[D5 RLix tiny num_gpus_per_actor manual CVD]
  D5G[D5 Function-top guarded import RayTaskError unwrap in post]

  D6[D6 Gate 1 2 2.5 3 M11.1 plus Gate 4 c d e f M11.2]

  Ctx --> D0
  AC --> D0
  AC --> D1A
  AC --> D1B
  AC --> D1C
  A --> D1A
  C --> D1A
  C --> D1B
  C --> D1C
  D0 --> D1A
  D0 --> D1B
  D0 --> D1C
  D1A --> D2A
  D1A --> D2B
  D1A --> D2C
  D1B --> D2C
  D1B --> D4D
  D1C --> D5A
  D2A --> D3A
  D2A --> D3B
  D2A --> D3G
  D2B --> D3C
  D2B --> D3D
  D2A --> D3E
  D2B --> D3E
  D2B --> D3F
  D3F --> D4C
  D3A --> D4A
  D3A --> D4B
  D3B --> D4B
  D3G --> D4B
  D3G --> D4F
  D3C --> D4C
  D3D --> D4C
  D2C --> D4D
  D3D --> D4E
  D3C --> D4E
  D4C --> D5A
  D4C --> D5B
  D4E --> D5B
  D5B --> D5E
  D5A --> D5D
  D3E --> D5C
  D2C --> D5F
  D4F --> D5G
  D4A --> D6
  D4C --> D6
  D4F --> D6
  D5A --> D6
  D5B --> D6
  D5G --> D6
```

| Decision | Chosen | Depends On (Ctx / A / C / Dn) | AC served | Audit |
| -------- | ------ | ----------------------------- | --------- | ----- |
| D0 | Goal: partial overlap, subset sleep/wake, CPU bucket cache, selective sync | Ctx | AC1, AC8, AC9 | scope alignment vs ROLL/NeMo F1–F12 |
| D1.A | RLix actor injection: 3 new actors `MilesPipeline / MilesCoordinator / MilesModelUpdateService`; manual-init `MilesCoordinator` | D0, A6, C1–C9 | AC1, AC9 | confirm no `super().__init__` to ROLL |
| D1.B | Shared PG: real `MilesPlacementProvider` adapter on shared `RollResourceManagerProxy` | D0, A1 | AC9 | provider receives proxy by injection; `WorkerPlacement` per-worker |
| D1.C | Milestoning: M11.1 cpu_serialize-only first build; M11.2 dual-pipeline + 0-active suspend; M11.4 LoRA; M11.5 hardening; M11.6 cuda_ipc | D0, C12 | AC1, AC9 | feature labels match milestone tags |
| D2.A | Sleep/wake/route module: F1+F2+F3 owned by `RolloutManager` + Miles Router | D1.A | AC3, AC6 | engine_index single dim (TP fan-out internal) |
| D2.B | Cache + sync module: F4+F5+F6 owned by service; cache owner = `_is_distributed_src_rank` | D1.A, A6 | AC1, AC4, AC11 | single slot; `_cache_lock` covers full transport |
| D2.C | RLix glue: F7–F12 (one merged adapter file allowed but capabilities preserved) | D1.A, D1.B | AC7, AC8 | namespace + reg + `DO_TIME_SHARING` preserved as 3 capabilities |
| D3.A | Sleep/wake: admission close → abort → drain (`/v1/loads`, A7) → release → post-sleep VRAM assert | D2.A | AC3 | drain reads `/v1/loads`, NOT `/server_info` |
| D3.B | Turn redispatch: snapshot/restore in `multi_turn.py`; `MAX_ATTEMPTS = total_engine_count`; exhaust → fail fast | D2.A, A9 | AC6 | counter is total (incl. sleeping) NOT active |
| D3.C | Two-path refresh: active in-flight (after_training) + expand sync (scheduler-driven), share one cache | D2.B, A8 | AC1, AC2 | pipeline does NOT call expand path |
| D3.D | Atomic sync unit: `service.sync_selected_workers` does (a)+(b)+(c) within one timeout | D2.B | AC1, AC4 | pipeline never calls finalize / set_weight_version directly |
| D3.E | Init pre-register: Step 6.6 pre-registers manager + service + base version + active bootstrap BEFORE Step 7 GEN | D2.A, D2.B, A5 | AC10, AC11, AC12 | Phase 5 before Phase 6; Gate 4 (e) timestamp test |
| D3.F | F4 sequence: build CPU cache → offload → destroy NCCL → sync, in this order | D2.B, A6 | AC4, AC5 | reordering deadlocks gather (build after offload) or serves stale (sync before offload) |
| D3.G | Router 0-active suspend: `_use_url` async + `asyncio.Condition` block-with-notify; bounded by `MILES_ROUTER_DISPATCH_WAIT_S`; timeout → HTTP 503 + `X-Miles-Preempt` sentinel; helpers stay sync, endpoints notify | D2.A, A11 | AC13 | NEVER raise generic RuntimeError from `_use_url`; NEVER raise `EnginePreemptedError` across HTTP (lives in `miles.rollout.base_types`, would not survive serialization) |
| D4.A | EngineInfo: 5-state (`shell/active/disabling/offloaded/loading`) single source of truth | D3.A | AC3, AC10 | no worker-side `is_engine_resident_on_gpu` flag |
| D4.B | Router admission: `/disable_worker` `/enable_worker` `/remove_worker` + metadata mutation on `/generate` only; `_use_url` is `async def` block-with-notify | D3.A, D3.B, D3.G | AC3, AC6, AC13 | disabled NOT in `_use_url`; health-check excludes disabled; `do_proxy` MUST `await self._use_url()` |
| D4.C | Version label: `version = _cache_ready_step`; single-shot `manager.set_weight_version`; updated inside `_resize_sync_lock` | D3.C, D3.D, D3.F | AC2, AC11 | per-bucket payload carries NO version field |
| D4.D | WorkerPlacement: per-worker, node-local gpu_ids, multi-node compatible | D1.B, D2.C, C7 | AC9 | `base_gpu_id=0` post-CVD local; never `wp.gpu_ids[0]` physical |
| D4.E | SyncSessionPlan: frozen dataclass; `target_handles` + per-engine masks; built once per sync at service entry | D3.C, D3.D | AC4 | cache_owner does NOT call back to service / manager |
| D4.F | Router preempt-sentinel HTTP contract: 503 + `X-Miles-Preempt: empty_active_set_timeout` header is the typed marker that crosses HTTP; client `_post` translates BEFORE retry loop into `EnginePreemptedError` | D3.G | AC13 | EnginePreemptedError NEVER lives inside the router process; client translation is the only valid bridge |
| D5.A | cpu_serialize: M11.1 colocate; ObjectRef → tmpfs → HTTP route | D4.C, D1.C, A3, A4, C12 | AC4 | F10 asserts transport |
| D5.B | Dynamic NCCL: TCP rendezvous (NOT `dist.new_group`) + warmup allreduce + `is_group_exist` no-op destroy | D4.C, D4.E | AC4, AC5 | sender H2D staging only for NCCL receivers |
| D5.C | v=-1 short-circuit: skip transport + finalize; only `manager.set_weight_version(-1, targets)` | D3.E | AC11 | Gate 4 (d) instrumented call counters |
| D5.D | SGLang fork patches: 4 patches under `external/sglang/3rdparty/cpu_serialize/` | D5.A | AC4 | rebase per SGLang bump |
| D5.E | master_port: `get_free_port()` + SharedStorage claim, never 0 | D5.B, C16 | AC9 | F10 forbids `port=0` |
| D5.F | Tiny GPU + manual CVD: RLix path uses `num_gpus_per_actor=0.01` + manual CVD + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + `local_rank=0` | D2.C | AC9 | no env mutation post-import |
| D5.G | `EnginePreemptedError` imported **outside the `try` path** in `_post` and `post()` (module-top preferred if cycle-free — verified for `miles.rollout.base_types`; otherwise function-top fallback; NEVER inside try — `UnboundLocalError` trap); module-top guarded `try/except ImportError: RayTaskError = ()` for Ray-absent envs; `post()` unwraps `RayTaskError.cause` BEFORE generic `Exception` fallback | D4.F | AC13 | module-top preferred + module-top RayTaskError guard is the recommended baseline; function-top fallback is acceptable if a future cycle is introduced |
| D6 | Gates: Gate 1/2/2.5/3 = M11.1; Gate 4 (c)+(d)+(e)+(f) = M11.2 happy path; cuda_ipc gate = M11.6 | D3, D5 | AC1, AC9 | Gate 2.5 covers dual-mask `tp>1`; Gate 4 (f) covers C20 router suspend with positive + negative + counter + distributed-POST sub-tests |

## 6. Evidence & Stop Conditions

This section **mirrors what the plan tells the agent**. tldr-plan does not invent new evidence or stop conditions — only surfaces what the plan defines and flags gaps the human auditor should fix in the plan before shipping.

#### 6.1 Evidence Required (per plan)

| ID | Binds to (D / A / C / M) | AC verified | Evidence the plan defines | Before Done? |
| -- | ------------------------ | ----------- | ------------------------- | ------------ |
| E1 | D3.A, A7 | AC3 | Gate 1 (a)/(b): `_preempted_engines` set+cleared; post-sleep VRAM under threshold (read SGLang `/server_info` `memory_usage` GB, NOT actor `torch.cuda`) | yes |
| E2 | D3.D, C16 | AC4 | Gate 2.5: mixed-mask sync (cpu_serialize + NCCL broadcast in same bucket); per-bucket barrier; warmup allreduce on every NCCL group CREATE; `is_group_exist` no-op DESTROY | yes |
| E3 | D3.C, A8 | AC1, AC2 | Gate 3: long-running fullasync GRPO with active in-flight refresh; quantify mis-attribution count vs in-flight × decode-step; trajectory `weight_versions` consumable by `--max-weight-staleness 2` | yes |
| E4 | D3.B, A9 | AC6 | Gate 3 (d): inject `sleep_partial` mid-turn; verify redispatch picks new engine and preserves completed turns; exhaustion raises `EnginePreemptedError` (NOT silent group recycle) | yes |
| E5 | D3.E, A5, C13 | AC11 | Gate 4 (d): instrumented assertions that `cache_owner_actor.run_sync_session` AND per-engine `finalize_weight_update` call counters remain at pre-expand values for v=-1 path | yes (M11.2) |
| E6 | D3.D, D4.C |  | Per-bucket payload audit: NO `weight_version` field; finalize call counter audit; `set_weight_version` audit | yes |
| E7 | C13 | AC7 | F10 startup assert that `args.load` and `args.ref_load` are in {None, `args.hf_checkpoint`} when `DO_TIME_SHARING` is true | yes |
| E8 | C12, C17, C18, C20 | AC7 | F10 startup asserts on RLix mode misconfiguration (transport, RadixTree, streaming, `MILES_ROUTER_DISPATCH_WAIT_S > 0`) | yes |
| E9 | A10, D3.F | AC5 | Gate 2.5 destroy/reload cycle ≥3 steps; verify VRAM after offload + no NCCL group leak | yes |
| E10 | D2.C | AC8 | Standalone `train_async.py` smoke test still passes when `RLIX_CONTROL_PLANE` is unset (no behavioral regression on the standalone path) | yes |
| E11 | D1.B, D1.C | AC9 | Gate 4 setup: two MILES pipelines admitted under one orchestrator share `RollResourceManagerProxy`; per-pipeline namespace isolation verified | yes (M11.2) |
| E12 | D3.E, D4.A, C14 | AC10 | Gate 4 (c): Pipeline B partial allocation — verify all of: init returns SUCCESS, `coordinator.get_active_engines()` returns active subset, shell engines have `handle is None`, router worker list contains only active subset, M1 subset assert passed, Step 7.3 consistency assert passed | yes (M11.2) |
| E13 | D3.E, A5 | AC12 | Gate 4 (e): timestamp test on coordinator + driver verifies `T1 (A.shrink returns) < T2 (B.expand starts) < T3 (B.request returns)` | yes (M11.2) |
| E14 | D3.G, D4.B, D4.F, D5.G, A11, C20 | AC13 | Gate 4 (f) with deterministic P1→P2→P3 ordering: **P1** — trigger `_before_training` and poll `GET /admission_state` (test-only diagnostic, gated by `MILES_ROUTER_TEST_HOOKS=1`; router runs in separate FastAPI process so test cannot read `self.enabled_workers` directly) until `enabled == [] AND dead == []` AND engine 1 still in `shell` (the race-eliminator barrier). **P2** — only after P1 confirms empty active set, inject background `/generate` via production `miles.utils.http_utils.post`; verify task is still pending after 1s (asyncio block-pending assertion). **P3** — trigger `_after_training` → expand → enable → verify suspended task resumes, completes, counter returns to 0 (visible via `/admission_state`, NOT direct `self.worker_request_counts` access). **Negative** — `MILES_ROUTER_DISPATCH_WAIT_S=2`, hold 3s; verify client `_post` raises `EnginePreemptedError` (NOT `httpx.HTTPStatusError`, NOT `RuntimeError`), response carried `503` + `X-Miles-Preempt: empty_active_set_timeout`, retry budget NOT consumed. **Counter** — two concurrent suspended dispatchers resume atomically through the lock with no `worker_request_counts` drift. **Distributed POST (conditional, NOT a Gate 4 hard gate)** — only run when target env has `_distributed_post_enabled=True` (M11.2 vast.ai dev gate typically OFF); when run, verify caller receives `EnginePreemptedError`, NOT `RayTaskError`, NOT fallback-then-retry log | yes (M11.2) |

#### 6.2 Stop Conditions (per plan)

The plan instructs the agent to stop and ask if:

- implementation requires touching out-of-scope files
- a lower-level implementation decision lacks a parent decision
- required tests force changing expected old behavior (standalone path or tool-error group recycle)
- the architecture insertion point appears wrong after code inspection (e.g. `MilesCoordinator` cannot avoid `super().__init__`)
- public API compatibility must be broken (e.g. existing `RolloutManager(args, pg=...)` standalone signature)
- assumption A1 / A5 / A6 / A11 fails during implementation (cache_owner reports zero or multiple owners; `_expand_workers` reads `_cache_ready_step is None` mid-init; 0-active state turns out to be unreachable in M11.2 — re-derive C20)
- the atomic-unit invariant (D3.D / E6) cannot be preserved without splitting RPCs
- v=-1 short-circuit cannot be implemented because base weights from `args.hf_checkpoint` differ from `build_cpu_bucket_cache(-1)` bytes (C13 must not be loosened)
- topology asserts in F10 conflict with feasible deployment (e.g. user reports vast.ai container has `--ipc=host` after all → revisit C12)
- C20 router suspend cannot be implemented without making sync helpers async (the spec mandates helpers stay sync; only endpoints + `_health_check_loop` notify)
- `EnginePreemptedError` cannot be imported at module top because of a future cycle AND function-top fallback also fails (e.g., signature constraints) — use the helper-pattern alternative documented in §Feature 3 §8

(The plan defines no stop condition for M11.2 multi-pipeline race scenarios — admission_epoch race, donor-shrink race — those are tagged "M11.2 follow-up, not Gate 4 pass criteria"; audit gap, see `## 7`. The 0-active-engine generation-path case is now first-class handled by C20 / AC13 / E14 — no longer a gap.)

## 7. Audit Checkpoints

Two lenses: **Intent** (does the plan reflect what the human wants?) and **Shippability** (will the agent reading the plan have to guess?).

- [ ] **Intent**: `## 1 Problem Context` accurately frames why MILES needs RLix scheduling — partial overlap multi-tenancy is the correct goal.
- [ ] **Intent**: `## 3.1 Out of Scope` correctly excludes MoE/EP, vLLM, PD, request-level migration — the human is intentionally leaving these out.
- [ ] **Intent**: `## 3.5 Strategy Comparison` choice of `cpu_serialize` for M11.1 reflects actual deployment target (vast.ai container) — not a temporary expedient.
- [ ] **Intent**: AC1–AC13 (§3.3) cover what the human auditor expects to receive at each milestone — no missing user-visible outcomes.
- [ ] **Intent**: D1.a / D1.b / D1.c (3 new RLix-side actors + adapter, manual `MilesCoordinator`, M11.x phasing) match the human's intended integration shape.
- [ ] **Intent**: assumptions **A5** (Phase-5-before-Phase-6), **A8** (bounded mis-attribution during active refresh), and **A11** (0-active-engine is legit M11.2 status) are risks the human is willing to bet on.
- [ ] **Intent**: M11.x phasing (M11.1 first, M11.2 dual-pipeline + 0-active suspend, M11.4 LoRA, M11.5 hardening, M11.6 cuda_ipc) reflects the human's actual delivery cadence.
- [ ] **Shippability**: hard constraints C1–C23 are present in the plan with concrete enforcement points (the agent will encounter and respect them).
- [ ] **Shippability**: every `Dn` in §5 cites ≥1 `AC` in §3.3 (no orphan decisions; a `Dn` with no `AC` is either over-engineering or a hidden goal).
- [ ] **Shippability**: every `AC` in §3.3 has at least one `D` citing it AND at least one `E` verifying it (no orphan promises, no unverifiable promises).
- [ ] **Shippability**: D3.d atomic sync unit + D5.c v=-1 short-circuit are described concretely enough in the plan that the agent will not split the RPC or skip the version-label publish (E2, E5, E6 enforce).
- [ ] **Shippability**: D3.f F4 sequence is unambiguous in the plan; the agent will not reorder steps or build cache after offload (E9 enforces).
- [ ] **Shippability**: cache owner discovery happens at init (Step 6.5 `report_cache_owner_role`) and NOT in sync hot path — A6 holds.
- [ ] **Shippability**: stop conditions (`## 6.2`) are present in the plan; the agent has clear halt points (mirroring §6.2).
- [ ] **Shippability**: evidence E1–E14 is achievable before the agent declares done — gate procedures are specified concretely (`/server_info` field names, `/v1/loads` endpoint, instrumented counter expectations, `T1<T2<T3` timestamp test, asyncio block-pending assertion, sentinel header inspection, distributed-POST configuration matrix).
- [ ] **Shippability**: M11.2 0-active-engine generation-path is first-class handled by C20 / AC13 / E14 — router blocks-with-notify (`asyncio.Condition`), timeout returns HTTP 503 + `X-Miles-Preempt` sentinel, client `http_utils._post` translates to `EnginePreemptedError` BEFORE retry loop, `http_utils.post` unwraps `RayTaskError` (imported under guarded `try/except ImportError: RayTaskError = ()`) for the distributed POST path. Implementation must touch FOUR pieces together: (1) router (`_use_url` async + `do_proxy` await + endpoint notify), (2) `http_utils._post` (sentinel detection — `EnginePreemptedError` imported **outside the try path**: module-top preferred if cycle-free, otherwise function-top fallback; NEVER inside try), (3) `http_utils.post` (`RayTaskError` unwrap before generic `Exception` fallback; import guarded so Ray-absent envs still load the module), (4) Gate 4 (f) with positive + negative + counter + distributed-POST sub-tests. Failure modes to reject in code review: `EnginePreemptedError` import inside try block → `UnboundLocalError` instead of `EnginePreemptedError`; eager unguarded `import ray.exceptions` at module top → breaks any caller that imports `http_utils` without Ray installed; `post()` patched but not `_post` (or vice versa) → silent partial regression.
- [ ] **Shippability gap**: the plan does not define explicit stop conditions for M11.2 multi-pipeline race scenarios (admission_epoch / donor-shrink). Decide whether to add them to the source plan or accept "M11.2 follow-up; manual `ray stop` recovery" as the agent's policy.
- [ ] **Shippability gap**: receiver crash tolerance is classified M11.5; verify the agent will not silently add fault-tolerant port management on the M11.1 path.

## Appendix A: Full Decision Trace

| Decision | Layer | Chosen | Rejected | Depends On | Unlocks | User Audit |
| -------- | ----- | ------ | -------- | ---------- | ------- | ---------- |
| D0 | D0 | partial overlap multi-tenancy | tp=1 MVP; standalone-only | Ctx | all features | scope freeze |
| D1.A | D1 | new `MilesPipeline / MilesCoordinator / MilesModelUpdateService`; manual init | subclass `PipelineCoordinator`; reuse `FullFinetunePipeline` | D0, A6 | F5+6, F7–F11 | confirm no `super().__init__` to ROLL coordinator |
| D1.B | D1 | real `MilesPlacementProvider` adapter on shared `RollResourceManagerProxy` | self-built PG; `RayVirtualCluster`-style wrapper | D0, A1 | F12 | adapter materializes ROLL `List[List[Dict]]` to `WorkerPlacement` |
| D1.C | D1 | M11.1 cpu_serialize-only first build then M11.2/4/5/6 | one-shot full parity; tp=1 → tp>1 staged | D0, C12 | all | feature labels match milestone tags |
| D2.A | D2 | F1+F2+F3 owned by `RolloutManager` + Miles Router | per-engine self-managed sleep | D1.A | F5+6 | engine_index single dim |
| D2.B | D2 | F4+F5+F6 owned by service; cache owner = `_is_distributed_src_rank` | dual cache slots; per-step versioned slots | D1.A, A6 | D3.D | single slot; `_cache_lock` covers full transport |
| D2.C | D2 | F7–F12 (one merged adapter file allowed) | merge namespace+lifecycle+flag into one capability | D1.A, D1.B | D6 | namespace + reg + `DO_TIME_SHARING` preserved |
| D3.A | D3 | admission close → abort → drain via `/v1/loads` → release → post-sleep VRAM assert | per-request id tracking | D2.A, A7 | F1, F2 | drain reads `slot["num_total_reqs"]`, NOT `/server_info` |
| D3.B | D3 | snapshot → retry, fail-fast on exhaustion | request-level deterministic migration | D2.A, A9 | F3 | commit-after-success; MAX = total_engine_count |
| D3.C | D3 | active in-flight + expand sync; same atomic unit | drain-then-sync; per-engine separate caches | D2.B, A8 | F5+6 | tolerated bounded mis-attribution |
| D3.D | D3 | `service.sync_selected_workers` does (a)+(b)+(c) | per-bucket version; pipeline-driven finalize; split RPCs | D2.B | F4, F5+6 | pipeline never calls finalize directly |
| D3.E | D3 | Step 6.6 pre-registers everything BEFORE Step 7 GEN | lazy registration after `_expand_workers` | D2.A, D2.B, A5 | F4, F5+6, F12 | Phase 5 before Phase 6 |
| D3.F | D3 | build → offload → destroy NCCL → sync | build after offload; sync before offload | D2.B, A6 | F4 | reordering deadlocks gather |
| D3.G | D3 | `_use_url` async + `_workers_changed: asyncio.Condition` block-with-notify; bounded by `MILES_ROUTER_DISPATCH_WAIT_S`; timeout → `_RouterDispatchTimeout` (router-local) → 503 + `X-Miles-Preempt` header | raise generic `RuntimeError` from `_use_url`; raise `EnginePreemptedError` from inside router process; retry-then-503 spillover | D2.A, A11 | M11.2 (Gate 4 (f)) | sync helpers + async endpoint notify; helpers stay `def`, endpoints carry `async with self._workers_changed: notify_all()` |
| D4.A | D4 | 5-state dataclass single dict | 5 parallel maps; `is_engine_resident_on_gpu` bool | D3.A | F2 | shell semantics for M11.2 partial allocation |
| D4.B | D4 | mutate JSON body in `do_proxy` (path == "generate") + admission state + `_use_url` async | header-only; SSE streaming; sync `_use_url` | D3.A, D3.B, D3.G | F3 | `Content-Encoding` stripped; payload re-serialized; `do_proxy` MUST `await self._use_url()` |
| D4.C | D4 | single cache, `_cache_ready_step` updated inside `_resize_sync_lock` | ROLL double-buffer; version bump per sync | D3.D, D3.F | F4, F5+6 | `sync_base_weights_to_active(step)` accepts step explicitly |
| D4.D | D4 | per-worker, node-local gpu_ids, multi-node compatible | global GPU id assumption; cluster wrapper | D1.B, D2.C, C7 | F12 | `base_gpu_id=0` post-CVD local |
| D4.E | D4 | frozen dataclass; `target_handles` + per-engine masks; built once at sync entry | service hot-path RPC to manager per bucket | D3.D | F4, F5+6 | cache_owner does NOT call back |
| D4.F | D4 | HTTP 503 + `X-Miles-Preempt: empty_active_set_timeout` is the typed marker that crosses HTTP boundary; client `_post` translates BEFORE `raise_for_status()` and BEFORE retry loop into `EnginePreemptedError` | raise `EnginePreemptedError` directly from router process (HTTP serialization loses type); retry on the sentinel response (burns 60-retry budget) | D3.G | M11.2 (Gate 4 (f)) | sentinel header is the contract; `multi_turn` redispatch + `fully_async` fatal-sentinel chain is reused intact |
| D5.A | D5 | `/dev/shm/miles_cpu_bucket_{uuid}.pt` + HTTP path | base64 in HTTP body; ObjectRef ID via subprocess `ray.init` | D4.C, A3, A4 | F4 (M11.1) | wrapper-owned cleanup |
| D5.B | D5 | per-sync CLASSIFY/CREATE/USE/DESTROY + warmup allreduce | persistent group; `dist.new_group` (cross-process invalid) | D4.C, D4.E | F4 (non-colocate) | per-bucket barrier before staging free |
| D5.C | D5 | service skips transport + finalize; only publishes label | run full transport on init bootstrap | D3.E | Gate 4 (d) | NO `run_sync_session` call; NO finalize; NO cache_owner GPU touch |
| D5.D | D5 | 4 patches under `external/sglang/3rdparty/cpu_serialize/` | upstream PR; vendored fork only | D5.A | F4 receiver | rebase per SGLang bump |
| D5.E | D5 | `get_free_port()` + SharedStorage claim | `master_port=0`; static config | D5.B, C16 | NCCL group | port 0 breaks multi-rank rendezvous |
| D5.F | D5 | RLix `num_gpus_per_actor=0.01` + manual CVD + NOSET env + `base_gpu_id=0` + `local_rank=0` | standalone fractional `0.4` reused; `wp.gpu_ids[0]` physical id | D2.C | F12 | no env mutation post-import |
| D5.G | D5 | `EnginePreemptedError` imported **outside the try path** for both `_post` and `post()` — module-top preferred if cycle-free (verified for `miles.rollout.base_types`), otherwise function-top fallback; module-top guarded `try/except ImportError: RayTaskError = ()` + `post()` unwraps `e.cause` BEFORE generic `Exception` fallback | import inside try block (UnboundLocalError trap); eager unguarded `import ray.exceptions` at module top (breaks Ray-absent envs); fallback-to-local on preempt (wastes round trip) | D4.F | Gate 4 (f) sub-tests | module-top preferred + module-top RayTaskError guard is the recommended baseline; function-top fallback acceptable if a future cycle is introduced |
| D6 | D6 | Gate 1/2/2.5/3 M11.1; Gate 4 (c)+(d)+(e)+(f) M11.2; cuda_ipc M11.6 | tp=1 MVP gate | D3, D5 | M11.1 sign-off | Gate 2.5 dual-mask `tp>1` coverage; Gate 4 (f) C20 four sub-tests |

## Appendix B: Full Module / File Boundary

| File / Module | Role | Allowed Change | Forbidden Responsibility | Parent Decision |
| ------------- | ---- | -------------- | ------------------------ | --------------- |
| [miles/ray/rollout.py](../external/miles/miles/ray/rollout.py) | RolloutManager + EngineInfo + routing lock | add `EngineInfo`, `shrink/expand_engines`, `activate_routing`, `set_weight_version`, subset APIs, `get_engine_handles`, `get_engine_count`, per-pipeline empty router on ctor | direct `service.X.remote` calls; per-request id tracking; `add_worker` outside `activate_routing`; `remove_worker` outside `shrink_engines` | D2.A, D4.A |
| [miles/backends/sglang_utils/sglang_engine.py](../external/miles/miles/backends/sglang_utils/sglang_engine.py) | SGLang Ray actor wrapper | add `update_weights_from_cpu_bucket`, `is_idle` (`/v1/loads`), `abort_all_requests`, post-sleep VRAM assert (server-side); M11.6 add `ipc_local_ranks` arg | reading actor-side `torch.cuda.memory_allocated()`; `os.environ['CUDA_VISIBLE_DEVICES']=...` after import | D3.A, D5.A |
| [miles/router/router.py](../external/miles/miles/router/router.py) | Miles Router admission state + 0-active suspend | add `/disable_worker` `/enable_worker` `/remove_worker`; mutate JSON body of `/generate` only; 4-dict worker lifecycle; `_use_url` filters `enabled - dead`; health-check excludes disabled; **`_use_url` becomes `async def` block-with-notify on empty active set**; `_workers_changed: asyncio.Condition`; `MILES_ROUTER_DISPATCH_WAIT_S` env-tunable timeout; `_RouterDispatchTimeout` local exception class; **`do_proxy` MUST `await self._use_url()` and translate `_RouterDispatchTimeout` to HTTP 503 + `X-Miles-Preempt: empty_active_set_timeout`**; load increment (`worker_request_counts[url] += 1`) stays INSIDE the condition lock so `_finish_url` accounting balances; every state-mutating endpoint (`add_worker` / `enable_worker` / `disable_worker` / `remove_worker`) ends with `async with self._workers_changed: self._workers_changed.notify_all()` (helpers themselves stay sync); `_health_check_loop` notifies on dead/recovered transitions | mutating non-`/generate` endpoints; admission_epoch race defense; polling disabled workers; raising `RuntimeError("no enabled live workers")` from `_use_url` (must suspend instead); raising `EnginePreemptedError` from inside the router process (the type does not survive HTTP — emit 503 + sentinel header instead); making `_use_url` helpers async (helpers stay sync; only endpoints/loop carry the notify) | D4.B, D3.G, D4.F |
| [miles/utils/http_utils.py](../external/miles/miles/utils/http_utils.py) (NEW row) | Async HTTP client + retry layer (local + Ray actor distributed POST) | `_post`: add `X-Miles-Preempt` sentinel detection BEFORE `raise_for_status()` and BEFORE the generic retry loop; raise `EnginePreemptedError`; add `except EnginePreemptedError: raise` clause ABOVE the generic `except Exception` so propagation is intact. **`EnginePreemptedError` must be imported outside the `try` path: module-top preferred if cycle-free** (verified for `miles.rollout.base_types`); otherwise function-top fallback. Either form is safe; importing inside `try` is the bug — Python treats `from X import Y` as local binding, so an exception before the import line leaves `EnginePreemptedError` unbound and `except EnginePreemptedError:` raises `UnboundLocalError`. `post()`: catch `RayTaskError` (imported under module-top `try/except ImportError: RayTaskError = ()` guard so Ray-absent envs still import the module; `except ()` matches nothing → branch is a safe no-op when Ray is missing), unwrap `e.cause`, re-raise `EnginePreemptedError` BEFORE the generic `except Exception` fallback | importing `EnginePreemptedError` inside the try block; eager unguarded `import ray.exceptions` at module top (breaks Ray-absent envs); falling back from distributed to local on `EnginePreemptedError` (must propagate, not retry); consuming the 60-retry budget on `EnginePreemptedError` | D4.F, D5.G, C20 |
| [miles/rollout/generate_hub/multi_turn.py](../external/miles/miles/rollout/generate_hub/multi_turn.py) | Multi-turn trajectory generator | drop `partial_rollout` assert; force `payload["stream"]=False`; snapshot-then-retry; `MAX_ATTEMPTS = total_engine_count`; raise `EnginePreemptedError` on exhaustion | swallow `EnginePreemptedError`; group-recycle as fallback; replay env steps on aborted turn | D3.B |
| [miles/rollout/base_types.py](../external/miles/miles/rollout/base_types.py) | Shared types | add `EnginePreemptedError`, `RLixRouterMetadataError` | extending `GenerateFnInput` schema | D3.B |
| [miles/rollout/generate_utils/generate_endpoint_utils.py](../external/miles/miles/rollout/generate_utils/generate_endpoint_utils.py) | Sample state mutation helpers | add `_snapshot_turn_state` / `_restore_turn_state` (length-only, not deep copy) | deep-copy entire sample | D3.B |
| [miles/ray/actor_group.py](../external/miles/miles/ray/actor_group.py) | RayTrainGroup | accept `worker_placements` + `num_gpus_per_actor=0.01`; `build_cpu_bucket_cache` fan-out; `collect_cache_owner_roles`; ctor self-cleanup; pass `local_rank=0` in RLix path | sharing actor handles across pipelines; using fractional `0.4` in RLix path | D2.C, D5.F |
| [miles/ray/train_actor.py](../external/miles/miles/ray/train_actor.py) | TrainRayActor | accept optional `local_rank` kwarg | break standalone `get_local_gpu_id()` fallback | D5.F |
| miles/ray/placement_provider.py (NEW) | ROLL ↔ MILES placement adapter | `MilesPlacementProvider` + `WorkerPlacement` dataclass; `get_all_rollout_engine_placements` (declared full table); `get_active_engine_indices`; `get_train_workers` | self-instantiate `RollResourceManagerProxy`; derive device mapping via `list(range(...))` | D1.B, D4.D |
| [miles/ray/placement_group.py](../external/miles/miles/ray/placement_group.py) | Standalone PG factory | accept `external_provider=None` (None → standalone unchanged) | branch standalone behavior | D2.C |
| miles/backends/megatron_utils/update_weight/cpu_bucket_cache.py (NEW) | CPU bucket build + lookup + version label | pinned CPU buckets; HF-converted bytes; single slot covering writes; `_cache_ready_step` label | multiple slots; raw Megatron state_dict; lazy build on sync hot path | D2.B, D4.C |
| [miles/backends/megatron_utils/actor.py](../external/miles/miles/backends/megatron_utils/actor.py) | MegatronTrainRayActor + sender API | add `build_cpu_bucket_cache(step)`, `report_cache_owner_role()`, `run_sync_session(plan)` (cache_owner holds `_cache_lock` in single method) | call back to service or manager during `run_sync_session`; split into multiple top-level RPCs | D3.D, D4.E |
| [miles/utils/arguments.py](../external/miles/miles/utils/arguments.py) | CLI args | add `miles_model_update_bucket_size_mb`, `miles_post_sleep_vram_threshold_gb`, `model_update_transport`; `MILES_ROUTER_DISPATCH_WAIT_S` is env-tunable (no arg) | add `device_mapping` args (derive only) | D5, C20 |
| miles/utils/rlix_hooks.py (NEW) | `RLixHooks` protocol + `NoOpRLixHooks` | import seam | importing RLix wire types into MILES code | D2.C |
| [examples/fully_async/fully_async_rollout.py](../external/miles/examples/fully_async/fully_async_rollout.py) | fully_async outer loop | accept `rlix_hooks=None`; call `begin/bump_completed/end_progress_batch`; `_FatalError` queue sentinel for `(EnginePreemptedError, RLixRouterMetadataError)` | importing `ProgressReport` or coordinator handle; reading `worker.buffer` (does not exist) | D3.C, A8 |
| [train_async.py](../external/miles/train_async.py) | Standalone entry | add fail-fast guards (`RLIX_CONTROL_PLANE` set → raise; partial-overlap topology + flag unset → raise) | branching RLix behavior internally | D2.C |
| examples/rlix/run_miles_rlix.py (NEW) | RLix entry driver | orchestrator + coordinator + pipeline.initialize_pipeline + main loop | top-level try/except + `ray.shutdown` (raise propagate; `ray stop` CLI) | D2.C |
| rlix/pipeline/miles_pipeline.py (NEW) | RLix-side `MilesPipeline` actor | sync `initialize_pipeline` (use `run(coro)`); train/infer cluster lifecycle; F10 validation (incl. `MILES_ROUTER_DISPATCH_WAIT_S > 0`); Phase 1 + Phase 2 cleanup | `async def initialize_pipeline`; mixing sync `_request_cluster_gpus` with asyncio.Lock | D1.A, D2.C, D3.E |
| rlix/pipeline/miles_coordinator.py (NEW) | `MilesCoordinator` (manual init, NOT subclass) | `bootstrap_active_engines`, `publish_cache_ready_step`, `register_model_update_resources`, `sync_base_weights_to_active(step)`, `resize_infer / _expand_workers`; manual init of `_resize_sync_lock`, `_active_engines_bootstrapped` flag; `sync_lora_weights` ABC stub | `super().__init__()` to `PipelineCoordinator` (would invoke ROLL config validators) | D1.A, D2.B |
| rlix/pipeline/miles_model_update_service.py (NEW) | sync atomic unit | per-bucket transport + finalize fan-out + `set_weight_version`, all in one timeout; v=-1 short-circuit; `master_port` claim; `SyncSessionPlan` build at sync entry; service holds both `cache_owner_actor` and `rollout_manager` handles | leak ObjectRef beyond sync barrier; persistent NCCL group across syncs; cuda_ipc on M11.1 path | D2.B, D3.D, D5.A, D5.B, D5.C |
| rlix/pipeline/miles_hooks.py (NEW) | `MilesRLixHooks` real impl | wire-format packing for `ProgressReport` | reading MILES internals | D2.C |
| external/sglang/3rdparty/cpu_serialize/*.patch (NEW) | SGLang fork patches | new HTTP route + scheduler dispatch + in-process `torch.load` + writer_lock acquire/release | persistent payload retention beyond request; async return / background task; bypassing `model_update_lock.writer_lock` | D5.A, D5.D |
| [miles/router/middleware_hub/radix_tree_middleware.py](../external/miles/miles/router/middleware_hub/radix_tree_middleware.py) | radix tree middleware | none in M11.1 (RLix mode forbids loading) | making it pass-through; `partial_rollout + radix_tree` compat | D2.A, C17 |
| [miles/utils/reloadable_process_group.py](../external/miles/miles/utils/reloadable_process_group.py) | NCCL teardown | none (already in place) | reimplementing NeMo-style `nccl_offload.py` | D3.F, A10 |

## Appendix C: Full Risk → Evidence Matrix

Risk-driven cut. §6 is decision-driven. Rows here are NOT duplicated in §6.

| Decision | Risk | Required Evidence | Stop Condition |
| -------- | ---- | ----------------- | -------------- |
| D3.F | gather deadlock or empty cache if order wrong | Gate 2.5 multi-step destroy/reload cycle; assertion `_cache_ready_step != None` before expand | hang or `_cache_ready_step is None` raise |
| D3.C | request-level mis-attribution beyond bounded in-flight × decode-step | Gate 3 quantifies count; engine publishes `_current_weight_version` after barrier | mis-attribution exceeds bound |
| D3.B | retry exhaustion in valid topology = scheduler/admission bug | Gate 1 multi-turn preempt resumes from aborted turn; Gate 3 `MAX = total_engine_count` | exhaustion in M11.1 4-GPU topology |
| D3.G | router suspend deadlocks (no expand fires) → indefinite block | `MILES_ROUTER_DISPATCH_WAIT_S` bounded timeout; Gate 4 (f) negative test verifies `EnginePreemptedError` raised | timeout fires + `EnginePreemptedError` not raised (sentinel header missing or client translation broken) |
| D4.A | shell→loading lazy ctor leaks Ray actor on failure | Gate 4 (M11.2) Pipeline B partial allocation; M4 self-cleanup unit test | actor handle leak after init failure |
| D4.B | gzip body or non-JSON skip silently in RLix mode | unit test: route stripping `Content-Encoding`; missing `miles_admission_disabled` raises `RLixRouterMetadataError` | router upgrade incomplete in production |
| D4.C | stale label inversion if not under `_resize_sync_lock` | Gate 3 concurrent expand + after_training; receiver sees new label after refresh | label < step after publish |
| D4.F | sentinel header dropped by intermediate proxy / response wrapper | Gate 4 (f) explicitly inspects response `headers["X-Miles-Preempt"]`; if absent, fall back to status_code==503 with body parse | sentinel header absent on real timeout response |
| D5.A | leak on crash mid-sync; concurrent receiver overflows `/dev/shm` | `try/finally os.unlink` covers HTTP error path; F10 startup `shutil.disk_usage` check | `/dev/shm` ENOSPC during sync |
| D5.B | port collision; group survival across syncs | Gate 2.5 warmup allreduce; SharedStorage claim; group destroy verified | persistent EADDRINUSE |
| D5.E | rendezvous failure on multi-rank | F10 fail-fast assertion | scheduler returns port=0 |
| D5.G | `_post` import-inside-try variant raises `UnboundLocalError` instead of `EnginePreemptedError` | unit test: simulate exception before import line and confirm `EnginePreemptedError` propagates intact | `UnboundLocalError` observed in test |
| C12 | cuda_ipc on vast.ai (no `--ipc=host` / `CAP_SYS_PTRACE`) | M11.1 fail-fast `model_update_transport == "cpu_serialize"`; M11.6 smoke-test capability check | starts on restricted container with cuda_ipc |
| C11 | `torch_memory_saver.pause()` vs flush race → segfault | F10 `assert not args.async_save` | flag enabled in RLix mode |
| C8 | F4 cache covers dense only | `assert expert_model_parallel_size == 1` and `moe_router_topk == 0` | EP > 1 enabled |
| C7 | `WorkerPlacement.placement_group` is per-node | `assert rollout_num_gpus_per_engine <= num_gpus_per_node` | engine TP/EP straddles nodes |
| F11/F12 cleanup | scheduler/Ray ledger split → next pipeline OOM | M4 hard cleanup on failure path; `release_cluster_gpus` gated on `actor_infer_allocated` | scheduler shows free GPU but Ray actor still holds CUDA context |
| Standalone partial-overlap drift | user runs partial overlap without `RLIX_CONTROL_PLANE=rlix` → silent OOM via standalone broadcast | `train_async.py` fail-fast `train_devices_subset_of_infer` guard | unguarded standalone run with overlap topology |

## Appendix D: Implementation Detail Trace

| Implementation Detail | Parent Decision | Reason | Drift Risk |
| --------------------- | --------------- | ------ | ---------- |
| `_routing_lock = asyncio.Lock` (manager-local, not actor-distributed) | D4.A | dispatch + state transition is short critical section; abort/drain/sleep run outside | Medium — must NOT reuse `RolloutManager.rollout_engine_lock` (cross-process distributed lock) |
| `_preempted_engines` retained across `disabling→offloaded→loading→active` | D4.A | preempt attribution outlives `EngineInfo.state` window | Medium — clearing too early reverts to ambiguous abort classification |
| `MAX_TURN_REDISPATCH_ATTEMPTS = args.rollout_num_gpus // args.rollout_num_gpus_per_engine` | D3.B | use total engines; `active_engine_count` collapses to 1 in shrink-to-1 cases | High — earlier draft used active count |
| `payload_bytes` parameter is auto-derefed `bytes`, not `ObjectRef` | D5.A, A3 | Ray runtime auto-derefs top-level ObjectRef at method boundary | High — `ray.get(payload_bytes)` raises TypeError |
| `do_proxy` JSON mutation guarded by `path == "generate"` | D4.B | other endpoints (`/model_info`, `/v1/loads`, `/health`) must keep schema | Medium — over-broad guard breaks healthchecks |
| Strip `Content-Encoding` after body mutation | D4.B | upstream may gzip; modified body must not carry stale encoding | Medium — easy to forget on patch update |
| `master_port` selected via `get_free_port()` then SharedStorage claim | D5.E | `master_port=0` cannot rendezvous multi-rank | High — looks fine on single-rank tests |
| `RayTrainGroup` `num_gpus_per_actor=0.01` in RLix mode | D5.F | offloaded train must release scheduler view of GPU | Medium — leaving 0.4 deadlocks Gate 4 |
| `base_gpu_id=0` for SGLang in RLix mode (NOT `wp.gpu_ids[0]`) | D4.D | post-CVD local view; physical id breaks SGLang `physical_gpu_id ∈ CVD list` validation | High — passing physical id "looks right" but is wrong |
| `coordinator.publish_cache_ready_step(-1)` only at init bootstrap | D4.C | active refresh path now updates `_cache_ready_step` inside `_resize_sync_lock`; init bootstrap is the only place a Pipeline B can be expanded before its first `after_training` | High — calling on every step risks lock-free inversion |
| `runtime_env={'env_vars': {'CUDA_VISIBLE_DEVICES': ...}}` at `.options()` time | D5.F | `cuInit()` lock timing: setting env after `import sglang/torch` is too late | High — actor body `os.environ[...]=...` is a common shortcut |
| F9 progress unit = `group`, not `trajectory` | D2.C | `len(data) < target_data_size` measures groups; trajectory-unit reports under-count by `n_samples_per_prompt` → premature shrink | Medium |
| F9 `_local_completed = initial_completed` (NOT derived from worker buffer scan) | D2.C | `AsyncRolloutWorker` exposes `output_queue` + `get_completed_groups()` only; `worker.buffer` etc. don't exist | Low |
| `_FatalError` is sentinel class, NOT `Exception` subclass | D3.B | otherwise outer try/except may swallow it | Medium |
| `MilesCoordinator` does NOT call `super().__init__` | D1.A | avoids ROLL `_validate_config_schema` / `_validate_offload_nccl` triggers on MILES args lacking those fields | High — easy to "just inherit" |
| `sync_lora_weights` ABC stub raises | D1.A | Coordinator ABC requires it; LoRA out of M11.1 scope | Low |
| Tmpfs file naming `miles_cpu_bucket_{uuid}.pt` | D5.A | grep-friendly leak detection (`ls /dev/shm/miles_cpu_bucket_*`) | Low |
| Per-bucket receiver invocation is **serial**, not parallel | D5.A | tmpfs peak = 1× bucket_size, not N× | Medium — parallelism looks like a perf win but breaks tmpfs budget |
| Single composite `cache_owner.run_sync_session(plan)` RPC (NOT 5 split RPCs) | D3.D, D4.E | `_cache_lock` cannot span multiple Ray RPCs; single method body holds lock for whole transport phase | High — natural temptation to split RPCs reintroduces lock-across-RPC anti-pattern |
| `_workers_changed: asyncio.Condition` notify happens at endpoint layer (NOT inside sync helpers) | D3.G | helpers stay sync per source-plan F3 contract; only async endpoints + `_health_check_loop` carry notify | High — making helpers async fragments the F3 admission state-machine |
| `_RouterDispatchTimeout` is a router-LOCAL exception class (NOT `EnginePreemptedError` from `miles.rollout.base_types`) | D3.G, D4.F | router process must not import `miles.rollout.base_types` (cycle risk + HTTP serialization loses Python type) | High — direct cross-module raise looks reasonable but breaks HTTP boundary |
| `_post`: `from miles.rollout.base_types import EnginePreemptedError` at FUNCTION TOP, NOT inside try | D5.G | Python treats import-as-binding-to-local; if exception fires before import line, `except EnginePreemptedError:` raises `UnboundLocalError` | High — looks like premature optimization to put import inside try; actually a correctness bug |
| `post()`: module-top `try: from ray.exceptions import RayTaskError; except ImportError: RayTaskError = ()` | D5.G | Ray-absent envs (CI / standalone tests) must still import `http_utils`; `except ()` matches nothing → branch is a safe no-op | Medium — eager unguarded `import ray.exceptions` breaks any caller that imports `http_utils` without Ray |

## Appendix E: Execution Anchors

### Allowed Changes

- New files: `miles/ray/placement_provider.py`, `miles/utils/rlix_hooks.py`, `examples/rlix/run_miles_rlix.py`, `miles/backends/megatron_utils/update_weight/cpu_bucket_cache.py`, `rlix/pipeline/miles_{pipeline,coordinator,model_update_service,hooks}.py`.
- SGLang vendored fork patches in dedicated dir: `external/sglang/3rdparty/cpu_serialize/{http_route,scheduler_dispatch,weight_sync_utils,engine_base}.patch`.
- Extensions only at named seams in MILES modules listed in Appendix B.
- New args in `miles/utils/arguments.py`: `miles_model_update_bucket_size_mb`, `miles_post_sleep_vram_threshold_gb`, `model_update_transport`. Env-tunable: `MILES_ROUTER_DISPATCH_WAIT_S` (default 60s, F10 asserts `> 0`).
- Feature labels gated by milestone: M11.1 only `cpu_serialize` colocate path; M11.2 dual-pipeline happy path + router 0-active suspend (C20 / AC13 / E14); M11.4 LoRA + multi-stream impl; M11.5 production hardening; M11.6 cuda_ipc adapter.

### Forbidden Changes

- Modifying standalone path (`train_async.py` body, `RolloutManager.update_weights_from_distributed/tensor`, existing `_send_to_colocated_engine` cuda_ipc).
- Adding device-mapping CLI args (must derive from existing `actor_num_nodes / actor_num_gpus_per_node / rollout_num_gpus`).
- Subclassing `PipelineCoordinator` for `MilesCoordinator`.
- Calling `service.sync_selected_workers`, `finalize_weight_update`, or `manager.set_weight_version` from pipeline or coordinator outside the documented atomic-unit / bootstrap entry points.
- Adding admission_epoch race defense, multi-pipeline orchestrator-driven cleanup, or graceful actor drain in M11.1 (all M11.2/M11.5 follow-ups).
- Adding selective P2P weight transfer, MoE/EP, or `partial_rollout + radix_tree` compatibility (out of scope).
- Adding `verify_model` debug validation in M11.1 receiver API surface.
- Mutating `os.environ['CUDA_VISIBLE_DEVICES']` after `import torch` / `import sglang`.
- Re-introducing per-request tracking (`_inflight_requests`) or worker-side `is_engine_resident_on_gpu` flag.
- Re-using `load_format="flattened_bucket"` for cpu_serialize (incompatible with ROLL wire format).
- Driver-level `ray.shutdown()` or top-level `try/except` in `run_miles_rlix.py` (rely on `ray stop` CLI).
- Splitting `cache_owner.run_sync_session(plan)` into multiple top-level Ray methods (would reintroduce lock-across-RPC anti-pattern).
- Raising `RuntimeError("no enabled live workers")` from router `_use_url` (must suspend with `_workers_changed` Condition; on timeout return HTTP 503 + sentinel header).
- Raising `EnginePreemptedError` from inside the router process (HTTP serialization loses Python type; sentinel header is the contract).
- Importing `EnginePreemptedError` inside `_post`'s try block (UnboundLocalError trap; must live outside the `try` path — module-top preferred if cycle-free, otherwise function-top fallback).
- Eager unguarded `import ray.exceptions` at module top in `http_utils.py` (breaks Ray-absent envs; use module-top guarded `try/except ImportError: RayTaskError = ()`).
- Falling back from distributed to local `_post` on `EnginePreemptedError` (must propagate, not retry); consuming the 60-retry budget on `EnginePreemptedError`.

### Stop Conditions

The plan instructs the agent to stop and ask if:

- implementation requires touching out-of-scope files (Appendix B Forbidden columns).
- a lower-level implementation decision lacks a parent decision in Appendix A (mark `unanchored` and surface).
- required tests force changing standalone MILES path or existing tool-error group-recycle behavior.
- the architecture insertion point appears wrong after code inspection (e.g. `MilesCoordinator` cannot avoid `PipelineCoordinator` super-init).
- public API compatibility must be broken (e.g. existing `RolloutManager(args, pg=...)` standalone signature).
- topology asserts in F10 conflict with feasible deployment (e.g. user reports vast.ai container has `--ipc=host` after all → revisit C12).
- atomic-unit invariant (transport + finalize + `set_weight_version` in single timeout) cannot be preserved without splitting RPCs.
- v=-1 short-circuit cannot be implemented because base weights from `args.hf_checkpoint` differ from `build_cpu_bucket_cache(-1)` bytes (C13 must not be loosened).
- C20 router suspend cannot be implemented without making sync helpers async (the spec mandates helpers stay sync; only endpoints + `_health_check_loop` notify).
- `EnginePreemptedError` cannot be imported at module top because of a future cycle AND function-top fallback also fails (e.g., signature constraints) — use the helper-pattern alternative documented in §Feature 3 §8.

(See `## 6.2` for the consolidated stop-conditions list with audit-gap callouts; this Appendix E list mirrors the same set in expanded form.)

### Done Criteria

#### Done(M11.1) = AC1 ∧ AC2 ∧ AC3 ∧ AC4 ∧ AC5 ∧ AC6 ∧ AC7 ∧ AC8

Gates 1, 2, 2.5, 3 all pass single-pipeline on 4-GPU `tp=2` partial-overlap with `multi_turn.generate + --max-weight-staleness 2 + cpu_serialize`. F10 asserts active in `MilesPipeline.initialize_pipeline()` (including `MILES_ROUTER_DISPATCH_WAIT_S > 0` for forward-compat with M11.2 C20). cache_owner discovered exactly once at init. Trajectory `weight_versions` consumable by `--max-weight-staleness`. Standalone `train_async.py` smoke test still passes when `RLIX_CONTROL_PLANE` is unset.

#### Done(M11.2) = AC9 ∧ AC10 ∧ AC11 ∧ AC12 ∧ AC13

Two MILES pipelines admitted under same orchestrator; share PG via `RollResourceManagerProxy`; partial allocation (c) validated; base v=-1 expand short-circuit (d) validated with instrumented call counters; donor-shrink-before-recipient ordering (e) validated with strict `T1 < T2 < T3`; **router 0-active suspend (f) validated end-to-end with positive (suspend + resume) + negative (timeout → sentinel → `EnginePreemptedError`) + counter (concurrent suspend/resume + lock atomicity) + distributed-POST (`_distributed_post_enabled` matrix) sub-tests**. Crash recovery = manual `ray stop`.

All `Implementation follow-up` items remain explicitly tagged with milestone (M11.2 / M11.4 / M11.5 / M11.6) and not silently merged into M11.1 scope.

## Appendix F: Activated Pattern Details

### F.1 Wire-Format Quick Reference

*Audit question: for each new schema field, what does it mean, who produces it, who consumes it?*

**`SyncSessionPlan` (frozen dataclass, built once at sync entry)** — `rlix/pipeline/miles_model_update_service.py`

| Field | Type | Source | Consumer | Notes |
| ----- | ---- | ------ | -------- | ----- |
| `sync_id` | str | service | cache_owner, all receivers | format `f"{pipeline_id}_step{step}_{uuid4().hex[:8]}"` |
| `version` | int | service (= `_cache_ready_step`) | cache_owner, manager.set_weight_version | `>= 0` (v=-1 short-circuit doesn't reach `run_sync_session`) |
| `group_name` | str | service | sender + NCCL receivers | `f"miles_model_update_{sync_id}"` |
| `master_addr` | str | service via `get_free_port()` + SharedStorage claim | sender + NCCL receivers | TCP rendezvous |
| `master_port` | int | same | same | NEVER 0 (C16) |
| `timeout_s` | float | service (`ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S`, default 150) | sender + receivers | wraps whole atomic unit |
| `target_handles` | dict[int, RayActorHandle] | service via `manager.get_engine_handles(target_engines)` | cache_owner | per-engine SGLang actor handles for cpu_serialize RPC |
| `cpu_serialize_local_ranks` | dict[int, list[int]] | service classification | each receiver wrapper | rank-local mask |
| `broadcast_local_ranks` | dict[int, list[int]] | service classification | each receiver wrapper | rank-local mask |
| `comm_ranks` | dict[int, int] | service | NCCL receivers only | sender = rank 0; receivers ≥ 1 |

**`update_weights_from_cpu_bucket` HTTP body** — `external/sglang/3rdparty/cpu_serialize/http_route.patch`

| Field | Type | Source | Consumer | Notes |
| ----- | ---- | ------ | -------- | ----- |
| `payload_path` | str | wrapper writes `/dev/shm/miles_cpu_bucket_{uuid}.pt` | SGLang HTTP route reads via `pathlib.Path(...).read_bytes()` then `torch.load(BytesIO, weights_only=True)` | wrapper owns cleanup via `try/finally os.unlink` |
| `load_format` | str | service | scheduler dispatch | NOT `"flattened_bucket"` |
| `flush_cache` | bool | service | scheduler | flush KV after load |
| `cpu_serialize_local_ranks` | list[int] | service | each TP worker | skip when `self.rank not in mask` |
| **NO `weight_version` field** | — | — | — | Fix #3: per-bucket payload carries no version label; version published once at sync exit via `manager.set_weight_version` |

**Router preempt sentinel HTTP response** — `miles/router/router.py` (do_proxy on `_RouterDispatchTimeout`)

| Field | Type | Source | Consumer | Notes |
| ----- | ---- | ------ | -------- | ----- |
| `status_code` | int (= 503) | router `do_proxy` on `_RouterDispatchTimeout` catch | `miles.utils.http_utils._post` | sentinel response shape |
| `headers["X-Miles-Preempt"]` | str (= "empty_active_set_timeout") | same | `_post` matches BEFORE `raise_for_status()` | HEADER is the contract — body is informational only |
| `body.error` | str (= "router_preempt") | same | logging only | not load-bearing |
| `body.reason` | str (= "empty_active_set_timeout") | same | logging only | mirrors header value |
| `body.wait_s` | float | same | logging only | echoes `MILES_ROUTER_DISPATCH_WAIT_S` |

**`ProgressReport` (RLix wire, fire-and-forget)** — `rlix/pipeline/miles_hooks.py`

| Field | Type | Source | Consumer | Notes |
| ----- | ---- | ------ | -------- | ----- |
| `pipeline_id` | str | F8 registration | central scheduler | namespace handle |
| `step_target_trajectories` | int | reporter (`= args.rollout_batch_size`) | scheduler | wire-name says trajectories but unit = group (matches fully_async wait) |
| `metrics["collected"]` | int | reporter raw | coordinator | NOT clamped at reporter |
| `metrics["completed"]` | int | coordinator (= `min(total_collected, total_required)`) | central scheduler | clamped at coordinator |
| `bucket` | int 0..50 | reporter (`floor(_local_completed/step_target_groups * 50)`) | scheduler | 2% gate at reporter |
| `current_train_step` | int | reporter (`= _progress_target_step`) | scheduler | step == target_weight_version in fully_async |
| `mode` | str | reporter | scheduler | `"train"` (M11.1); M11.4 multi-stream uses other values |
| `new_batch` | bool | reporter | scheduler | `True` only at `begin_progress_batch`; `False` thereafter |

### F.2 Sub-system State Machines

*Audit question: are subsystem lifecycles complete and all transitions accounted for?*

**`EngineInfo.state` (RolloutManager) — 5 states**:

```mermaid
stateDiagram-v2
  [*] --> shell: ctor (slot reserved, no actor, no GPU)
  shell --> loading: expand_engines (lazy _create_sglang_actor + onload, M11.2 partial alloc)
  offloaded --> loading: expand_engines (wake_up)
  loading --> active: activate_routing (open routing + add_worker)
  active --> disabling: shrink_engines (admission close)
  disabling --> offloaded: drain + release_memory_occupation (post-sleep VRAM assert)
```

`shell` is terminal-on-the-left for shrink — never `active → shell`. The `_preempted_engines` set spans `disabling → offloaded → loading → active` for preempt attribution and is cleared after `wake_up` cycle completes.

**Miles Router worker admission — 4 dicts (`enabled_workers`, `dead_workers`, `worker_request_counts`, `worker_failure_counts`)**:

```mermaid
stateDiagram-v2
  [*] --> registered: add_worker
  registered --> enabled: default
  enabled --> disabled: disable_worker (admission close, sleep)
  disabled --> enabled: enable_worker (admission open, wake)
  enabled --> dead: health probe fail
  registered --> [*]: remove_worker
  disabled --> [*]: remove_worker
  dead --> [*]: remove_worker
```

Health monitor probes only `enabled - dead` (NOT disabled). `_use_url` selects from `enabled - dead`. `enable_worker` does NOT revive `dead`. **Every state transition (`add` / `enable` / `disable` / `remove` / `mark dead` / `recovered`) MUST notify `_workers_changed: asyncio.Condition` so suspended `_use_url` waiters re-evaluate** (C20).

**Router `_use_url` dispatch lifecycle (M11.2 C20 suspend mechanism)**:

```mermaid
stateDiagram-v2
  [*] --> select: do_proxy entry, await self._use_url()
  select --> dispatched: valid set non-empty, +1 worker_request_counts, return URL
  select --> suspended: valid set empty, asyncio.Condition.wait_for predicate
  suspended --> select: workers_changed.notify_all from add/enable/recover endpoint
  suspended --> timeout: MILES_ROUTER_DISPATCH_WAIT_S elapsed
  timeout --> sentinel_503: raise _RouterDispatchTimeout, do_proxy emits 503 plus X-Miles-Preempt
  sentinel_503 --> [*]: client _post translates to EnginePreemptedError
  dispatched --> [*]: response, _finish_url decrements
```

### F.3 Cross-cutting Role Matrix

*Audit question: for each `tp_rank × transport_kind`, which receiver path runs and which mask gates it?*

| Transport | Mask field | Colocate? | Sender path | Receiver guard | Milestone |
| --------- | ---------- | --------- | ----------- | -------------- | --------- |
| `cpu_serialize` | `cpu_serialize_local_ranks` | Yes (same GPU) | `engine.update_weights_from_cpu_bucket.remote(payload_ref, ...)` → wrapper auto-deref → tmpfs file → HTTP `POST /update_weights_from_cpu_bucket` | wrapper checks `self.rank in cpu_serialize_local_ranks`; SGLang HTTP handler under `model_update_lock.writer_lock` | **M11.1** |
| NCCL broadcast | `broadcast_local_ranks` | No (cross-GPU) | per-bucket H2D staging on cache_owner GPU → `dist.broadcast(staging, src=0, group=tmp)` → free staging | `broadcast_parameter` checks `self.rank in broadcast_local_ranks`; `destroy_collective_group` uses `is_group_exist` no-op guard for colocate-only ranks | **M11.1 + always** |
| `cuda_ipc` | `ipc_local_ranks` | Yes (same GPU) | new adapter: CPU cache → per-bucket H2D staging → `MultiprocessingSerializer.serialize` IPC handle → `engine.update_weights_from_tensor.remote(serialized, ipc_local_ranks=...)` | `update_weights_from_tensor` checks `self.rank in ipc_local_ranks` (M11.6 extends signature) | **M11.6** |

Mixed receivers in the same bucket (M11.1 example: colocate `cpu_serialize` + non-colocate NCCL broadcast for partial overlap) → sender does H2D staging once for NCCL path; cpu_serialize receivers do not depend on the staging buffer. Each mask is checked independently per receiver.

### F.4 Source-axis Crosswalk: Feature → Decision

The source plan organizes work by **Feature number** (F1–F12). tldr-plan organizes by **Decision layer** (D0–D6). This table lets an auditor familiar with the source plan jump from `Fn` (their mental handle) to the right `D` rows here.

| Feature | Primary decision(s) | Secondary | Notes |
| ------- | ------------------- | --------- | ----- |
| F1 SGLang sleep/wake (sleep_level via tags) | D3.A | D5.F (post-sleep VRAM read SGLang `/server_info`) | tag-based API; `release_memory_occupation` keeps TP NCCL alive (A2) |
| F2 Selective engine sleep/wake (per-engine subset) | D3.A, D4.A | D2.A | EngineInfo 5-state machine; `_routing_lock` short critical section; `_preempted_engines` attribution window |
| F3 Generation routing skip sleeping engines + turn redispatch + **router 0-active suspend** | D3.B, D4.B, D3.G, D4.F, D5.G | D2.A | router admission API; metadata injection on `/generate`; `multi_turn.py` snapshot/restore; `MAX = total_engine_count`; **§7 `_use_url` async block-with-notify; §8 client `_post` sentinel translation; §9 client `post()` `RayTaskError` unwrap with guarded import** |
| F4 Training-side weight caching (CPU bucket cache + `_cache_ready_step`) | D3.F, D4.C, D2.B | D5.A, D5.B | HF-format gather; single owner = `pp0+dp0+tp0+cp0` (A6, C15); single slot |
| F5+F6 Two-path weight refresh (active in-flight + expand sync) + version accounting | D3.C, D3.D | D4.C, D4.E, D5.C | atomic sync unit; `sync_base_weights_to_active(step)` updates `_cache_ready_step` inside `_resize_sync_lock`; v=-1 short-circuit |
| F7 Per-pipeline Ray namespace isolation | D2.C | D1.A | every named RLix actor MUST carry `namespace=` and `name=` |
| F8 Pipeline registration lifecycle | D2.C, D3.E | D1.A | driver does `allocate → register → admit` then create coordinator |
| F9 Progress reporting | D2.C | (Appendix F.1 ProgressReport schema) | reporter raw `collected` + 2% gate; coordinator clamped `completed`; group unit, NOT trajectory |
| F10 Partial GPU topology validation | C1–C23 (all of §3.2) | D5.F | F10 is the enforcement surface for hard constraints (incl. C20 `MILES_ROUTER_DISPATCH_WAIT_S > 0`, C21 fullasync, C22 Megatron parallelism divisibility, C23 infer_device_mapping divisibility) |
| F11 Conditional RLix behavior flag (`DO_TIME_SHARING`) | D2.C | D1.A | dual entry: `train_async.py` + `examples/rlix/run_miles_rlix.py` |
| F12 Shared PG cluster | D1.B, D4.D, D5.F | D2.C | `MilesPlacementProvider` real adapter; `WorkerPlacement` per-worker; `num_gpus_per_actor=0.01` + manual CVD |

### F.5 Expanded Milestone Notes

- **M11.1 first build deployment target**: vast.ai restricted container (no `--ipc=host` / `CAP_SYS_PTRACE`). Forces `model_update_transport == "cpu_serialize"` (C12). Multi-node DP capable but each engine node-local (C7). Receiver dual-mask (cpu_serialize + broadcast) covered by Gate 2.5. **C20 router suspend code lands in M11.1 even though it does not fire** — single-pipeline + C3 declared topology guarantees ≥1 active engine; the suspend path is dead code in M11.1 but must be PRESENT to keep the router state machine coherent for M11.2.
- **M11.2 NOT-pass-criteria** (tagged but explicitly out of Gate 4 scope): admission_epoch race defense (production multi-pipeline only); multi-pipeline orchestrator-driven cleanup (manual `ray stop` is acceptable for M11.1/M11.2); graceful actor drain replacing `ray.kill` (force-kill timeout adequate for happy path).
- **M11.2 PASS criteria additions for C20**: Gate 4 (f) sub-test matrix MUST cover positive (suspend + resume + counter parity), negative (timeout → 503 + sentinel header → client `EnginePreemptedError`), AND distributed-POST (`_distributed_post_enabled` ∈ {False, True} both verified). Implementing only the local path silently regresses production deployments using Ray actor distribution.
- **M11.6 cuda_ipc adapter**: writes a NEW adapter (CPU cache → per-bucket H2D staging → IPC handle serialize, ~50–80 lines). Does NOT reuse standalone `_send_to_colocated_engine` because that path depends on live `dist.gather_object(group=ipc_gather_group)` which conflicts with F4 destroy-NCCL ordering. Standalone cuda_ipc path remains unchanged.
- **Capability check (M11.6)**: NO fragile env detection (`CAP_SYS_PTRACE` / `--ipc=host` heuristics). Use startup smoke test: `MultiprocessingSerializer.serialize(small_tensor, output_str=True)` → `MultiprocessingSerializer.deserialize(handle)` cross-process round-trip; failure prompts user to switch back to `cpu_serialize`.
