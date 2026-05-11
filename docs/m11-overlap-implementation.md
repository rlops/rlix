# Real M11.2 — Overlap-Pool Dual-Pipeline (reviewer guide)

> Reviewer-friendly walkthrough for the **overlap** variant of M11.2. The
> previously shipped M11.2 used disjoint pools (P1=[0,1], P2=[2,3]) per Codex's
> Option A. This guide covers the **real** M11.2 per `plans/miles-port-unified-plan.md`
> §M11.2 Gate 4 — overlap GPUs, donor-shrink-before-receiver-expand, Phase B
> init under contention.
>
> Source plan: `/Users/zhenyulin/.claude/plans/yeah-i-want-overlap-stateless-plum.md`
> (Codex APPROVE_WITH_NOTES, 0 blockers).
>
> Iteration / debug log: `plans/m11-2-overlap-log.md`

## §1 What "overlap" means here

Two MilesPipelines share at least one physical GPU on their `actor_infer`
mapping. The rlix scheduler must arbitrate which pipeline holds the GPU at any
given moment via the donor-shrink-before-receiver-expand path at
`rlix/scheduler/scheduler.py:1027-1073`.

Minimum smoke topology (4×A40):

```
P1: train=[0]   infer=[0,1,2]   exclusive [0],   shared [1,2]
P2: train=[3]   infer=[1,2,3]   exclusive [3],   shared [1,2]
```

## §2 The two prerequisites we already had

1. **Donor-shrink IS implemented** — `rlix/scheduler/scheduler.py:1027-1073`.
   Cross-pipeline donor search at L1031 iterates global `active_allocations`.
   Plan-level test: `tests/test_gap_ratio.py::test_two_pipelines_donor_shrink`
   (L316-409).
2. **C20 0-active suspend IS implemented** — `miles/miles/router/router.py:42-49,
   533-549`. Router blocks dispatch when `_candidate_set()` is empty;
   `_notify_workers_changed` wakes it on `add/enable/disable/remove_worker`.

There is **no** C4 topology check rejecting shared `cluster_device_mappings` —
verified by exhaustive search.

## §3 What "real M11.2" requires that isn't there yet

The unified plan §M11.2 Gate 4(c) requires *Pipeline B init under contention
waits for full INIT allocation, initializes all SGLang engines, then offloads
all without routing or sync*. Today engines come up `state="active"`
(`miles/miles/ray/rollout.py:609-613`) and self-register with the local router
during `_init_normal` (`miles/backends/sglang_utils/sglang_engine.py:339-358`).
P2's INIT would OOM on shared GPUs that P1's active engines are holding.

We chose **Option β** (over the Option α that Codex Q1 flagged as SHAKY):

- Driver sets `MILES_INIT_DEFER_ADD_WORKER=1` in per-pipeline runtime_env.
- `sglang_engine._init_normal` skips the `/add_worker` HTTP POST when that env is set.
- `RolloutManager._init_engine_info_table` writes `EngineInfo(state="loading", ...)` instead of `"active"` when the env is set.
- `MilesPipeline._init_phase_b_infer` immediately drives `finish_init_offload(all)` after `start_rollout_servers` — engines transition `loading → offloaded`, VRAM released via `release_memory_occupation`.
- `bootstrap_active_engines(frozenset())` instead of full set; post-INIT active set is empty.
- Skip post-init `sync_base_weights_to_active(-1)` — the first runtime `_expand_workers` does base v=-1 sync via the F40 Runtime branch (`miles_coordinator.py:498-512`).

This keeps `RolloutManager.__init__` unchanged (avoids the full F22 shell-init
architectural rebuild) while still giving Gate 4(c) semantics: post-INIT engines
are `offloaded`, router `enabled_workers` is empty.

## §4 The three invariants (Codex Q5)

| ID | Where | Check |
|---|---|---|
| Q5-a | `MilesPipeline._init_phase_b_infer` end | `ray.get(rollout_manager.get_router_enabled_workers.remote())` returns empty set. Raise `RuntimeError` if not. |
| Q5-b | `RolloutManager.shrink_engines` | `/disable_worker` POST against local router precedes `release_memory_occupation`. Log line `"shrink_engines: disabled router workers prior to release engine_indices=…"` for harness grep. |
| Q5-c | smoke harness (`grep_overlap_log.sh`) | Parse `mp1_infer=` and `mp2_infer=` log keys, assert `len(set(p1) & set(p2)) > 0`. Fail with non-zero exit if smoke ran disjoint (prevents accidental regression). |

## §5 R04-F1 prerequisite

The HIGH finding from `m11-review.review-report.md` (`miles/utils/rlix_train_loop.py`
has no `try/finally` around `before_step / train / after_step`) ships **first**
as Phase 1 of this work. Codex Q4 explicitly required this — without it, every
overlap-smoke OOM leaks the scheduler ledger and forces `ray stop`, making
iteration un-debuggable.

The fix uses a `release_train_only` shim (NOT a naive `after_step` call on
failure — `_after_training` builds and publishes a CPU bucket BEFORE release,
so calling it on a partial train double-publishes). `release_train_only` calls
`_notify_release_cluster_gpus` directly.

## §6 PASS bar (Phase 4)

Every smoke iteration must satisfy ALL 7 to be considered PASS:

1. `EXIT_CODE=0`
2. Both `shutdown_hard complete pipeline_id=` log lines present
3. `set(P1_infer) ∩ set(P2_infer) ≠ ∅` (overlap actually happened)
4. ≥1 `donor_shrink` event AND ≥1 `F40 Runtime` event, **donor-shrink precedes matching receiver F40 Runtime expand within 0-3 seconds**
5. ≥1 `disabled router workers prior to release` log per pipeline per shrink
6. Post-run `nvidia-smi memory.used` residual ≤200 MiB per GPU
7. `ray status` shows 0 leaked actors

## §7 How to read the attempt log

`plans/m11-2-overlap-log.md` is append-only. Each attempt has:

- `## Attempt N (UTC YYYY-MM-DD HH:MM) — <one-line outcome>`
- Branch heads at attempt time
- Topology actually used
- Pre/post `nvidia-smi`
- `EXIT_CODE`
- Grep results for the 7 PASS-bar signals
- `### Bug B-N — <title>` sub-blocks for each new bug found (repro, log, root cause, fix sketch, fixed-by commit hash)
- `### Codex review for commit <sha>` sub-blocks for each Codex pass

When debugging, search the log for the relevant `B-N` ID; each bug's fix commit
appears in `Implementation I-N — <title>` entries under the owning phase.

## §8 File map

Implementation lives across two repos:

### rlix (`zhenyu/miles-mvp-e2e`)

| File | What changes | Phase |
|---|---|---|
| `rlix/pipeline/miles_pipeline.py` | release_train_only shim; finish_init_offload call; bootstrap empty; router-empty invariant; physical→local log line | 1, 3c, 3e |
| `rlix/pipeline/miles_coordinator.py` | F10 hatch gated behind env (eventually removed) | 3d |
| `plans/m11-2-overlap-log.md` (NEW) | attempt + debug log | 0 |
| `docs/m11-overlap-implementation.md` (NEW) | this guide | 0 |
| `scripts/run_smoke_dual.sh` | CLI args for explicit pool mappings | 4 |
| `scripts/grep_overlap_log.sh` (NEW) | 7-condition PASS-bar harness | 4 |

### miles (`zhenyu/m11-mvp-test`)

| File | What changes | Phase |
|---|---|---|
| `miles/utils/rlix_train_loop.py` | try/finally with release_train_only on failure | 1 |
| `miles/backends/sglang_utils/sglang_engine.py` | env-gated `/add_worker` skip | 3a |
| `miles/ray/rollout.py` | env-gated state="loading"; get_router_enabled_workers helper; shrink_engines pre-release disable_worker | 3b, 3e, 3f |
| `examples/rlix/run_miles_dual.py` | explicit --p1-train/--p1-infer/--p2-train/--p2-infer CLI; structured log keys | 2 |
