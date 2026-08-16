# Plan: Enable NCCL broadcast path for miles weight update (rlops/rlix#42)

Status: DRAFT v8 (2026-08-16) — implementation in progress
v8 (implementation-discovered constraint): miles `assert_rlix_topology`
C1 gate REQUIRES train ⊂ infer (partial overlap is the RLix-mode
contract), so the planned M4 run (b) disjoint-pool topology cannot start
at all, and a fully-disjoint all-broadcast e2e is unreachable without
relaxing the C1 gate (out of scope — rlix_validation untouched per plan
C1). AC7/E9 narrowed: strict `broadcast` mode is verified by (a) E3 unit
matrix (incl. rejection + subset-target success), (b) an e2e REJECTION
check on the overlap harness (mode=broadcast → classification fail-fast,
log captured). The all-broadcast positive e2e is deferred as new O7
until disjoint topologies are admitted by the M11 validation contract.
Strict mode remains usable today for sync targets that avoid colocate
engines (E3 subset case).
v7: addresses codex round-5 (1 high): the v6 uniform engine-GPU-count
assumption was declared (O6) but not enforced — SGLang server groups can
override `num_gpus_per_engine` per group and the existing C7 checks do not
compare resolved group TP against `rollout_num_gpus_per_engine`. Fixed
with a rlix-side startup uniformity guard (miles still untouched beyond
actor.py): before `register_model_update_resources`, rlix validates the
resolved SGLang config — every server group's `num_gpus_per_engine` equals
`rollout_num_gpus_per_engine` — and fail-fasts otherwise (new E10).
v6 (user redirection): transport is a USER-SELECTED mode, not silent
auto-classification — rlix-side env `RLIX_MILES_UPDATE_TRANSPORT ∈
{cpu_serialize (default) | broadcast | auto}`; strict `broadcast` never
mixes (fail-fast on colocate targets); `auto` is the only mixing mode and
is what the mandated overlap M4 uses. Minimal-change pass: miles-side
changes shrink to `actor.py` + tests ONLY (dropped the planned
`rollout.py` manager method — rlix derives per-engine GPU counts from its
own `rollout_num_gpus_per_engine` arg); single env replaces the separate
kill-switch.
v5: addresses codex round-3 (1 medium): the harness hardcodes
`--sglang-mem-fraction-static 0.30` (run_smoke_dual.sh:125), contradicting
the mandated 0.8 — resolved by parameterizing the harness mem fraction via
env (default 0.30 preserved; M4 runs 0.8). Also documents the existing
miles C11 transport-flag gate + S2 VRAM gate discovered during the fix.
v4: addresses codex round-2 (2 high): C9 re-aligned to the ACTUAL
run_smoke_dual.sh topology (P1 train [0] / infer [0,1,2]; P2 train [3] /
infer [1,2,3]); explicit timeout hierarchy (sender budgets strictly inside
the service session deadline) + port-claim ownership on abnormal exit
(leak-and-log, never release under a possibly-live TCP store).
v2: M4 e2e re-specified per user requirement — dual-pipeline overlap topology
(not single-pipeline disaggregated), GPU memory limit 0.8.
v3: addresses codex adversarial review (2 high, 2 medium): sender-owned NCCL
teardown + bounded rendezvous timeouts + partial-receiver-failure abort
semantics; pre-committed M4 GPU split + mixed-class startup assertion;
memory preflight + adaptive staging.
Issue: https://github.com/rlops/rlix/issues/42 — "Weight update in miles currently
only support cpu_serialize, we need to support broadcast for distributed setting."

Branches:
- rlix: `zhenyu/miles-nccl-broadcast` (based on `zhenyu/miles-mvp-e2e` @ b4e0cf6)
- miles: `zhenyu/miles-nccl-broadcast` (based on `zhenyu/m11-mvp-test` @ 6ff8df3)

---

## 1. Problem context

### Current state

The M11 RLix-mode weight sync chain is:

```
MilesCoordinator (rlix)                 — sync_base_weights_to_active / _expand_workers
  → MilesModelUpdateService (rlix)      — builds SyncSessionPlan, atomic timeout, port claim
    → MegatronTrainRayActor.run_sync_session (miles, cache_owner)  — single composite RPC
      → Path A: _dispatch_cpu_serialize_bucket  — torch.save bytes → engine tmpfs → HTTP  [WORKS]
      → Path B: _dispatch_nccl_broadcast        — dynamic NCCL group broadcast            [BLOCKED]
        → SGLangEngine.setup_collective_group / broadcast_parameter / destroy_collective_group  [EXIST, dead code]
```

Two deliberate fail-fast guards block Path B today:

1. rlix `miles_model_update_service.py:199` — `sync_selected_workers` raises
   `NotImplementedError` when `broadcast_local_ranks` is non-empty.
2. miles `megatron_utils/actor.py:968` — `_dispatch_nccl_broadcast` raises
   `NotImplementedError` unconditionally; the receiver-side fan-out code below the
   raise (setup / per-bucket broadcast_parameter / destroy) is dead code.

The missing piece both guards protect against: **sender-side NCCL** on the
cache_owner (join the dynamic group as rank 0, `dist.broadcast` each bucket
tensor, destroy the group). Without it, receivers would block forever inside
SGLang `/init_weights_update_group` waiting for rank 0.

Additionally, neither rlix call site (`sync_base_weights_to_active`,
`_expand_workers` F40 runtime branch) ever passes `broadcast_local_ranks` —
there is no classification logic deciding which engines need broadcast.

### Reference implementation (do not modify, reuse pattern only)

miles standalone (non-RLix) mode already ships a working sender-side NCCL path:
`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py`
(`UpdateWeightFromDistributed`, `connect_rollout_engines_from_distributed`,
`update_weights_from_distributed`). Key proven pattern:

1. dispatch receiver `init_weights_update_group` `.remote()` refs (do NOT ray.get yet)
2. sender `init_process_group(backend="nccl", init_method=tcp://addr:port, rank=0, world_size)`
   — this blocks until all receivers join, which is why step 1 must not block
3. `ray.get(refs)` only after the sender has joined
4. per transfer: dispatch receiver metadata RPC refs (async) → sender
   `dist.broadcast(param, src=0, group, async_op=True)` + wait → `ray.get(refs)`
5. teardown: receiver `destroy_weights_update_group` refs + sender
   `dist.destroy_process_group` + `ray.get(refs)`

### Why broadcast matters (issue motivation)

- cpu_serialize routes weights CPU → tmpfs → HTTP → engine. It requires
  sender and receiver on the **same node** (tmpfs path handoff) and pays
  serialize/deserialize + disk + HTTP cost per bucket.
- NCCL broadcast is GPU→GPU, required for multi-node ("distributed setting"
  in the issue) and substantially faster for disaggregated single-node
  topologies (train pool ∩ infer pool = ∅, e.g. M11.2 P2 `[2,3]` vs train `[0,1]`).

## 2. Assumptions

- A1: SGLang receiver admin routes behave per the existing standalone path:
  `/init_weights_update_group` registers `tp_size` consecutive NCCL ranks
  starting at `rank_offset`; `/update_weights_from_distributed` blocks until
  the named tensors arrive via NCCL from rank 0.
- A2: `bucket.params` is an insertion-ordered dict (Python ≥3.7), so the
  `names`/`dtypes`/`shapes` metadata lists and the sender's broadcast order
  agree by construction.
- A3: The cache_owner train actor has a live CUDA context and enough free GPU
  memory to stage one bucket (default cap 512 MB,
  `miles_model_update_bucket_size_mb`) during transport.
- A4: MVP validation targets a single node, engines with
  `nodes_per_engine == 1`; multi-node engines (node_rank > 0 shards) are out
  of MVP scope (their `setup_collective_group` returns `{}` today).
- A5: `cluster_device_mappings` (actor_train / actor_infer physical GPU lists)
  and `rollout_num_gpus_per_engine` are sufficient to derive engine_index ↔
  physical-GPU sets, as already done in
  `MilesPipeline._wait_for_overlap_engines_offloaded`.
- A6: Known reviewer focus (PR rlops/RL#3 history): NCCL rendezvous deadlock
  ordering and global-vs-local rank confusion are the two highest-risk bug
  classes for this change family.

## 3. Hard constraints

- C1 (minimize-upstream-miles, tightened in v6): miles-side changes are
  confined to `miles/backends/megatron_utils/actor.py`
  (`_dispatch_nccl_broadcast` + in-method helpers — all RLix-port-added
  code) and tests. NOTHING else in miles changes: no `rollout.py`, no
  `arguments.py`, no `rlix_validation.py` (C11/S2 gates untouched), no
  `sglang_engine.py`, and never any pristine upstream line (including
  `UpdateWeightFromDistributed`).
- C2 (F04 single composite RPC): no new top-level Ray methods on the train
  actor for transport; sender NCCL lives inside `run_sync_session` as
  in-method helpers.
- C3 (F26/C16): `master_port != 0`; port claimed via SharedStorage before the
  plan is sent (already implemented — keep).
- C4 (F21): exactly one `set_weight_version` publish per sync, by the service
  only (unchanged).
- C5 (fail-fast): heterogeneous/invalid plans raise; no silent fallback from
  broadcast to cpu_serialize at transport time — and (v6) no silent mode
  switching either: the user-selected transport mode is honored exactly or
  the run refuses to start (strict `broadcast` + colocate target =
  startup rejection, never a quiet downgrade to mixing).
- C6 (atomicity): the whole transport remains inside the service's single
  `asyncio.wait_for` timeout. **Enforcement split (codex v3)**: the
  service-side `asyncio.wait_for` + `ray.cancel` CANNOT interrupt a sender
  blocked inside a native NCCL/HTTP call; the real bound is sender-side
  timeouts, so the sender task always unwinds by itself and its own
  `finally` performs the teardown. Service-level `inflight_refs` covers
  only service-issued refs (top-level RPC, manager calls); it is NOT
  claimed to cover the sender's nested receiver refs.
  **Deadline hierarchy (codex v4)** — one shared value is NOT a valid
  hierarchy; the budgets nest strictly:
  - session deadline = plan `timeout_s` (service `asyncio.wait_for`,
    default 150 s) — outermost;
  - sender internal budgets are fractions of `timeout_s` chosen so the
    sender's worst case (rendezvous + all per-bucket collectives +
    teardown grace) completes strictly inside the session deadline:
    rendezvous ≤ 0.2×, transport total ≤ 0.6×, teardown grace ≤ 0.1×,
    safety margin ≥ 0.1× (exact constants are implementation detail; the
    invariant "sender worst-case unwind < session deadline" is the
    requirement);
  - consequence: the service deadline fires only when the sender is truly
    wedged past its own budgets (native call refusing to unwind), which is
    the fail-fast pipeline-death path — not a normal cleanup path.
  **Port-claim ownership on abnormal exit (codex v4, refined v8-r1)**:
  the claim is released only after sender teardown is acknowledged —
  i.e. after the `run_sync_session` ref resolves (success OR exception;
  the sender's `finally` has run by then). Sender-ref resolution is
  tracked independently of claim presence, so a cancellation landing in
  the post-resolution release window still releases (fire-and-forget)
  instead of leaking. Only if the service session deadline expires
  while the sender ref is UNRESOLVED AND the session had a non-empty
  broadcast set is the claim **intentionally leaked and logged loudly**,
  never deleted — the TCP store may still be bound to that port, and the
  existing ROLL-backend precedent treats a leaked claim as safer than a
  collision. Fresh per-session `get_free_port` picks prevent rendezvous
  reuse; leaked claims die with the SharedStorage actor (fail-fast
  lifecycle). cpu_serialize-only sessions keep today's release-on-timeout
  (no TCP store exists).
- C7 (no-commit): no commit/push without explicit user instruction; codex
  review approval required before sign-off.
- C8 (world-size accounting): NCCL group ranks are per-GPU, not per-engine:
  `world_size = 1 + Σ gpu_count(broadcast engines)`; engine `rank_offset`s
  form a cursor. The current rlix per-engine assignment is only valid for
  TP=1 and must be generalized.
- C9 (user-mandated test topology): M4 e2e validation MUST run the
  dual-pipeline **overlap** topology (M11.2-style, env-driven
  `MILES_DUAL_P*` via `scripts/run_smoke_dual.sh`) with GPU memory limit
  0.8 (SGLang `mem_fraction_static=0.8`). Single-pipeline disaggregated
  smoke is NOT an acceptable substitute. **Topology = the harness's actual
  checked-in defaults (codex v4 — the v3 draft had invented a split that
  contradicted the harness)**, verified against `scripts/run_smoke_dual.sh`
  lines 68-71: `MILES_DUAL_P1_TRAIN=0`, `MILES_DUAL_P1_INFER=0,1,2`,
  `MILES_DUAL_P2_TRAIN=3`, `MILES_DUAL_P2_INFER=1,2,3` (4 GPUs, TP=1,
  1 GPU/engine). Expected classification: P1 → e0@gpu0 cpu_serialize,
  e1@gpu1 + e2@gpu2 broadcast; P2 → e0@gpu1 + e1@gpu2 broadcast, e2@gpu3
  cpu_serialize. Both pipelines are mixed-transport by construction of the
  real harness envs — the topology needs no harness change. **Memory
  limit needs one (codex v5)**: the harness hardcodes
  `--sglang-mem-fraction-static 0.30` (`run_smoke_dual.sh:125`), so the
  mandated 0.8 would silently not apply. M4 parameterizes that line via
  env — `MILES_SMOKE_MEM_FRACTION` (default `0.30`, preserving existing
  harness behavior) — and the M4 invocation exports
  `MILES_SMOKE_MEM_FRACTION=0.8`. E4 evidence must include the effective
  mem-fraction value echoed from the launch log; a 0.30 run does NOT
  satisfy AC6. The topology env values are the source of truth: the
  startup assertion (before the first training step)
  logs the actual env-derived train/infer mappings AND the actual
  scheduler grant used for classification, then verifies each pipeline's
  active sync target set contains ≥1 broadcast-classified AND ≥1
  cpu_serialize-classified engine; the smoke is INVALID (fail-fast, not
  skip) if the assertion does not hold or if the logged mappings diverge
  from the harness envs.

## 4. Acceptance criteria

- AC1: A sync session whose plan carries a non-empty `broadcast_local_ranks`
  completes end-to-end — no `NotImplementedError` on either side; targeted
  engines receive all buckets via NCCL and serve the published weight version.
  (Derives from: Goal, A1, A2)
- AC2: All-cpu_serialize behavior is unchanged: existing unit tests and the
  colocate smoke pass without modification to expectations. (Derives from:
  Goal, C5)
- AC3: NCCL rank/world_size accounting is correct for TP>1 engines
  (rank-offset cursor, per-GPU world size), verified at unit level. (Derives
  from: C8, A1)
- AC4: Transport is user-selected and its topology inputs are
  startup-validated: the v7 uniformity guard rejects resolved SGLang
  configs whose server groups diverge from
  `rollout_num_gpus_per_engine` before any classification runs. Mode via
  rlix-side env
  `RLIX_MILES_UPDATE_TRANSPORT ∈ {cpu_serialize | broadcast | auto}`,
  default `cpu_serialize` (= today's behavior, zero change until opt-in):
  `cpu_serialize` → all engines cpu_serialize; `broadcast` → ALL engines
  broadcast, with classification-time fail-fast rejection if any target
  engine's GPU intersects the pipeline's train pool (NCCL cannot form a
  group with duplicate physical GPUs — sender-colocate targets are
  unservable); `auto` → topology classification (engine GPUs ∩ train pool
  = ∅ → broadcast, else cpu_serialize; the only mode that mixes). Mode is
  logged once per pipeline at startup. (Derives from: Goal, A5, C5)
- AC5: Any abnormal end of a broadcast session — receiver failure during
  rendezvous (dies / rejects / hangs), mid-bucket exception, or sender
  budget expiry — unwinds within the C6 deadline hierarchy (sender
  worst-case < session deadline), executes the sender-owned teardown
  (destroy both sides, cancel nested receiver refs), and then releases
  the SharedStorage port claim (release strictly after teardown ack). On
  the residual wedged-sender path (service session deadline fires first)
  the claim is leaked-and-logged, never deleted. No live group, claimed
  port under a live TCP store, or unbounded block survives into the next
  sync. (Derives from: C6, C3)
- AC6: E2E dual-pipeline overlap smoke on a 4-GPU vast.ai instance (C9
  pre-committed topology, GPU memory limit 0.8, mode `auto` — the overlap
  topology requires the mixing mode by NCCL's duplicate-GPU constraint)
  exercises **mixed-transport sync sessions** — broadcast for engines
  outside each pipeline's granted train GPUs, cpu_serialize for overlap
  engines — log-verified per pipeline, with the C9 mixed-class startup
  assertion passing, **zero GPU OOM events** across the run, the existing
  dual-smoke pass-bar met, and EXIT_CODE=0. (Derives from: Goal, A4, A5,
  C9)
- AC7 (narrowed in v8): Strict `broadcast` mode honors its no-mix
  contract, verified by (a) the E3 unit matrix — colocate target →
  rejection naming the engines; all-disjoint and colocate-avoiding
  subset targets → all-broadcast; and (b) an e2e rejection check on the
  overlap harness: `RLIX_MILES_UPDATE_TRANSPORT=broadcast` fails fast at
  the first classification with the actionable error (log captured). The
  all-broadcast POSITIVE e2e is O7-deferred: miles `assert_rlix_topology`
  C1 requires train ⊂ infer, so no startable RLix topology can make
  every engine broadcast-eligible today. (Derives from: Goal, A4, A5, C5)

## 5. Milestones

- M1 — miles sender-side NCCL transport. Implement sender join (bounded
  timeout) + per-bucket broadcast with memory preflight + sender-owned
  teardown in `_dispatch_nccl_broadcast`; remove the miles-side guard;
  unit tests (E1, E7, E8) with mocked engine handles + fake dist.
  Delivers: AC1 (miles half), AC5 (miles half).
- M2 — rlix service unlock + rank accounting. Remove the service raise; fix
  `comm_ranks`/`world_size` to the per-GPU cursor scheme (uniform stride
  from `rollout_num_gpus_per_engine`, injected — no manager RPC, v6);
  C6 claim/deadline rules; unit tests. Delivers: AC1 (rlix half), AC3,
  AC5 (rlix half).
- M3 — rlix transport-mode wiring. `RLIX_MILES_UPDATE_TRANSPORT` env
  (three modes, default cpu_serialize), v7 uniformity startup guard,
  strict-broadcast startup rejection, auto classification, wiring at both
  call sites; unit tests (E3, E10). Delivers: AC4, AC2.
- M4 — vast.ai e2e validation (v8): (a) dual-pipeline overlap smoke per
  C9 (mem 0.8, mode `auto`) — mixed sessions; cpu_serialize legs +
  existing dual pass-bar double as the AC2 regression confirmation;
  (b') strict-mode e2e REJECTION check on the same overlap harness
  (mode `broadcast` → classification fail-fast, log captured; fast run,
  no training). Delivers: AC6, AC7, AC2 confirmation.

## 6. Decision trace

- D0 (Goal): unblock the NCCL broadcast transport for miles RLix-mode weight
  sync so disaggregated/distributed topologies can sync weights without
  tmpfs/same-node coupling. AC served: AC1, AC4, AC6.
- D1 (Architecture): keep the existing three-layer shape — service builds the
  plan, `run_sync_session` is the single composite RPC (C2), SGLang engines
  stay passive receivers. Sender NCCL is added strictly inside the miles
  cache_owner; classification is added strictly inside the rlix coordinator.
  No protocol keys added or removed from the plan dict (semantics of
  `comm_ranks` refined per C8). AC served: AC1, AC2. Depends on: C1, C2.
- D2 (Module boundary):
  - miles `megatron_utils/actor.py`: `_dispatch_nccl_broadcast` gains the
    sender path; new private helper(s) only (e.g. `_sender_join_group`,
    `_broadcast_bucket`). No change to `UpdateWeightFromDistributed`.
  - miles `ray/rollout.py`: NOT touched (v6 — the previously planned
    `get_engine_gpu_counts` manager method is dropped; rlix already holds
    `rollout_num_gpus_per_engine` in miles_args, uniform per pipeline and
    validated by the existing C7-engine gate, so per-engine GPU counts are
    derived rlix-side with zero new miles surface).
  - rlix `miles_model_update_service.py`: remove raise; rank-cursor
    `comm_ranks` with uniform per-engine stride from the injected
    per-engine GPU count; `world_size = 1 + per_engine × len(broadcast_set)`.
  - rlix `miles_coordinator.py`: transport-mode resolution
    (`RLIX_MILES_UPDATE_TRANSPORT`, default cpu_serialize) +
    `_classify_broadcast_engines(target, mode) -> frozenset[int]` (strict
    `broadcast` raises on any colocate target; `auto` splits; logged once
    at startup) + wiring at both call sites; topology inputs threaded
    through `register_model_update_resources` (train GPU set, infer
    mapping, per-engine count).
  - rlix `miles_pipeline.py`: pass the topology inputs at registration
    (already knows `cluster_device_mappings` + miles_args). **Uniformity
    startup guard (v7)**: before `register_model_update_resources`
    (phaseB step5), validate the resolved SGLang config that
    `_validate_topology` already receives — require every server group's
    `num_gpus_per_engine` to equal `miles_args.rollout_num_gpus_per_engine`
    (single supported RLix shape); any divergence (per-group TP override,
    heterogeneous groups) → fail-fast with an error naming the offending
    group. This makes the O6 exclusion enforceable and keeps the
    args-derived rank cursor sound without any miles-side query.
  - rlix `scripts/run_smoke_dual.sh` (v5): parameterize line 125 as
    `--sglang-mem-fraction-static ${MILES_SMOKE_MEM_FRACTION:-0.30}` —
    behavior-preserving default; M4 exports 0.8. No other harness change.
  AC served: AC1, AC3, AC4. Depends on: C1, C2, A5.
- D3 (Runtime behavior): per-sync dynamic NCCL group, strict ordering to
  avoid rendezvous deadlock (A6): (1) dispatch receiver
  `setup_collective_group` refs without blocking; (2) sender
  `init_process_group(rank=0, timeout=<rendezvous budget, C6 hierarchy>)`
  — bounded, so a receiver that dies/rejects/hangs mid-rendezvous cannot
  block rank 0 forever; on timeout the sender raises and its `finally`
  tears down (codex v3, partial-rendezvous abort); (3) `ray.get` the setup
  refs — a receiver setup failure surfaces here as an exception and takes
  the same abort path; (4) per bucket: dispatch `broadcast_parameter` refs
  → stage bucket tensors on GPU → `dist.broadcast(src=0)` in metadata
  order (collectives bounded by the transport budget, C6 hierarchy) →
  `ray.get` the bucket refs; (5) sender-owned teardown in the sender's own `finally`
  (codex v3): dispatch receiver `destroy_collective_group` refs (tolerating
  the existing 400→no-op guard) + sender `dist.destroy_process_group`,
  executed on success, exception, AND timeout paths alike — the sender
  actor, not the service, owns nested-ref lifecycle. Hard fallback when a
  native call refuses to unwind even past its timeout: fail-fast per the
  system contract — the pipeline dies and is re-registered (no in-place
  recovery attempted). Existing pause/finalize/publish steps in the
  service are unchanged. Both transport paths may coexist in one session
  (cpu_serialize set and broadcast set are disjoint subsets of the
  target). AC served: AC1, AC2, AC5. Depends on: A1, A2, A6, C6.
- D4 (Data/state shape): plan dict keys unchanged (`comm_ranks` becomes the
  per-engine rank_offset cursor; `world_size` becomes 1 + per_engine ×
  |broadcast_set| — consistent with what `setup_collective_group` already
  forwards to SGLang as `rank_offset`/`world_size`). Per-engine GPU count
  comes from rlix's own `miles_args.rollout_num_gpus_per_engine` (uniform
  per pipeline, C7-engine-validated) — no new miles surface (v6).
  Transport-mode env `RLIX_MILES_UPDATE_TRANSPORT ∈ {cpu_serialize
  (default) | broadcast | auto}` replaces the earlier separate kill-switch
  env — one control, user-explicit, no silent mixing outside `auto` (v6).
  **Relation to the existing miles `--model-update-transport` flag**: that
  flag (choices `cuda_ipc`/`cpu_serialize`, gated by miles
  `rlix_validation.py` C11 which forces `cpu_serialize` in RLix mode)
  governs only the **colocate-leg** transport inside miles. The rlix env
  is a separate rlix-side control; the C11 gate, the flag definition, and
  the harness's `--model-update-transport cpu_serialize` argument all
  remain untouched (C1). AC served: AC3, AC4. Depends on: C8, C5.
- D5 (Implementation details):
  - Staging with memory preflight (codex v3): buckets are CPU tensors
    (cache contract). Before staging each bucket, query
    `torch.cuda.mem_get_info()`; if free < bucket_size + margin (margin
    default 1 GiB, env-overridable), degrade to tensor-by-tensor staging —
    `dist.broadcast` is per-tensor anyway, so whole-bucket staging is only
    a batching optimization and degradation changes no wire behavior. Free
    staged memory before the next bucket → peak GPU overhead ≈ 1 bucket
    (healthy) or ≈ 1 tensor (degraded) (A3).
  - Warmup (F25 comment in actor.py): after group create, broadcast a
    1-element sentinel tensor before bucket 0 — receivers ignore it? —
    `unknown`: whether SGLang's route tolerates an unannounced warmup
    collective; resolve during M1 against the reference path (the standalone
    path does NOT warm up, so default is NO warmup unless e2e shows NCCL
    lazy-init stalls; decision recorded in M1).
  - dtype strings already normalized (`str(dtype).replace("torch.", "")`) on
    the metadata side; sender broadcasts raw tensors — no cast.
  - Cancellation ownership (codex v3, replaces the v1/v2 claim that
    service `inflight_refs` suffices): the service's `inflight_refs` +
    `_release_port_claim` cover only service-issued refs (top-level
    `run_sync_session` ref, manager calls, port claim). The sender's
    nested receiver refs are owned by the sender: tracked in a local list
    inside `_dispatch_nccl_broadcast`, cancelled (`ray.cancel`) plus
    group-destroyed in the sender's `finally` on any exit path. Port-claim
    release follows the C6 v4 ownership rule: release after the sender ref
    resolves (teardown acknowledged); on session-deadline expiry with a
    broadcast session, leak-and-log, never delete.
  - miles-side sender must NOT hold any lock beyond the existing
    `_cache_lock` (already held for the whole transport in
    `run_sync_session`).
  AC served: AC1, AC5. Depends on: A2, A3, C6.
- D6 (Evidence): see §7.

## 7. Evidence and stop conditions

### Evidence required

- E1 (M1, AC1/AC5): miles unit tests — sender dispatch ordering (setup refs
  before sender join; bucket metadata refs before broadcast; destroy in
  finally on success AND on mid-bucket exception), using mocked handles +
  monkeypatched `dist`/`init_process_group`. Existing
  `tests/test_miles_pipeline.py` style.
- E2 (M2, AC1/AC3): rlix unit tests — service no longer raises for non-empty
  broadcast set; plan carries cursor comm_ranks + per-GPU world_size with
  uniform per-engine stride (e.g. per_engine=2, broadcast engines {0,1} →
  ranks {0:1, 1:3}, world_size 5; per_engine=1 degenerates to the dense
  case); count injected from args, no manager RPC.
- E3 (M3, AC4/AC2): rlix unit tests — mode resolution: default/unset →
  cpu_serialize (all engines, AC2 regression shape); `broadcast` + any
  colocate target → raises at classification time; `broadcast` + all
  disjoint → all broadcast; `auto` → overlap engine cpu_serialize,
  disjoint engine broadcast (mix); invalid env value → fail-fast error.
- E4 (M4, AC6): dual overlap smoke (C9, mem 0.8) log evidence — effective
  `--sglang-mem-fraction-static 0.8` echoed from the launch log (a 0.30
  run does not satisfy AC6); C9
  mixed-class startup assertion passes for both pipelines; for each
  pipeline: ≥1 `sync_selected_workers_done ... broadcast=[...]` line with
  non-empty broadcast set AND non-empty cpu_serialize set (mixed session);
  SGLang `init_weights_update_group` success; ≥1 training step completes
  after a broadcast sync; zero CUDA OOM events in all logs; EXIT_CODE=0.
- E5 (M4, AC2): same dual run — existing dual-smoke pass-bar conditions
  (`scripts/grep_overlap_log.sh` 7-condition harness) unchanged and green;
  cpu_serialize legs of every mixed session complete normally.
- E6 (M1+M2, AC5): unit tests — (a) normal/exception paths: port claim
  released only AFTER the sender ref resolves (teardown ack ordering
  asserted); miles sender's `finally` cancels its nested receiver refs and
  destroys the group on both sides; (b) wedged-sender path (service
  session deadline fires with sender ref unresolved, broadcast set
  non-empty): claim is NOT deleted — leak-and-log asserted;
  cpu_serialize-only session keeps release-on-timeout; (c) budget
  hierarchy: sender internal budgets derived from `timeout_s` satisfy
  "worst-case unwind < session deadline".
- E7 (M1, AC5): unit test — receiver-fails-during-setup (mocked
  `setup_collective_group` raising / hanging past the patched timeout):
  sender aborts with a bounded error (no indefinite block), runs teardown,
  and surfaces the failure to the service; also covers a mid-bucket
  receiver exception taking the same abort path.
- E8 (M1, AC1): unit test — memory-preflight degradation: with
  `mem_get_info` patched to report low free memory, staging switches to
  tensor-by-tensor and the broadcast sequence/metadata order is unchanged.
- E9 (M4, AC7, narrowed v8): strict-mode e2e rejection evidence — overlap
  harness + `RLIX_MILES_UPDATE_TRANSPORT=broadcast` fails fast at the
  first classification with the actionable colocate error (log
  captured); the strict-mode POSITIVE matrix (all-disjoint, subset
  targets) is covered at unit level by E3 (see O7 deferral).
- E10 (M3, AC4): unit tests for the v7 uniformity guard — resolved SGLang
  config with a server-group `num_gpus_per_engine` override diverging
  from `rollout_num_gpus_per_engine` → startup fail-fast naming the
  offending group; uniform config passes; guard runs before
  `register_model_update_resources`.

### Stop conditions (agent must stop and ask)

- S1: Any change would touch pristine upstream miles lines (C1 conflict).
- S2: E2E requires an SGLang server-side route change (receiver routes turn
  out not to match A1) — surface findings first.
- S3: The warmup question (D5) cannot be resolved by observation on the vast
  instance within one debugging session.
- S4: Any commit/push/PR action (C7 — needs explicit user instruction).

## 8. Out of scope

- O1: Multi-node engines (`nodes_per_engine > 1`) and cross-node tmpfs — MVP
  is single-node validation (A4); the rank-cursor scheme is designed
  multi-node-ready but not e2e-verified here.
- O2: Performance benchmarking cpu_serialize vs broadcast (issue asks for
  capability, not perf numbers).
- O3: Changes to miles standalone (non-RLix) update paths
  (`UpdateWeightFromDistributed`, `UpdateWeightP2P`, colocate IPC).
- O4: LoRA/multi-adapter sync over broadcast.
- O5: Dynamic re-classification mid-training beyond what the two existing
  call sites already do per-sync.
- O6 (v6, enforcement added v7): heterogeneous per-engine GPU counts
  (e.g. prefill TP2 / decode TP4 mixes) are out of scope — and, per codex
  round-5, this exclusion is ENFORCED, not merely declared: a rlix-side
  startup uniformity guard (see D2) fail-fasts any resolved SGLang config
  whose server groups override `num_gpus_per_engine` away from
  `rollout_num_gpus_per_engine`. If heterogeneous engines land later, the
  documented extension is the miles manager read method
  (`engine_gpu_counts` property already exists) — deliberately NOT added
  now per minimal-change.
- O7 (v8): fully-disjoint train/infer topologies (and therefore the
  all-broadcast positive e2e). miles `assert_rlix_topology` C1 requires
  train ⊂ infer as the M11 partial-overlap contract; admitting disjoint
  pools is a scheduler/validation design change, not a transport change.
  When that lands, run the deferred disjoint dual smoke with
  `RLIX_MILES_UPDATE_TRANSPORT=broadcast` as the positive AC7 e2e.

## 9. Risks

- R1 (A6): rendezvous deadlock if any `ray.get` lands between receiver
  dispatch and sender join → covered by E1 ordering tests. Partial-receiver
  failure during rendezvous (the non-happy-path variant, codex v3) →
  bounded sender timeout + abort semantics (D3), covered by E7.
- R2 (A6, C8): rank misassignment for TP>1 → covered by E2; e2e M4 runs TP=1
  so unit coverage is the only TP>1 gate (accepted for MVP).
- R3 (A3): GPU OOM while staging a bucket on a busy train GPU. C9 tightens
  this risk: mem limit 0.8 on 16 GB GPUs leaves ≈3 GB nominal headroom,
  and fragmentation/NCCL buffers/copy temporaries erode it further (codex
  v3). Mitigation is now designed-in, not deferred: memory preflight +
  tensor-by-tensor degradation (D5, E8) + zero-OOM as an AC6 hard
  condition (E4). Defense-in-depth note (v5): miles already ships an S2
  startup gate (`rlix_validation.py`) rejecting configs where bucket_size
  + transport scratch ≥ estimated post-wake free VRAM when non-colocate
  engines exist — S2 catches misconfiguration at startup; the D5
  preflight catches runtime erosion. Both stay.
- R4: leaked NCCL group after a failed session poisons the next session's
  group_name — group names are per-sync (`miles_{pipeline}_{sync_id}`), so a
  leak wastes resources but cannot collide; sender-owned
  destroy-in-finally + bounded timeouts (D3) + E6/E7 bound the exposure.
  Port-claim rule (v4, replaces the v3 note codex flagged): release only
  after sender teardown ack; on the wedged-sender path the claim is
  leaked-and-logged (existing ROLL-backend precedent: leak is safer than
  collision under a possibly-live TCP store). Residual cost of a leak:
  one stale SharedStorage key until job restart — accepted.
- R5: `flush_cache`/finalize interaction with paused engines differs under
  broadcast — finalize flow is transport-agnostic (unchanged code path);
  watched in M4 logs.

## 10. M4 evidence record (2026-08-16, appended post-run)

Instance: vast 47858280, 4x A100-SXM4-80GB, fork-baseline image (torch
2.9.1+cu129 / sglang 0.5.10 / ray 2.54.1). Logs on instance:
`/root/smoke_b_reject.log`, `/root/smoke_a_auto08.log`,
`/root/smoke_a_final.log`.

- E9 (run b'): `RLIX_MILES_UPDATE_TRANSPORT=broadcast` on the overlap
  harness → registration logged `transport mode=broadcast ...
  train_gpus=[0] infer_gpus=[0,1,2]`; first classification raised the
  actionable error ("cannot serve colocate engines [0] ... Use 'auto'");
  SMOKE_EXIT=1. PASS.
- E4 (run a, final): `auto` + `MILES_SMOKE_MEM_FRACTION=0.8` +
  `RLIX_ASSERT_MIXED_TRANSPORT=1` → C9 assertion logged+passed for both
  pipelines (P1 broadcast=[1,2]/cpu_serialize=[0]; P2
  broadcast=[0,1]/cpu_serialize=[2]); `mem_fraction_static=0.8`
  confirmed in engine logs; zero CUDA OOM; SMOKE_EXIT=0. Route-level
  mixed-transport attribution (first run): `init_weights_update_group`
  200 ×10, `update_weights_from_distributed` 200 ×11,
  `update_weights_from_cpu_bucket` 200 ×6,
  `destroy_weights_update_group` 200 ×10; per-pid GPU attribution
  matches the predicted classification; P2 same-session mix confirmed by
  timestamp adjacency (cpu_bucket 11:39:36-38 → distributed 11:39:39).
  PASS.
- E5: `grep_overlap_log.sh` 7-condition harness → RESULT: PASS on both
  the first auto run and the final assertion-enabled run.
- Unit evidence: miles 16/16 (E1/E7/E8 + env guard); rlix 32/32
  (E2/E3/E6/E10 + C9 assertion). Regression: rlix suite 91 passed (1
  pre-existing ray-2.54 API failure, verified on pristine base); miles
  test_miles_pipeline.py 6/6.
- Codex implementation reviews: r1 (1 high NCCL bound + 1 medium claim
  window) fixed; r2 (1 high env setdefault) fixed; r3 approve; r4
  (assertion delta) approve.

## 11. Post-delivery user-directed expansion: O7 partially delivered (2026-08-16, appended)

User directive ("我要强制跑nccl" + "不管用户怎么设置gpu拓扑，你都应该支持的吧")
unlocked the dedicated-train (fully-disjoint) topology ahead of the O7
deferral, expanding the miles diff beyond the original C1 boundary
(actor.py + tests) into `examples/rlix/run_miles_dual.py` — still
RLix-port-added code, per the user's standing minimize-upstream rule.

Changes (codex delta rounds 5-6, r6: approve):
- `run_miles_dual.py::_overlap_pools_from_env` + `_build_pipeline`: the
  per-pipeline invariant is now two-family — subset (overlap, tested) or
  fully-disjoint (dedicated train cards, experimental warning); partial
  intersections stay rejected with an actionable error (scheduler
  shrink/grant accounting unverified for them).
- rlix `miles_pipeline._wait_for_overlap_engines_offloaded`: filters
  granted train GPUs to the infer mapping before deriving engine indices
  (disjoint grants previously produced nonsense indices like -1, masked
  by a broad exception catch).
- Correction to the v8 note: miles C1 (F10) checks the LOCAL count-derived
  shape, not physical placement — dual-mode physical mappings were only
  blocked by the driver's own asserts, which is why this unlock needed no
  `rlix_validation.py` change. The v8 claim "unreachable without relaxing
  the C1 gate" was wrong at the physical level.
- Status: strict `broadcast` all-NCCL on the disjoint dual topology
  (P1 train [0] / infer [1,2]; P2 train [3] / infer [1,2]) is
  user-verification-pending (user runs it themselves); unit + codex
  gates green.

## 12. Disjoint-topology debug: concurrent-sync port collision (2026-08-16, appended)

First user-run of the dedicated-train topology crashed: sender CUDA
"invalid argument" + receivers losing the rendezvous TCP store mid
ncclUniqueId exchange. Root cause chain (three stacked facts):
miles `RayActor.get_free_port` scans deterministically from 20000, so
both pipelines' cache_owners pick the SAME port; the SharedStorage
port-claim protocol was silently skipped in miles mode (ROLL's actor
does not exist there — the F26/C16 protection was never active); the
overlap topology masked this by scheduling serialization, while
dedicated-train pipelines finish training steps simultaneously and sync
CONCURRENTLY — two NCCL TCP stores cross-wired on port 20000.

Fix (codex delta rounds 7-8, r8: approve):
- New `_PortClaimStore` detached Ray actor ('rlix:miles_port_claims',
  RLIX_NAMESPACE) — rlix-owned fallback backend for the claim protocol;
  fallback triggers ONLY on ImportError (no roll) or ray.get_actor
  ValueError (actor absent); any other lookup failure fails the sync
  fast (no silent skip, no split claim namespace).
- Claim tuple carries the accepting store handle; all three release
  paths (awaited, nowait, leak-path unpack) use the carried handle.
- Collision retry now re-picks with get_free_port(start_port=port+1) —
  the contested port is claim-reserved but not OS-bound, so the old
  plain rescan returned the same port until the retry budget exhausted.
- Evidence bar for the user's rerun: two distinct master_port values in
  the log + zero cpu_bucket routes + EXIT 0. rlix suite 35/35.

## 13. O7 continued: dual-disjoint scheduler gap + single-disjoint unlock (2026-08-16, appended)

Second user-run finding (after §12's port fix and the tms hook lesson):
dual + disjoint + SHARED infer cards deadlocks at P2 init. Chain: with
dedicated train cards nothing ever shrinks P1's activated engines off the
shared GPUs (in overlap topologies the pipeline's own train reclaims
them, driving the shrink/planned-release rotation); the driver stages
init before training loops, so P1 emits no progress signals; P2's
init-time GENERATION request (lowest priority, progress-driven rotation)
waits forever. **Known limitation, NOT fixed here**: admitting
idle-GEN-yields-to-pending-init is scheduler-regime design work (O7
follow-up). On 4 GPUs: dual ⇒ use overlap+auto (validated); all-NCCL ⇒
single-pipeline disjoint.

Also learned (user runs): (a) tms hook `preload` breaks sender staging
while the train actor is offloaded (stale cached allocator pointers →
cudaErrorInvalidValue at tensor.to); `torch` hook is the validated mode
for broadcast + offload_train — debug_pipeline's "auto" resolves to
preload on A100 and must be overridden. (b) On shared infer cards under
DISJOINT (no time-sharing), co-resident engine pools must divide the
card: mem_fraction ≤ ~0.9/N (dual → 0.4); the 0.8 mandate applies to
overlap (time-shared) topologies where ≤1 pool is awake per card.

Delivered (codex delta rounds 9-10, r10: approve):
`run_miles_rlix.py::_build_cluster_device_mappings` gains the
`MILES_SINGLE_TRAIN_GPUS` / `MILES_SINGLE_INFER_GPUS` explicit-mapping
override (two-family invariant; env lengths must equal the CLI-derived
actor/rollout counts, fail-fast otherwise) + 9-case test matrix in
miles tests/test_single_mapping_override.py. Minimal all-NCCL topology
on 4 GPUs: single pipeline, train [0] / infer [1,2], mem 0.8, mode
broadcast — user-verification pending.

## 14. Sync-under-load flush-timeout fix (2026-08-16, appended; codex r11-r16, r16: approve)

Field failure (user's 20-rollout dual auto run): after_training →
sync_base_weights_to_active → finalize /flush_cache TimeoutError. Root
cause: under the fully-async rollout, generation for the next step is
already live on the target engines and the router keeps dispatching
during the pause/flush window — the queue never drains (2-rollout smokes
had empty queues). Transport-agnostic (cpu path shares finalize).

Fix — the sync-under-load bracket (rlix miles_coordinator
sync_base_weights_to_active) hardened over six codex rounds:
- quiesce BEFORE the sync, mirroring shrink_engines: per-engine
  unregister_from_router (admission close) + manager _abort_engines
  (router re-dispatches aborted requests with the NEW weights);
- quiesce lives INSIDE the try so all failures reach cleanup (r11);
- re-admission via the NEW manager-side atomic
  RolloutManager.register_router_if_active (state check + idempotent
  /add_worker in ONE serialized manager call; manager is
  max_concurrency-1, so it cannot interleave with shrink_engines) —
  NOT activate_routing (loading-only INIT transition, raises for active
  engines, r11) and NOT a read-then-register pair (TOCTOU, r13); the
  method filters manually because _resolve_engine_indices raises for
  the exact concurrently-shrunk engines it must skip (found via the
  r15 contract test);
- concurrently-shrunk engines are never re-admitted (intent ∩ live
  state, r12) and are warning-logged;
- a re-registration FAILURE after a successful sync escalates to
  RuntimeError (never report healthy-but-unroutable, r14); a failed
  sync keeps its original exception as primary;
- registration-time capability probe (empty-list call) fails loud on
  new-rlix/old-miles version skew before any sync quiesces engines
  (r15);
- abort-idempotency cache reset is an independent best-effort finally
  step so future genuine shrinks re-abort.

miles diff grows by the one manager method (rollout.py RLix section) +
contract test. Tests: rlix AST ordering + behavioral concurrency/
escalation (105 total green), miles contract 4 cases (29 total green).
User e2e verification of the 20-rollout dual run pending.

## 15. Final e2e verification — 20-rollout dual overlap run (2026-08-16, user-run)

Config: debug_pipeline.py, mode=dual, transport=auto, tms hook torch,
mem_fraction 0.8, num_rollout=20, topology P1 [0]/[0,1,2] + P2
[3]/[1,2,3], 4x A100-SXM4-80GB. Evidence panel from /root/logs/run.log:

- NCCL group creations: 45 (`backend=nccl`, `world_size=3`, per-sync
  group names `miles_{pipeline}_{sync_id}`).
- Mixed transport across the full run: `update_weights_from_distributed`
  200 x48 (NCCL legs) + `update_weights_from_cpu_bucket` 200 x46
  (colocate legs) + `destroy_weights_update_group` 200 x44.
- Port anti-collision IN ACTION: master_port=20000 x27 AND
  master_port=20001 x18 — concurrent pipeline syncs claimed distinct
  rendezvous ports via the _PortClaimStore fallback (§12 fix verified
  under load).
- Zero CUDA OOM, zero "Timeout while flushing cache" (§14 bracket
  verified under 20-rollout fully-async load), zero colocate
  rejections; 36 /generate requests served across cycles.

This is the definitive under-load validation on top of §10's 2-rollout
smokes: every fix from §12-§14 exercised in one run.
