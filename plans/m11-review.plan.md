# PLAN: M11 (single + dual pipeline rlix-mode E2E) review

> Precondition artifact for the `code-review-plan` skill. Compact-first;
> the skill reads this and emits `m11-review.review.md` with bounded
> shards + adversarial prompts + test cases per shard.

## Summary

Two RL training milestones for rlix-mode (Ray-based GPU time-sharing
controller hosting miles' Megatron + SGLang stack):

- **M11.1** — single MilesPipeline with partial-overlap GPU sharing
  (train ⊂ infer). Qwen2.5-0.5B GRPO `--num-rollout 2`. ✅ EXIT_CODE=0
  on vast.ai 4xRTX5090 (10 fix-retest cycles).
- **M11.2** — two concurrent MilesPipelines on disjoint per-pipeline
  pools (P1=[0,1] partial-overlap, P2=[2,3] partial-overlap, no
  cross-pipeline contention). Same model + dataset. ✅ EXIT_CODE=0 on
  vast.ai 4xA40 (4 fix-retest cycles).

The full reviewer-friendly walkthrough lives at
`docs/m11-implementation-guide.md` (17-feature catalog).

## Files changed (in scope for review)

### rlix branch `zhenyu/miles-mvp-e2e` (HEAD `5dc4e43`)

| File | Lines | Feature(s) |
|---|---|---|
| `rlix/pipeline/miles_pipeline.py` | 67-70 (cluster_id) · 133-241 (Phase A) · 242-411 (Phase B incl. step_target_estimate) · 418-505 (`_wait_for_overlap_engines_offloaded`) · 566-596 (`_before_training`) · 598-660 (`_after_training`) · 717-721 (public Ray hooks) · 833-884 (`_build_placement_provider`) | F6 init bootstrap · F7 runtime hooks · F8 nvidia-smi probe · F9 with_ref · F11 engine-index conversion · F13 cluster_device_mappings · F17 step_target_estimate |
| `rlix/pipeline/miles_coordinator.py` | ~470 (`_expand_workers` no-op) | F10 already-active no-op |
| `rlix/pipeline/miles_model_update_service.py` | 288-318 | F3 pause/finalize/continue around weight sync |
| `docs/m11-implementation-guide.md` (NEW) | full | reviewer-friendly catalog |
| `docs/tms-fixes.md` | full | 5 tms fixes deep-dive |
| `plans/m11-e2e-test-log.md` | full (append-only) | M11.1 attempts 0–10 verification log |
| `plans/m11-2-dual-pipeline-log.md` | full (append-only) | M11.2 attempts 0–4 verification log |

### miles branch `zhenyu/m11-mvp-test` (HEAD `6126e01`)

| File | Lines | Feature(s) |
|---|---|---|
| `miles/ray/rollout.py` | 220-235 (`MILES_ROLLOUT_BASE_PORT` env var) · 911-942 (shrink_engines pause_generation) · 1616-1628 (rlix-mode override forces `enable_memory_saver=True`) | F1 enable_memory_saver · F2 pause_generation pre-release · F14 port collision env var |
| `examples/rlix/run_miles_rlix.py` | full (~265 lines) · 191-208 (coordinator name + namespace) | F6 init driver · F16 coordinator name fix |
| `examples/rlix/run_miles_dual.py` | full (~370 lines, NEW) | F12 dual driver |
| `miles/utils/rlix_train_loop.py` | full (~150 lines) | F7 reusable async loop |

### Smoke scripts (gitignored — local + vast only)

| File | Purpose |
|---|---|
| `scripts/run_smoke_e2e.sh` | M11.1 single-pipeline smoke (`--offload-rollout`, `MILES_TMS_HOOK_MODE=torch`) |
| `scripts/run_smoke_dual.sh` | M11.2 dual-pipeline smoke (`ulimit -n 65536`, per-pipeline shape) |
| `scripts/run_smoke_with_watchdog.sh` | watchdog wrapper |

## Hard constraints

The review must respect these scope guards (no findings against them):

1. **Source is authoritative** — F22 strict shell→offloaded→active
   contract is deferred (M11.3); M11.1 uses an "engines come up active"
   hatch (F10).
2. **Multi-pipeline 3+** — out of scope per Codex M11.2 scope review.
3. **save / eval at final rollout** — broken by `_after_training`'s
   offload; smoke uses `--save ""` and skips eval. Not a defect.
4. **Vast image hotpatch** — `_with_region_config` nested-region bug in
   tms 0.0.9 entrypoint.py is patched on the vast image only, not in
   repo. Tracked.
5. **No new pytest tests required** — gap is acknowledged; smoke run
   is the verification surface for this milestone.

## Decision trace

- **D0** — *Why driver-from-outside?* `MilesPipeline` has step hooks
  but no `run()` method. Driver lives in miles `examples/rlix/` (not in
  rlix repo) so rlix doesn't import miles training internals (clean
  layering).
- **D1** — *Why partial overlap and not separate pools for M11.1?* User
  explicitly said "purpose of rlix is to use fewer gpu and make it more
  efficient". Disjoint pools defeat the purpose.
- **D2** — *Why disjoint pools for M11.2?* C4 topology check + F22
  shell-init not done. Disjoint per-pipeline pool with internal
  partial-overlap is the minimum-viable PASS that still exercises the
  rlix scheduler bookkeeping for two pipelines without cross-pipeline
  GPU contention.
- **D3** — *Why pause_generation, not flush_cache + sleep?* Tested 0.5s
  sleep workaround in M11.1 attempt 6; flush_cache hangs 60s anyway
  because the SGLang scheduler thread keeps re-emitting Triton kernels.
  `pause_generation(mode="retract")` is the proper API.
- **D4** — *Why nvidia-smi probe and not engine-state-only wait?*
  `state="offloaded"` returns BEFORE the CUDA driver returns memory
  to the OS pool. Tested in M11.1 attempt 5: state was "offloaded"
  but free_GB=1.97. nvidia-smi is the actual "memory free to OS"
  signal.
- **D5** — *Why `with_ref=args.use_kl_loss or kl_coef!=0`?* Standalone
  miles uses the same condition (`miles/ray/placement_group.py:192`).
  Without it, `train_actor`'s ref_log_probs computation is gated off
  and `policy_loss_function` crashes on `torch.cat(NoneType, dim=int)`.

## Acceptance criteria

The reviewer fleet's findings should answer:

1. **Correctness**: do all 17 features land their stated invariant?
   (e.g., does Feature 11's offset arithmetic actually produce the
   right local engine index for an arbitrary infer pool?)
2. **Race safety**: any new races introduced by the pause_generation
   wrapping? double-pause? continue_generation called when no pause?
3. **Resource safety**: do the rlix-mode forks of `start_rollout_servers`
   and `shrink_engines` leak GPU memory in failure paths (e.g.,
   pause_generation 200 OK but release_memory_occupation crashes —
   does continue_generation still run)?
4. **Multi-pipeline correctness**: per-pipeline `MILES_ROLLOUT_BASE_PORT`
   guarantees no port reuse on Ray worker restarts? `cluster_device_mappings`
   correctly threaded through `_build_placement_provider` even when
   `system_envs` mutates `pipeline_config`?
5. **Driver semantics**: does `run_miles_dual.py` correctly deep-copy
   args (no per-pipeline arg pollution)? Does `_split_pools_for_dual`
   handle odd GPU counts safely?
6. **Test coverage gap**: is the smoke run sufficient for M11.1 / M11.2,
   or are there scenarios (e.g., shrink mid-decode-of-final-iteration)
   not exercised?

## Out of scope for this review

- Standalone (non-rlix) miles paths — unchanged
- Megatron internals (FP8, MoE, etc.) — orthogonal
- SGLang internals — patched at API boundary only (pause_generation,
  release_memory_occupation, finalize_weight_update)
- ROLL `RollResourceManagerProxy` — used as intended; not modified

## Evidence

- `docs/tms-fixes.md` — 5 tms fixes with reproduction traces
- `plans/m11-e2e-test-log.md` — M11.1 attempts 0–10 with timestamped
  log evidence per attempt
- `plans/m11-2-dual-pipeline-log.md` — M11.2 attempts 0–4
- Archived smoke logs: `~/Downloads/m11-e2e-success-1778067894.tar.gz`
  (M11.1) + `~/Downloads/m11-2-dual-success-1778123778.tar.gz` (M11.2)

## Stop conditions

The `code-review-plan` skill should emit shards. Reviewer fleet stops
when:
- All 17 features have at least one shard touching them
- Each adversarial prompt has been answered (yes / no / out-of-scope)
- Final aggregated `m11-review.review-report.md` has zero CRITICAL,
  documents HIGH/MEDIUM/LOW with severity
