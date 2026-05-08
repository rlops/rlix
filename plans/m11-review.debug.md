# M11 Review — Debug Log

Source plan: `plans/m11-review.plan.md`
Code review brief: `plans/m11-review.review.md`
Review report: `plans/m11-review.review-report.md`
Code under review: rlix `zhenyu/miles-mvp-e2e@5dc4e43` + miles `zhenyu/m11-mvp-test@6126e01`
Vast target: instance 36267892 (`ssh4.vast.ai:27892`, 4xA40)

**Append-only log of post-review verification runs.**

---

## Current status

- Final recommendation from review-report.md: **NEEDS_FIX** (1 HIGH = R04-F1, 3 MEDIUM)
- Verification plan: §10 Annex A.3 (M11.2 dual smoke post-engine-index-fix), §10.4 (R06 destructive naming test)
- Tests-passing evidence pending: fresh M11.2 dual smoke at HEAD `5dc4e43` / `6126e01` (E6 from review-report §3)

---

## Run 1 — M11.2 dual smoke (post-fix)

- **Goal**: Close evidence gap E6 — confirm M11.2 dual smoke still PASSes after Feature 13 (engine-index conversion fix, commit 549cfbd) lands. M11.2 attempt 4's PASS log preceded the fix.
- **Commands**: §10.3 of `m11-review.review.md` (run on vast)
- **Vast**: instance 36267892, 4xA40, direct SSH `root@154.42.3.11:19145` (jump host `ssh4.vast.ai:27892` was unreachable; vast direct host worked)
- **Repos at run time**: rlix `5dc4e43` ✓, miles `6126e0187` ✓ (matches review brief)
- **Started**: 2026-05-08 14:58 UTC (vast time) under `tmux new-session -d -s smoke`
- **Watchdog config**: `SILENCE_LIMIT=900`, `RUN_LIMIT=3600`, log at `/root/logs/m11-2-rerun.log` (smoke output redirected to `/root/logs/run.log` by the script)
- **Completed**: 2026-05-08 15:08:40 UTC (~10 min wall clock from launch to dual shutdown_hard)
- **Status**: PASS
- **Findings**:
  - `EXIT_CODE=0` ✓
  - mp1 (`miles_e42d3b58a41f`) and mp2 (`miles_df5809b2097b`) both logged `training loop complete` (15:08:30 / 15:08:40) and `shutdown_hard complete` (both at 15:08:40).
  - `grep -c KeyError` → 0; `grep -c engine_index` → 0 (Feature 13 fix is non-regressing on a fresh dual smoke).
  - Both pipelines completed 2 rollouts each (`rollout_id=0` and `rollout_id=1`) through the full bracket: `before_step → train → after_step` with CPU offload markers visible (`after_offload_train: 91.60–94.39 GB`).
  - No `Failed to resolve actor`, no Tracebacks, no `OOM`, no `Killed`.
  - SGLang quiescence sequence observed end-to-end: `pause_generation → flush_cache → continue_generation → update_weight_version` all 200 OK.
- **Result**: ✅ Closes evidence gap E6 from `m11-review.review-report.md` §3. R02 + R05 sign-off no longer blocked on a pre-fix log.

---

## Run 2 — R06 destructive naming test

- **Goal**: Confirm Feature 16 invariant (coordinator naming convention) is load-bearing. Corrupt the name in `run_miles_rlix.py` and verify the scheduler emits "Failed to resolve actor".
- **Commands**: §10.4 of `m11-review.review.md`
- **Started**: 2026-05-08 15:12 UTC under `tmux new-session -d -s baduname`; rerun at 16:25 UTC under `baduname2` after first attempt aborted because `scripts/run_smoke_e2e.sh` was missing on vast (it's gitignored locally — scp'd over before the rerun).
- **Corruption**: `name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` → `name=f"bad_coordinator_{pipeline_id}"` at `examples/rlix/run_miles_rlix.py:201`
- **Backup**: `/tmp/run_miles_rlix.py.bak` on vast — restored at 16:32 UTC; diff post-restore = empty ✓
- **Log**: `/root/logs/m11-bad-name.log`
- **Completed**: 2026-05-08 16:26:34 UTC (smoke crashed during init before reaching coordinator-creation)
- **Status**: INCONCLUSIVE
- **Findings**:
  - Driver reached F10 startup validation (`engines=4, per_engine=1, train=2, infer=4, overlap=2`) and `cluster_device_mappings` computed correctly.
  - Driver attached to Ray cluster (`Connected to Ray cluster ... 172.17.0.2:6379`).
  - **8 seconds later** Ray crashed: `core_worker_process.cc:216: Failed to register worker to Raylet: IOError: Failed to read data from the socket: End of file`. Same pattern observed on both attempts (15:44:21 and 16:26:34 UTC).
  - This Raylet IOError aborts `run_miles_rlix` **before** the driver creates the (deliberately mis-named) `MilesCoordinator` actor and **before** the scheduler attempts to resolve it. The destructive test could not be exercised on this instance.
  - Independent of this test, `ray status` reports the GCS unreachable after the crash — leftover Ray state from the prior dual smoke is the suspected root cause (the script's `ray stop --force` + `ray start --head` did not produce a fully clean cluster on this image).
- **Result**: ⚠️ INCONCLUSIVE on this instance. **Does not invalidate the R06 verdict.** The R06 sidecar (`plans/m11-review.review-report/R06.md`) cites M11.1 attempt 4's primary `Failed to resolve actor` failure plus the matching commit 7b83be5 fix as the authoritative load-bearing evidence; subsequent attempt 10 (full E2E PASS) and M11.2 attempt 4 (dual-pipeline PASS) prove the convention is stable. The destructive test was a corroborative sanity check, not a critical-path verification. Re-run on a fresh-boot instance (or after a hard `ray stop` cycle that clears `/tmp/ray*`) is the natural follow-up.

### Restoration verification

```
# on vast at 16:32 UTC
cd /root/miles && diff /tmp/run_miles_rlix.py.bak examples/rlix/run_miles_rlix.py
# (empty diff)
grep -n 'name=f' examples/rlix/run_miles_rlix.py
# 201:            name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}",
```

---

## Summary for review-report consumers

- **E6 (m11-review.review-report.md §3)**: ✅ CLOSED. Run 1 PASS at HEAD `5dc4e43` / `6126e01` confirms Feature 13 engine-index fix is non-regressing.
- **Annex A.4 R06 destructive test**: ⚠️ INCONCLUSIVE due to Ray init issue unrelated to the naming invariant. R06 verdict (PASS_WITH_NOTES) stands on M11.1 attempt 4's primary evidence.
- **Final review-report recommendation (NEEDS_FIX)** unchanged: the single HIGH finding (R04-F1, GPU not released if `train()` raises mid-iteration) was a static-analysis finding; happy-path Run 1 cannot exercise that path. Tracking carries into M11.x production hardening per the recommendation.
