# M11 RLix-mode E2E Test Log

Append-only iteration log for the Qwen2.5-0.5B GRPO `--num-rollout 2` smoke
test on vast.ai 4xGPU under the rlix-controlled path.

Plan reference: `/Users/zhenyulin/.claude/plans/humming-sleeping-pumpkin.md`.

---

## Attempt 0 — pre-test wiring landed (no run yet)

- Date: 2026-05-04
- Branch heads (pre-push):
  - rlix: `zhenyu/miles-mvp-e2e` — adds `MilesPipeline` accessors + public `before_training`/`after_training` aliases.
  - miles: `zhenyu/m11-mvp-test` — replaces iter-16 stub with full driver
    (`examples/rlix/run_miles_rlix.py`) and adds reusable async loop helper
    (`miles/utils/rlix_train_loop.py`).
- Vast instance: not yet provisioned for this attempt.
- Outcome: wiring committed. Driver flow:
  1. `rlix.init(create_if_missing=True)` → orchestrator + auto scheduler
  2. `allocate_pipeline_id` → `register_pipeline` → `admit_pipeline`
  3. Build `MilesPipelineConfig` dataclass (writeable `system_envs`)
  4. Create `MilesCoordinator` named actor in `RLIX_NAMESPACE`
  5. `coord.create_pipeline_actor` → `MilesPipeline` actor
  6. `pipeline.initialize_pipeline(coord)` → Phase A + Phase B
  7. Pull `train_group` / `rollout_manager` via accessors
  8. `train_group.set_rollout_manager(rollout_manager)` (single asyncio.run)
  9. `coord.sync_base_weights_to_active(-1)` for v=-1 base push
  10. `run_async_train_loop` with rlix-corrected ordering
      (rollout dispatch AFTER `after_step`)
  11. `pipeline.shutdown_hard`
- Codex KT + plan review: applied feedback on
  - loop ordering (dispatch next rollout AFTER `after_step`, not before)
  - base sync via `coord.sync_base_weights_to_active(-1)` (not direct `train_group.update_weights`)
  - `--save ""` to disable save (final-rollout trigger fires regardless of `--save-interval`)
  - public hook aliases on `MilesPipeline`
  - missing CUDA / NCCL env vars in vast runbook
- Known limitations / deferred:
  - `args.start_rollout_id` defaults to 0 in rlix loop (Phase A discards init() return); fine for clean checkpoint smoke.
  - Save / eval at final rollout disabled by `--save ""` and skipping eval inside the loop helper.
  - Single-pipeline only (F22 dual-pipeline deferred).
- Next: commit, push, rsync to vast, run smoke.

---
