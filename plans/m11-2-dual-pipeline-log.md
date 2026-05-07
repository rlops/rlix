# M11.2 RLix-mode Dual-Pipeline E2E Test Log

Append-only iteration log for the Qwen2.5-0.5B GRPO `--num-rollout 2`
DUAL-pipeline smoke test on vast.ai 4xRTX5090.

Plan reference: Codex M11.2 scope review (recommends Option A — disjoint
pools, minimum-viable PASS without F22 shell-init).

## Topology (Option A — disjoint pools, per-pipeline partial overlap)

```
Pipeline 1: train=[0],  infer=[0,1]   (1 train + 1 non-overlap)
Pipeline 2: train=[2],  infer=[2,3]   (1 train + 1 non-overlap)
                                       │
                                       └─ both satisfy C4
                                          (≥1 non-overlap engine)
```

Each pipeline uses 2 GPUs total; the two pipelines share NO physical GPU.
The `actor_num_gpus_per_node=1` per-pipeline shape exercises Megatron at
its smallest viable footprint (no tp/pp/cp parallelism for 0.5B).

## Pre-test code (committed, unpushed)

- rlix `zhenyu/miles-mvp-e2e` head: d97178e — feat(rlix): thread
  cluster_device_mappings into _build_placement_provider for M11.2
- miles `zhenyu/m11-mvp-test` head: 992fa26 — feat(miles): M11.2
  dual-pipeline driver + per-pipeline rollout base port

## Files added / changed for M11.2

- `miles/examples/rlix/run_miles_dual.py` (new) — dual-pipeline driver
- `rlix_miles/scripts/run_smoke_dual.sh` (new, gitignored) — smoke runner
- `rlix_miles/rlix/pipeline/miles_pipeline.py` — `_build_placement_provider`
  reads `cluster_device_mappings` from pipeline_config when set
- `miles/miles/ray/rollout.py` — `start_engines` reads
  `MILES_ROLLOUT_BASE_PORT` env var (default 15000) so two MilesPipeline
  actors don't race on `find_available_port`

## Vast instance state

Instance 36156578 (machine 25132) STUCK in EXITED state. Multiple
`vastai start instance 36156578` attempts (2026-05-06 11:46–~12:18)
all return "Required resources are currently unavailable, state change
queued." `vastai search offers machine_id=25132` returns no offers —
the host's GPUs are fully reserved by another tenant.

Decision: keep waiting on machine 25132. Will not pivot to a new
4xRTX5090 offer because (1) the user explicitly said "turn on instance
by yourself" referring to the existing instance, (2) re-setup on a
new instance (model weights, repos, tms hotpatch) would take 30-60
min and lose the existing storage spend, (3) the dual implementation
is committed + pushed and can be exercised whenever capacity returns.

The dual-pipeline implementation IS complete:
- rlix `zhenyu/miles-mvp-e2e` head: d97178e — pushed
- miles `zhenyu/m11-mvp-test` head: 992fa26 — pushed
so any later operator can SSH into 36156578 once it boots, run
`bash /root/rlix/scripts/run_smoke_dual.sh` (after rsync), and verify.


## ✅ M11.2 dual-pipeline PASS on A40 instance (2026-05-07 03:15:49)

After ~12 hours of waiting on machine 25132 (host saturated), user
provided a new 4xA40 instance (ssh4.vast.ai:27893). Setup + run took
~33 min total.

### Setup performed on the new instance
1. Cloned rlix to /root/rlix at branch `zhenyu/miles-mvp-e2e` head d97178e
2. Switched /root/miles to `zhenyu/m11-mvp-test` head 992fa26 → 6126e01
3. `pip install -e .` for both miles + rlix
4. Installed missing deps: ROLL (`git+https://github.com/rlops/ROLL.git`)
   and `tg4perfetto`
5. `hf download Qwen/Qwen2.5-0.5B + dapo-math-17k + aime-2024`
6. Megatron checkpoint conversion (`tools/convert_hf_to_torch_dist.py`)
7. `cp -r Qwen2.5-0.5B_torch_dist Qwen2.5-0.5B_miles` for actor `--load`

### Iteration history (all on the A40 instance)

#### Attempt 1 — `_split_devices` not on remote
- Cause: my refactor to `_split_pools_for_dual` was uncommitted; remote
  miles HEAD had old function name.
- Fix: commit + push (HEAD 6126e01), git pull on remote.

#### Attempt 2 — `Failed to register worker to Raylet (errno=24)`
- Cause: ulimit -n soft cap = 1024, raylet hits EMFILE under 2 pipelines.
- Fix: `ulimit -n 65536` at top of `scripts/run_smoke_dual.sh`.

#### Attempt 3 — `ModuleNotFoundError: No module named 'roll'`
- Cause: ROLL not installed; miles' `MilesPipeline._build_placement_provider`
  imports `roll.distributed.scheduler.resource_manager.RollResourceManagerProxy`.
- Fix: `pip install 'roll @ git+https://github.com/rlops/ROLL.git'`
  + `pip install tg4perfetto`.

#### Attempt 4 — ✅ PASS
- 03:11:00 mp1 init complete (engines=2, train=[0], infer=[0,1])
- 03:14:49 mp2 init complete (engines=2, train=[2], infer=[2,3])
- 03:14:49 both pipelines pre-loop dispatch rollout_id=0
- 03:15:15 mp1 rollout 0 train+after_step done
- 03:15:16 mp2 rollout 0 train+after_step started (KeyError on _wait_for_overlap_engines_offloaded — non-fatal, returns early)
- 03:15:25 mp1 rollout 0 step4 after_step done
- 03:15:34 mp1 rollout 1 train_group.train done
- 03:15:43 mp1 rollout 1 step4 after_step done; mp1 training loop complete
- 03:15:48 mp2 rollout 1 step4 after_step done; mp2 training loop complete
- 03:15:49 shutdown_hard complete for both pipelines
- **EXIT_CODE=0**

### Non-fatal warning to fix as follow-up
`_wait_for_overlap_engines_offloaded` for MP2 logged
`KeyError('unknown engine_index 2')`. Cause: it converted physical GPU
IDs to engine local indices via `g // per_engine`, which works for M11.1
single-pipeline (pool starts at GPU 0) but fails for M11.2 P2 (pool
starts at GPU 2 → `2 // 1 = 2`, which doesn't exist). The function
returns early on the warning, so the run completes — but Phase 2
nvidia-smi probe is skipped, meaning if MP2 had to wait for engine
offload the train wake_up could OOM. Not exercised in this disjoint-pool
smoke (no cross-pipeline GPU contention).

### Follow-up commit
- rlix `zhenyu/miles-mvp-e2e`: fix `_wait_for_overlap_engines_offloaded`
  to convert physical GPU IDs to local engine indices using infer pool's
  min physical GPU as offset (instead of assuming pool starts at 0).

### Final commit chain
- rlix `zhenyu/miles-mvp-e2e`:
  - d97178e: thread cluster_device_mappings into _build_placement_provider
  - (next): fix engine_index conversion in _wait_for_overlap_engines_offloaded
- miles `zhenyu/m11-mvp-test`:
  - 992fa26: M11.2 dual-pipeline driver + per-pipeline rollout base port
  - 6126e01: dual driver — base args carry per-pipeline shape, satisfies C4

### Key lessons (multi-pipeline-specific)
1. **`ulimit -n 65536`** required before any Ray operation; default 1024 raylet SIGABRT under 2 pipelines + ~20 SGLang sub-processes.
2. **`MILES_ROLLOUT_BASE_PORT` env var per-pipeline** prevents the find-free-port race on `start_engines`.
3. **Per-pipeline `cluster_device_mappings`** must thread through to `_build_placement_provider` so each pipeline's MilesPlacementProvider uses its own physical GPU subset (not `range(rollout_num_gpus)`).
4. **Driver must derive per-pipeline arg shape from base args** (don't override actor_num_gpus_per_node with whole-machine view) — base args = per-pipeline shape, driver maps to physical GPUs via cluster_device_mappings.
5. **Local engine indices, not physical GPU IDs**: any function that talks to RolloutManager.get_engine_states/etc. needs `(physical_gpu - infer_pool_first) // per_engine`, not `physical_gpu // per_engine`.

### Total wall-clock for M11.2
~33 min from SSH-in to EXIT_CODE=0 (vs M11.1's ~3.5 hr); the prior 5 lessons compounded.
