# torch_memory_saver (tms) fixes for partial-overlap rlix mode

Captured during the M11 smoke that took the rlix-controlled Qwen2.5-0.5B GRPO loop
to a green `--num-rollout 2` run on a 4xRTX5090 vast.ai instance (2026-05-06).

These five issues all interact with `torch_memory_saver` (the library SGLang and
Megatron use to swap GPU memory in/out so train and infer can share the same
physical GPUs). Each section gives the symptom, the root cause, and the actual
patch with file/line references.

---

## 1. SGLang launched without `enable_memory_saver=True` → release is a no-op

### Symptom
`SGLangEngine.release_memory_occupation` returns 200 OK and the rollout-manager
state machine flips to `offloaded`, but `nvidia-smi` still reports the same
process holding ~30 GB on the overlap GPUs. The next train wake_up then OOMs:

```
[Rank 0] free_GB=1.97, used_GB=29.43
[torch_memory_saver.cpp] CUresult error: 2 (out of memory)
  file=csrc/utils.h func=cu_mem_create line=194
```

### Root cause
SGLang's `release_memory_occupation` only frees memory that was allocated
inside a torch_memory_saver region. SGLang creates those regions only when
the `ServerArgs.enable_memory_saver=True` flag is set at engine start.

Two miles-side gates control that flag:

1. `miles/backends/sglang_utils/sglang_engine.py:1057`
   ```python
   "enable_memory_saver": args.offload_rollout,
   ```
   So `--offload-rollout` must be passed.
2. `miles/ray/rollout.py:1586` (pre-fix)
   ```python
   needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
   if args.offload_rollout and not needs_offload:
       overrides.setdefault("enable_memory_saver", False)
   ```
   miles' static `rollout_pg_offset` math places engines after the train pool
   (`group_abs_start = 2`, `megatron_num_gpus = 2`, `2 < 2 → False`), so even
   with `--offload-rollout` set, miles overrides `enable_memory_saver=False`
   for what it thinks is a disjoint pool. This is correct for miles' colocate
   model but wrong for rlix's `cluster_device_mappings = {actor_train: [0,1],
   actor_infer: [0,1,2,3]}` partial-overlap topology, where the rlix scheduler
   can preempt those same engines.

### Fix
1. Add `--offload-rollout` to the smoke command in
   `scripts/run_smoke_e2e.sh` (next to the existing `--offload-train`).
2. Patch `miles/ray/rollout.py:1586` to force `needs_offload=True` under rlix:
   ```python
   needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
   if args.offload_rollout and os.environ.get("RLIX_CONTROL_PLANE") == "rlix":
       needs_offload = True
   ```
   Verification: log shows `enable_memory_saver=True` in `ServerArgs` for
   every engine, and the next nvidia-smi probe reports `min=28.20 GB` free
   immediately after `state="offloaded"`.

---

## 2. SGLang scheduler not quiescent → `release_memory_occupation` races Triton kernels (or `flush_cache` hangs)

### Symptom A — Triton race (no synchronization)
```
write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
[2026-05-06 11:10:42] SIGQUIT received. signum=None, frame=None.
```

### Symptom B — flush_cache hang (with a naive sleep)
```
TimeoutError: Timeout while flushing cache.
```
After a 60-second loop in `SGLangEngine.flush_cache` waiting for `/flush_cache`
to return 200.

### Root cause
`shrink_engines` and `MilesModelUpdateService.sync_selected_workers`
both follow this sequence:
1. abort in-flight requests (`/abort_request`)
2. wait until `is_idle` → True (request queue drained)
3. call `release_memory_occupation` (in shrink) or `finalize_weight_update`
   (in sync), which internally calls `flush_cache`

The problem is that `is_idle=True` only means the request queue is empty. The
SGLang scheduler thread can still have an in-flight Triton kernel launched
for the last decode iteration (e.g. `write_req_to_token_pool_triton` against
the persistent token pool buffer). `release_memory_occupation` then moves
those persistent buffers to CPU mid-iteration → the kernel sees a CPU
tensor and crashes. With the fully-async rollout, even the data-return path
does not synchronously quiesce the engine — `flush_cache` then loops 60 s
because `/flush_cache` keeps returning non-200.

### Fix
Both call sites: wrap `release_memory_occupation` / `finalize_weight_update`
with `pause_generation(mode="retract")` (and, in the weight-sync path, a
matching `continue_generation` after).

`pause_generation` is the proper SGLang API: it retracts in-flight decode
iterations and blocks until the scheduler reaches a safe checkpoint. Once
paused, both `flush_cache` and the actual memory release run against a
quiescent engine.

#### Shrink path — `miles/ray/rollout.py:911`
```python
# Step 3.5 (rlix-mode safety): pause the SGLang scheduler with
# mode="retract" before release_memory_occupation, so any in-flight
# Triton kernel finishes against GPU-resident persistent buffers
# before release moves them to CPU.
if os.environ.get("RLIX_CONTROL_PLANE") == "rlix":
    try:
        ray.get([h.pause_generation.remote(mode="retract") for h in handles])
    except Exception as exc:
        logger.warning("shrink_engines: pause_generation pre-release failed: %r", exc)
# Step 4: release memory.
ray.get([h.release_memory_occupation.remote(tags=None) for h in handles])
```

#### Weight-sync path — `rlix/pipeline/miles_model_update_service.py:288`
```python
# (4.pre, rlix-mode safety) pause each engine's scheduler before
# finalize_weight_update.
try:
    pause_refs = [h.pause_generation.remote(mode="retract") for h in handles.values()]
    await _ray_get(pause_refs)
except Exception as exc:
    logger.warning("[MilesModelUpdateService] pause_generation pre-finalize failed: %r", exc)
finalize_refs = [h.finalize_weight_update.remote() for h in handles.values()]
await _ray_get(finalize_refs)
# (4.post) resume the engines so subsequent generate calls work.
try:
    cont_refs = [h.continue_generation.remote() for h in handles.values()]
    await _ray_get(cont_refs)
except Exception as exc:
    logger.warning("[MilesModelUpdateService] continue_generation post-finalize failed: %r", exc)
```

Verification: log shows
```
SGLangEngine ... POST /pause_generation HTTP/1.1 200 OK
SGLangEngine ... POST /release_memory_occupation HTTP/1.1 200 OK
SGLangEngine ... GET /flush_cache HTTP/1.1 200 OK
```
sequenced cleanly, and `_wait_for_overlap_engines_offloaded` returns
within 0.5 s of the state flip.

> **Note**: a sleep-only workaround (a 0.5 s grace between drain and release)
> moved the failure mode but did not fix it — `pause_generation` is the
> correct API and what landed in the final tree.

---

## 3. Double `tms.resume` → `Cannot resume allocation that is not paused`

### Symptom
After a successful first wake_up, a second one fires immediately and crashes:
```
[2026-05-06 11:24:26] timer.py:24 - Timer wake_up start
[2026-05-06 11:24:26] memory_utils after wake_up: free_GB=24.39, used_GB=6.97
[2026-05-06 11:24:26] timer.py:32 - Timer wake_up end (elapsed: 0.9s)
[2026-05-06 11:24:26] timer.py:24 - Timer wake_up start          ← second!
[torch_memory_saver.cpp] Cannot resume allocation that is not paused.
  tag=default ptr=13287555072 file=csrc/core.cpp func=resume line=155
```
Megatron actor exits SYSTEM_ERROR; `train_group.train()` raises ActorDiedError.

### Root cause
Two unconditional wake_ups in the same step:

1. `MilesPipeline._before_training` called `self._run_async(self._train_group.onload())`,
   which broadcasts `wake_up` to every Megatron actor.
2. `MegatronTrainRayActor.train()` (`miles/backends/megatron_utils/actor.py:357`)
   internally calls `self.wake_up()` when `args.offload_train=True`.

Standalone `train_async.train` (the reference) only triggers the second
path — there is no explicit onload before `actor_model.train(...)`. The
rlix-mode code added the redundant call defensively; tms then sees the
region as already resumed when the second resume arrives.

### Fix
Drop the explicit `train_group.onload()` from `MilesPipeline._before_training`
and let `train()` do its single wake_up.

`rlix/pipeline/miles_pipeline.py:558` (post-fix):
```python
# Do NOT call train_group.onload() here: when args.offload_train is
# set, MegatronTrainRayActor.train() already wakes up internally
# (see miles/backends/megatron_utils/actor.py:357). Adding an onload
# here causes a double wake_up — the second call hits
# `[torch_memory_saver.cpp] Cannot resume allocation that is not
# paused` because tms regions are already resumed. Leave the actor
# offloaded; rlix_train_loop will dispatch train() which wakes up
# safely.
```

Verification: log shows exactly one `Timer wake_up end` per train step
(0.6 s–0.9 s elapsed) and `train_group.train done` follows cleanly.

---

## 4. tms hook mode segfault on CUDA 12.9 / Blackwell

### Symptom
With tms 0.0.9 default hook mode (`preload` — LD_PRELOAD libc malloc), the
process segfaults during `build_cpu_bucket_cache` (Phase A step 4) on CUDA
12.9 / Blackwell. No clean traceback; the worker crashes during
`named_params_and_buffers` while reading GPU weights.

### Root cause
`hook_mode="preload"` intercepts every libc `malloc`/`free` in the process. On
Blackwell + CUDA 12.9 with the new mempool internals, that interception
sometimes hits an allocation path inside the CUDA runtime / driver that the
preload hook cannot safely wrap, causing a segfault on the next free.

`hook_mode="torch"` switches tms to PyTorch's `CUDAPluggableAllocator`
path, which only wraps torch-owned allocations — narrower catchment but
safe on the new toolchain.

### Fix
Read `MILES_TMS_HOOK_MODE` from the environment **before** the first tms
call (`build_cpu_bucket_cache` triggers `_ensure_initialized`).

`miles/backends/megatron_utils/actor.py:96-111`:
```python
if args.offload_train:
    # MILES_TMS_HOOK_MODE=torch switches torch_memory_saver into
    # PyTorch's CUDAPluggableAllocator path, avoiding the LD_PRELOAD
    # libc malloc hook that segfaults during build_cpu_bucket_cache on
    # CUDA 12.9 / Blackwell. Must be set BEFORE any tms call that
    # triggers _ensure_initialized.
    import os as _os
    mode = _os.environ.get("MILES_TMS_HOOK_MODE")
    if mode in ("torch", "preload"):
        logger.info(f"Set torch_memory_saver.hook_mode to {mode!r}")
        torch_memory_saver.hook_mode = mode  # type: ignore[assignment]
```

`scripts/run_smoke_e2e.sh` exports `MILES_TMS_HOOK_MODE=torch` so the actor
picks it up via Ray `runtime_env` (forwarded from
`examples/rlix/run_miles_rlix.py`'s `pipeline_runtime_env_vars`).

Companion escape hatch: `MILES_SKIP_TMS_PAUSE=1` (also wired in
`actor.py:212-253`) turns sleep/wake_up into a near no-op for cases where
the model fits without aggressive offload — useful for bringing up new
hardware before chasing tms bugs.

---

## 5. tms `_with_region_config` clobbers nested regions (vast image patch)

### Symptom
During Megatron DDP init, `torch_memory_saver` is entered nested (outer
training region + inner allgather region). The inner region's
`_with_region_config` overwrote the outer's still-active config, leaving the
allocator with the wrong tag once the inner exited. The next `pause(tag=...)`
then either silently no-op'd or raised "tag not found".

### Root cause
`torch_memory_saver/entrypoint.py::_with_region_config` unconditionally
swapped its module-global region config on entry and restored on exit. When
nesting, the restore at inner-exit reverted to the **outer's pre-entry**
config (which was already swapped out by the outer's own entry), losing the
outer's active config for the rest of its scope.

### Fix (vast image only — not in repo, applied at runtime)
Patch the installed `torch_memory_saver/entrypoint.py` so `_with_region_config`
becomes a no-op when the saver is already inside an interesting region:

```python
# /usr/local/lib/python3.12/dist-packages/torch_memory_saver/entrypoint.py
@contextmanager
def _with_region_config(...):
    if torch_memory_saver._inside_interesting_region:
        # already inside outer region; skip nested config swap
        yield
        return
    # original swap/restore logic
    ...
```

This is a vast-image-only hotpatch. Upstream fix tracked in
`https://github.com/fzyzcjy/torch_memory_saver/issues` (file as M11.5
follow-up). The repo pinning
`pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b`
in `scripts/run_smoke_e2e.sh` ensures the same source build that the patch
was authored against.

---

## Verifying tms is healthy mid-run

Useful log lines to grep when debugging a fresh image:

| Line | What it tells you |
|------|------------------|
| `Set torch_memory_saver.hook_mode to 'torch'` | Hook mode applied before init (fix #4 active) |
| `enable_memory_saver=True` in SGLang `ServerArgs` | Engines will track allocations (fix #1 active) |
| `Engine group 'regular' ... needs_offload=True` | rlix-mode override fired (fix #1 active) |
| `POST /pause_generation HTTP/1.1 200 OK` before `/release_memory_occupation` | Quiescence before release (fix #2 active) |
| `_wait_for_overlap_engines_offloaded: OS-level GPU mem free min=28+ GB` | Memory actually returned to OS (fix #1 effective) |
| Exactly one `Timer wake_up end` per train step | No double wake_up (fix #3 active) |

Failure signatures to watch for:

| Line | Likely root cause |
|------|------------------|
| `Pointer argument cannot be accessed from Triton (cpu tensor?)` | Missing `pause_generation` before release (fix #2 regressed) |
| `Timeout while flushing cache` | Same as above; `/flush_cache` looping 60 s |
| `Cannot resume allocation that is not paused` | Double wake_up (fix #3 regressed) |
| `CUresult error: 2 (out of memory) in cu_mem_create` after `state="offloaded"` | `enable_memory_saver=False` somewhere (fix #1 regressed) |
| Segfault during `build_cpu_bucket_cache` with no Python traceback | `MILES_TMS_HOOK_MODE` not set or set after first tms call (fix #4 regressed) |
