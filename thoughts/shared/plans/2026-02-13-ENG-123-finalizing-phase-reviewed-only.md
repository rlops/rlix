# ENG-123 Finalizing Phase: Reviewed Angles Only

**Date:** 2026-02-13  
**Scope:** Only angles that were actually reviewed and validated with code evidence.

---

## Additional Validated Issues (From Parallel Review)

### Issue A: SharedStorage Cleanup Contract Breaks (ROLL_multi_pipeline branch)
**Status:** 🟢 **ADDRESSED IN CURRENT REPO - LOW RISK**

**Assessment:** This issue primarily affects the separate `ROLL_multi_pipeline` branch, not the current integrated codebase.

| Aspect | Current Repo Status |
|--------|---------------------|
| `delete_port_claims` method | ✅ **EXISTS** in `roll/distributed/scheduler/storage.py` |
| `delete_prefix` method | ✅ **EXISTS** in `roll/distributed/scheduler/storage.py` |
| Port claim value | ✅ Stores `pipeline_id` when in SchedRL mode |
| Rendezvous key | ✅ Uses `f"{pipeline_id}:{cluster_name}"` format |

**Risk Remaining:**
- Non-SchedRL mode (without `PIPELINE_ID`) stores `True` as value - but we fail-fast at import when `SCHEDRL_CONTROL_PLANE=schedrl` without `PIPELINE_ID`
- Conditional risk only for non-SchedRL deployments

**Verdict:** Downgraded from Critical to **Low/Documentation**.

---

### Issue B: Admission Gating Missing in Scheduler
**Status:** 🔴 **CONFIRMED - REQUIRES FIX**

**The Problem:**
The SchedRL scheduler's shrink/expand execution methods don't call the admission control methods that exist in the adapter.

| Method | Current Behavior | Required Behavior |
|--------|-----------------|-------------------|
| `_execute_shrink_calls` | Only calls `shrink_workers` | Should call `close_admission` first |
| `_execute_expand_calls` | Only calls `expand_workers` | Should call `open_admission` after |

**Evidence:**
- `schedrl/scheduler/scheduler.py:840` - only calls `shrink_workers`
- `schedrl/scheduler/scheduler.py:876` - only calls `expand_workers`
- `roll/schedrl_adapter/adapter.py:272` - directly calls `RequestScheduler.shrink_workers`, no suspend

**Impact:**
- During shrink: New requests can still be routed to workers being shut down
- During expand: Newly added workers may not be ready to accept requests

**Fix Required:**
1. Add `close_admission` call before `shrink_workers` in `_execute_shrink_calls`
2. Add `open_admission` call after `expand_workers` in `_execute_expand_calls`

---

### Issue C: VLLM Offload Gated by Colocation (Issue 86)
**Status:** 🟡 **NEEDS TARGETED CONFIRMATION**

**The Concern:**
GPU memory may not be freed during shrink because offload is gated by `is_actor_infer_colocated`.

**Current Assessment:**
- Many pipelines intentionally set `batch.meta_info["is_offload_states"] = False` during compute phases
- Haven't found the exact site where `is_actor_infer_colocated` controls `offload_states` skip
- May be conflating intentional compute-phase offload disabling with colocation gating

**To Confirm This Issue:**
1. Find exact code path: `shrink_workers` → `offload_states` → colocation gate
2. Reproduce: non-colocated actor_infer + scheduler shrink/release
3. Confirm weights remain resident when they should be freed

**Verdict:** Downgraded from Critical to **"Needs Reproduction"**.

---

### Issue D: Hidden Globals Not Pipeline-Scoped
**Status:** 🟡 **PARTIALLY ADDRESSED - ACCEPTABLE RISK**

| Global | Status | Risk Level |
|--------|--------|------------|
| `_global_limiters` | ✅ **FIXED** - Now keyed by `f"{pipeline_id}:{tag}"` | None |
| `GlobalCounter` actor | ✅ **FIXED** - Uses `f"{pipeline_id}_DynamicSchedulerRequestCounter"` | None |
| `PROCESS_GROUP_MANAGER` | ⚠️ Still module-level singleton | Low (Ray process isolation) |
| `os.environ` mutations (driver) | ⚠️ Plausible but needs evidence | Unknown |

**Evidence of Fixes:**
- `env_action_limiter.py:122` - `_global_limiters` now uses `key = f"{pipeline_id}:{tag}"`
- `async_generate_scheduler.py:409` - `GlobalCounter` name includes `pipeline_id`

**Remaining Risk:**
- `PROCESS_GROUP_MANAGER` in `globals.py:21` is still a pure singleton
- **Mitigation:** Ray actors run in separate processes, so this only affects CP usage within same process

**Verdict:** Downgraded from Critical to **Low** - acceptable given Ray's process model.

---

### Issue E: Logging Infrastructure Missing
**Status:** 🟡 **QUALITY ISSUE - NOT CORRECTNESS**

| Gap | Status |
|-----|--------|
| Fail-fast structured context | ❌ NOT implemented |
| Critical decision logging | ❌ NOT implemented |
| Structured logging library | ❌ NONE exists |
| Current method | `sys.stderr.write()` only |

**Assessment:**
- This is a debugging/observability quality issue
- Not a correctness blocker - fail-fast exceptions still occur
- Would help with multi-pipeline debugging but isn't required for functionality

**Verdict:** Keep as **Medium/Low priority** - nice to have, not required.

---

### Issue F: Driver-Level Config Bleed
**Status:** 🟡 **NEEDS CONCRETE EVIDENCE**

**The Concern:**
Driver-level `os.environ` mutations could bleed across pipelines.

**Current Assessment:**
- The adapter correctly uses `runtime_env` for worker isolation
- Most `os.environ` mutations appear to be in workers (isolated) or use `system_envs`
- Need specific evidence of driver-side mutations that affect multiple pipelines

**To Confirm This Issue:**
1. Find specific `os.environ[...] = ...` calls in driver code paths
2. Demonstrate that these mutations affect multiple pipeline configurations
3. Show that the adapter's `runtime_env` isolation doesn't prevent the bleed

**Verdict:** Downgraded to **"Needs Evidence"**.

---

## Summary of Issue Status Changes

| Issue | Original Severity | Updated Assessment | Action Required |
|-------|-------------------|-------------------|-----------------|
| SharedStorage cleanup | 🔴 Critical | 🟢 Addressed | Document non-SchedRL risk |
| Admission gating | 🔴 Critical | 🔴 **Confirmed** | **Fix in scheduler** |
| VLLM offload gating | 🔴 Critical | 🟡 Needs Reproduction | Find exact gating site |
| Hidden globals | 🔴 Critical | 🟢 Mostly Fixed | Only PROCESS_GROUP_MANAGER remains |
| Logging infra | 🟡 Medium | 🟡 Quality issue | Nice to have |
| Config bleed | 🟡 Medium | 🟡 Needs Evidence | Find concrete mutation sites |

**The one real confirmed gap to fix is (B) Admission gating in the scheduler shrink/expand flow.**

---

## Angles Reviewed with Code Evidence

### 1. Multi-Pipeline Isolation (Ray Namespace + Naming)

#### 1.1 Actor Namespace Coverage
**Tested:** Start 2+ pipelines; enumerate actors; assert per-pipeline actors in correct namespace.

**Negative Test:** Unset `ROLL_RAY_NAMESPACE` / `PIPELINE_ID`; confirm fail-fast.

**Status:** ⚠️ CONDITIONAL
- With per-pipeline `ROLL_RAY_NAMESPACE`, actors are isolated
- **Risk:** Env bootstrap failure → namespace collision

**Bugs Found:**
- **GlobalLimiter** (`roll/utils/env_action_limiter.py:75-80`): Actor name `f"GlobalLimiter_{tag}"` lacks `pipeline_id` (relies on per-pipeline namespace isolation).
- **model_update_locker** (`megatron/model_update.py:362`): Fixed name `"model_update_locker"` (relies on per-pipeline namespace isolation).

Both rely solely on namespace isolation; defense-in-depth would add `pipeline_id` to names.

---

#### 1.2 Name Uniqueness
**Tested:** Confirm names include `pipeline_id`; confirm `get_if_exists=True` cannot cross pipelines.

**Actor Audit:**
| Actor | Name Format | Status |
|-------|-------------|--------|
| RolloutScheduler | `"RolloutScheduler-train"` / `"RolloutScheduler-val"` (namespaced) | ✅ |
| RequestScheduler | `f"{pipeline_id}_request_scheduler_{mode}"` (namespaced) | ✅ |
| GroupQueueManager | `f"{pipeline_id}_group_queue_manager_{mode}"` (namespaced) | ✅ |
| RewardScheduler | `f"RewardScheduler-{reward.name}"` (namespaced) | ✅ |
| GlobalDatasetManager | `"val_dataset_manager"` (namespaced) | ✅ |
| GlobalLimiter | `f"GlobalLimiter_{tag}"` | ⚠️ RISK |
| GlobalCounter | `f"{pipeline_id}_..."` | ✅ |
| model_update_locker | `"model_update_locker"` | ⚠️ RISK |

---

#### 1.3 "Job-Global by Design" Audit
**Tested:** Verify only SharedStorage and orchestrator/scheduler are job-global.

**Status:** ✅ CORRECT
- SharedStorage: `global_storage_namespace`
- Orchestrator/Scheduler: `schedrl` namespace
- All else: per-pipeline namespaces

---

#### 1.4 Import-Time Env Hazards
**Tested:** Verify `ROLL_RAY_NAMESPACE` set before `roll.*` imports.

**Status:** ⚠️ NEEDS VALIDATION
- `RAY_NAMESPACE = os.environ.get("ROLL_RAY_NAMESPACE", "roll")` binds at import
- Bootstrap invariant: assert env var before import

---

### 2. SharedStorage / Rendezvous / Port-Lock Correctness

#### 2.1 Rendezvous Key Scoping
**Tested:** Two pipelines, same `cluster_name`; assert no key overwrite.

**Status:** ✅ CORRECT
```python
rendezvous_key = f"{self.pipeline_id}:{self.cluster_name}"
```

---

#### 2.2 Port-Lock Lifecycle
**Tested:** Create/shrink/expand/kill cycles; assert no key accumulation.

**Status:** ✅ CORRECT (validated)
- Key: `MASTER_ADDR_PORT:{master_addr}:{master_port}`
- Cleanup uses value-based filtering, not prefix
- `pipeline_id` stored as value; matched during cleanup

---

#### 2.3 Namespace Split
**Tested:** SharedStorage in `GLOBAL_STORAGE_NAMESPACE`; no shadow instances.

**Status:** ✅ CORRECT
- Single SharedStorage actor per job
- All pipelines share same instance

---

### 3. Orchestrator ↔ Scheduler ↔ Adapter Init Sequencing

#### 3.1 Orchestrator Registration Flow
**Tested:** Register → get pipeline_id → inject runtime env → start adapter.

**Status:** ✅ CORRECT (contract-based)
- Adapter docs specify: register + admit before creating adapter
- If contract violated → fail-fast

---

#### 3.2 ResourceManager Topology Init
**Tested:** `SCHEDRL_REQUIRED_GPUS_PER_NODE` handling.

**Bugs Found:**

##### Bug 3.2.1: Broad `except Exception` (FIXED ✅)
**Location:** `scheduler.py:233-236`

**OLD:**
```python
except Exception:  # Catches ALL exceptions
    required_gpus_per_node = int(await self._resource_manager.init_topology.remote())
```

**NEW (FIXED):**
```python
except RuntimeError as e:
    if "init_topology" in msg or "not initialized" in msg:
        raise RuntimeError("ResourceManager topology not initialized...") from e
    raise  # Re-raise other errors
```

---

##### Bug 3.2.2: ResourceManager Re-Init (FIXED ✅)
**Location:** `resource_manager.py:20-23`

**NEW (FIXED):**
```python
def init_topology(self, *, required_gpus_per_node: int | None = None) -> int:
    if self.required_gpus_per_node is not None:
        raise RuntimeError("ResourceManager topology already initialized")
```

---

### 4. Topology Validation (Registration-Time) + Canonicalization

#### 4.1 Canonicalization Rules
**Tested:** GPU IDs unsorted/random order → canonical ordered mapping.

**Status:** ✅ CORRECT
```python
canonical = sorted(int(x) for x in device_mapping)
```

---

#### 4.2 GPU ID Sanity
**Tested:** Out-of-range, non-int, duplicates, empty mapping.

**Status:** ✅ CORRECT
- Type: `isinstance(gpu, int)`
- Range: `gpu < 0 or gpu >= num_gpus`
- Duplicates: `len(device_mapping) != len(set(device_mapping))`
- Empty: Only allowed for "reward" cluster

---

#### 4.3 TP Group Formation
**Tested:** `len(device_mapping) % tp_size != 0` → fail-fast; contiguous groups.

**Status:** ✅ CORRECT
```python
if len(canonical) % tp_size != 0:
    raise ValueError(...)
# Contiguity check:
expected = list(range(group[0], group[0] + tp_size))
if group != expected:
    raise ValueError("Non-contiguous TP group")
```

---

#### 4.4 Node Boundary Constraints
**Tested:** TP group within `required_gpus_per_node` boundaries.

**Status:** ✅ CORRECT
```python
if tp_size <= required_gpus_per_node:
    start_node = group[0] // required_gpus_per_node
    end_node = group[-1] // required_gpus_per_node
    if start_node != end_node:
        raise ValueError("TP group crosses node boundary")
```

---

#### 4.5 Actor-Infer Overlap Exceptions
**Tested:** Overlap allowed where intended; validation doesn't reject legitimate overlaps.

**Status:** ✅ CORRECT
```python
# Allows actor_infer to overlap with any cluster
# Only rejects non-actor_infer cluster overlaps
if cluster_name != "actor_infer":
    overlap = used_non_infer & used
    if overlap:
        raise ValueError(...)
```

---

### 5. Scheduler Planning Invariants (Execution-Plan Validation)

#### 5.1 Execution Plan Validator Coverage
**Tested:** Violate each invariant; ensure validation rejects.

**Status:** ✅ 11 CONDITIONS IMPLEMENTED
1. Operation uniqueness
2. State transitions
3. Mutual exclusivity
4. GPU state consistency
5. DP-rank activity checks
6. Device mapping boundaries
7. Capacity limits
8. Conservation
9. No double-free
10. DP-rank overlap
11. Completion consistency

---

#### 5.2 Shrink-to-Zero Contract
**Tested:** Shrinking to `active_dp_ranks == ∅` allowed and treated as suspended.

**Status:** ✅ FIXED

**Fix:** allow `min_bundles=0` when `target_gpu_count <= 0` so shrink-to-zero is reachable.
```python
if state.target_gpu_count <= 0:
    min_bundles = 0
else:
    min_bundles = max(1, state.target_gpu_count // state.tp_size)
```

---

#### 5.3 Fairness/Gap-Ratio Logic
**Tested:** Many pipelines, different progress; verify no starvation.

**Status:** ⚠️ CONDITIONAL

##### Bug 5.3.1: Zero Target Weight Edge Case (P1)
**Location:** `scheduler.py:1013`

**Code:**
```python
if not _receiver_eligible(p) or total_target_weight == 0:
    p.target_ratio = 0.0
    p.target_gpu_count = 0
```

**Problem:** When all pipelines have `remaining == 0`, everyone gets `target_gpu_count = 0`. GPUs may sit idle.

**Conditional:** Only a bug if pending requests exist but don't contribute weight.

---

### 6. Shrink/Expand Correctness (Locks, Ordering, No Races)

#### 6.1 Atomic Shrink Ordering (6-Step)
**Tested:** Inject in-flight requests during shrink; assert drain/abort behavior.

**Status:** ⚠️ PARTIAL

**Note:** Failure-atomicity for shrink/expand is not treated as a requirement (assume adapter RPCs succeed; fail-fast otherwise).

---

#### 6.2 Expand Ordering Invariant
**Tested:** "Receiver loaded before apply" ordering.

**Status:** ✅ FIXED
- `_execute_expand_calls()` added
- Calls `adapter.expand_workers.remote()`

---

#### 6.3 Concurrent Shrink+Expand
**Tested:** Overlapping operations; serialize or fail-fast.

**Status:** ✅ CORRECT
- Serialized by `self._lock`
- RPCs executed outside lock

---

### 7. Request Identity / Retry Semantics

#### 7.1 Attempt Increment on ABORT
**Tested:** Force ABORT; verify `attempt` increments, request_id changes.

**Status:** ✅ CORRECT
```python
elif stop_reason == GenerateStopReason.ABORT:
    if os.environ.get("SCHEDRL_CONTROL_PLANE", "") == "schedrl":
        self.rollout_cache.attempt += 1
```

---

#### 7.2 Abort is Only Allowed Retry Trigger
**Tested:** Non-abort failures don't silently spin.

**Status:** ✅ CORRECT
- Retry only on `ABORT`
- Other failures fail-fast

---

#### 7.3 Cross-Pipeline request_id Uniqueness
**Tested:** Same traj/turn across pipelines; unique due to pipeline_id prefix.

**Status:** ✅ CORRECT
```python
return f"{pipeline_id}:{traj_id}:{turn_id}:{attempt}"
```

---

### 8. Progress Reporting & Replanning Triggers

#### 8.1 2% Banding Correctness
**Tested:** Only 2% boundary crossings reach scheduler.

**Status:** ⚠️ EMITTER RESPONSIBILITY
- Scheduler stores every report
- Banding should happen in ROLL emitter

---

#### 8.2 Completion Semantics
**Tested:** Final update at >=100% triggers transitions.

**Status:** ⚠️ CONTRACT ISSUE
- If completion via explicit RPC, OK
- If via progress=1.0, needs handling

---

#### 8.3 Multi-Pipeline Load
**Tested:** Many pipelines spamming progress; scheduler responsive.

**Status:** ⚠️ PERF ISSUE
- `scheduling_cycle()` holds `_lock` for extended periods
- Progress reports may queue up

---

### 9. Selective Model Update (Phase 4)

**Status:** ⚠️ NOT IMPLEMENTED IN SCHEDRL
- Group name scoping: Not in SchedRL state
- `finally:` teardown: Not implemented
- `cache_lock`: Not implemented
- Payload validation: Not implemented
- ModelUpdateService: Not implemented

These are ROLL-side concerns, not SchedRL scheduler bugs.

---

### 10. Kill / Cleanup / Lifecycle Robustness

#### 10.1 kill_pipeline
**Tested:** All per-pipeline actors terminated; PGs destroyed; storage cleaned.

**Status:** ✅ CORRECT
- Named actors: killed by name
- Unnamed actors: force-killed after timeout
- Placement groups: removed by prefix
- Storage: `delete_port_claims` + `delete_prefix`

---

#### 10.2 LogMonitorListener atexit
**Tested:** Library mode does not `ray.shutdown()` from worker-side.

**Status:** ✅ FIXED

**Fix Applied:**
```python
def _schedrl_disable_ray_cluster_lifecycle() -> bool:
    # ENG-123: do not let per-pipeline workers stop the job-global Ray cluster.
    # Use SCHEDRL_CONTROL_PLANE as the source-of-truth (SCHEDRL_LIBRARY_MODE may be false in future service mode).
    if os.environ.get("SCHEDRL_CONTROL_PLANE", "") == "schedrl":
        return True
    return os.environ.get("SCHEDRL_LIBRARY_MODE", "0") == "1"

class LogMonitorListener:
    def stop(self):
        if _schedrl_disable_ray_cluster_lifecycle():
            # Safe path - only close file handlers
            StdPublisher.close_file_handlers()
            return
        # DANGER: Only reached if NOT in SchedRL mode
        ray.shutdown()
        subprocess.run("ray stop --force", shell=True)

    def start(self):
        if _schedrl_disable_ray_cluster_lifecycle():
            return  # Safe - no atexit registration
        atexit.register(self.stop)  # Only for non-SchedRL mode
```

**Key Changes:**
1. Uses `SCHEDRL_CONTROL_PLANE` as source-of-truth (as requested)
2. Falls back to `SCHEDRL_LIBRARY_MODE` for backwards compatibility
3. When `SCHEDRL_CONTROL_PLANE=schedrl`, cluster lifecycle is disabled

**Adapter Sets Both:**
```python
# roll/schedrl_adapter/adapter.py:31-33
"SCHEDRL_CONTROL_PLANE": "schedrl",
"SCHEDRL_LIBRARY_MODE": "1",  # Backwards-compatible override
```

---

### Other Uses of SCHEDRL_LIBRARY_MODE vs SCHEDRL_CONTROL_PLANE

| File | Uses | Status |
|------|------|--------|
| `log_monitor.py` | Both, prefers `SCHEDRL_CONTROL_PLANE` | ✅ Fixed |
| `initialize.py` | Both, prefers `SCHEDRL_CONTROL_PLANE` | ✅ Fixed |
| `adapter.py` | Sets both | ✅ Correct |

**Recommendation:** All checks should follow the pattern:
```python
if os.environ.get("SCHEDRL_CONTROL_PLANE", "") == "schedrl":
    return True  # In SchedRL mode
return os.environ.get("SCHEDRL_LIBRARY_MODE", "0") == "1"  # Fallback
```

---

#### 10.3 ExceptionMonitor Shared Across Pipelines
**Tested:** Verify pipeline-scoped, not job-global.

**Status:** ⚠️ CONDITIONAL
- Fixed name `"ExceptionMonitor"` with `get_if_exists=True`
- Isolated by per-pipeline namespace
- Safe if namespace invariant holds

---

#### 10.4 Crash Mid-Operation
**Tested:** Crash during shrink/expand; verify fail-fast with clear diagnostics.

**Status:** ✅ CORRECT
```python
except Exception as e:
    await self._fail_fast_shutdown(reason=f"scheduler_cycle_failed: {type(e).__name__}: {e}")
    raise
```

---
