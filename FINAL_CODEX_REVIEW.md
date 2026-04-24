# Final Codex Review — F4 &amp; F6

## (1) Doc Accuracy
**Verdict**: PARTIAL
**Issue**: `IMPLEMENTATION.md` overstates the receiver path for F4/F6 by claiming inline tensor reconstruction was eliminated, but `vllm_backend.update_parameter_in_bucket()` still reconstructs tensors inline in the `cuda_ipc` branch.

## (2) F4 Implementation Completeness
**Verdict**: PARTIAL
**Issue**: The CPU bucket cache is implemented, but `_cache_ready_step` publication still happens later in `BucketCacheLifecycle.mark_promoted()` under a separate pipeline lock instead of under the sender `_cache_lock` as required by the port plan.

## (3) F6 Implementation Completeness
**Verdict**: PARTIAL
**Issue**: The expand path still syncs shrunk ranks before any explicit wake/load step and then only calls `expand_sampler(skip_load=True)`, so the port plan’s wake → sync → finalize → activate sequence is not fully implemented.
