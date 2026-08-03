#!/usr/bin/env python3
"""Minimal memory-offload mock test (~30 s), v2.

Mirrors MegatronTrainRayActor.sleep()/wake_up() mechanics
(miles/backends/megatron_utils/actor.py:212-258):
  - torch_memory_saver.hook_mode set BEFORE first tms call (tms-fixes.md #4)
  - allocate tensors inside a tms region
  - clear_memory equivalent (empty_cache) BEFORE pause, like actor.sleep()
  - pause  -> physical pages unmapped; resume -> remapped
Two variants: plain region (data may be discarded) and
enable_cpu_backup=True region (data must survive, Megatron-style).

Whole-GPU nvidia-smi is the source of truth: on vast containers
--query-compute-apps PIDs live in the HOST pid namespace, so per-process
attribution is expected to fail here (same fail-open path as
miles/utils/gpu_probe.py).

Run: MILES_TMS_HOOK_MODE=torch CUDA_VISIBLE_DEVICES=0 python mock_offload_test.py [alloc_gib]
"""
import os
import subprocess
import sys
import time

ALLOC_GIB = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
PID = os.getpid()


def smi(query, extra):
    return subprocess.check_output(
        ["nvidia-smi", f"--query-{query}={extra}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()


def gpu0_used():
    line = smi("gpu", "index,memory.used").splitlines()[0]
    return int(line.split(",")[1])


def report(label):
    import torch

    torch.cuda.synchronize()
    time.sleep(0.5)
    used = gpu0_used()
    print(
        f"[{label:<22}] whole_gpu0={used:6d} MiB  torch_reserved={int(torch.cuda.memory_reserved() / 2**20):6d} MiB",
        flush=True,
    )
    return used


def pid_namespace_check():
    lines = [l for l in smi("compute-apps", "pid,used_memory").splitlines() if l.strip()]
    pids = [int(l.split(",")[0]) for l in lines]
    print(f"compute-apps pids visible: {pids}  (this pid={PID}, match={PID in pids})", flush=True)


def run_variant(torch_memory_saver, torch, cpu_backup):
    name = "cpu_backup" if cpu_backup else "plain"
    n_elem_half = int(ALLOC_GIB * 2**30 / 2 / 2)  # bf16, 2 tensors
    kwargs = {"enable_cpu_backup": True} if cpu_backup else {}
    with torch_memory_saver.region(tag="default", **kwargs):
        tensors = [torch.empty(n_elem_half, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
        for i, t in enumerate(tensors):
            t.fill_(float(i + 1))
    # checksum WITHOUT big temporaries: sum a small slice per tensor
    before = [float(t[:1024].float().sum()) for t in tensors]
    alloc = report(f"{name}:after_alloc")

    # actor.sleep() equivalent: clear_memory (empty_cache) then pause
    torch.cuda.empty_cache()
    report(f"{name}:after_empty_cache")
    torch_memory_saver.pause(tag=None)
    paused = report(f"{name}:after_pause")

    torch_memory_saver.resume(tag=None)
    resumed = report(f"{name}:after_resume")
    after = [float(t[:1024].float().sum()) for t in tensors]
    data_ok = before == after

    released = alloc - paused
    print(
        f"  -> {name}: released_by_pause={released} MiB "
        f"(expect ~{int(ALLOC_GIB * 1024)}), data={'PRESERVED' if data_ok else 'DISCARDED'}",
        flush=True,
    )
    del tensors
    torch.cuda.empty_cache()
    return released, data_ok


def main():
    t0 = time.time()
    mode = os.environ.get("MILES_TMS_HOOK_MODE", "torch")

    from torch_memory_saver import torch_memory_saver

    torch_memory_saver.hook_mode = mode
    print(f"tms hook_mode={mode!r}  alloc={ALLOC_GIB} GiB  pid={PID}", flush=True)

    import torch

    torch.zeros(1, device="cuda")
    base = report("baseline(cuda ctx)")
    pid_namespace_check()

    rel_plain, ok_plain = run_variant(torch_memory_saver, torch, cpu_backup=False)
    rel_backup, ok_backup = run_variant(torch_memory_saver, torch, cpu_backup=True)

    final = report("final(after cleanup)")
    print("-" * 72, flush=True)
    print(f"cuda context baseline           : {base} MiB", flush=True)
    print(f"irreducible residual at end     : {final} MiB (context + allocator metadata)", flush=True)
    print(f"elapsed: {time.time() - t0:.1f}s", flush=True)

    want = ALLOC_GIB * 1024 * 0.9
    ok = rel_plain > want and rel_backup > want and ok_backup
    print(f"OFFLOAD_TEST_{'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
