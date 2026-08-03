#!/usr/bin/env python3
"""SGLang engine offload micro-test (~2-3 min).

Launches one SGLang engine with enable_memory_saver=True (same flag
miles passes when --offload-rollout, see tms-fixes.md #1), then:
  release_memory_occupation -> measure whole-GPU residual
  resume_memory_occupation  -> measure recovery
Run: CUDA_VISIBLE_DEVICES=0 python mock_sglang_offload_test.py
"""
import subprocess
import sys
import time


def gpu0_used():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0].split(",")[1])


def report(label):
    time.sleep(1.0)
    used = gpu0_used()
    print(f"[{label:<18}] whole_gpu0={used:6d} MiB", flush=True)
    return used


def main():
    t0 = time.time()
    base = report("baseline")

    import sglang as sgl

    engine = sgl.Engine(
        model_path="/root/Qwen2.5-0.5B",
        mem_fraction_static=0.30,
        enable_memory_saver=True,
        disable_cuda_graph=False,
        skip_server_warmup=True,
    )
    loaded = report("engine_loaded")

    out = engine.generate("1+1=", {"max_new_tokens": 4, "temperature": 0})
    print(f"sanity generate: {out['text']!r}", flush=True)
    after_gen = report("after_generate")

    engine.release_memory_occupation()
    released = report("after_release")

    engine.resume_memory_occupation()
    resumed = report("after_resume")

    engine.shutdown()
    final = report("after_shutdown")

    print("-" * 60, flush=True)
    print(f"engine footprint       : {after_gen - base} MiB", flush=True)
    print(f"freed by release       : {after_gen - released} MiB", flush=True)
    print(f"residual after release : {released} MiB (over baseline {released - base} MiB)", flush=True)
    print(f"recovered by resume    : {resumed - released} MiB", flush=True)
    print(f"elapsed: {time.time() - t0:.1f}s", flush=True)

    ok = (after_gen - released) > (after_gen - base) * 0.5 and released - base < 3000
    print(f"SGLANG_OFFLOAD_TEST_{'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
