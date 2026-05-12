"""
RLix Multi-Pipeline GPU Scheduling Experiment
==============================================
Model:     Qwen/Qwen2.5-0.5B-Instruct
Algorithm: GRPO (agentic, no critic)
Env:       SimpleSokoban (6×6, 1 box)

Runs 4 experiment scenarios and measures wall time and GPU utilization:

  A   single_ft          — 1 full-finetune pipeline
  B   dual_ft            — 2 full-finetune pipelines sharing 4 GPUs
  C   single_lora        — 1 multi-LoRA pipeline (2 adapters, shared base)
  D   ft_plus_lora       — 1 full-finetune + 1 multi-LoRA pipeline concurrently

Usage
-----
  # Run one scenario
  python examples/run_rlix_experiment.py --scenario A
  python examples/run_rlix_experiment.py --scenario B

  # Run all scenarios sequentially with comparison table
  python examples/run_rlix_experiment.py --scenario all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False


# ---------------------------------------------------------------------------
# GPU Monitor
# ---------------------------------------------------------------------------

@dataclass
class GPUStats:
    avg_util: float = 0.0
    peak_mem_mb: int = 0
    per_gpu: Dict[int, Dict] = field(default_factory=dict)


class GPUMonitor:
    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._samples: List[List] = []  # [[util0, util1, ...], ...]
        self._peak_mem: Dict[int, int] = {}

    def start(self) -> None:
        if not NVML_OK:
            return
        self._running = True
        self._samples = []
        self._peak_mem = {}
        n = pynvml.nvmlDeviceGetCount()
        self._peak_mem = {i: 0 for i in range(n)}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        n = pynvml.nvmlDeviceGetCount()
        while self._running:
            row = []
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                m = pynvml.nvmlDeviceGetMemoryInfo(h).used // (1024 * 1024)
                row.append(u)
                if m > self._peak_mem[i]:
                    self._peak_mem[i] = m
            self._samples.append(row)
            time.sleep(self._interval)

    def stop(self) -> GPUStats:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if not self._samples:
            return GPUStats()
        n = len(self._samples[0])
        per_gpu = {}
        total_avg = 0.0
        for i in range(n):
            avg = sum(row[i] for row in self._samples) / len(self._samples)
            per_gpu[i] = {"avg_util": round(avg, 1), "peak_mem_mb": self._peak_mem.get(i, 0)}
            total_avg += avg
        overall_avg = total_avg / n if n else 0.0
        peak = max(self._peak_mem.values()) if self._peak_mem else 0
        return GPUStats(avg_util=round(overall_avg, 1), peak_mem_mb=peak, per_gpu=per_gpu)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS = {
    "A": {
        "label": "Single Full-Finetune",
        "config_names": "full_finetune_pipeline1",
        "description": "1 FT pipeline on GPUs 0-1 (train+ref), GPUs 0-3 (infer)",
    },
    "B": {
        "label": "Dual Full-Finetune",
        "config_names": "full_finetune_pipeline1,full_finetune_pipeline2",
        "description": "2 FT pipelines: P1 train 0-1, P2 train 2-3; infer shared 0-3",
    },
    "C": {
        "label": "Single Multi-LoRA",
        "config_names": "multi_lora_pipeline1",
        "description": "1 multi-LoRA pipeline (2 adapters) on GPUs 0-1/0-3",
    },
    "D": {
        "label": "FT + Multi-LoRA Concurrent",
        "config_names": "full_finetune_pipeline1,multi_lora_pipeline2",
        "description": "FT pipeline (GPUs 0-1) + LoRA pipeline (GPUs 2-3) sharing infer GPUs 0-3",
    },
    "E": {
        "label": "Qwen2.5-0.5B Single FT (Megatron)",
        "config_names": "full_finetune_pipeline1",
        "description": "1 Qwen2.5-0.5B FT pipeline (megatron_train) on GPUs 0-1, infer GPUs 0-3",
    },
    "F": {
        "label": "Qwen2.5-0.5B Dual FT (Megatron)",
        "config_names": "full_finetune_pipeline1,full_finetune_pipeline2",
        "description": "2 Qwen2.5-0.5B pipelines: P1 train 0-1, P2 train 2-3; infer shared 0-3",
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: str
    label: str
    wall_time_s: float
    gpu_stats: GPUStats
    success: bool
    error: str = ""


def run_scenario(scenario: str, examples_dir: Path) -> ScenarioResult:
    cfg = SCENARIO_CONFIGS[scenario]
    print(f"\n{'='*70}")
    print(f"  Scenario {scenario}: {cfg['label']}")
    print(f"  Configs : {cfg['config_names']}")
    print(f"  GPUs    : {cfg['description']}")
    print(f"{'='*70}")

    cmd = [
        sys.executable,
        str(examples_dir / "start_multi_pipeline_test.py"),
        "--config_name", cfg["config_names"],
    ]

    monitor = GPUMonitor()
    monitor.start()
    t0 = time.time()
    success = True
    error = ""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(examples_dir.parent),
            timeout=3600,
            capture_output=False,
        )
        if result.returncode != 0:
            success = False
            error = f"exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        success = False
        error = "timeout after 3600s"
    except Exception as e:
        success = False
        error = str(e)

    wall = time.time() - t0
    stats = monitor.stop()

    status = "OK" if success else f"FAILED ({error})"
    print(f"\nScenario {scenario} done: {wall:.0f}s  {status}")
    return ScenarioResult(scenario=scenario, label=cfg["label"], wall_time_s=wall,
                          gpu_stats=stats, success=success, error=error)


def print_table(results: List[ScenarioResult]) -> None:
    print("\n" + "=" * 72)
    print("  RLIX MULTI-PIPELINE EXPERIMENT — RESULTS")
    print("=" * 72)
    header = f"  {'Scen':<4}  {'Label':<28}  {'Wall':>8}  {'AvgUtil':>8}  {'PeakMem':>9}"
    print(header)
    print("  " + "─" * 68)
    for r in results:
        status = "" if r.success else " FAILED"
        print(f"  {r.scenario:<4}  {r.label:<28}  {r.wall_time_s:>7.0f}s  "
              f"{r.gpu_stats.avg_util:>7.1f}%  {r.gpu_stats.peak_mem_mb:>8} MB{status}")
        for i, gs in sorted(r.gpu_stats.per_gpu.items()):
            print(f"        GPU {i}: avg {gs['avg_util']:>5.1f}%  peak {gs['peak_mem_mb']:>6} MB")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="A",
                        help="Scenario to run: A, B, C, D, or 'all'")
    args = parser.parse_args()

    examples_dir = Path(__file__).resolve().parent

    if args.scenario.lower() == "all":
        scenarios = list(SCENARIO_CONFIGS.keys())
    else:
        scenarios = [s.strip().upper() for s in args.scenario.split(",")]
        for s in scenarios:
            if s not in SCENARIO_CONFIGS:
                print(f"Unknown scenario: {s!r}. Choose from {list(SCENARIO_CONFIGS)}")
                sys.exit(1)

    results = []
    for s in scenarios:
        results.append(run_scenario(s, examples_dir))
        # Brief pause between scenarios to let Ray and GPU memory settle
        if s != scenarios[-1]:
            print("Waiting 30s between scenarios...")
            time.sleep(30)

    print_table(results)


if __name__ == "__main__":
    main()
