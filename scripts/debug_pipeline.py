#!/usr/bin/env python3
"""rlix training-pipeline debug launcher — a plain Python file you can open,
edit, and run (on the vast instance), including under an IDE debugger.

Usage on the instance:
    python /root/debug_pipeline.py                 # run with Config below
    python /root/debug_pipeline.py --dry-run       # print argv/env, run nothing
    python -m pdb /root/debug_pipeline.py          # step into the driver

Everything you would tweak lives in `Config` right below — edit and rerun.
The driver runs IN-PROCESS (via runpy), so breakpoints set in miles/rlix
code work when you launch this file from a debugger.
"""

from __future__ import annotations

import math
import os
import runpy
import subprocess
import sys
import time
from dataclasses import dataclass, field


# ======================================================================
# Edit me
# ======================================================================
@dataclass
class Config:
    # --- what to run -------------------------------------------------
    mode: str = "single"          # "single" or "dual"
    num_rollout: int = 2          # training cycles
    extra_args: list[str] = field(default_factory=list)  # appended to the driver argv

    # --- GPU topology --------------------------------------------------
    # single mode: train GPUs = range(train_gpus), infer GPUs = range(infer_gpus)
    # (derived in run_miles_rlix.py::_build_cluster_device_mappings — train may
    # be a subset of infer, that overlap is the shrink/grant/expand handoff).
    #   train_gpus=1, infer_gpus=2 -> train=[0]   infer=[0,1]     (GPUs 2,3 idle)
    #   train_gpus=2, infer_gpus=4 -> train=[0,1] infer=[0,1,2,3] (all 4 GPUs, M11.1 topo)
    train_gpus: int = 1
    infer_gpus: int = 2
    # dual mode: explicit per-pipeline GPU index lists (comma strings).
    #   defaults = M11.2 overlap topology, all 4 GPUs, shared infer on [1,2]
    dual_p1_train: str = "0"
    dual_p1_infer: str = "0,1,2"
    dual_p2_train: str = "3"
    dual_p2_infer: str = "1,2,3"

    # --- batch sizes -----------------------------------------------------
    # Megatron constraint: global_batch_size % (micro_batch(1) * DP) == 0,
    # where DP = number of train GPUs. And each rollout must produce enough
    # samples: rollout_batch_size * n_samples_per_prompt >= global_batch_size.
    # None -> auto-derived from the topology (smallest legal debug values).
    global_batch_size: int | None = None   # auto: = train GPUs (dual: per-pipeline train GPUs)
    rollout_batch_size: int | None = None  # auto: = global_batch_size / n_samples_per_prompt
    n_samples_per_prompt: int = 1

    # --- memory offload knobs ----------------------------------------
    # "auto": preload where it is safe (non-Blackwell, or Blackwell with
    #         cu13+ torch wheels), torch on Blackwell + cu12.x (the tms
    #         0.0.9 preload SIGSEGV combo — RTX 50xx / RTX PRO 6000 / B100
    #         on the fork-baseline image).
    tms_hook_mode: str = "auto"           # "auto" | "preload" | "torch"
    residual_threshold_gb: float | None = None  # None -> code default (13.0 pre-#31, 7.0 after)

    # --- paths (instance layout) -------------------------------------
    miles: str = "/root/miles"
    rlix: str = "/root/rlix"
    megatron: str = "/root/Megatron-LM"
    model_hf: str = "/root/Qwen2.5-0.5B"
    model_torch_dist: str = "/root/Qwen2.5-0.5B_torch_dist"
    model_miles: str = "/root/Qwen2.5-0.5B_miles/"
    prompt_data: str = "/root/dapo-math-17k/dapo-math-17k.jsonl"
    eval_data: str = "/root/aime-2024/aime-2024.jsonl"

    # --- runtime ------------------------------------------------------
    restart_ray: bool = True      # kill + restart the local ray head first
    num_gpus: int = 4
    # Mirror all output (incl. ray/Megatron C-level writes) to this file
    # while still printing to the terminal. Previous log is rotated to
    # <name>.prev.log at launch. Set to None to disable.
    log_file: str | None = "/root/logs/run.log"


CFG = Config()
# ======================================================================


def build_env(cfg: Config) -> None:
    """Set process env (inherited by ray workers via raylet)."""
    os.environ["PYTHONPATH"] = f"{cfg.miles}:{cfg.rlix}:{cfg.megatron}"
    os.environ["RLIX_CONTROL_PLANE"] = "rlix"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    nvlink = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.count("NVLink")
    os.environ["NCCL_NVLS_ENABLE"] = "1" if nvlink > 0 else "0"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["MILES_SKIP_NODE_PG_PIN"] = "1"
    # This image's ray ships an OTel metrics exporter whose background gRPC
    # thread getenv()s while actor startup setenv()s -> glibc environ race ->
    # SIGSEGV during actor creation (seen 2026-07-09 on MilesModelUpdateService).
    # Metrics are useless for debugging; kill the racing thread at the source.
    os.environ["RAY_enable_metrics_collection"] = "false"
    hook = cfg.tms_hook_mode
    if hook == "auto":
        import torch

        cc_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        cuda = torch.version.cuda or "0"
        blackwell_pre_cu13 = cc_major >= 10 and int(cuda.split(".")[0]) < 13
        hook = "torch" if blackwell_pre_cu13 else "preload"
        print(f"tms hook auto-selected: {hook} (cc_major={cc_major}, cuda={cuda})")
    os.environ["MILES_TMS_HOOK_MODE"] = hook
    if hook == "preload":
        # Guard escape hatch — needed for preload on Blackwell+cu13 (segfault
        # verified gone there, 2026-07-05 audit). NOT set in torch mode: on
        # Blackwell + cu12.x the guard's raise is protecting you from a real
        # SIGSEGV; bypassing it would crash build_cpu_bucket_cache.
        os.environ["MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL"] = "1"
    if cfg.residual_threshold_gb is not None:
        os.environ["MILES_MAX_RESIDUAL_GPU_MEM_GB"] = str(cfg.residual_threshold_gb)
    if cfg.mode == "dual":
        os.environ["MILES_INIT_DEFER_ADD_WORKER"] = "1"
        os.environ["MILES_DUAL_P1_TRAIN"] = cfg.dual_p1_train
        os.environ["MILES_DUAL_P1_INFER"] = cfg.dual_p1_infer
        os.environ["MILES_DUAL_P2_TRAIN"] = cfg.dual_p2_train
        os.environ["MILES_DUAL_P2_INFER"] = cfg.dual_p2_infer
    for p in (cfg.miles, cfg.rlix, cfg.megatron):
        if p not in sys.path:
            sys.path.insert(0, p)


def model_args(cfg: Config) -> list[str]:
    """MODEL_ARGS from miles' qwen2.5-0.5B.sh (single source of truth)."""
    out = subprocess.run(
        ["bash", "-c", f'source {cfg.miles}/scripts/models/qwen2.5-0.5B.sh && printf "%s\\n" "${{MODEL_ARGS[@]}}"'],
        capture_output=True, text=True, check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def build_argv(cfg: Config) -> tuple[str, list[str]]:
    driver = os.path.join(
        cfg.miles, "examples/rlix",
        "run_miles_dual.py" if cfg.mode == "dual" else "run_miles_rlix.py",
    )
    # Resolve batch sizes against the topology. Each dual pipeline is an
    # INDEPENDENT Megatron world: its DP = len(its own train mapping)
    # (P1=[0] -> DP=1, P2=[3] -> DP=1 — the two never form one DP=2 group).
    # Both pipelines receive the SAME global_batch_size, so it must be
    # divisible by BOTH pipelines' DP -> use the LCM (max is wrong: with
    # train pools of 3 and 2 GPUs, global=3 divides 3 but not 2).
    def _n(mapping: str) -> int:
        return len([x for x in mapping.split(",") if x.strip()])

    if cfg.mode == "dual":
        dp = math.lcm(_n(cfg.dual_p1_train), _n(cfg.dual_p2_train))
        # CLI topology args must AGREE with the dual mappings: the driver
        # overrides them per pipeline when the MILES_DUAL_* envs are present,
        # but if the envs ever go missing (e.g. driver launched directly),
        # the DISJOINT fallback uses the CLI values — a CLI/mapping mismatch
        # is exactly what produced the "DP=2 vs global=1" Megatron assert.
        cli_train_gpus = max(_n(cfg.dual_p1_train), _n(cfg.dual_p2_train))
        cli_infer_gpus = max(_n(cfg.dual_p1_infer), _n(cfg.dual_p2_infer))
    else:
        dp = cfg.train_gpus
        cli_train_gpus = cfg.train_gpus
        cli_infer_gpus = cfg.infer_gpus
    global_bs = cfg.global_batch_size if cfg.global_batch_size is not None else dp
    if global_bs % dp != 0:
        raise SystemExit(
            f"Config error: global_batch_size={global_bs} not divisible by "
            f"micro_batch(1) * DP({dp}) — Megatron will assert. Pick a multiple of {dp}."
        )
    rollout_bs = (
        cfg.rollout_batch_size
        if cfg.rollout_batch_size is not None
        else max(1, -(-global_bs // cfg.n_samples_per_prompt))  # ceil div
    )
    if rollout_bs * cfg.n_samples_per_prompt < global_bs:
        raise SystemExit(
            f"Config error: rollout_batch_size({rollout_bs}) * n_samples_per_prompt"
            f"({cfg.n_samples_per_prompt}) = {rollout_bs * cfg.n_samples_per_prompt} "
            f"< global_batch_size({global_bs}) — a train step would starve for samples."
        )

    argv = [
        driver,
        *model_args(cfg),
        "--hf-checkpoint", cfg.model_hf,
        "--ref-load", cfg.model_torch_dist,
        "--load", cfg.model_miles,
        "--save", "",
        "--save-interval", "100", "--eval-interval", "100",
        "--eval-prompt-data", "aime", cfg.eval_data,
        "--n-samples-per-eval-prompt", "1", "--eval-max-response-len", "1024", "--eval-top-p", "1",
        "--prompt-data", cfg.prompt_data,
        "--input-key", "prompt", "--label-key", "label", "--apply-chat-template", "--rollout-shuffle",
        "--rm-type", "deepscaler",
        "--num-rollout", str(cfg.num_rollout),
        "--rollout-batch-size", str(rollout_bs), "--n-samples-per-prompt", str(cfg.n_samples_per_prompt),
        "--rollout-max-response-len", "256", "--rollout-temperature", "1",
        "--global-batch-size", str(global_bs), "--balance-data",
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--advantage-estimator", "grpo", "--use-kl-loss", "--kl-loss-coef", "0.0",
        "--kl-loss-type", "low_var_kl", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--optimizer", "adam", "--lr", "1e-6", "--lr-decay-style", "constant",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", "512",
        "--sglang-mem-fraction-static", "0.90",
        "--rollout-num-gpus", str(cli_infer_gpus), "--rollout-num-gpus-per-engine", "1",
        "--use-miles-router",
        "--rollout-function-path", "examples.fully_async.fully_async_rollout.generate_rollout_fully_async",
        "--offload-train", "--offload-rollout",
        "--moe-router-topk", "0",
        "--model-update-transport", "cpu_serialize",
        "--num-gpus-per-node", str(cfg.num_gpus),
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(cli_train_gpus),
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        *cfg.extra_args,
    ]
    return driver, argv


def restart_ray(cfg: Config) -> None:
    subprocess.run(["ray", "stop", "--force"], capture_output=True)
    for pat in ("sglang", "raylet", "gcs_server", "ray::", "run_miles"):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(5)
    subprocess.run(["bash", "-c", "rm -rf /tmp/ray /tmp/raylet* /tmp/plasma*"], capture_output=True)
    time.sleep(2)
    subprocess.run(
        ["ray", "start", "--head", "--node-ip-address", "127.0.0.1",
         "--num-gpus", str(cfg.num_gpus), "--disable-usage-stats",
         "--dashboard-host", "0.0.0.0", "--dashboard-port", "8265"],
        check=True,
    )


def main() -> None:
    cfg = CFG
    dry = "--dry-run" in sys.argv

    build_env(cfg)
    driver, argv = build_argv(cfg)

    print(f"mode={cfg.mode} hook={cfg.tms_hook_mode} "
          f"threshold={cfg.residual_threshold_gb or '<code default>'} num_rollout={cfg.num_rollout}")
    print(f"driver: {driver}")
    if dry:
        print("argv:")
        for a in argv[1:]:
            print(f"  {a}")
        return

    # Raise fd limit (two pipelines + SGLang subprocesses exhaust the 1024 default).
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    if cfg.log_file:
        # fd-level tee: dup stdout/stderr through `tee` so C-level writes
        # (ray workers' forwarded logs, Megatron banners) land in the file
        # too — sys.stdout redirection alone would miss them.
        os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
        if os.path.exists(cfg.log_file):
            os.replace(cfg.log_file, cfg.log_file.replace(".log", "") + ".prev.log")
        tee = subprocess.Popen(["tee", cfg.log_file], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), 1)
        os.dup2(tee.stdin.fileno(), 2)
        print(f"logging to {cfg.log_file} (previous run -> .prev.log)")

    if cfg.restart_ray:
        restart_ray(cfg)

    os.chdir("/root")  # ray workers inherit CWD; keep it neutral
    sys.argv = argv
    # In-process execution: your debugger's breakpoints inside miles/rlix fire.
    runpy.run_path(driver, run_name="__main__")


if __name__ == "__main__":
    main()
