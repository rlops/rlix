#!/bin/bash
# M11.2 RLix-mode dual-pipeline E2E smoke runner.
#
# Real M11.2 (Option β / overlap topology, this update):
#   P1: train=[0]   infer=[0,1,2]   exclusive [0],   shared [1,2]
#   P2: train=[3]   infer=[1,2,3]   exclusive [3],   shared [1,2]
# Driver reads MILES_DUAL_P{1,2}_{TRAIN,INFER} envs; if any unset,
# falls back to disjoint Option A behavior.
#
# Each pipeline runs Qwen2.5-0.5B GRPO --num-rollout 2 concurrently via
# asyncio.gather in run_miles_dual.py. The shared infer GPUs [1,2]
# exercise the donor-shrink-before-receiver-expand path in the rlix
# scheduler (rlix/scheduler/scheduler.py:1027-1073).
#
# Option β contract (MILES_INIT_DEFER_ADD_WORKER=1):
#   - SGLang engines skip /add_worker at init (sglang_engine.py:_init_normal)
#   - EngineInfo state lands "loading" not "active" (rollout.py:_init_engine_info_table)
#   - MilesPipeline._init_phase_b_infer drives finish_init_offload(all)
#     → engines transition loading → offloaded, VRAM released
#   - bootstrap_active_engines(frozenset()) — empty post-INIT
#   - F40 Runtime branch handles every subsequent expand
#
# This script is rsynced to /root/rlix/scripts/run_smoke_dual.sh.

set -e

# Bump file descriptor limit before any Ray operation. The default 1024
# soft limit causes raylet to SIGABRT with errno=24 (EMFILE) once two
# pipelines + ~20 SGLang sub-processes start opening sockets.
ulimit -n 65536

# CRITICAL: cd /root (not /root/miles). Ray spawns workers inheriting
# CWD; /root/miles has a ghost `ray/` dir from a bad earlier rsync that
# would shadow the pip ray pkg's `ray._private` submodule in workers.
cd /root

# ---- env vars (same as M11.1 smoke; identity vars are per-pipeline) ----
# Ghost dirs from a prior bad rsync (ray/, utils/, etc. at /root/miles/)
# were renamed to /root/miles/_ghost_from_bad_rsync/, so /root/miles is
# now safe in PYTHONPATH (Ray actors need it to find examples.fully_async).
export PYTHONPATH=/root/miles:/root/rlix:/root/Megatron-LM
export RLIX_CONTROL_PLANE=rlix
export CUDA_DEVICE_MAX_CONNECTIONS=1
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    export NCCL_NVLS_ENABLE=1
else
    export NCCL_NVLS_ENABLE=0
fi
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MILES_TMS_HOOK_MODE=torch
# Also needed for the m11.1 smoke flow that the dual driver shares.
export MILES_SKIP_TMS_PAUSE=1
export MILES_SKIP_NODE_PG_PIN=1

# Real M11.2 Option β switch + overlap topology (Codex KT review).
# Comment any of these out to fall back to disjoint Option A behavior.
export MILES_INIT_DEFER_ADD_WORKER=1
export MILES_DUAL_P1_TRAIN=0
export MILES_DUAL_P1_INFER=0,1,2
export MILES_DUAL_P2_TRAIN=3
export MILES_DUAL_P2_INFER=1,2,3

echo "=== branch heads ==="
( cd /root/miles && git rev-parse HEAD 2>/dev/null || echo "miles@(rsync)" )
( cd /root/rlix  && git rev-parse HEAD 2>/dev/null || echo "rlix@(rsync)"  )

echo "=== env vars ==="
echo "RLIX_CONTROL_PLANE=$RLIX_CONTROL_PLANE  NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE  CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"

echo "=== ray cleanup + start ==="
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f sglang 2>/dev/null || true
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
pkill -9 -f ray:: 2>/dev/null || true
pkill -9 -f run_miles 2>/dev/null || true
sleep 5
rm -rf /tmp/ray /tmp/raylet* /tmp/plasma* 2>/dev/null || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "=== launching run_miles_dual ==="
source /root/miles/scripts/models/qwen2.5-0.5B.sh

# Real M11.2 OVERLAP topology (when MILES_DUAL_P*_{TRAIN,INFER} envs set above):
#   P1: train=[0]   infer=[0,1,2]   exclusive [0],   shared [1,2]
#   P2: train=[3]   infer=[1,2,3]   exclusive [3],   shared [1,2]
# Driver derives per-pipeline shape (actor_num_gpus_per_node, rollout_num_gpus)
# from the mapping lengths. CLI args below carry the DEFAULTS used when
# overlap envs are unset (Option A disjoint fallback): per-pipeline
# train=1 / infer=2 / num_gpus_per_node=4.
python /root/miles/examples/rlix/run_miles_dual.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/Qwen2.5-0.5B \
  --ref-load /root/Qwen2.5-0.5B_torch_dist \
  --load /root/Qwen2.5-0.5B_miles/ \
  --save "" \
  --save-interval 100 --eval-interval 100 \
  --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl \
  --n-samples-per-eval-prompt 1 --eval-max-response-len 1024 --eval-top-p 1 \
  --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt --label-key label --apply-chat-template --rollout-shuffle \
  --rm-type deepscaler \
  --num-rollout 2 --rollout-batch-size 8 --n-samples-per-prompt 4 \
  --rollout-max-response-len 2048 --rollout-temperature 1 \
  --global-batch-size 32 --balance-data \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --use-dynamic-batch-size --max-tokens-per-gpu 4096 \
  --rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 \
  --use-miles-router \
  --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async \
  --offload-train --offload-rollout \
  --moe-router-topk 0 \
  --model-update-transport cpu_serialize \
  --num-gpus-per-node 4 \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash
