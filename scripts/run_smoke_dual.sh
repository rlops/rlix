#!/bin/bash
# M11.2 RLix-mode dual-pipeline E2E smoke runner.
# Two MilesPipelines on disjoint GPU pools (P1=[0,1], P2=[2,3]) on a
# 4xRTX5090 vast.ai instance. Each runs Qwen2.5-0.5B GRPO --num-rollout 2
# concurrently via asyncio.gather in run_miles_dual.py.
#
# Topology rationale (per Codex M11.2 scope review): Option A (disjoint
# pools) is the minimum-viable PASS that exercises:
#   - Concurrent allocate_pipeline_id / register_pipeline / admit_pipeline
#   - Two MilesCoordinator + MilesPipeline pairs in separate Ray namespaces
#   - Two RolloutManager actors on disjoint port windows (15000 vs 16000)
#   - Two concurrent rlix_train_loops (asyncio.gather)
#   - Clean shutdown_hard for both pipelines
# Without requiring the deferred F22 shell-init contract.
#
# This script is rsynced to /root/rlix/scripts/run_smoke_dual.sh.

set -e

# Bump file descriptor limit before any Ray operation. The default 1024
# soft limit causes raylet to SIGABRT with errno=24 (EMFILE) once two
# pipelines + ~20 SGLang sub-processes start opening sockets.
ulimit -n 65536

cd /root/miles

# ---- env vars (same as M11.1 smoke; identity vars are per-pipeline) ----
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

echo "=== branch heads ==="
( cd /root/miles && git rev-parse HEAD 2>/dev/null || echo "miles@(rsync)" )
( cd /root/rlix  && git rev-parse HEAD 2>/dev/null || echo "rlix@(rsync)"  )

echo "=== env vars ==="
echo "RLIX_CONTROL_PLANE=$RLIX_CONTROL_PLANE  NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE  CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"

echo "=== ray cleanup + start ==="
ray stop --force >/dev/null 2>&1 || true
pkill -9 sglang 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "=== launching run_miles_dual ==="
source /root/miles/scripts/models/qwen2.5-0.5B.sh

# BASE args carry the PER-PIPELINE topology shape:
#   actor_num_gpus_per_node = per-pipeline train size (1)
#   rollout_num_gpus        = per-pipeline infer pool size (2)
# num_gpus_per_node = whole-machine GPU count (4) — driver splits this
# into 2 disjoint per-pipeline pools:
#   P1: train=[0],  infer=[0,1]  (1 train + 1 non-overlap, satisfies C4)
#   P2: train=[2],  infer=[2,3]  (1 train + 1 non-overlap, satisfies C4)
python -m examples.rlix.run_miles_dual \
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
