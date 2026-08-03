#!/bin/bash
# M11 memory-offload AUDIT smoke — single pipeline, low host-RAM variant.
#
# Why single pipeline: this host has 15 GB RAM + 8 GB swap; the dual
# smoke (2 Megatron actors + 6 SGLang engines) peaks ~28 GB host RAM and
# Ray's memory monitor kills workers at 95%. One pipeline with
# train=[0] infer=[0,1] still exercises the audited path: the train GPU
# overlaps the infer pool, so every cycle runs
# shrink_engines -> _wait_for_overlap_engines_offloaded -> train wake_up
# -> train sleep -> expand_engines on GPU 0.
#
# Deltas vs run_smoke_e2e.sh:
#   - MILES_SKIP_TMS_PAUSE NOT set: torch_memory_saver.pause() is the
#     offload under audit; A5000 is Ampere sm_86 (pre-Blackwell), safe.
#   - RAY_memory_monitor_refresh_ms=0: rely on the 8 GB swap instead of
#     Ray's 95% OOM killer.
#   - ray --object-store-memory 500MB: nothing large goes through plasma.
#   - Tiny rollout/batch sizes (from run_smoke_dual.sh) to bound memory.

set -e
ulimit -n 65536
cd /root

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
export MILES_SKIP_NODE_PG_PIN=1
export RAY_memory_monitor_refresh_ms=0

echo "=== branch heads ==="
( cd /root/miles && git rev-parse HEAD 2>/dev/null || echo "miles@(rsync)" )
( cd /root/rlix  && git rev-parse HEAD 2>/dev/null || echo "rlix@(rsync)"  )

echo "=== env vars ==="
echo "RLIX_CONTROL_PLANE=$RLIX_CONTROL_PLANE NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE MILES_TMS_HOOK_MODE=$MILES_TMS_HOOK_MODE MILES_SKIP_TMS_PAUSE=${MILES_SKIP_TMS_PAUSE:-<unset>}"
echo "AUDIT_T0=$(date +%s)"

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
  --object-store-memory 500000000 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "=== launching run_miles_rlix (audit: train=[0] infer=[0,1]) ==="
source /root/miles/scripts/models/qwen2.5-0.5B.sh

python /root/miles/examples/rlix/run_miles_rlix.py \
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
  --num-rollout 2 --rollout-batch-size 1 --n-samples-per-prompt 1 \
  --rollout-max-response-len 256 --rollout-temperature 1 \
  --global-batch-size 1 --balance-data \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --use-dynamic-batch-size --max-tokens-per-gpu 512 \
  --sglang-mem-fraction-static 0.30 \
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
