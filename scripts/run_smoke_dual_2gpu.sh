#!/bin/bash
# Adapted M11.2 2-GPU overlap smoke (user request — 2ppl partial overlap).
#
# Topology (user spec):
#   P1: train=[0]   infer=[0,1]
#   P2: train=[1]   infer=[0,1]
# Both pipelines share both physical GPUs for inference; train is disjoint
# (P1 → GPU0, P2 → GPU1). Each pipeline's inference engines co-locate with
# (a) its own train (P1: infer GPU0 = train GPU0), and
# (b) the peer pipeline's train (P1's infer GPU1 = P2's train).
# This exercises the donor-shrink-before-receiver-expand path in
# rlix/scheduler/scheduler.py:1027-1073.
#
# Goal: smoke / pass-through (just need the loop to complete one iteration
# without crash). Hence --num-rollout 1, save-interval/eval-interval pushed
# out of range.

set -e

# Bump file descriptor limit before any Ray operation.
ulimit -n 65536

cd /root

# ---- env vars ----------------------------------------------------------
: "${HF_TOKEN:?set HF_TOKEN before running}"
export HF_TOKEN
export PYTHONPATH=/root/miles:/root/rlix:/root/ROLL:/root/Megatron-LM
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
export MILES_SKIP_TMS_PAUSE=1
export MILES_SKIP_NODE_PG_PIN=1
# Reduce VRAM fragmentation under tight 2-GPU + cross-pipeline overlap.
# Cannot use expandable_segments — incompatible with miles' TorchMemorySaver.
# garbage_collection_threshold lets the caching alloc release unused cache
# before failing with OOM; max_split_size_mb keeps blocks small for tight reuse.
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:64

# Option β switch — deferred /add_worker until Phase B finish_init_offload.
export MILES_INIT_DEFER_ADD_WORKER=1

# 2-GPU overlap topology (user request).
export MILES_DUAL_P1_TRAIN=0
export MILES_DUAL_P1_INFER=0,1
export MILES_DUAL_P2_TRAIN=1
export MILES_DUAL_P2_INFER=0,1

echo "=== branch heads ==="
( cd /root/miles && git rev-parse HEAD 2>/dev/null || echo "miles@?" )
( cd /root/rlix  && git rev-parse HEAD 2>/dev/null || echo "rlix@?" )

echo "=== env vars ==="
echo "RLIX_CONTROL_PLANE=$RLIX_CONTROL_PLANE  NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"
echo "P1 train=$MILES_DUAL_P1_TRAIN infer=$MILES_DUAL_P1_INFER"
echo "P2 train=$MILES_DUAL_P2_TRAIN infer=$MILES_DUAL_P2_INFER"

echo "=== ray cleanup ==="
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f sglang 2>/dev/null || true
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
pkill -9 -f run_miles 2>/dev/null || true
sleep 3
rm -rf /tmp/ray /tmp/raylet* /tmp/plasma* 2>/dev/null || true
sleep 2

echo "=== ray start ==="
ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "=== pre-run nvidia-smi ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=== launching run_miles_dual (2-GPU overlap) ==="
source /root/miles/scripts/models/qwen2.5-0.5B.sh

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
  --num-rollout 1 --rollout-batch-size 8 --n-samples-per-prompt 4 \
  --rollout-max-response-len 512 --rollout-temperature 1 \
  --global-batch-size 32 --balance-data \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --use-precision-aware-optimizer --optimizer-cpu-offload \
  --optimizer-offload-fraction 1.0 \
  --use-dynamic-batch-size --max-tokens-per-gpu 128 \
  --rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.3 \
  --sglang-disable-cuda-graph \
  --sglang-disable-piecewise-cuda-graph \
  --sglang-chunked-prefill-size 1024 \
  --sglang-max-prefill-tokens 1024 \
  --sglang-max-running-requests 4 \
  --use-miles-router \
  --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async \
  --offload-train --offload-rollout \
  --moe-router-topk 0 \
  --model-update-transport cpu_serialize \
  --num-gpus-per-node 2 \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash

ec=$?
echo "=== EXIT_CODE=$ec ==="
echo "=== post-run nvidia-smi ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv
exit $ec
