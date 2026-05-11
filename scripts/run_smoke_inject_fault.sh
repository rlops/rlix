#!/bin/bash
# Phase 1 R04-F1 verification smoke (single pipeline + injected fault).
# Confirms that a train() raise releases the actor_train scheduler
# allocation via release_train_only (Phase 1) so the ledger is empty
# after the driver exits non-zero.
#
# Usage:
#   bash /root/rlix/scripts/run_smoke_inject_fault.sh
# Then verify:
#   tail -50 /tmp/miles_inject_fault.log
#   nvidia-smi
#   ray status   # expect: 0 leaked actors

set -e
ulimit -n 65536
# CRITICAL: cd /root (not /root/miles). Ray spawns workers inheriting
# CWD; /root/miles has a ghost `ray/` dir from a bad earlier rsync that
# would shadow the pip ray pkg's `ray._private` submodule in workers.
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
export MILES_SKIP_TMS_PAUSE=1
export MILES_SKIP_NODE_PG_PIN=1

# Phase 1 fault injection — raises inside train_group.train at first rollout.
export MILES_INJECT_TRAIN_FAULT=1

echo "=== branch heads ==="
( cd /root/miles && git rev-parse HEAD 2>/dev/null || echo "miles@(rsync)" )
( cd /root/rlix  && git rev-parse HEAD 2>/dev/null || echo "rlix@(rsync)"  )

echo "=== ray cleanup + start ==="
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f sglang 2>/dev/null || true
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
pkill -9 -f ray:: 2>/dev/null || true
pkill -9 -f run_miles 2>/dev/null || true
sleep 5
# Clear leftover Ray socket/state from prior crashed smoke iterations.
# Symptom this targets: "Failed to register worker to Raylet: IOError:
# End of file" (Ray worker can't connect because the raylet socket
# from a prior session is stale).
rm -rf /tmp/ray /tmp/raylet* /tmp/plasma* 2>/dev/null || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "=== launching run_miles_rlix (expects non-zero exit by design) ==="
source /root/miles/scripts/models/qwen2.5-0.5B.sh
set +e
# Direct path invocation (not `python -m`) so sys.path does NOT include
# /root/miles (where the ghost `ray/` from a prior bad rsync shadows the
# pip ray package).
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
  --rollout-num-gpus 4 --rollout-num-gpus-per-engine 1 \
  --use-miles-router \
  --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async \
  --offload-train --offload-rollout \
  --moe-router-topk 0 \
  --model-update-transport cpu_serialize \
  --num-gpus-per-node 4 \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash \
  2>&1 | tee /tmp/miles_inject_fault.log
EXIT=$?
set -e
echo "EXIT_CODE=$EXIT  (non-zero expected; 0 = injection didn't fire)"

echo "=== nvidia-smi post-run ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a /tmp/miles_inject_fault.log
echo "=== ray status post-run ==="
ray status 2>&1 | tee -a /tmp/miles_inject_fault.log || true

echo "=== R04-F1 invariant checks ==="
INJECT=$(grep -c "MILES_INJECT_TRAIN_FAULT=1 — injected failure" /tmp/miles_inject_fault.log || true)
RELEASE=$(grep -c "release_only done" /tmp/miles_inject_fault.log || true)
LEAK=$(grep -c "release_only=None" /tmp/miles_inject_fault.log || true)
echo "  injected_raise_log_lines=$INJECT (expect >=1)"
echo "  release_only_done_log_lines=$RELEASE (expect >=1)"
echo "  release_only_None_log_lines=$LEAK (expect 0)"
if [ "$INJECT" -lt 1 ] || [ "$RELEASE" -lt 1 ] || [ "$LEAK" -ne 0 ]; then
    echo "R04-F1 VERIFICATION: FAIL"
    exit 1
fi
echo "R04-F1 VERIFICATION: PASS (try/finally bracket released allocation)"
