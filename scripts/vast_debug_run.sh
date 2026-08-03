#!/bin/bash
# rlix training-pipeline debug runner — run ON the vast instance.
#
# Usage:
#   bash /root/vast_debug_run.sh [single|dual] [extra driver args...]
#
# Env knobs (all optional):
#   HOOK=preload|torch     tms hook mode              (default preload)
#   THRESH=<GiB>           MILES_MAX_RESIDUAL_GPU_MEM_GB override (default: code default)
#   NUM_ROLLOUT=<n>        training cycles            (default 2)
#   SILENCE_LIMIT=<s>      watchdog: kill after this much log silence (default 300)
#   RUN_LIMIT=<s>          watchdog: hard wall-clock cap             (default 1800)
#   SAMPLER=0|1            1 Hz whole-GPU + per-process memory sampler (default 1)
#   MILES=/root/miles RLIX=/root/rlix LOG_DIR=/root/logs   tree/log locations
#
# Examples:
#   bash vast_debug_run.sh single                      # minimal 1-pipeline debug loop
#   HOOK=torch THRESH=13 bash vast_debug_run.sh dual   # rollback-mode dual smoke
#   NUM_ROLLOUT=5 bash vast_debug_run.sh single --rollout-max-response-len 512
#
# Topologies: single = train[0], infer[0,1] (self-overlap on GPU 0)
#             dual   = P1 train[0]/infer[0,1,2] + P2 train[3]/infer[1,2,3] (overlap [1,2])
# Ends with EXIT_CODE=<n> as the last line of $LOG_DIR/run.log and a result summary.

set -uo pipefail

MODE="${1:-single}"; shift || true
HOOK="${HOOK:-preload}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
SILENCE_LIMIT="${SILENCE_LIMIT:-300}"
RUN_LIMIT="${RUN_LIMIT:-1800}"
SAMPLER="${SAMPLER:-1}"
MILES="${MILES:-/root/miles}"
RLIX="${RLIX:-/root/rlix}"
LOG_DIR="${LOG_DIR:-/root/logs}"
LOG="$LOG_DIR/run.log"

mkdir -p "$LOG_DIR"
for f in run gpu_samples; do
  [ -f "$LOG_DIR/$f.log" ] && mv "$LOG_DIR/$f.log" "$LOG_DIR/$f.prev.log"
done

# ---- sampler ----
if [ "$SAMPLER" = "1" ] && [ -f /root/audit_gpu_sampler.sh ]; then
  pkill -f "[a]udit_gpu_sampler" 2>/dev/null
  OUT="$LOG_DIR/gpu_samples.log" setsid nohup bash /root/audit_gpu_sampler.sh >/dev/null 2>&1 < /dev/null &
  echo "sampler pid=$! -> $LOG_DIR/gpu_samples.log"
fi

# ---- env (mirrors scripts/run_smoke_dual.sh conventions) ----
ulimit -n 65536
cd /root
export PYTHONPATH="$MILES:$RLIX:/root/Megatron-LM"
export RLIX_CONTROL_PLANE=rlix
export CUDA_DEVICE_MAX_CONNECTIONS=1
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && export NCCL_NVLS_ENABLE=1 || export NCCL_NVLS_ENABLE=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MILES_SKIP_NODE_PG_PIN=1
export MILES_TMS_HOOK_MODE="$HOOK"
export MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL=1   # harmless on cu13+/pre-Blackwell
[ -n "${THRESH:-}" ] && export MILES_MAX_RESIDUAL_GPU_MEM_GB="$THRESH"

if [ "$MODE" = "dual" ]; then
  export MILES_INIT_DEFER_ADD_WORKER=1
  export MILES_DUAL_P1_TRAIN=0 MILES_DUAL_P1_INFER=0,1,2
  export MILES_DUAL_P2_TRAIN=3 MILES_DUAL_P2_INFER=1,2,3
  DRIVER="$MILES/examples/rlix/run_miles_dual.py"
  TOPO_ARGS=(--rollout-num-gpus 2 --actor-num-gpus-per-node 1)
else
  DRIVER="$MILES/examples/rlix/run_miles_rlix.py"
  TOPO_ARGS=(--rollout-num-gpus 2 --actor-num-gpus-per-node 1)
fi

echo "=== debug run: mode=$MODE hook=$HOOK thresh=${THRESH:-<code default>} num_rollout=$NUM_ROLLOUT"
echo "=== heads: miles=$(git -C "$MILES" rev-parse --short HEAD 2>/dev/null || echo rsync) rlix=$(git -C "$RLIX" rev-parse --short HEAD 2>/dev/null || echo rsync)"

# ---- ray restart ----
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f sglang 2>/dev/null; pkill -9 -f raylet 2>/dev/null
pkill -9 -f gcs_server 2>/dev/null; pkill -9 -f "ray::" 2>/dev/null
pkill -9 -f run_miles 2>/dev/null
sleep 5; rm -rf /tmp/ray /tmp/raylet* /tmp/plasma* 2>/dev/null; sleep 2
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 >/dev/null

# ---- launch under watchdog ----
source "$MILES/scripts/models/qwen2.5-0.5B.sh"
(
python "$DRIVER" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/Qwen2.5-0.5B \
  --ref-load /root/Qwen2.5-0.5B_torch_dist \
  --load /root/Qwen2.5-0.5B_miles/ \
  --save "" --save-interval 100 --eval-interval 100 \
  --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl \
  --n-samples-per-eval-prompt 1 --eval-max-response-len 1024 --eval-top-p 1 \
  --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt --label-key label --apply-chat-template --rollout-shuffle \
  --rm-type deepscaler \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size 1 --n-samples-per-prompt 1 \
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
  --rollout-num-gpus-per-engine 1 \
  "${TOPO_ARGS[@]}" \
  --use-miles-router \
  --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async \
  --offload-train --offload-rollout \
  --moe-router-topk 0 \
  --model-update-transport cpu_serialize \
  --num-gpus-per-node 4 --actor-num-nodes 1 \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash \
  "$@"
echo "EXIT_CODE=$?"
) >"$LOG" 2>&1 &
RUN_PID=$!

START=$(date +%s); LAST_SIZE=0; LAST_CHANGE=$START
while kill -0 $RUN_PID 2>/dev/null; do
  NOW=$(date +%s)
  SIZE=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  [ "$SIZE" != "$LAST_SIZE" ] && { LAST_SIZE=$SIZE; LAST_CHANGE=$NOW; }
  if [ $((NOW - LAST_CHANGE)) -gt "$SILENCE_LIMIT" ] || [ $((NOW - START)) -gt "$RUN_LIMIT" ]; then
    echo "=== WATCHDOG: silent=$((NOW-LAST_CHANGE))s elapsed=$((NOW-START))s — killing ===" | tee -a "$LOG"
    kill -9 $RUN_PID 2>/dev/null
    pkill -9 -f run_miles 2>/dev/null; pkill -9 -f sglang 2>/dev/null
    ray stop --force >/dev/null 2>&1
    echo "EXIT_CODE=124" >> "$LOG"
    break
  fi
  sleep 10
done
wait $RUN_PID 2>/dev/null

pkill -f "[a]udit_gpu_sampler" 2>/dev/null

# ---- summary ----
echo
echo "================ RESULT ================"
tail -1 "$LOG"
echo "tracebacks: $(grep -c Traceback "$LOG" 2>/dev/null || echo 0)"
echo "--- gate readings ---"
# new gate (PR#17+): "whole-GPU mem used max=..."; old gate: "OS-level GPU mem free min=..."
grep -E "whole-GPU mem used|OS-level GPU mem free" "$LOG" | sed 's/\x1b\[[0-9;]*m//g' | sed 's/.*INFO:rlix/INFO:rlix/' | tail -6
echo "--- training loops ---"
grep -E "training loop complete|WATCHDOG" "$LOG" | sed 's/\x1b\[[0-9;]*m//g' | tail -4
echo "--- errors (if any) ---"
grep -E "RuntimeError|OutOfMemoryError|ActorDiedError" "$LOG" | sed 's/\x1b\[[0-9;]*m//g' | head -4
echo "full log: $LOG   gpu samples: $LOG_DIR/gpu_samples.log"
