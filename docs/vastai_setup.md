# Qwen2.5-0.5B GRPO Training Setup Guide
### Vast.ai + Miles + SGLang + Ray (4x GPU Async)

---

## Step 1: Launch Vast.ai Instance

Go to the following URL and select the template:

```
https://cloud.vast.ai?ref_id=93591&template_id=aad3af111541bdb77d4807f3e2a7a81b
```

SSH into the instance once it's running.

---

## Step 2: Setup Repository & Dependencies

```bash
cd /root/miles
git pull
pip install -e . --no-deps
```

---

## Step 3: Download Model & Datasets

```bash
# Base model
hf download Qwen/Qwen2.5-0.5B --local-dir /root/Qwen2.5-0.5B

# Training dataset
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Eval dataset
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

---

## Step 4: Load Model Config

```bash
source scripts/models/qwen2.5-0.5B.sh
```

---

## Step 5: Convert HuggingFace Checkpoint to Torch Dist Format

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-0.5B \
    --save /root/Qwen2.5-0.5B_torch_dist
```

---

## Step 6: Create Training Script

Save the following as `scripts/run-qwen2.5-0.5B_4xgpu_async.sh`:

```bash
#!/bin/bash
# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
set -ex
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=0,1,2,3
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Qwen2.5-0.5B model architecture args
MODEL_ARGS=(
   --swiglu
   --num-layers 24
   --hidden-size 896
   --ffn-hidden-size 4864
   --num-attention-heads 14
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --group-query-attention
   --num-query-groups 2
   --vocab-size 151936
)

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-0.5B
   --ref-load /root/Qwen2.5-0.5B_torch_dist
   --load /root/Qwen2.5-0.5B_miles/
   --save /root/Qwen2.5-0.5B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size 32
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 4096
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2   # 2 GPUs for SGLang rollout engine
   --rollout-num-gpus 2              # total rollout GPUs
   --use-miles-router
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
```

---

## Step 7: Launch Training

```bash
bash scripts/run-qwen2.5-0.5B_4xgpu_async.sh
```

---

## Key Config Summary

| Category | Key Settings |
|---|---|
| **Model** | Qwen2.5-0.5B, 24 layers, hidden=896, 14 attn heads, GQA (2 groups) |
| **Training** | GRPO, lr=1e-6, global batch=32, KL loss (low_var_kl) |
| **Rollout** | 3000 rollouts, 4 samples/prompt, max 2048 tokens, temp=1.0 |
| **Eval** | AIME-2024, every 20 steps, 4 samples/prompt, max 4096 tokens |
| **Parallelism** | TP=1, PP=1, 2 GPUs for SGLang engine, 2 GPUs for training actor |
| **GPU Setup** | 4x GPU total, NVLink auto-detected, Ray head node on 127.0.0.1 |
| **Checkpointing** | Save every 20 steps to `/root/Qwen2.5-0.5B_miles/` |
