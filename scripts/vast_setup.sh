#!/bin/bash
# M11 offload-audit instance setup — follows docs/smoke-test-runbook.md
set -x
exec > /root/setup.log 2>&1

echo "=== STEP 3: miles branch ==="
cd /root/miles
git remote add rlops https://github.com/rlops/miles.git 2>/dev/null
git fetch rlops zhenyu/m11-mvp-test
git checkout -B zhenyu/m11-offload-audit rlops/zhenyu/m11-mvp-test
pip install -e . --no-deps

echo "=== STEP 4: rlix clone ==="
cd /root
if [ ! -d /root/rlix ]; then git clone https://github.com/rlops/rlix.git; fi
cd /root/rlix
git fetch origin zhenyu/miles-mvp-e2e
git checkout -B zhenyu/m11-offload-audit origin/zhenyu/miles-mvp-e2e
pip install -e . --no-deps

echo "=== STEP 5: ROLL ==="
python -c "import roll" 2>/dev/null || pip install "roll @ git+https://github.com/rlops/ROLL.git" --no-deps
pip list 2>/dev/null | grep -i -E '^(roll|tg4perfetto|sglang|torch|ray) '
python -c "import tg4perfetto" 2>/dev/null || pip install tg4perfetto

echo "=== STEP 6: model + datasets ==="
[ -f /root/Qwen2.5-0.5B/config.json ] || hf download Qwen/Qwen2.5-0.5B --local-dir /root/Qwen2.5-0.5B
[ -d /root/dapo-math-17k ] && [ -n "$(ls /root/dapo-math-17k 2>/dev/null)" ] || hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
[ -d /root/aime-2024 ] && [ -n "$(ls /root/aime-2024 2>/dev/null)" ] || hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024

echo "=== STEP 7: checkpoint conversion ==="
if [ ! -d /root/Qwen2.5-0.5B_torch_dist ]; then
  cd /root/miles
  source scripts/models/qwen2.5-0.5B.sh
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint /root/Qwen2.5-0.5B \
      --save /root/Qwen2.5-0.5B_torch_dist
fi

echo "=== VERSIONS ==="
python -c "import sglang; print('sglang', sglang.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cc', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'nocuda')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mkdir -p /root/logs
echo "=== SETUP_DONE ==="
