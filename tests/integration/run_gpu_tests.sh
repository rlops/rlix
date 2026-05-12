#!/usr/bin/env bash
# Run GPU integration tests on Vast.ai instance.
# Usage: bash run_gpu_tests.sh
# Run from the rlix repo root on the remote instance.
set -euo pipefail

RLIX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RLIX_ROOT"

echo "=== rlix GPU integration tests ==="
echo "Working dir: $RLIX_ROOT"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"

# Install minimal deps if not present
python3 -c "import torch" 2>/dev/null || pip install torch --quiet
python3 -c "import transformers" 2>/dev/null || pip install transformers --quiet
python3 -c "import pytest" 2>/dev/null || pip install pytest --quiet

# Pre-download model so tests don't timeout on first run
echo ""
echo "=== Pre-downloading Qwen2.5-0.5B ==="
python3 - <<'PYEOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Downloading tokenizer...")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print("Downloading model weights...")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="bfloat16", low_cpu_mem_usage=True)
print("Download complete.")
PYEOF

echo ""
echo "=== Running GPU integration tests ==="
python3 -m pytest tests/integration/test_bucket_cache_gpu.py -v \
    --tb=short \
    --no-header \
    -p no:cacheprovider \
    2>&1 | tee /tmp/gpu_test_results.txt

echo ""
echo "=== Test summary ==="
tail -5 /tmp/gpu_test_results.txt
