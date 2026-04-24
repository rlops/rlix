#!/usr/bin/env bash
# Run all Gate 2.5 tests on the Vast.ai instance.
# Usage: bash tests/integration/run_gate2_5.sh
# Must be run from rlix repo root with .venv activated.
set -euo pipefail

echo "================================================================"
echo "Gate 2.5 Test Suite"
echo "================================================================"
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

echo ""
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs available: $N_GPUS"
echo ""

# ----------------------------------------------------------------
# Part 1: NCCL destroy/re-init (2 GPUs)
# ----------------------------------------------------------------
echo "================================================================"
echo "Part 1: Megatron NCCL destroy/re-init stability (2 GPUs)"
echo "================================================================"

if python3 -c "from megatron.core import parallel_state" 2>/dev/null; then
    torchrun --nproc-per-node=2 \
        tests/integration/test_gate2_5_nccl_destroy.py
    echo ""
    echo "Part 1: DONE"
else
    echo "SKIP Part 1: megatron-core not installed"
    echo "  Install: pip install megatron-core"
fi

echo ""

# ----------------------------------------------------------------
# Part 2: Selective sync via dynamic NCCL group (2 GPUs)
# ----------------------------------------------------------------
echo "================================================================"
echo "Part 2: Selective sync dynamic NCCL group (2 GPUs)"
echo "================================================================"

torchrun --nproc-per-node=2 \
    tests/integration/test_gate2_5_selective_sync.py

echo ""
echo "Part 2: DONE"
echo ""

# ----------------------------------------------------------------
# Part 3: Real Qwen2.5-0.5B train + weight sync (4 GPUs)
# ----------------------------------------------------------------
echo "================================================================"
echo "Part 3: Qwen2.5-0.5B training + bit-exact weight sync (4 GPUs)"
echo "================================================================"

if [ "$N_GPUS" -lt 4 ]; then
    echo "SKIP Part 3: requires 4 GPUs (found $N_GPUS)"
else
    torchrun --nproc-per-node=4 \
        tests/integration/test_gate2_5_qwen_train_sync.py
    echo ""
    echo "Part 3: DONE"
fi

echo ""
echo "================================================================"
echo "ALL GATE 2.5 TESTS COMPLETE"
echo "================================================================"
