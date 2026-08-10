#!/usr/bin/env python3
"""最简 GPU memory offload 测试 — 分配 2GB → offload → 恢复,~10 秒跑完。

用法(在 vast 实例上):
    CUDA_VISIBLE_DEVICES=0 python test_offload_simple.py
"""
import subprocess
import torch
from torch_memory_saver import torch_memory_saver

torch_memory_saver.hook_mode = "torch"  # 必须在第一次 tms 调用之前设置


def gpu_used():
    """当前 GPU 0 整卡已用显存 (MiB),问 nvidia-smi,不问 torch。"""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.splitlines()[0])


torch.zeros(1, device="cuda")  # 初始化 CUDA context
print(f"1) 基线 (CUDA context)      : {gpu_used():6d} MiB")

with torch_memory_saver.region(tag="default"):
    x = torch.ones(2 * 1024**3 // 2, dtype=torch.bfloat16, device="cuda")  # 2 GiB
torch.cuda.synchronize()
print(f"2) 分配 2GB 之后            : {gpu_used():6d} MiB")

torch.cuda.empty_cache()
torch_memory_saver.pause()  # ← offload:物理显存被释放
torch.cuda.synchronize()
print(f"3) offload (pause) 之后     : {gpu_used():6d} MiB   ← 应该回到基线附近")

torch_memory_saver.resume()  # ← 恢复:物理显存重新映射
torch.cuda.synchronize()
print(f"4) 恢复 (resume) 之后       : {gpu_used():6d} MiB   ← 应该回到 2GB 水平")
