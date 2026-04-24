# Task 2 — CPU Bucket Cache + 选择性权重同步 (F4, F6-transport)

**规格文档**: [nemorl-port-plan.md](https://github.com/rlops/rlix/blob/nemo/plans/nemorl-port-plan.md) — Feature 4 + Feature 6  
**Gate**: 2.5 — 全部 6 个 GPU 集成测试通过（4× RTX A5000）  
**代码分支**: `task2-bucket-cache` (rlix) · `rlix-task2` / `main` (NeMo 子模块)

---

## Feature 4 — 训练侧 CPU Bucket Cache

### 规格要求 → 实现位置

| 规格要求 | 实现文件 | 说明 |
|---------|---------|------|
| 所有 TP/PP/CP/EP rank 参与 gather，只有 cache owner 存储 | `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py` → `build_latest_bucket_cache()` | owner = pp0/dp0/tp0/cp0，非 owner drain iterator 但不存储 |
| 打包为 canonical `List[BucketRecord]`（512字节对齐 uint8） | `rlix/pipeline/bucket_cache.py` → `BucketRecord`, `_bucket_named_tensors()` | 包含 `param_names`, `shapes`, `dtypes`, `offsets`, `used_bytes`, `cpu_uint8_bucket` |
| 接收侧 unpack 还原各 tensor | `rlix/pipeline/bucket_cache.py` → `unpack_bucket_record()` | 用 `torch.empty(0, dtype=dtype).element_size()` 计算字节宽度，避免 uint8 slice 非法 view |
| `_cache_ready_step` 原子更新（版本指针） | `rlix/pipeline/bucket_cache.py` → `VersionedBucketCache.promote()` | 两指针设计：`_latest_cached` / `_active_cached`，promote 后 GC 旧版本 |
| 生命周期追踪 | `rlix/pipeline/bucket_cache_lifecycle.py` → `BucketCacheLifecycle` | `build_latest_bucket_cache.remote()` → `promote_active_checkpoint.remote()` → `mark_promoted()` |
| `bucket_size_bytes` 必须显式配置，禁止隐式默认 | `megatron_policy_worker.py` → `_rlix_get_bucket_size_bytes()` | 未配置则 `raise RuntimeError`，读取 `RLIX_BUCKET_SIZE_BYTES` 或 `worker.cfg['rlix']['bucket_size_bytes']` |
| 单个 param > bucket_size_bytes → fail fast | `megatron_policy_worker.py` → `build_latest_bucket_cache()` | append 前检查，匹配 ROLL `send_recv_utils.py` 的 assert 模式 |
| host RAM 检查：2 × model_bytes < 80% available | `megatron_policy_worker.py` → `build_latest_bucket_cache()` | 用实际打包后的 `total_bytes`，而非 per-bucket 大小 |
| `_cache_lock` 贯穿 cache lookup → transport → NCCL teardown | `megatron_policy_worker.py` → `selective_sync_active_cache()` | `with cache._cache_lock:` 覆盖整个 bucket 循环 + sender 侧 NCCL destroy |
| Pipeline 层 init / post-train 调用序列 | `rlix/pipeline/full_finetune_pipeline.py` | init: `build_latest_bucket_cache(-1)` → `promote_active_checkpoint(version=-1)` → `mark_promoted(-1)` |

### 关键设计决策

- **两指针缓存**（`_latest_cached` / `_active_cached`）：比规格要求的单槽 `_cache_ready_step` 更安全，防止并发 build/promote 竞争
- **receiver 侧 IPC 路径不走 CPU 中转**：`cuda_ipc` 模式直接 `rebuild_cuda_tensor()` 得到 GPU tensor，无 GPU→CPU→GPU roundtrip
- **receiver rank mask 用 `self.rank`**：不用 `dist.get_rank()`，因为 ipc_local_ranks 是 vLLM worker 本地 rank，非分布式 rank

---

## Feature 6 — 选择性权重同步（两条刷新路径）

### 规格要求 → 实现位置

| 规格要求 | 实现文件 | 说明 |
|---------|---------|------|
| `coordinator.sync_base_weights_to_active()` — training loop 刷新 active ranks | `rlix/pipeline/coordinator.py` + `rlix/protocol/coordinator.py` | 持 `_resize_sync_lock`，snapshot `_active_infer_dp_ranks`，直接调 `ModelUpdateService.sync_selected_workers()` |
| `_expand_workers()` — expand 时刷新 woken ranks | `rlix/pipeline/full_finetune_pipeline.py` → `_expand_workers()` | 顺序：sync → finalize → **version publish（先于 routing 激活）** → expand_sampler |
| ModelUpdateService 6-phase 同步流程 | `rlix/pipeline/model_update_service.py` → `sync_selected_workers()` | Phase 1: NCCL setup / Phase 2: sender dispatch / Phase 3: receiver teardown / Phase 4: verify |
| IPC vs NCCL broadcast 路由分类 | `model_update_service.py` → `_build_comm_plan_for_sender()` | 按 (node_rank, gpu_rank) 判断是否同一物理 GPU，同 GPU → IPC，跨 GPU → NCCL |
| **CUDA IPC**（同一物理 GPU，不能建 NCCL group） | `megatron_policy_worker.py` → `selective_sync_active_cache()` | `get_handle_from_tensor(staging_buf)` 产生 IPC handle，随 payload 发给 receiver |
| **CUDA IPC receiver**（零拷贝） | `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py` → `update_parameter_in_bucket()` | `rebuild_cuda_tensor(*ipc_args)` 直接拿到 GPU tensor，无 CPU 中转 |
| **NCCL broadcast**（跨 GPU，tp > 1） | `megatron_policy_worker.py` → `selective_sync_active_cache()` | stage CPU→GPU → `dist.broadcast(staging_buf, group=nccl_group)` |
| 动态 NCCL group 创建/销毁 | `megatron_policy_worker.py` → `setup_collective_group()` / `destroy_collective_group()` | sender 在 `_cache_lock` 内 destroy；receiver 侧由 ModelUpdateService Phase 3 触发 |
| 全部 6 个 receiver API | `vllm_backend.py` + `vllm_generation.py` | `setup_collective_group`, `update_parameter_in_bucket`, `broadcast_parameter`, `destroy_collective_group`, `verify_model`, `finalize_weight_update` |
| vllm_generation pass-through 必须 await sub-worker | `vllm_generation.py` 全部 6 个方法 | 每个方法内 `ray.get(futures)` 确保 outer barrier 语义正确 |
| **finalize_weight_update** — pipeline 所有，worker 执行 | `full_finetune_pipeline.py` | sync 返回后，pipeline 对每个 synced rank 调 `finalize_weight_update.remote()`；ModelUpdateService 不调 |
| version publish 必须在 routing 激活**之前** | `full_finetune_pipeline.py` → `_expand_workers()` | `set_weight_version.remote(v)` → `expand_sampler(skip_load=True)` 顺序固定 |
| trajectory collector 版本通知 | `vllm_backend.py` / `grpo.py` / `full_finetune_pipeline.py` | grpo.py 将 collector 注册为命名 Ray actor `rlix:trajectory_collector:{id}`；pipeline 通过 `_get_trajectory_collector()` 懒加载后调 `set_weight_version` |
| port claim 在 teardown 完成后释放，失败时故意泄漏 | `model_update_service.py` | receiver teardown（Phase 3）完成后才 `_release_master_port_claim()`，异常时 finally 不 release |

### 版本号语义

```
train step 3 完成:  _cache_ready_step = 3
active refresh:    _current_weight_version = 3  （无 bump）
                   collector.set_weight_version(3)
later expand:      collector.set_weight_version(3)  （同一版本，无 bump）
```

两条路径刷新的权重相同，版本号相同，避免双重递增。

### transport 模式选择

| 模式 | 场景 | 机制 |
|------|------|------|
| `cuda_ipc` | 同物理 GPU（colocated） | `get_handle_from_tensor()` → IPC handle → `rebuild_cuda_tensor()` |
| `cpu_serialize` | 跨 GPU（默认） | CPU uint8 bucket dict → Ray RPC → `pin_memory().to(device)` |
| NCCL broadcast | 跨 GPU，tp > 1 | `dist.broadcast()` on dynamic group `[sender] + [infer_ranks]` |

> **规格约束**（line 316）：NCCL 无法在同一物理 GPU 的两个进程之间建组。同 GPU 的 colocated worker **必须** 走 CUDA IPC，这是正确性要求，不是性能优化。

---

## 文件索引

### rlix 主仓库（`zhenyulincs/rlix`）

```
rlix/pipeline/bucket_cache.py               BucketRecord, VersionedBucketCache, pack/unpack
rlix/pipeline/bucket_cache_lifecycle.py     BucketCacheLifecycle（版本追踪）
rlix/pipeline/model_update_service.py       ModelUpdateService（Ray actor，6-phase 同步）
rlix/pipeline/coordinator.py               sync_base_weights_to_active()（具体实现）
rlix/pipeline/full_finetune_pipeline.py    _expand_workers, finalize, version publish
rlix/protocol/coordinator.py              抽象协议接口
```

### NeMo 子模块（`zhenyulincs/RL`，分支 `rlix-task2` / `main`）

```
nemo_rl/models/policy/workers/megatron_policy_worker.py
    build_latest_bucket_cache()           — 所有 rank gather，owner 打包存储
    promote_active_checkpoint()           — 切换 active 指针
    selective_sync_active_cache()         — sender 主逻辑（IPC + NCCL）
    setup_collective_group()              — 加入动态 NCCL group
    destroy_collective_group()            — 销毁动态 NCCL group

nemo_rl/models/generation/vllm/vllm_backend.py
    update_parameter_in_bucket()          — receiver IPC 路径（CUDA IPC / cpu_serialize）
    broadcast_parameter()                 — receiver NCCL broadcast 路径
    finalize_weight_update()              — post-bucket hook（FP8 等）
    verify_model()                        — 可选验证
    setup_collective_group()              — receiver 侧加入 NCCL group
    destroy_collective_group()            — receiver 侧销毁 NCCL group

nemo_rl/models/generation/vllm/vllm_generation.py
    （以上 6 个方法的 Ray actor pass-through，每个内部 ray.get(futures) 确保 barrier）

nemo_rl/algorithms/grpo.py
    trajectory_collector 注册为命名 Ray actor: rlix:trajectory_collector:{pipeline_id}
```

---

## 测试文件说明

### 单元测试（无 GPU / Ray）

```bash
python -m pytest tests/test_bucket_cache.py            # BucketRecord pack/unpack
python -m pytest tests/test_bucket_cache_lifecycle.py  # 版本追踪、promote、GC
python -m pytest tests/test_model_update_service.py    # comm plan、finalize 归属
python -m pytest tests/test_nemo_rl_pipeline.py        # _expand_workers 顺序
# 期望：53 passed
```

### Gate 2.5 集成测试（需要 4× GPU，torchrun）

```bash
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1   # PCIe 硬件（无 NVLink）

# 1. NCCL destroy/re-init 稳定性（2 GPU）
torchrun --nproc-per-node=2 tests/integration/test_gate2_5_nccl_destroy.py

# 2. NCCL proper-subset group broadcast（4 GPU）
#    验证: group=[0,2,3] 是 world=[0,1,2,3] 的真子集，不会 hang
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_selective_sync.py

# 3. Megatron TP=2 训练 + per-shard NCCL 同步（4 GPU）
#    group[0,2] 同步 shard0，group[1,3] 同步 shard1
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_megatron_tp.py

# 4. Qwen2.5-0.5B 真实模型训练 + 同步（4 GPU，需 HF 缓存）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_qwen_train_sync.py

# 5. 双 pipeline 交替同步，A≠B 权重隔离（4 GPU）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_full.py

# 6. F6 顺序验证：sync→finalize→version_publish→activate（4 GPU）
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_feature6.py
```

全部 6 个应输出 `ALL GATE 2.5 * CHECKS PASSED`，exit 0。

### F6.3 / F4.4 / F6.6 专项测试（单 GPU）

```bash
# CUDA IPC 跨进程零拷贝传输
python tests/integration/test_gate2_5_cuda_ipc.py

# bucket_size_bytes 配置检查（未配置 → RuntimeError；过大 → RAM fail-fast）
python tests/integration/test_gate2_5_bucket_size_guard.py

# version publish 顺序验证（set_weight_version 在 expand_sampler 之前）
python tests/integration/test_gate2_5_trajectory_collector.py
```

### 快速使用示例

```python
# 在测试或调试时手动构造 bucket cache 并验证 pack/unpack
import torch
import sys
sys.path.insert(0, ".")  # rlix repo root

from rlix.pipeline.bucket_cache import (
    _bucket_named_tensors,
    unpack_bucket_record,
    VersionedBucketCache,
)

# 1. 打包
named_tensors = [("fc1.weight", torch.randn(256, 256)),
                 ("fc2.weight", torch.randn(256, 256))]
record = _bucket_named_tensors(named_tensors)
print(f"packed: {record.cpu_uint8_bucket.numel()} bytes")

# 2. 缓存
cache = VersionedBucketCache()
cache.build_latest(step=1, buckets=[record])
cache.promote(version=1)

# 3. 读取（持锁）
with cache._cache_lock:
    buckets = cache.get_active_buckets()

# 4. 解包还原
for bucket in buckets:
    for name, tensor in unpack_bucket_record(bucket):
        print(f"  {name}: {tensor.shape}, {tensor.dtype}")

# 5. 验证 bit-exact
import hashlib
def h(t): return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:8]

orig = {name: h(t) for name, t in named_tensors}
recv = {name: h(t) for name, t in unpack_bucket_record(buckets[0])}
assert orig == recv, f"mismatch: {orig} vs {recv}"
print("bit-exact ✓")
```

---

## 已知待实现项

| 项目 | 原因 |
|------|------|
| `wake_up_partial()` / `activate_dp_ranks()` | Feature 2（VllmGeneration sleep/wake API）尚未实现，当前用 ROLL 的 `expand_sampler(skip_load=True)` 等效替代 |
| ZMQ ping-pong 双缓冲 IPC | NeMo RL 环境未安装 `zmq`；Ray RPC 实现等效功能 |
| `_cache_ready_step` 在 sender `_cache_lock` 下发布 | 跨 Ray actor 架构约束：training worker 锁 ≠ pipeline 的 lifecycle 锁，不可共享 |

---

## 环境配置

```bash
# 克隆（含子模块）
git clone https://github.com/zhenyulincs/rlix.git --recurse-submodules
cd rlix

# 安装依赖
pip install uv && uv sync

# 必须显式配置（无隐式默认值）
export RLIX_BUCKET_SIZE_BYTES=$((256 * 1024 * 1024))   # 256 MB per bucket
export RLIX_MODEL_UPDATE_TRANSPORT=cpu_serialize         # 或 cuda_ipc（同 GPU colocated）
```
