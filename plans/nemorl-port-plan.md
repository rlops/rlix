# NeMo RL 整合进 RLix — 方案

---

## GOAL

**一句话：让 NeMo RL 的 async GRPO 训练可以被 RLix 调度器管理，实现多个训练 pipeline 之间的 GPU 时分复用。**

核心原语是 **partial overlapping**：inference worker 占据的 GPU 是 training worker 的超集。当需要 training 时，重叠部分的 inference worker "sleep" 释放 GPU 给 training；非重叠 GPU 上的 inference 继续运行（async 核心价值）。training 完成后 worker "wake_up" 恢复 inference。

---

## 范围

### In Scope

- **推理引擎：仅 vLLM**
- **训练后端：仅 Megatron**（`megatron_cfg.enabled=true`）
- **算法：异步 GRPO 优先**（`async_grpo_train()`, `grpo.py:2365`）
  - `max_trajectory_age_steps` 控制 off-policy 程度，等价于 ROLL 的 `async_generation_ratio` (`base_config.py:453`)
  - async 模式要求 non-colocated inference (`grpo.py:2448`)，天然适配 Partial Overlap
- **Pipeline 类型：仅 Full Finetune**
- **资源模式：Partial Overlap**
- **并行度（按 backend 分开说明）**。RLix `cluster_tp_configs` 表示的是 **backend-specific 的 per-worker GPU width**，不是 cluster 总 GPU 数。当前 scope 下：
  - **Megatron `actor_train`**：`TP / PP / CP / EP` 任意组合 in scope。注册时 `tp_size=1`（ROLL 行为：Megatron workers 各自占 1 GPU，并行度通过 NCCL groups 实现，`worker_config.py:231` 强制 `num_gpus_per_worker=1`）
  - **vLLM `actor_infer`**：仅 `TP` in scope。注册时 `tp_size=vllm_tp`
  Scheduler 只对 generation cluster 计算 `max_dp_workers`，training cluster 的 `tp_size` 仅用于 device_mapping 验证。EP 不影响注册，只影响 Feature 4 的权重 cache build（PP collective gather 中 EP-aware 处理，见 `model_update.py:128-158`；cache owner 存储 gather 后的完整模型）。这不是 RLix 的全局硬编码规则；如果未来接入别的 training backend，应由对应 adapter 定义其 per-worker GPU width。Gate 测试受限于 2 GPU，仅覆盖 `tp<=2, pp=1, cp=1, ep=1`；更高维组合依赖更大机器验证。

### Out of Scope

- ❌ SGLang backend（sleep/wake 是空实现 TODO）
- ❌ DTensor/FSDP2 训练后端（`dtensor_cfg`）
- ❌ Multi-LoRA pipeline
- ❌ Megatron generation backend
- ❌ DPO / SFT / Distillation
- ❌ 同步 `grpo_train()` (`grpo.py:1306`) — partial overlap 对 sync 模式无价值
- ❌ NeMo-Gym 环境（`_should_use_nemo_gym` 路径）— 当前聚焦 calculator 标准路径（`run_async_multi_turn_rollout`）。NeMo-Gym resize 方案见下方"Future: NeMo-Gym shard preemption"

---

## 方案：逐 Feature 从 ROLL 移植到 NeMo RL

以下每个 Feature 是 ROLL + RLix 所需的所有独立能力。对每个 Feature 说明：ROLL 怎么做的 → NeMo RL 现状 → 移植方案。

---

### Feature 1: vLLM sleep/wake with level=2

**作用：** 释放 inference worker 的 GPU VRAM（weights + KV cache），腾给 training worker 使用。

#### ROLL 怎么做的

- `roll/third_party/vllm/__init__.py:42` — 创建 vLLM 引擎时 `enable_sleep_mode=True`
- `roll/distributed/strategy/vllm_strategy.py:582` — `offload_states(level)` 调用 `self.model.offload_states(self.sleep_level)`
- `roll/distributed/strategy/vllm_strategy.py:569` — `load_states()` 调用 `self.model.load_states()`
- `roll/third_party/vllm/worker.py:500` — vLLM < 0.8.5 的 buffer 兼容（保存/恢复 named_buffers）
- sleep_level 从 config 传入，RLix 模式下为 2

#### NeMo RL 现状

- `vllm_worker.py:986` — `sleep()` 存在，但 `self.llm.sleep(level=1)` **硬编码为 1**（line 1009）
- `vllm_worker_async.py:1135` — `sleep_async()` 同样硬编码 level=1（line 1154）
- `vllm_generation.py:733-782` — `prepare_for_generation()` / `finish_generation()` 仅用于 colocated 场景
- 使用 `run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"]` — 每 DP shard 仅在 TP rank-0 worker 执行
- NeMo RL 用 vLLM 0.17.0 >> 0.8.5，不需要 buffer 兼容逻辑
- **vLLM TP NCCL communicator 不受影响**（D5 已验证）：vLLM native sleep/wake 只管 CuMem，TP process groups 保持有效
- **Training 后端：仅 Megatron**（`megatron_cfg.enabled`，`lm_policy.py:86`），有 TP/PP/CP NCCL groups
- **`offload_nccl` 是硬性要求，不只是 Gate 3 验证项**：在 RLix 中，"release GPUs" 是 scheduler 记账状态，不是 actor teardown。长生命周期 training actors 会保持 NCCL communicator buffers 驻留在 GPU 上，除非显式销毁。RLix 当前通过 `_validate_offload_nccl` (`coordinator.py:136`) 强制 `offload_nccl=True`。NeMo coordinator 分支（Feature 8）必须保留等价验证，且 NeMo RL 的 Megatron training workers 需要在 training 结束后显式销毁 NCCL communicator groups 释放 GPU VRAM，否则 inference wake_up 会 OOM

#### 移植方案

1. `vllm_worker.py:1009` — `self.llm.sleep(level=1)` → `self.llm.sleep(level=self._sleep_level)`，`_sleep_level` 从 config 传入
2. `vllm_worker_async.py:1154` — 同上
3. 新增 `enable_sleep_mode=True` 到 vLLM 引擎创建参数（如果 NeMo RL 未默认启用）
4. 改动量：~10 行

---

### Feature 2: Selective DP shard sleep/wake (partial sleep/wake)

**作用：** 只 sleep 重叠 GPU 上的 inference worker，非重叠 GPU 继续 generation。

#### ROLL 怎么做的

- `roll/pipeline/base_worker.py:527` — `InferWorker.offload_states_partial(target_dp_ranks)` — 对指定 DP ranks 调用 `offload_states`
- `roll/pipeline/base_worker.py:494` — `InferWorker.load_states_partial(target_dp_ranks)` — 对指定 DP ranks 调用 `load_states`
- `roll/distributed/scheduler/generate_scheduler.py:1885` — `shrink_workers(dp_ranks)` — 从 active routing 中移除 + 调用 offload
- `roll/distributed/scheduler/generate_scheduler.py:1973` — `expand_workers(dp_ranks)` — 调用 load + 恢复 active routing
- `roll/distributed/scheduler/rollout_scheduler.py:1088,1138` — `shrink_sampler/expand_sampler` 包装层

#### NeMo RL 现状

- **不存在** partial sleep/wake。`prepare_for_generation` / `finish_generation` 是全量操作
- 无 shard-leader 级别的选择性执行机制
- `RayWorkerGroup.get_dp_leader_worker_idx(dp_shard_idx)` (line 404) 可用于定位 DP shard 的 leader worker
- `_worker_metadata[i]["dp_shard_idx"]` 可用于找到 DP shard 的所有 worker（含 TP tied workers）

#### 移植方案

**执行粒度决策（必须先明确）：**

NeMo RL 现有 sleep/wake 用 `run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"]`（只在 DP leader 上执行）。这意味着 vLLM 的 `LLM.sleep()` 在 leader 上调用后，通过 `collective_rpc` 内部传播到同 TP group 的其他 worker。

**选择方案：DP-leader-only 调用**（与 NeMo RL 现有模式一致）。理由：
- NeMo RL 的 `prepare_for_generation` / `finish_generation` 已验证此路径工作正确
- vLLM 的 `collective_rpc` 负责 TP 内部传播，无需外部逐 worker 调用
- `run_on_dp_shard_leaders` 实现为"对指定 DP ranks 的 leader workers 执行"，**不是**对所有 TP-tied workers 执行

因此 `run_on_dp_shard_leaders` 语义 = 对每个目标 DP rank 调用 leader worker，与现有 `run_all_workers_single_data(..., run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"])` 一致，只是限定了 DP rank 子集。

1. `VLLMGeneration` 新增 `sleep_partial(dp_ranks, level=2)` 和 `wake_up_partial(dp_ranks)`
2. `VLLMGeneration` 新增 `_active_dp_ranks: Set[int]` 状态追踪
3. `VLLMGeneration` 新增 `_preempted_shards: Set[int]` — abort 窗口期间的子集，用于 error 分类
4. `RayWorkerGroup` 新增 `run_on_dp_shard_leaders(dp_ranks, fn, *args, **kwargs)` — 对指定 DP shard 的 **leader worker** 执行（vLLM 内部传播到 TP peers）
5. 内部仅需 leader index 列表：`[self.get_dp_leader_worker_idx(r) for r in dp_ranks]`。**不新增** `_get_all_workers_for_dp_shard()`

**`sleep_partial` 必须 abort-drain-sleep，不能直接 sleep in-flight 请求：**

```python
async def sleep_partial(self, dp_ranks: List[int], level: int = 2):
    # 1. 阻止新请求分发到这些 shards
    self._preempted_shards |= set(dp_ranks)

    # 2. Worker 内部 abort 所有 running requests（无需从 generation 层传 request IDs）
    self.run_on_dp_shard_leaders(dp_ranks, "abort_all_requests")

    # 3. Drain：轮询 vLLM 已有的 engine metric 直到 idle
    #    vllm_worker_async.py:241 已在采集 vllm:num_requests_running
    for rank in dp_ranks:
        await self._wait_engine_idle(rank)  # poll worker.is_idle() → num_requests_running == 0

    # 4. Engine idle，安全 sleep
    self.run_on_dp_shard_leaders(dp_ranks, "sleep", level=level)
    self._active_dp_ranks -= set(dp_ranks)
```

**不需要 per-request tracking** — 不引入 `_inflight_requests: Dict[int, Set[str]]`，不修改 request ID 生成路径。Worker 内部通过 engine API 获取 running request IDs 并 abort，drain 用现有 `vllm:num_requests_running` metric 确认 idle。

**不能跳过 abort 直接 sleep** — vLLM engine 在处理请求时被 sleep 会导致对已 offload 的 GPU memory 的访问，crash 整个 worker。abort 先让 engine 干净地丢弃请求，drain 确认 engine idle，然后 sleep 安全执行。

被 abort 的请求在 caller 侧收到异常，`_async_generate_base` 检查 `dp_rank in self._preempted_shards` 分类为 `ShardPreemptedError`（见 Feature 3）。

6. 改动量：~150 行

---

### Feature 3: Generation routing skip sleeping shards

**作用：** Generation 只分发到 active DP shards，跳过 sleeping 的。

#### ROLL 怎么做的

- `generate_scheduler.py` 的 `active_dp_ranks` set 控制 routing — shrink 时移除，expand 时添加
- `RequestScheduler` 的 `_select_dp_rank()` 只从 active_dp_ranks 中选择
- shrink 时还会 abort 正在 sleeping shard 上执行的 in-flight requests（abort + retry 语义）

#### NeMo RL 现状

- **Sync generation** (`generate`, line 465): `run_all_workers_sharded_data` 无条件分发到所有 DP shards
- **Async generation**：`generate_async()` 调用私有 helper `_async_generate_base()`；round-robin 逻辑在 `_async_generate_base`（`vllm_generation.py:559`）里，将请求发到单个 DP shard（`current_generate_dp_shard_idx` mod `dp_size`）
- `AsyncTrajectoryCollector` 使用 `generate_async` — 每次调用只用一个 DP shard

#### 移植方案

async 模式优先，两部分改动：

**1. Round-robin 跳过 sleeping shards：**

```python
# 在 _async_generate_base round-robin 中跳过 sleeping shards
# 状态关系：_active_dp_ranks 是 canonical 集合（Feature 2 定义）
#   sleeping = all_dp_ranks - _active_dp_ranks（派生，不单独存储）
#   _preempted_shards ⊆ sleeping（abort 窗口期间的子集，用于 error 分类）
while not self._active_dp_ranks:
    await self._wait_for_active_dp_shards()
while self.current_generate_dp_shard_idx not in self._active_dp_ranks:
    self.current_generate_dp_shard_idx = (self.current_generate_dp_shard_idx + 1) % self._dp_size
```

`_wait_for_active_dp_shards()` 由 `activate_dp_ranks()` / `sleep_partial()` 维护的 condition/event 驱动。语义是“collector 在 shrink-to-zero 期间阻塞等待 scheduler expand”，**不是**抛异常让后台线程崩掉。

**2. In-flight generation preemption：abort-drain-sleep + `ShardPreemptedError` 信号机制**

Shrink 时 **不能直接 sleep in-flight 请求** — vLLM engine 在处理请求时被 sleep 会访问已 offload 的 GPU memory，crash worker。必须 abort-drain-sleep（见 Feature 2 `sleep_partial` 实现）。

**信号机制（abort → `ShardPreemptedError` 传播路径）：**

**不需要 per-request ID tracking。** Abort 和 drain 都在 worker 内部完成（见 Feature 2），`VLLMGeneration` 层不需要知道具体 request ID。需要新增：

1. **`vllm_worker_async.py` 新增 `abort_all_requests()`**：worker 内部从 engine 获取所有 running request IDs 并调用 `engine.abort()`，无需外部传入 IDs
2. **`vllm_worker_async.py` 新增 `is_idle() -> bool`**：检查已有的 `vllm:num_requests_running` metric（line 241），返回是否为 0
3. **Error 转换**：abort 后 caller 的 `await worker_task` 收到异常。**不能 `except Exception:` 全吞**。在 `_async_generate_base` 的 result handler 中：检查 `dp_rank in self._preempted_shards`，如果是则 raise `ShardPreemptedError`；否则原样抛出（真实 bug 不吞）

```python
# _async_generate_base 中，request 完成/失败时的处理
try:
    result = await worker_task
except Exception as e:
    if dp_rank in self._preempted_shards:
        raise ShardPreemptedError(dp_rank) from e
    raise  # 非 preempt 错误，原样抛出
```

**关键约束**：`_preempted_shards` 标记在 abort 之前设置（Feature 2 `sleep_partial` step 1），所以 error 转换不会有”shard 已 sleep 但未标记”的窗口。`wake_up_partial` 清除标记。

**3. Targeted retry（放在 `_async_generate_base`）**

最简单且与调用路径最一致的做法，是把 retry 直接放进 `_async_generate_base`：它已经是所有 async generation 的单一分发点，single-turn / multi-turn 最终都经过这里。不再新增 rollout wrapper，也不改 `rollouts.py`。

示意（直接包住 `_async_generate_base` 内部现有 dispatch + await 逻辑）：

```python
# In _async_generate_base
# 注意：这不是 fail-fast 的例外 — 是 shard re-dispatch（重新分发到不同 shard），
# 不是 retry 同一个失败操作。语义等价于 ROLL RequestScheduler 的 request migration。
MAX_SHARD_REDISPATCH_ATTEMPTS = 3  # 上限 = dp_size 即可（每个 shard 最多试一次）
for attempt in range(MAX_SHARD_REDISPATCH_ATTEMPTS):
    try:
        (updated_message_log, generated_tokens, input_lengths, gen_metrics,
        ) = await async_generate_response_for_sample_turn(...)
        break
    except ShardPreemptedError:
        if attempt == MAX_SHARD_REDISPATCH_ATTEMPTS - 1:
            raise
        # Shard was aborted — re-dispatch to next active shard via round-robin
        continue
```

**已完成 turns 的工作完全保留**（env 交互结果 + message_log 累积），只重做 aborted turn 的 generate 调用。比 ROLL 的 abort+retry 简单得多 — ROLL 需要在 `RequestScheduler` 层面做 request 级别的迁移，而 NeMo RL 的 retry 粒度是单个 turn 的 generate call。

sync `generate` 的 sharded dispatch 修改 out of scope（sync 模式不适配 partial overlap）。

改动量：~50 行（routing skip + error 转换 + `_async_generate_base` 内 retry；无 per-request tracking）

---

### Feature 4: Training-side weight caching (CPU bucket cache + `_cache_ready_step`)

**作用：** Training 完成后，在 training worker 侧缓存最新权重到 CPU，供 expand 时快速同步到 inference worker。

#### ROLL 怎么做的

- `roll/distributed/executor/worker.py:363` — `build_latest_bucket_cache(checkpoint_version)` — 将当前模型参数序列化为 CPU bucket cache（raw bytes + metadata），存储在 training worker 上
- `roll/distributed/executor/worker.py:387` — `promote_active_checkpoint(checkpoint_version)` — 标记哪个 version 是当前 active 的，供下次 expand 使用
- **调用时机**（`rlix/pipeline/full_finetune_pipeline.py`）：
  - Init 阶段 (line 289-301): `build_latest_bucket_cache(-1)` → `promote_active_checkpoint(-1)` — 初始 base model cache
  - 每次 train_step 后 (line 1008-1013): `promote_active_checkpoint(checkpoint_version)` — 标记训练后的最新权重
- 底层实现在 `roll/distributed/strategy/megatron_strategy.py:1994` — `promote_active_checkpoint` 将 bucket cache 的 active pointer 切换到新 version

#### NeMo RL 现状

- **不存在**。NeMo RL 的 `refit_policy_generation()` (`grpo.py:1097`) 直接做权重传输：
  - ZMQ IPC 路径 (line 1157): `policy.stream_weights_via_ipc_zmq()` → `policy_generation.update_weights_via_ipc_zmq()`
  - NCCL broadcast 路径 (line 1172): `policy.broadcast_weights_for_collective()` → `policy_generation.update_weights_from_collective()`
- 传输是**同步且全量**的 — 每次 refit 都从 training actor 实时读取并传输所有参数

#### 移植方案

**问题：refit 时训练权重在哪里？**

NeMo RL 的 `refit_policy_generation` 在发送时需要训练权重 **在 GPU 上**：
- ZMQ IPC 路径（colocated）：从 GPU 创建 CUDA IPC handle 发送（`utils.py:272,295`）
- NCCL broadcast 路径（non-colocated）：从 GPU 上的模型参数 broadcast（`megatron_policy_worker.py:1105`）

在 partial overlap 中，训练 GPU = 重叠 GPU = inference 需要 wake_up 的 GPU。这造成 **OOM**：
- Expand 需要 inference workers wake_up（占用重叠 GPU 的 VRAM）
- Refit 需要 training weights 留在重叠 GPU 上发送
- 两者不能同时占用同一 GPU 的 VRAM → OOM

**这正是 ROLL 引入 bucket cache 的原因：**
- `build_latest_bucket_cache` 在训练完成后将权重 **缓存到 CPU**
- `offload_states` 释放训练 GPU VRAM
- Expand 时 inference wake_up 占用 GPU
- `ModelUpdateService.sync_selected_workers` 从 **CPU cache** 发送权重到 inference workers

**结论：路径 A（复用原生 refit）不可行** — refit 要求发送端权重在 GPU 上，而 GPU 已被 inference wake_up 占用。

**方案：CPU bucket cache + selective sync + dual transport（参照 ROLL ModelUpdateService）**

需要 selective sync 和 IPC path 的两个原因：

1. **Selective sync 是正确性要求** — 全量 broadcast 要求所有 inference workers（含非重叠 GPU 上正在 generation 的 workers）参与 NCCL collective → 必须暂停所有 generation → 违背 async 核心价值。Selective sync 只推送到刚 woken 的 overlap shards，非重叠 shards 继续 generation 不受影响。

2. **CUDA IPC 是正确性要求** — partial overlap 中，training worker 和 inference worker 在同一物理 GPU 上（overlap GPUs）。NCCL 无法对同一 GPU 上的两个 rank 建组。必须走 CUDA IPC zero-copy 路径。

**NeMo RL 已有两条 transport 实现：**
- **CUDA IPC**（colocated 路径）：sender `get_handle_from_tensor()` → ZMQ → receiver `rebuild_cuda_tensor_from_ipc()` → `model_runner.model.load_weights()`（`vllm_backend.py:164`, `policy/utils.py:250`）
- **NCCL broadcast**（non-colocated 路径）：`packed_broadcast_producer/consumer` via `model_update_group`（`packed_tensor.py:39,98`）

**两条路径都已完整实现且经过验证**。我们只需要构建 routing 层（`ModelUpdateService`）决定每个 target device 走哪条路径 — transport 本身无需修改。

**实现方案：**

引入简化版 `ModelUpdateService`（参照 ROLL `rlix/pipeline/model_update_service.py`），复用 NeMo RL 现有 transport（versioning 安全性分析见下文）：

1. 每次 `train_step` 后，构建 CPU bucket cache（**单 cache owner 模式，与 ROLL 一致**）：
   - **所有 TP/PP/CP/EP ranks 参与 collective gather**（`gather_all_hf_weights`，内部使用 PP collectives 将所有 pipeline stages 的权重汇聚）。EP-aware：expert 参数通过 `get_expert_tensor_parallel_group()` + `get_expert_model_parallel_group()` gather；non-expert 参数通过 `get_tensor_model_parallel_group()` gather（`model_update.py:128-158`）。
   - **仅 cache owner（pp0/dp0/tp0/cp0）存储 gather 后的完整模型 CPU buckets**（`megatron_strategy.py:1049-1065`）。其他 ranks 参与 collective 但丢弃结果（drain generator to keep collective moving，`megatron_strategy.py:1918-1939`）。
   - 打包到 **CPU bucket buffer**（`device=”cpu”`），cache 是完整模型（非 per-shard）
   - 启动时估算 `total_cpu_cache_bytes`（cache owner 上的**单份完整模型**大小），超过 host RAM budget 直接 fail fast
2. Offload training GPU（释放全部 VRAM）
3. Expand 时：wake_up target inference workers（仅 overlap shards）
4. `ModelUpdateService.sync_selected_workers(tgt_dp_ranks)` — **单 sender**（cache owner）推送到 woken shards：
   - Sender 由 `_select_global_sender_rank()`（`model_update_service.py:90`）确定 — 返回 pp0/dp0/tp0/cp0
   - **关键约束：不能”整模型回灌到 sender GPU 再发”**。必须 **逐 bucket CPU→GPU stage**，每个 bucket 传完立即释放 staging buffer，控制 peak VRAM
   - `bucket_size_bytes` 必须是显式配置，不是隐式默认值；初始化时用”wake_up 后剩余 VRAM”做上界检查，确保 `bucket_size_bytes + transport scratch` 小于 overlap GPU 的可用余量
   - **同 GPU**（overlap GPU，training 和 inference colocated）→ 逐 bucket stage → 复用现有 **ZMQ IPC** 路径（`stream_weights_via_ipc_zmq` / `update_weights_via_ipc_zmq`）
   - **跨 GPU**（如果 target worker 有 TP 跨 GPU 的 rank）→ 逐 bucket stage → 复用现有 **NCCL broadcast** 路径（`packed_broadcast_producer/consumer`）
5. 非重叠 GPU 上的 inference workers **不参与**，继续 generation

**Cache 安全性：3 个不变量**

跳过 ROLL 的完整 checkpoint versioning，采用单槽 `_cache_ready_step`。安全性依赖：
1. **单 writer**：training hook 写 `_cache_ready_step`（`after_training` 中 `build_cpu_bucket_cache(step)` 完成后原子更新）
2. **单 reader 路径**：expand 读 `_cache_ready_step`（`_expand_workers` → `sync_selected_workers`）
3. **顺序契约**：`before_training(step+1)` 阻塞到前一个 `after_training(step)` 触发的 expand 完成后才返回（由 `request_cluster_gpus` 的 blocking `ray.get` 保证）。Gate 3 需验证此不变量。

**comm_plan 分类逻辑** 复用 ROLL 的 `_build_comm_plan_for_sender()`（为单个 cache owner 构建 plan，按 `(node_rank, gpu_rank)` 分类 IPC vs broadcast）。

改动量：~200 行（简化版 ModelUpdateService routing 层 + CPU bucket build。transport 实现复用现有代码）

---

### Feature 5: Scheduler-driven shrink/expand (resize_infer + generation allocation lifecycle)

**作用：** RLix scheduler 通过 coordinator 异步驱动 inference cluster 的 shrink/expand，pipeline 不直接控制。

#### ROLL 怎么做的

- **Coordinator** (`rlix/pipeline/coordinator.py:502-547`): `resize_infer(dp_ranks_to_remove, dp_ranks_to_add)` — 在 `_resize_sync_lock` 下调用 pipeline actor 的 `resize_infer`
- **Pipeline** (`rlix/pipeline/full_finetune_pipeline.py:1062-1071`): `resize_infer()` 转发到 `_shrink_workers()` 或 `_expand_workers()`
- **_shrink_workers** (line 440-449): `rollout_scheduler.shrink_sampler(dp_ranks, skip_offload=False)` — 路由移除 + vLLM sleep
- **_expand_workers** (line 451-463): `rollout_scheduler.expand_sampler(dp_ranks, skip_load=False)` — vLLM wake_up + `ModelUpdateService.sync_selected_workers()` 权重同步 — **expand 和 weight sync 是原子操作**
- `_resize_sync_lock` (line 519) 保护 resize 和 weight sync 互斥，防止 race condition

- **Pipeline run() 循环与 scheduler 的交互**（`full_finetune_pipeline.py`）：
  - Pipeline 调用 `_request_cluster_gpus(actor_train, ACTOR_TRAINING)` 请求 training GPU
  - Scheduler **异步** 调用 `coordinator.resize_infer(remove=overlap_ranks)` 做 shrink
  - Pipeline 训练完成后调用 `_notify_release_cluster_gpus(actor_train)` 释放
  - Scheduler **异步** 调用 `coordinator.resize_infer(add=overlap_ranks)` 做 expand + weight sync
  - Pipeline 不直接调用 shrink/expand — 完全由 scheduler 驱动

#### NeMo RL 现状

- **不存在**任何 RLix / scheduler 集成。`grpo.py` 中无 hook、无外部控制面
- `async_grpo_train()` (line 2365) 是闭环循环 — generation 在 `AsyncTrajectoryCollector` 后台运行，training loop 顺序执行

#### 移植方案
NeMo RL 适配器需要实现 `NemoRLFullFinetunePipeline`，包含：

1. **`actor_infer` 必须成为 RLix 的一等 `GENERATION` cluster**，不能只在训练前后请求 `actor_train`
   - RLix scheduler 只会对“已 active / pending 的 generation cluster”做 gap-ratio planning、background rebalance、以及 scheduler-driven `resize_infer`
   - 因此需要给 NeMo RL 增加显式的 generation allocation lifecycle：至少在 async collector 启动时发起一次 `request_gpus(cluster_id=actor_infer, priority=GENERATION, step_target_estimate=...)`，并在 pipeline 结束时释放
   - **不需要每个 training step 都 release / re-request generation**；async collector 是长生命周期，推荐保留一个长期的 `actor_infer` allocation，由 progress + scheduler 背景 rebalancing 来调节 active DP ranks

2. **`resize_infer(dp_ranks_to_remove, dp_ranks_to_add)`** — 与 ROLL 的 pipeline adapter 一致
   - shrink: `vllm_generation.sleep_partial(dp_ranks, level=2)`
   - expand: `vllm_generation.wake_up_partial(dp_ranks)` + refit（见 Feature 6）

3. **在 `async_grpo_train()` 中插入 training-side hook**（2 个）：
   - `before_training(step)` → `request_cluster_gpus(actor_train)` — scheduler 异步 shrink
   - `after_training(step)` → `build_cpu_bucket_cache(step)` + `self._cache_ready_step = step` + `notify_release_cluster_gpus(actor_train)` — scheduler 异步 expand + refit
   - generation 侧不是“无 hook”，而是需要独立的 collector lifecycle hook：`on_generation_start()` / `on_generation_stop()`（或等价入口）来持有/释放 `actor_infer` allocation，并持续上报 demand（见 Feature 9）
   - **ownership**：`NemoRLFullFinetunePipeline` 持有 `trajectory_collector` actor handle 和 `_current_weight_version`；`resize_infer()` / `_expand_workers()` 在同一个 pipeline actor 内更新它们，避免 training loop 和 resize path 各自维护版本状态

4. **NemoRLConfigBridge** — 将 NeMo RL config 转换为 coordinator 期望的 config 对象，**并在同一文件中提供声明式 registration helper**（生成 `cluster_device_mappings` / `cluster_tp_configs`）。共享 `PipelineCoordinator` 路径会读取以下属性（`coordinator.py:197-252`），bridge 必须全部提供：

   | 属性 | 来源 | 用途 |
   |------|------|------|
   | `actor_train` | NeMo Megatron config | `_validate_config_schema` |
   | `actor_infer` | NeMo vLLM config | `_validate_config_schema` |
   | `actor_train.device_mapping` | 从 `cluster_device_mappings["actor_train"]` | `_validate_offload_nccl` 用于判断 GPU-active（line 153-154）；**缺失则 offload_nccl 校验被静默跳过** |
   | `actor_infer.device_mapping` | 从 `cluster_device_mappings["actor_infer"]` | 同上 |
   | `actor_infer.strategy_args.strategy_name` | 强制 `"vllm"` | `_validate_vllm_sleep_level` 前置检查（`coordinator.py:122`）；**缺失则 validator 静默 return，sleep_level 校验被完全跳过** |
   | `actor_infer.strategy_args.strategy_config.sleep_level` | 强制 =2 | `_validate_vllm_sleep_level` |
   | `actor_train.offload_nccl` / `actor_infer.offload_nccl` | 强制 =True | `_validate_offload_nccl` |
   | `verify_model_after_sync` | 默认 False | Post-sync weight verification |
   | `num_gpus_per_node` | NeMo cluster config | `RollResourceManagerProxy` 构造 |
   | `pipeline_cls` | `"rlix.pipeline.nemo_rl_pipeline.NemoRLFullFinetunePipeline"` | `create_pipeline_actor` 动态加载 |

   如果遗漏任何属性，mixed-mode 启动时 coordinator 会在 ROLL pipeline 正常通过但 NeMo RL pipeline 在同一代码路径上 AttributeError 崩溃

5. **Init bootstrap 序列**（参照 `full_finetune_pipeline.py:274-431`）：
   ```
   initialize_pipeline():
     proxy = RollResourceManagerProxy(num_gpus_per_node=...)  # 查找 shared PG singleton

     # Phase 1: 在 scheduler INITIALIZATION 优先级下初始化 training
     request_cluster_gpus(actor_train, INITIALIZATION)
     # world_size = total Megatron workers, NOT dp_size.
     # Megatron: 1 GPU per worker (worker_config.py:231), so workers = len(train_devices) = dp*tp*pp*cp*ep
     train_pg = proxy.allocate_placement_group(world_size=len(train_devices), device_mapping=train_devices)
     policy.initialize(pg_alloc=train_pg)  # Megatron workers 调度到 shared PG
     build_cpu_bucket_cache(-1)            # 基础模型权重 → CPU cache
     self._cache_ready_step = -1           # 初始 base cache 标记为 ready，供首次 expand 读取
     offload_training_gpu()                # 释放 training GPU VRAM
     destroy_nccl_groups()                 # offload_nccl 等价
     release_cluster_gpus(actor_train)

     # Phase 2: 在 scheduler INITIALIZATION 优先级下初始化 inference
     request_cluster_gpus(actor_infer, INITIALIZATION)
     infer_pg = proxy.allocate_placement_group(world_size=infer_dp_size, device_mapping=infer_devices)
     policy_generation.initialize(pg_alloc=infer_pg)  # vLLM workers 调度到 shared PG
     offload_inference_gpu()               # vLLM sleep(level=2)
     release_cluster_gpus(actor_infer)

     # Phase 3: 创建 ModelUpdateService + shrink-to-zero
     create_model_update_service(src=policy, tgt=policy_generation)
     shrink_all_dp_ranks()                 # 禁用所有 generation routing，等 scheduler grant
   ```
   **注意：RLix 模式下不创建 `RayVirtualCluster`（见 Feature 12）。不做这个 bootstrap，首次 expand 没有 CPU cache → inference workers 醒来后没有权重。**
   **保持 ROLL 对齐：保留显式 `initialize_pipeline()` + `_ensure_initialized()` 形状。** Coordinator 创建 pipeline actor 时不做重初始化；pipeline 的公开入口（`resize_infer()` / train hook path）先走 `_ensure_initialized()`，与现有 `RollFullFinetunePipeline` 生命周期一致。

改动量：~600 行（pipeline adapter + init bootstrap + generation lease management + merged config/registration bridge）

---

### Feature 6: Expand + refit atomic under resize lock

**作用：** Expand（wake_up sleeping workers）和权重同步必须原子执行，防止 woken workers 用 stale weights 服务 generation。

#### ROLL 怎么做的

- `_expand_workers()` 调用 `rollout_scheduler.expand_sampler(dp_ranks, skip_load=False)` — 内部原子执行 wake_up + `ModelUpdateService.sync_selected_workers()`
- 整个 expand 在 `coordinator._resize_sync_lock` 下运行（`coordinator.py:519-546`）
- Training loop 中 train_step 后立即调用 `promote_active_checkpoint`（line 1008-1013），让 cache 就绪供 expand 使用（ROLL 的双槽 versioning；NeMo RL 简化为单槽 `_cache_ready_step` 指针）
- Expand 时 `sync_selected_workers` 只推送到刚醒来的 dp ranks（selective sync）

#### NeMo RL 现状

- `async_grpo_train` 的 refit 流程 (`grpo.py:2860-2880`):
  ```
  trajectory_collector.prepare_for_refit()    # 暂停新 generation
  refit_policy_generation(policy, policy_generation)  # 全量权重同步
  trajectory_collector.resume_after_refit()   # 恢复 generation
  ```
- `prepare_for_refit` 在 `in_flight_weight_updates=True` 时不等 pending generation 完成（`async_utils.py:558-567`）
- refit 对所有 worker 全量执行 — 无 selective sync

#### 移植方案

Expand path 的 `resize_infer(dp_ranks_to_add)` 实现。**版本状态由 pipeline actor 持有**，并且 **collector version 必须先更新、后激活 routing**：

```python
def _expand_workers(self, dp_ranks_to_add):
    # 注意：此时 training weights 已 snapshot 到 CPU bucket（Feature 4）
    #       training GPU 已 offload，VRAM 空闲

    # 1. 不把新增 ranks 暴露给 routing，active set 暂时保持不变
    vllm_generation.mark_dp_ranks_inactive(dp_ranks_to_add)

    # 2. Wake up sleeping workers（仅 overlap shards，占用重叠 GPU VRAM — 因 training 已 offload 所以不 OOM）
    vllm_generation.wake_up_partial(dp_ranks_to_add)

    # 3. Selective sync：只推送到 woken shards（Feature 4）
    #    非重叠 GPU 上的 active workers 继续 generation，不做全局 pause
    model_update_service.sync_selected_workers(tgt_dp_ranks=dp_ranks_to_add)
    #    内部：CPU bucket → GPU → CUDA IPC（同 GPU）或 NCCL broadcast（跨 GPU）

    # 4. 先更新 collector 的 weight_version；必须同步等待完成，避免新 shard
    #    刚激活就用新权重生成、却被旧 version 打标签
    new_weight_version = self._current_weight_version + 1
    ray.get(self._trajectory_collector.set_weight_version.remote(new_weight_version))
    self._current_weight_version = new_weight_version

    # 5. Sync 成功且 collector version 更新完成后，才把这些 ranks 加回 active routing
    vllm_generation.activate_dp_ranks(dp_ranks_to_add)
```

整个操作在 `coordinator._resize_sync_lock` 下执行（由 coordinator.resize_infer 保证）。

**关键不变量：** `weight_version` 由 pipeline actor 独占维护（单 writer）。pseudocode 中 `set_weight_version` 用 `ray.get` 同步等待 collector 生效，**后** 才 `activate_dp_ranks` — 确保新 shard 进入 routing 前 collector 已看到新 version。不能复用全局 `prepare_for_refit()` / `resume_after_refit()`，因为那会暂停所有 generation。

改动量：~80 行（expand path 整合 selective sync）

---

### Feature 7: Per-pipeline Ray namespace isolation

**作用：** 多个 pipeline 共存时，Ray actor 命名隔离，防止冲突。

#### ROLL 怎么做的

- 每个 pipeline 有独立 Ray namespace（通过 env var `ROLL_RAY_NAMESPACE` 传入，ROLL 内部再派生 `RAY_NAMESPACE`）
- Actor 名称带 `pipeline_id` 前缀
- `full_finetune_pipeline.py:376-390` 校验 namespace 和 pipeline_id 匹配
- 通过 `runtime_env` 传递 pipeline identity env vars

#### NeMo RL 现状

- 无 namespace 隔离概念
- 所有 Ray actors 在默认 namespace

#### 移植方案

Env vars 是必要但不充分的。真正的隔离来自 actor 创建时指定 namespace。需要：

1. **Coordinator actor** 在 `get_pipeline_namespace(pipeline_id)` 中创建（参照 `coordinator.py:194`）
2. **Pipeline actor** (`NemoRLFullFinetunePipeline`) 在同一 namespace 创建（参照 `coordinator.py:277`）
3. **ModelUpdateService actor** 在同一 namespace 创建（参照 `full_finetune_pipeline.py:409-411`）
4. **审计所有 NeMo RL 命名 child actors** — NeMo RL 的 `AsyncTrajectoryCollector` 和 `ReplayBuffer` 是 Ray actors（`grpo.py:2496,2519`），需要确保它们也在 pipeline namespace 内创建，否则多 pipeline 会冲突
5. 通过 `runtime_env` 传递 `pipeline_identity_env_vars()` 给所有 actor（`rlix/utils/env.py:24`）：`PIPELINE_ID` + `ROLL_RAY_NAMESPACE` + `RLIX_CONTROL_PLANE`。注意：ROLL 代码在 import time 读取 `ROLL_RAY_NAMESPACE`，再导出内部 `RAY_NAMESPACE`；缺失会 fail fast

改动量：~60 行

---

### Feature 8: Pipeline registration lifecycle

**作用：** Pipeline 必须向 RLix orchestrator 注册 GPU 拓扑，才能参与调度。

#### ROLL 怎么做的

- 三步注册流程（`rlix/orchestrator/orchestrator.py:195-253`）：
  1. `allocate_pipeline_id(pipeline_type)` → 返回 `ft_abc123def456` 格式的 ID
  2. `register_pipeline(pipeline_id, ray_namespace, cluster_tp_configs, cluster_device_mappings)` → 向 scheduler 注册 GPU 拓扑
  3. `admit_pipeline(pipeline_id)` → scheduler 开始为该 pipeline 分配 GPU
- `cluster_device_mappings` 格式：`{"actor_train": [0,1,2,3], "actor_infer": [0,1,2,3,4,5,6,7]}`
- `cluster_tp_configs` 格式：`{"actor_train": 1, "actor_infer": 2}` — 每个 cluster 的 TP size

#### NeMo RL 现状

- 无注册概念。`setup()`（定义在 `grpo.py:216`）内部直接创建并使用 `RayVirtualCluster`：colocated 路径在 `grpo.py:430`，non-colocated 路径在 `grpo.py:509,522`
- GPU topology 在 `VirtualCluster._bundle_ct_per_node_list` 中

#### 移植方案

注册是 driver 侧 declarative contract — 从 NeMo config 计算 `cluster_device_mappings` / `cluster_tp_configs`，不需要活 PG。

```python
# Driver 脚本 — 声明式注册
cluster_device_mappings = {
    "actor_train": list(range(train_gpus_per_node)),    # [0,1,2,3]
    "actor_infer": list(range(total_gpus_per_node)),    # [0,1,2,3,4,5,6,7]
}
cluster_tp_configs = {
    "actor_train": 1,                                    # Megatron: 固定 1 GPU/worker（bridge canonicalize）
    "actor_infer": vllm_cfg.get("tensor_parallel_size", 1),  # vLLM: tp
}
orchestrator.allocate_pipeline_id()
orchestrator.register_pipeline(pipeline_id, ray_namespace, cluster_tp_configs, cluster_device_mappings)
orchestrator.admit_pipeline(pipeline_id)
coordinator = PipelineCoordinator(pipeline_config=nemo_config)
coordinator.create_pipeline_actor()
# Pipeline actor 内部 init bootstrap — 见 Feature 5 第 5 点
```

Coordinator 保持不变 — 正常创建 `RollResourceManagerProxy` singleton。PG 共享和 worker bundle mapping 见 Feature 12。`RayVirtualCluster` 在 RLix 模式下不使用（见 Feature 12）。

改动量：~60 行（merged config/registration helper + pipeline 内部 bundle mapping helper）

---

### Feature 9: Progress reporting

**作用：** Pipeline 向 RLix scheduler 报告 generation demand 进度，scheduler 据此做 gap-ratio planning（决定何时触发 shrink）。

#### ROLL 怎么做的

- `RolloutScheduler` 每 2% 进度变化时发送 `ProgressReport`（`rollout_scheduler.py:601-635`）
- Report 包含：`pipeline_id`, `step_target_trajectories`, `collected`, `bucket` (0-50), `current_train_step`
- Fire-and-forget 发给 coordinator → 转发给 scheduler
- Scheduler 的 gap-ratio planner 用 progress 决定何时触发 shrink（在采样需求波谷时）

#### NeMo RL 现状

- 无 progress reporting
- `async_grpo_train` 循环内有 `step` 计数器和 `replay_buffer.size()` 但不对外暴露

#### 移植方案

不能直接把 `async_grpo_train` 的 `step / total_steps / replay_buffer.size()` 映射成 RLix progress。

RLix scheduler 的 gap-ratio planner 吃的是 **generation demand**：`ProgressReport.step_target_trajectories` + `metrics["completed"]`，scheduler 内部计算 `remaining = max(step_target - completed, 0)`（`scheduler.py:840`），用于判断当前 pipeline 还差多少 rollout work 来决定何时触发 shrink。训练步数不是这个信号。

**NeMo RL 的 continuous collector 没有离散 batch 边界** — `AsyncTrajectoryCollector._collection_loop`（`async_utils.py:392`）是 daemon thread，持续从 dataloader 拉 batch、生成轨迹、push 到 `ReplayBuffer`。training loop 通过 `replay_buffer.sample(num_prompt_groups=num_prompts_per_step)`（`grpo.py:2646`）轮询拉取，直到有足够样本。不存在显式的 "generation batch start/end"。

**因此不能用离散 batch API，改用连续快照模型：**

RLix `ProgressReport` 本身就是 point-in-time 快照（不需要 begin/end lifecycle）。映射：

| RLix 字段 | NeMo RL 对应值 | 来源 |
|-----------|---------------|------|
| `step_target_trajectories` | `num_prompts_per_step` | `master_config["grpo"]["num_prompts_per_step"]`（`grpo.py:2454`）— 一个 training step 需要的 prompt groups 数 |
| `metrics["completed"]` | `min(buffer_valid_count, num_prompts_per_step)` | `ReplayBuffer` 中 age 窗口内（`max_trajectory_age_steps`）的有效条目数，cap 到 target |

**Demand window = inter-training-step collection period：**
- `resume_after_refit()` 后 collector 恢复，buffer 开始为下一个 step 积累
- buffer 有效条目逐渐增长 → `completed` 从 0 趋近 `num_prompts_per_step`
- `replay_buffer.sample()` 成功时 training step 开始 → 下一轮 shrink

**上报时机：** 每次轨迹 push 到 `ReplayBuffer` 后（`_run_prompt_group_worker` 调用 `push_with_wait_signal`，`async_utils.py:695`），检查 2% 进度变化（与 ROLL `rollout_scheduler.py:607` 的 bucket 机制一致），fire-and-forget 发送 `ProgressReport`。

**实现：**

```python
# AsyncTrajectoryCollector 构造时注入 rlix_hooks（不依赖全局单例）
class AsyncTrajectoryCollector:
    def __init__(self, ..., rlix_hooks=None):
        self._rlix_hooks = rlix_hooks or NoOpRLixHooks()
        self._last_progress_bucket = -1  # 2% granularity

# _run_prompt_group_worker 中，push 成功后上报
def _run_prompt_group_worker(self, ...):
    ...
    replay_buffer.push_with_wait_signal.remote(trajectory, ...)
    # 上报 progress（fire-and-forget, 2% 变化阈值）
    valid_count = ray.get(replay_buffer.valid_count.remote(
        current_weight_version=self._weight_version,
        max_age_steps=self._max_trajectory_age_steps,
    ))
    completed = min(valid_count, self._num_prompts_per_step)
    bucket = int(completed / self._num_prompts_per_step * 50)
    if bucket != self._last_progress_bucket:
        self._last_progress_bucket = bucket
        self._rlix_hooks.report_progress(
            step_target_trajectories=self._num_prompts_per_step,
            completed=completed,
        )

# prepare_for_refit 时 clear progress（避免 scheduler 误认为 pipeline 有 backlog）
def prepare_for_refit(self):
    ...
    self._rlix_hooks.clear_progress()
    self._last_progress_bucket = -1
```

`NemoRLRLixHooks.report_progress` 构造 `ProgressReport` 并 fire-and-forget 发给 coordinator。`clear_progress` 调用 `coordinator.clear_progress(pipeline_id)`（`scheduler.py:571`）。

**hooks 放置：保留独立 `rlix_hooks.py` 小模块。** 原因不是“为了抽象而抽象”，而是 import 方向：`AsyncTrajectoryCollector` / `grpo.py`（NeMo 侧）需要拿到 `NoOpRLixHooks` 默认实现，而 RLix pipeline 侧需要提供真实实现。把 protocol/no-op 放在独立 seam file，可避免 NeMo 侧反向 import `nemo_rl_pipeline.py`。

改动量：~40 行

---

### Feature 10: Partial GPU topology validation

**作用：** 验证 GPU 拓扑满足 partial overlap 要求，在启动时 fail fast。

#### ROLL 怎么做的

- `_validate_partial_gpu_config()` (`agentic_pipeline.py:770-894`) 检查：
  1. `train_devices ⊂ infer_devices`（训练 GPU 是推理 GPU 的子集）
  2. `infer_dp_size >= 2`（至少 2 个 DP shard，否则无法 partial）
  3. `async_generation_ratio > 0`（必须是 async 模式）
  4. TP/PP/EP compatibility
  5. 至少 1 个 DP rank 在 shrink 后保持 active
  6. Colocated mode 禁止 async（`async_generation_ratio == 0`）

#### NeMo RL 现状

- 无 partial overlap 验证
- `async_grpo_train` 只验证 `not colocated_inference`（`grpo.py:2448`）

#### 移植方案

在 `NemoRLFullFinetunePipeline.initialize_pipeline()` 中添加验证：

```python
assert train_devices.issubset(infer_devices), "partial overlap requires train ⊂ infer"
assert infer_dp_size >= 2, "partial overlap requires dp >= 2"
assert async_grpo_enabled, "partial overlap requires async GRPO"
# NeMo RL 内部一致性检查（与 RLix 注册无关 — 注册用 tp_size=1 for Megatron）
# tp*pp*cp*ep 不是 model-parallel width（EP 是 DP 的细分），但作为 divisibility check
# 等价于验证 (1) dp 为整数 且 (2) dp % ep == 0（expert_data_parallel 为整数）
megatron_parallelism_product = tp_size * pp_size * cp_size * ep_size
assert len(train_devices) % megatron_parallelism_product == 0, (
    f"train device_mapping ({len(train_devices)}) must divide evenly by "
    f"tp*pp*cp*ep ({megatron_parallelism_product})"
)
assert len(infer_devices) % vllm_tp_size == 0, (
    f"infer device_mapping ({len(infer_devices)}) must divide evenly by vllm_tp_size ({vllm_tp_size})"
)
assert len(infer_devices - train_devices) >= vllm_tp_size, (
    "at least 1 full inference DP rank must stay active after shrink"
)
```

`megatron_parallelism_product = tp * pp * cp * ep` 是 NeMo RL 内部一致性检查（divisibility check，非 model-parallel width）。RLix 注册用 `cluster_tp_configs["actor_train"] = 1`（bridge canonicalize），scheduler 不依赖 training 并行度。

改动量：~30 行

---

### Feature 11: Conditional RLix behavior flag

**作用：** NeMo RL 代码在 standalone 和 RLix 模式下行为不同，需要一个 flag 控制。

#### ROLL 怎么做的

- `DO_TIME_SHARING` 常量（`roll/utils/constants.py`）— 从 `RLIX_CONTROL_PLANE` env var 派生
- 用于：
  - 跳过 `ray.shutdown()`（library mode 下 Ray 生命周期由 RLix 控制）
  - 启用 pipeline-scoped actor naming
  - 启用 progress reporting
  - 选择 `RollFullFinetunePipeline`（RLix 版）vs `AgenticPipeline`（standalone 版）

#### NeMo RL 现状

- 无此概念。`grpo_train` / `async_grpo_train` 总是 standalone 运行

#### 移植方案

需要 **hooks + flag 双管齐下**。No-op hooks 只能覆盖"添加行为"的场景，但 RLix 模式还需要**改变或跳过**现有行为：

| 行为 | Standalone 模式 | RLix 模式 | 控制方式 |
|------|----------------|----------|---------|
| `ray.shutdown()` | 正常执行 | 跳过（RLix 管 Ray 生命周期） | Flag |
| Train step 后 | 直接进入 refit | CPU bucket cache build + offload training GPU + destroy NCCL groups | Flag + Hook |
| Weight sync | `refit_policy_generation()` 全量同步 | 跳过原生 refit — 由 scheduler expand 触发 `ModelUpdateService.sync_selected_workers()` | Flag |
| `prepare_for_generation()` / `finish_generation()` | 全量 sleep/wake（colocated） | 跳过 — sleep/wake 由 scheduler `resize_infer` 驱动 | Flag |
| Progress reporting | 无 | `AsyncTrajectoryCollector` 上报 demand | Hook |
| Generation allocation | 无 | 持有 `actor_infer` GENERATION allocation | Hook |

**实现：** `RLIX_CONTROL_PLANE` env var → `DO_TIME_SHARING` 常量（与 ROLL 一致）。在 `async_grpo_train()` 中：

```python
DO_TIME_SHARING = os.environ.get("RLIX_CONTROL_PLANE") == "rlix"

# 训练后
if DO_TIME_SHARING:
    build_cpu_bucket_cache(step)      # RLix: cache for expand
    self._cache_ready_step = step     # RLix: 单槽 ready 指针（非 ROLL 双槽 versioning）
    offload_training_gpu()            # RLix: free GPU for inference
    destroy_nccl_groups()             # RLix: free communicator buffers（见下方复杂度说明）
    hooks.after_training(step)        # RLix: notify scheduler → expand
else:
    refit_policy_generation(...)      # Standalone: 原生 refit
```

**`destroy_nccl_groups()` 复杂度说明：**

ROLL 通过 `ReloadableProcessGroup` monkey-patch 统一托管 NCCL groups（`roll/utils/offload_nccl.py`）；NeMo RL 不复用这套基础设施，而是走更直接的 Megatron helper 路径：

```python
def destroy_megatron_nccl_groups():
    """Local helper — 不修改上游 Megatron。"""
    from megatron.core import parallel_state
    # 1. 收集 parallel_state 中所有非 None 的 process groups
    # 2. 过滤出 NCCL backend groups（排除 Gloo）
    # 3. 去重 handles
    # 4. 对每个调用 torch.distributed.destroy_process_group(pg)
    # 5. 调用 destroy_model_parallel() 清理 Megatron 缓存的 global state
    # Wake / 下次训练 / checkpoint / eval 前：调用 initialize_model_parallel(...) 重建
```

已知风险只保留两点：`parallel_state` 可能不是唯一 owner；反复 `destroy → initialize` 在长生命周期 worker 中需要 Gate 2.5 验证。如果 NeMo RL 在 offload 状态下做 checkpoint/export/eval，必须先 reload comm state。

改动量：~40 行（flag 检查 + 行为分支）+ ~50 行（helper + re-init）

---

### Feature 12: Shared PG cluster for partial overlap

**作用：** Partial overlap 需要 training 和 inference workers 共享同一组 GPU（overlap GPUs 上两种 worker 共存）。NeMo RL 现有的 colocated/non-colocated 二选一模式无法表达这种拓扑。

#### ROLL 怎么做的

- ROLL 创建 **一组 PG** 覆盖所有 GPU，不同 role（actor_train, actor_infer）通过 `device_mapping` 映射到 PG 中的不同 GPU 子集
- PG 本身不变 — 只有 worker 状态（active/sleeping）随 shrink/expand 变化
- 同一个 PG bundle 上可以有 training worker + inference worker（通过 sleep/wake 交替占用 GPU）

#### NeMo RL 现状

`grpo.py:setup()` 只有两种资源形态：
- **Colocated** (line 430-440): `train_cluster = inference_cluster = cluster` — 完全共享一套 `RayVirtualCluster`
- **Non-colocated** (line 509-528): training / inference 各自独立 `RayVirtualCluster`

它们都不能表达 partial overlap：前者是“全重叠”，后者是“零重叠”。而且 RLix mixed deployment 下，ROLL 与 NeMo RL 必须共享同一套 PG；如果 NeMo RL 另建 `RayVirtualCluster`，就会和 `RollResourceManagerProxy` 的 shared PG 冲突。

#### 移植方案

**统一策略：RLix 模式下始终用 `RollResourceManagerProxy` 的 shared PGs，不创建 `RayVirtualCluster`。**

无论是 NeMo-only 还是 ROLL+NeMo 混合，所有 pipeline 共享同一组 PG：

1. Coordinator 创建 `RollResourceManagerProxy`（singleton，覆盖所有 GPU）— 对 NeMo 和 ROLL 一视同仁
2. NeMo RL 的 `initialize_pipeline()` 通过 `RollResourceManagerProxy.allocate_placement_group(device_mapping=...)` 获取 PG handle
3. NeMo RL workers 调度到 shared PG 上，与 ROLL workers 使用完全相同的 PG 基础设施

```python
# NeMo RL initialize_pipeline 内：
proxy = RollResourceManagerProxy(num_gpus_per_node=num_gpus_per_node)

# inference workers: 全部 GPU
infer_pg_alloc = proxy.allocate_placement_group(
    world_size=infer_dp_size, device_mapping=list(range(total_gpus))
)

# training workers: overlap GPU 子集
# world_size = total Megatron workers (1 GPU each, worker_config.py:231), NOT dp_size
train_pg_alloc = proxy.allocate_placement_group(
    world_size=len(train_device_mapping), device_mapping=train_device_mapping
)
```

Coordinator 保持不变 — 所有 backend 都用 `RollResourceManagerProxy`。`RayVirtualCluster` 仅用于 standalone 模式，RLix 模式下完全不用。

**Worker 创建适配**：`allocate_placement_group` 返回 `(node_rank, gpu_rank, placement_group)` → pipeline 私有 helper 转换为 `RayWorkerGroup` 的 `bundle_indices_list`。Training workers 只在 overlap GPU 对应的 bundle 上创建。

改动量：~90 行（shared-PG 接入 + pipeline 内 bundle mapping）

---

## 测试策略

### 验证环境

vast.ai 2x 3060（$0.15/hr）= **2 GPU**。所有 Gate 按 2 GPU 设计。

### 测试工作负载

NeMo RL 自带的 **calculator multiturn async GRPO example**（简单、有单元测试可参考）。

### 分步验证

1. **先跑通单个 NeMo RL pipeline** — 验证 Feature 1-3 + Feature 6 + Feature 8-10 + Feature 12（partial sleep/wake + routing + refit + registration + progress + validation + shared PG）
2. **再加第二个 pipeline** — 验证 Feature 5 + Feature 7 + Feature 11（scheduler-driven resize + namespace isolation + conditional flag），类似 `examples/` 目录里 ROLL 的双 pipeline setup

### Gate 1: partial sleep/wake 基础 (dp=2, tp=1)

```
配置：2 GPU, dp=2, tp=1, async_engine=True, async GRPO
测试：
1. 初始化 2 个 vLLM generation worker
2. 先验证 shared-PG bundle mapping 正确：dp_shard 0 → GPU 0，dp_shard 1 → GPU 1
3. generate_async — round-robin 到 2 个 worker
4. sleep_partial([1], level=2) — GPU 1 VRAM 释放
5. generate_async — 自动跳过 sleeping shard，只用 worker 0
6. wake_up_partial([1]) — GPU 1 VRAM 恢复
7. generate_async — round-robin 恢复到 2 个 worker
预期：全部通过，无 crash
```

### Gate 2: TP group sleep/wake (dp=1, tp=2)

```
配置：2 GPU, dp=1, tp=2, async_engine=True
测试：验证 sleep_partial 在 dp=1 时不能 sleep 唯一 shard，TP group NCCL 无错误
```

### Gate 2.5: NCCL selective sync + Megatron NCCL destroy/re-init (dp=1, tp=2)

```
配置：2 GPU, dp=1, tp=2, async GRPO, calculator example (小模型)
测试：完整 training → offload → expand → sync 周期，验证 tp=2 路径：
1. Megatron training step (tp=2 TP NCCL groups active)
2. build_cpu_bucket_cache — 所有 TP/PP/CP/EP ranks 参与 gather，只有 cache owner 存储完整 CPU cache
3. destroy_megatron_nccl_groups() — 销毁 TP NCCL communicators，验证 GPU VRAM 释放
4. vLLM wake_up (tp=2 collective_rpc 传播)
5. sync_selected_workers — 验证 NCCL broadcast transport 路径（跨 GPU TP ranks）
6. 下一轮 training 前 initialize_model_parallel() — 重建 TP NCCL groups
7. 连续跑 3+ step，验证 destroy/re-init 循环稳定性
预期：无 NCCL 错误，无 VRAM 泄漏（每轮 peak VRAM 稳定），权重正确
关键：这是唯一覆盖 NCCL broadcast transport 和 Megatron NCCL lifecycle 的 gate。
      dp=1 意味着没有 partial overlap（所有 GPU 都 overlap），但足以验证 transport 和 NCCL 生命周期。
```

### Gate 3: 单 pipeline 端到端 async GRPO

```
配置：2 GPU, dp=2, tp=1, async GRPO, calculator example
测试：完整 async training loop — `actor_infer` 持有长期 GENERATION allocation，
     generation 在后台持续，training 时 shrink dp[1]，
     expand 后 selective sync + 原子激活新 shard，非重叠 shard 无全局 pause，
     `after_training(step)` 触发 expand 未完成前，`before_training(step+1)` 不进入下一轮 train，
     且 collector 的 version / active-rank 可见性保持一致，无 stale weight generation
```

### Gate 4: 双 NeMo RL pipeline 调度

```
配置：2 GPU, dp=2, tp=1, 两个 NeMo RL async GRPO pipeline
测试：两个 NeMo RL pipeline 共享 GPU，通过 RLix scheduler 交替获得 training GPU，
     Perfetto trace 确认 GPU 时分复用
PG：两个 pipeline 共享 RollResourceManagerProxy 的 shared PGs
```

### Gate 5: ROLL + NeMo RL 混合调度

```
配置：2 GPU, dp=2, tp=1, 1 个 ROLL full_finetune pipeline + 1 个 NeMo RL async GRPO pipeline（场景 B：混合）
测试：
1. ROLL pipeline 正常启动，RollResourceManagerProxy 创建 shared PGs
2. NeMo RL pipeline 复用 shared PGs，不创建 RayVirtualCluster
3. 两个 pipeline 交替获得 training GPU，通过 RLix scheduler 调度
4. Perfetto trace 确认两个不同框架的 pipeline GPU 时分复用
PG：NeMo RL workers 调度到 RollResourceManagerProxy 的 shared PGs 上
这是最终验证 — 证明 RLix 可以统一调度不同 RL 框架
```

---

## 文件改动总清单

### NeMo RL 侧

| 文件 | Feature | 改动 | 行数 |
|------|---------|------|------|
| `vllm_worker.py` | F1 | sleep level 参数化 (:1009) | +5 |
| `vllm_worker_async.py` | F1, F2 | sleep level 参数化 (:1154) + `abort_all_requests()` 方法（内部获取 running IDs）+ `is_idle()` 方法（检查 `vllm:num_requests_running` metric） | +15 |
| `vllm_generation.py` | F2, F3 | `sleep_partial()`（abort-drain-sleep via engine idle check）, `wake_up_partial()`, `_active_dp_ranks`, `_preempted_shards`, `ShardPreemptedError` 转换, async routing skip, `_async_generate_base` 内 targeted retry | +150 |
| `worker_groups.py` | F2 | `run_on_dp_shard_leaders()`（仅 leader 路径，不加 `_get_all_workers_for_dp_shard()`） | +20 |
| `megatron_policy_worker.py` | F4 | CPU bucket build（参与 PP collective gather，仅 cache owner 存储） | +60 |
| `nccl_offload.py` (**新增**) | F1, F11 | Megatron NCCL group 手动 destroy/reload（从 `parallel_state` 收集 NCCL groups + `torch.distributed.destroy_process_group` + re-init；不用 `destroy_model_parallel()` 因其不支持反复调用） | +90 |
| `grpo.py` | F5, F11 | `async_grpo_train()` training hook 调用点 + `DO_TIME_SHARING` 行为分支 | +60 |
| `async_utils.py` | F9 | `AsyncTrajectoryCollector` 连续快照 progress 上报（2% bucket 阈值, `prepare_for_refit` 时 clear）+ `ReplayBuffer` 新增 `valid_count(current_weight_version, max_age_steps)` 方法 | +60 |
| `rlix_hooks.py` (**新增**) | F5, F9 | `RLixHooks` protocol + `NoOpRLixHooks` 默认实现（NeMo/RLix 共享 import seam） | +30 |

### RLix 侧

| 文件 | Feature | 改动 | 行数 |
|------|---------|------|------|
| `nemo_rl_pipeline.py` (**新增**) | F5, F6, F8, F10, F12 | NemoRLFullFinetunePipeline（含 resize_infer, expand+selective sync, registration, validation, shared-PG `bundle_indices_list` helper） | +420 |
| `nemo_rl_model_update_service.py` (**新增**) | F4, F6 | 简化版 ModelUpdateService（selective sync, CUDA IPC + NCCL, 无 versioning） | +200 |
| `nemo_rl_config_bridge.py` (**新增**) | F5, F8 | ConfigBridge + 声明式 registration helper（见下方必须提供的属性清单） | +100 |

### 测试

| 文件 | 改动 | 行数 |
|------|------|------|
| `tests/test_partial_sleep_wake.py` (**新增**) | Feature 1-3 单元测试 | +150 |
| `tests/test_nemo_rl_pipeline.py` (**新增**) | Feature 5-6 集成测试 | +200 |

**总计：~1650 行**

---

## 时间线

```
Week 1: Feature 1-4 — vLLM sleep/wake + partial + routing + CPU weight cache
  ├── Day 1: Feature 1 — sleep_level 参数化
  ├── Day 2-3: Feature 2 — run_on_dp_shard_leaders + sleep_partial/wake_up_partial
  ├── Day 4: Feature 3 — generate_async routing skip + `_async_generate_base` 内 targeted retry
  ├── Day 5: Feature 4 — CPU weight snapshot + broadcast from CPU
  └── Day 6: Gate 1 + Gate 2 + Gate 2.5 测试（Gate 2.5 验证 NCCL broadcast + Megatron destroy/re-init）

Week 2-3: Feature 5-12 — RLix 适配器
  ├── Day 1-2: Feature 12+8+10 — shared PG cluster + merged config/registration bridge + validation
  ├── Day 3-4: Feature 5 — NemoRLFullFinetunePipeline + hooks
  ├── Day 5-6: Feature 6 — expand + refit 原子操作
  ├── Day 7: Feature 7+11 — namespace isolation + conditional flag
  ├── Day 8: Feature 9 — progress reporting
  └── Day 9-10: Gate 3 (单 pipeline) + Gate 4 (双 NeMo RL pipeline)

Week 4: 打磨 + Gate 5
  ├── Day 1-2: Gate 5 — ROLL + NeMo RL 混合调度（最终验证）
  ├── Day 3-4: 防御性 assertion + 所有 Gate 回归
  └── Day 5: 文档更新

Out of scope:
  └── sync grpo_train() / sync generate() — see Out of Scope section
```

---

## Future: NeMo-Gym shard preemption

**目标：** 让 NeMo-Gym 的 async GRPO 训练也支持 shard resize（当前 out of scope，标准 calculator 路径优先）。

**问题：** NeMo-Gym 通过 HTTP 访问同一组 vLLM workers（`dp_openai_server_base_urls`），与标准路径的 Ray actor 调用不同。标准路径可直接 `engine.abort(req_id)`，但 NeMo-Gym 的 HTTP 连接由 `nemo_gym` 包的 aiohttp client 持有（`server_utils.py:157-205`），我们无法从 vLLM server 侧主动关闭。

**方案：503 middleware + 强制断连（Option 1）**

不修改 nemo-gym 包，在 vLLM HTTP server 侧拦截：

1. **Preemption middleware**：在 NeMo RL 自己的 FastAPI app 上添加中间件（`vllm_worker_async.py:627`，**不需要 patch vLLM**）。NeMo RL 已经创建独立的 `FastAPI()` app 并用自己的 uvicorn server 运行（line 641-647），代码注释（line 625-626）明确预留了 middleware 扩展点。中间件检查 per-shard `_preempted` flag，flag 为 True 时：
   - 新请求：立即返回 HTTP 503，不进入 engine
   - 已有连接：强制关闭 TCP 连接 → 触发 vLLM 的 `@with_cancellation` decorator（`vllm PR #11190`）→ 内部调用 `engine.abort(request_id)`

2. **NeMo-Gym 天然兼容 503**：`NeMoGymAsyncOpenAI._request()`（`openai_utils.py:479-508`）对 503 做 **无限重试**（503 ∈ `RATE_LIMIT_ERROR_CODES`，每次 retry `max_num_tries += 1`），0.5s 间隔。shard 恢复后重试自然成功。

3. **Shard-aware URL routing**：当前 NeMo-Gym 通过 cookie 将 session pin 到固定 `base_url`（`app.py:427-436`，round-robin 分配）。需要增加：
   - vLLM server 在 503 response 中携带 header 提示"此 shard 不可用"
   - 或 NeMo-Gym `_resolve_client()` 在收到 503 后自动 failover 到下一个 `base_url`
   - 最简方案：NeMo-Gym 的 `_request()` 已有重试循环，只需在重试时 round-robin 切换 `base_url`（需小改 `openai_utils.py`）

4. **Flag 控制路径**：`sleep_partial` 设置 `_preempted` flag → middleware 拦截 → 503 + 断连 → drain 确认 engine idle → safe sleep。`wake_up_partial` 清除 flag → middleware 放行 → 重试成功。

**与标准路径的对比：**

| | 标准路径（Ray actor） | NeMo-Gym 路径（HTTP） |
|---|---|---|
| Abort 机制 | `engine.abort(req_id)` via Ray RPC | 503 middleware + TCP 断连 → `@with_cancellation` → `engine.abort` |
| Retry | `ShardPreemptedError` → `_async_generate_base` retry loop | aiohttp 503 无限重试（已有） |
| 改动范围 | `vllm_generation.py` + `vllm_worker_async.py` | vLLM HTTP server middleware + `openai_utils.py` failover（可选） |

**依赖：** 标准路径的 Feature 1-3 先完成（sleep/wake + partial + routing），HTTP middleware 在此基础上扩展。

**估算：** ~100 行（middleware ~40, flag 控制 ~30, URL failover ~30）

---

## 附录：ROLL 参考代码

| 组件 | 路径 |
|------|------|
| **RLix** | |
| Scheduler 核心 | `rlix/scheduler/scheduler.py` |
| ROLL 适配器 | `rlix/pipeline/full_finetune_pipeline.py` |
| Coordinator（含校验） | `rlix/pipeline/coordinator.py` (校验 :81, :113, :136; resize :502-547) |
| ModelUpdateService | `rlix/pipeline/model_update_service.py` |
| **ROLL** | |
| shrink/expand 定义 | `agentic_pipeline.py` — `_shrink_workers`:237, `_expand_workers`:256 |
| shrink/expand 实现 | `generate_scheduler.py:1885,1973` |
| rollout shrink/expand | `rollout_scheduler.py:1088,1138` |
| Worker partial offload/load | `base_worker.py` — `load_states_partial`:494, `offload_states_partial`:527 |
| vLLM sleep/wake | `vllm_strategy.py` — `load_states`:569, `offload_states`:582 |
| vLLM worker lifecycle | `third_party/vllm/worker.py` — `WorkerBase`:118, sleep/wake:336-536 |
| async generation ratio | `base_config.py:453` — `async_generation_ratio: float` |
| **NeMo RL** | |
| async GRPO 训练循环 | `grpo.py:2365` — `async_grpo_train()` |
| async refit 协调 | `grpo.py:2860-2880` — `prepare_for_refit` → `refit` → `resume_after_refit` |
| AsyncGRPOConfig | `grpo.py:111` — `max_trajectory_age_steps`, `in_flight_weight_updates` |
| vLLM Worker sleep (sync) | `vllm_worker.py:986` — hardcoded level=1 at :1009 |
| vLLM Worker sleep (async) | `vllm_worker_async.py:1135` — hardcoded level=1 at :1154 |
| vLLM Generation lifecycle | `vllm_generation.py:733-782` |
| Worker Group + dp_leader | `worker_groups.py:404` — `get_dp_leader_worker_idx()` |
| worker_metadata dp_shard_idx | `worker_groups.py` — `_worker_metadata[i]["dp_shard_idx"]` |
| async round-robin 核心 | `vllm_generation.py:559` — `_async_generate_base()` 中的 `current_generate_dp_shard_idx` |
| refit ZMQ IPC | `grpo.py:1157` — colocated + vLLM |
| refit NCCL broadcast | `grpo.py:1172` — non-colocated |
| colocated 资源分支 | `grpo.py:419-444` |
| non-colocated 资源分支 | `grpo.py:446+` |
