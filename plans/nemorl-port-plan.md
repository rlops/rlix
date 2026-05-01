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
  - `max_trajectory_age_steps` 控制 replay buffer 的 lookahead / age window，用于限制可消费轨迹的步龄；**不是** ROLL `async_generation_ratio`（pool-based mixing / staleness tolerance, `base_config.py:453`）的等价物
  - 两者语义不同，但**不影响本移植方案**：RLix 自己管理 resize lifecycle，port 不依赖把 NeMo RL 的 age window 映射成 ROLL 的 generation pool mixing 语义
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
4. 对齐 ROLL 加入 **idempotency guard**：worker 本地维护 `is_model_in_gpu`（或等价状态），`sleep` / `sleep_async` 仅在当前仍驻留 GPU 时执行，`wake_up` / `prepare_for_generation` 仅在当前已 offload 时执行。重复 resize 命令视为 no-op，避免 double-sleep / double-wake
5. 改动量：~20 行

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
4. `VLLMGeneration` 新增 `_routing_lock`（`asyncio.Lock`）— 串行化“读 active set + 选择 shard + 提交 dispatch”与“更新 active set + abort/drain/sleep”。这是 ROLL `RequestScheduler.routing_lock` 的语义等价物；只保护 set mutation 不够，必须保护整个 compound operation，避免 TOCTOU
5. `RayWorkerGroup` 新增 `run_on_dp_shard_leaders(dp_ranks, fn, *args, **kwargs)` — 对指定 DP shard 的 **leader worker** 执行（vLLM 内部传播到 TP peers）
6. 内部仅需 leader index 列表：`[self.get_dp_leader_worker_idx(r) for r in dp_ranks]`。**不新增** `_get_all_workers_for_dp_shard()`

**`sleep_partial` 必须 abort-drain-sleep，不能直接 sleep in-flight 请求：**

这条 abort-drain-sleep 路径**只用于 scheduler-driven resize / shrink**。普通权重更新不走这条路径；RLix 模式下权重同步发生在 expand 时的 selective sync（见 Feature 6）。

```python
async def sleep_partial(self, dp_ranks: List[int], level: int = 2):
    # 1. 在 routing lock 下：从 routing 中移除 + 标记 preempted。
    #    必须把“更新 active set + 后续 dispatch 不再选中这些 shards”串行化，
    #    对齐 ROLL RequestScheduler.routing_lock 语义，避免 dispatch/shrink TOCTOU。
    async with self._routing_lock:
        self._active_dp_ranks -= set(dp_ranks)
        self._preempted_shards |= set(dp_ranks)

    # 2. Worker 内部 abort 所有 running requests（无需从 generation 层传 request IDs）
    self.run_on_dp_shard_leaders(dp_ranks, "abort_all_requests")

    # 3. Drain：轮询 vLLM 已有的 engine metric 直到 idle
    #    vllm_worker_async.py:241 已在采集 vllm:num_requests_running
    for rank in dp_ranks:
        await self._wait_engine_idle(rank)  # poll worker.is_idle() → num_requests_running == 0

    # 4. Engine idle，安全 sleep
    self.run_on_dp_shard_leaders(dp_ranks, "sleep", level=level)

    # 5. Fail fast: post-offload VRAM must actually drop.
    #    对齐 ROLL 的 post-offload sanity check，避免“逻辑上 slept、实际上显存没释放”。
    self.run_on_dp_shard_leaders(dp_ranks, "assert_post_sleep_memory_below_threshold")
```

**不需要 per-request tracking** — 不引入 `_inflight_requests: Dict[int, Set[str]]`，不修改 request ID 生成路径。Worker 内部通过 engine API 获取 running request IDs 并 abort，drain 用现有 `vllm:num_requests_running` metric 确认 idle。

**不能跳过 abort 直接 sleep** — vLLM engine 在处理请求时被 sleep 会导致对已 offload 的 GPU memory 的访问，crash 整个 worker。abort 先让 engine 干净地丢弃请求，drain 确认 engine idle，然后 sleep 安全执行。

**Post-offload memory assertion（新增硬约束）：** `sleep_partial()` 成功返回前，目标 shard leader 必须验证 `torch.cuda.memory_allocated() < post_sleep_vram_threshold_bytes`（默认按 ROLL 使用 1 GiB 量级阈值；做成显式配置更稳妥）。若 sleep 后显存未降到阈值以下，直接 fail fast，而不是让 scheduler 误以为该 GPU 已可供 training / wake_up 复用。

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
while True:
    await self._wait_for_active_dp_shards()
    async with self._routing_lock:
        if not self._active_dp_ranks:
            continue
        while self.current_generate_dp_shard_idx not in self._active_dp_ranks:
            self.current_generate_dp_shard_idx = (self.current_generate_dp_shard_idx + 1) % self._dp_size
        dp_rank = self.current_generate_dp_shard_idx
        # dispatch 目标必须在 lock 内决定并提交，避免 shrink 在选择后、dispatch 前移除该 rank
        worker = self._select_worker_for_dp_rank(dp_rank)
        break
```

`_wait_for_active_dp_shards()` 由 `activate_dp_ranks()` / `sleep_partial()` 维护的 condition/event 驱动。语义是“collector 在 shrink-to-zero 期间阻塞等待 scheduler expand”，**不是**抛异常让后台线程崩掉。

**必须引入 routing lock，不是可选优化。** 否则会出现：
1. `_async_generate_base` 读到 `dp_rank = X`
2. `sleep_partial()` 并发把 `X` 从 `_active_dp_ranks` 移除并开始 abort/drain
3. request 仍被 dispatch 到正在 drain/sleep 的 shard

这与 ROLL `generate_one_request()` / `shrink_workers()` 共享同一把 `routing_lock` 的设计意图一致：锁保护的是 dispatch 决策的原子性，而不只是 set mutation。

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
MAX_SHARD_REDISPATCH_ATTEMPTS = self._dp_size  # 上限 = dp_size（每个 shard 最多试一次）
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

**多 turn 轨迹允许跨 `weight_version`**。约束只需要做到“单个 turn 的 generate 调用是 pure 的”；如果 resize 发生在 turn 边界之间，后续 turn 落到新版本权重是允许的。上面的 targeted retry 恰好满足这个语义：只重做被 abort 的当前 turn，不回滚已完成 turns。

**Retry safety invariant（必须写明）：** 上述 turn-level retry 只对“side effect 在 successful generation 之后才提交”的 rollout 路径安全。当前 scope 内的 calculator / 标准 `run_async_multi_turn_rollout` 路径满足这一点：先完成 assistant turn generation，再调用 env step；aborted turn 不会执行 env step，因此重试同一 turn 不会重复提交 side effect。ROLL 内建的 agentic env-manager 也是同一模式：`GenerateStopReason.ABORT` 仅增加 attempt，`env.step()` 只在 `FINISH` / `MAX_LENGTH` 后执行。**但这不是框架层的普遍保证。** 如果未来扩展到 NeMo-Gym 或其他 stateful tool/env 路径，必须保持同样的 commit point（abort/preempt 发生在 side effect 之前），否则需要显式 idempotency key / dedupe 机制后才能启用当前的 turn retry 语义。

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
- **ZMQ colocated 路径**：复用 `plans/cpu_serialize.md` 已落地的 sender/receiver bucket payload 约定。
  - `model_update_transport="cuda_ipc"`：sender `get_handle_from_tensor(buffer)`，payload = `(ipc_handle, param_names, used_bytes)`；receiver `rebuild_cuda_tensor_from_ipc()` 后按 `param_names + state_dict_info` 切回各 tensor（`policy/utils.py:285-314`, `vllm_backend.py:197-234`）。
  - `model_update_transport="cpu_serialize"`：sender 将 `buffer[:used_bytes]` DMA 到 pinned CPU tensor，再以 `send_multipart([b"cpu_serialize", pickle.dumps((param_names, used_bytes)), torch.save({\"bucket\": pinned})])` 发送；receiver `torch.load(...)` 后同样按 `param_names + state_dict_info` 重建 tensor（`policy/utils.py:287-313`, `vllm_backend.py:201-234`）。
- **NCCL broadcast**（non-colocated 路径）：复用 `packed_broadcast_producer/consumer` 的现有 packed-tensor 格式。producer 发送拼接后的 `torch.uint8` bucket；consumer 侧 metadata 是 `(name, shape, dtype, offset, tensor_size)`，据此 split/view 回各 tensor（`packed_tensor.py:39-94,113-199`）。

**结论：Feature 4 不再需要单独发明 bucket format。** 需要做的是：
1. 复用上述两条已存在的 payload/bucket 格式；
2. 在 `ModelUpdateService` 中补上 CPU cache 生命周期、bucket 级路由、sender-side `_cache_lock`、以及 `_cache_ready_step` version 发布语义。

**实现方案：**

引入简化版 `ModelUpdateService`（参照 ROLL `rlix/pipeline/model_update_service.py`），复用 NeMo RL 现有 transport（versioning 安全性分析见下文）：

1. 每次 `train_step` 后，构建 CPU bucket cache（**单 cache owner 模式，与 ROLL 一致**）：
   - **所有 TP/PP/CP/EP ranks 参与 collective gather**（`gather_all_hf_weights`，内部使用 PP collectives 将所有 pipeline stages 的权重汇聚）。EP-aware：expert 参数通过 `get_expert_tensor_parallel_group()` + `get_expert_model_parallel_group()` gather；non-expert 参数通过 `get_tensor_model_parallel_group()` gather（`model_update.py:128-158`）。
   - **仅 cache owner（pp0/dp0/tp0/cp0）存储 gather 后的完整模型 CPU buckets**（`megatron_strategy.py:1049-1065`）。其他 ranks 参与 collective 但丢弃结果（drain generator to keep collective moving，`megatron_strategy.py:1918-1939`）。
   - 打包到 **CPU bucket buffer**（`device="cpu"`），cache 是完整模型（非 per-shard）
   - **Bucket format specification（单一 canonical cache record）**：cache owner 存 `List[BucketRecord]`，每个 `BucketRecord` 至少包含 `param_names`, `shapes`, `dtypes`, `used_bytes`, `cpu_uint8_bucket`（contiguous CPU tensor / bytes）。colocated ZMQ 路径直接复用它生成 `cpu_serialize` multipart payload；跨 GPU NCCL 路径复用同一 bucket 顺序和 `names/shapes/dtypes`，接收端按现有 packed-tensor 语义 split/view 回 tensor。**不要为两条路径维护两套不一致的 bucket layout**
   - 启动时估算 `total_cpu_cache_bytes`（cache owner 上的**单份完整模型**大小），超过 host RAM budget 直接 fail fast
2. Offload training GPU（释放全部 VRAM）
3. Expand 时：wake_up target inference workers（仅 overlap shards）
4. `ModelUpdateService.sync_selected_workers(tgt_dp_ranks)` — **单 sender**（cache owner）推送到 woken shards：
   - Sender 由 `_select_global_sender_rank()`（`model_update_service.py:90`）确定 — 返回 pp0/dp0/tp0/cp0
   - **关键约束：不能”整模型回灌到 sender GPU 再发”**。必须 **逐 bucket CPU→GPU stage**，每个 bucket 传完立即释放 staging buffer，控制 peak VRAM
   - `bucket_size_bytes` 必须是显式配置，不是隐式默认值；初始化时用”wake_up 后剩余 VRAM”做上界检查，确保 `bucket_size_bytes + transport scratch` 小于 overlap GPU 的可用余量
   - **同 GPU**（overlap GPU，training 和 inference colocated）→ 逐 bucket stage → 复用现有 **ZMQ IPC** 路径（`stream_weights_via_ipc_zmq` / `update_weights_via_ipc_zmq`）
   - **跨 GPU**（如果 target worker 有 TP 跨 GPU 的 rank）→ 逐 bucket stage → **动态 NCCL group**（见下文）
5. 非重叠 GPU 上的 inference workers **不参与**，继续 generation

**跨 GPU broadcast 的动态 NCCL group 生命周期（参照 ROLL `ModelUpdateService`）：**

NeMo RL 现有 `model_update_group`（`StatelessProcessGroup`，init 时创建，`vllm_backend.py:56`）是覆盖所有 training + inference workers 的静态 group，**不能复用**：
- 参与者包含非重叠 GPU 上正在 generation 的 workers — 参与 collective 会暂停他们
- 每次 expand 的 target worker set 不同 — 静态 group 无法表达

**方案：每次 `sync_selected_workers` 调用动态创建临时 NCCL group，sync 完成后销毁。** 与 ROLL 的 `_build_comm_plan_for_sender` → `setup_collective_group` → `destroy_collective_group` 完全对齐。

生命周期（每次 `sync_selected_workers` 调用一次完整循环）：

```
1. CLASSIFY: _build_comm_plan_for_sender(sync_id, src_rank, tgt_dp_ranks)
   - 对每个 target device: (node_rank, gpu_rank) 匹配 sender → IPC path
   - 不匹配 → broadcast path, 加入 tgt_ranks_in_group
   - 若 tgt_ranks_in_group 非空 → 需要 NCCL group

2. CREATE: 动态创建临时 NCCL group
   - group_name = f"selective_model_update_{pipeline_id}_{uuid4().hex[:8]}_src{src_rank}"
     （pipeline_id + per-call uuid 避免跨 pipeline 和跨调用冲突）
   - master_addr = sender node IP（缓存）
   - master_port = 临时端口（OS ephemeral port, SharedStorage 原子 claim 避免多 pipeline 冲突）
   - world_size = 1 (sender) + len(tgt_ranks_in_group) (receivers)
   - 并行 fire setup_collective_group.remote() 到所有 broadcast receivers + sender
   - 内部：TCP rendezvous → PrefixStore(group_name) → _new_process_group_helper
   - warmup allreduce 验证 group 工作正常
   - ray.get(setup_refs) — barrier 等待所有方完成 init

3. USE: 逐 bucket broadcast
   - sender: collective.broadcast(gpu_staged_bucket, src_rank=0, group_name, async_op=True)
   - receivers: worker.broadcast_parameter.remote(group_name, ...) — 阻塞接收
   - 每个 bucket 传完: handle.wait() + ray.get(recv_refs) — barrier 确认传输完成

4. DESTROY: sync 完成后立即销毁
   - sender: collective.destroy_collective_group(group_name)
     → dist.destroy_process_group(pg) + 清理 name maps
   - receivers: ray.get([w.destroy_collective_group.remote(group_name) for w in broadcast_workers])
     **注意**：receiver-side `destroy_collective_group` 必须有 no-op guard（`is_group_exist` 检查），
     因为 IPC-only ranks 从未 join group（参照 `worker.py:640`）
   - finally: **条件释放 port claim** — 仅当 `sync_completed=True` 时释放；
     失败时 **intentionally leak** port claim 避免 remote worker 仍持有端口时的冲突
     （参照 `model_update_service.py:370`，Tao 在 hardening 中加入的 pattern）
```

**什么时候需要 broadcast path？** 仅当 target inference worker 有 TP 跨 GPU 的 rank（`tp_size > 1` 且 TP peer GPU 不与 sender 共 GPU）。对于 `tp=1`（Gate 1-4 覆盖的场景），所有 overlap workers 都与 sender 在同一 GPU → 全部走 IPC path → **不需要 NCCL group**。`tp=2` 且 overlap 时至少一个 TP rank 在不同 GPU → 需要 broadcast path → 需要动态 NCCL group（Gate 2.5 验证）。

**实现复用：** `_build_comm_plan_for_sender` 直接参照 ROLL `model_update_service.py:100-226` 的分类逻辑。`init_collective_group` / `destroy_collective_group` 可复用 ROLL 的 `roll/utils/collective/collective.py`（已在 submodule 中）或用 PyTorch 原生 `dist.new_group` / `dist.destroy_process_group` 实现（更轻量，不依赖 ROLL utilities）。NeMo RL 的 `StatelessProcessGroup` **不复用** — 它是一次性 init 用的，没有动态 create/destroy 能力。

**Cache 安全性：4 个不变量**

跳过 ROLL 的完整 checkpoint versioning，采用单槽 `_cache_ready_step`。安全性依赖：
1. **单 writer**：training hook 写 `_cache_ready_step`（`after_training` 中 `build_cpu_bucket_cache(step)` 完成后原子更新）
2. **单 reader 路径**：expand 读 `_cache_ready_step`（`_expand_workers` → `sync_selected_workers`）
3. **顺序契约**：`before_training(step+1)` 阻塞到前一个 `after_training(step)` 触发的 expand 完成后才返回（由 `request_cluster_gpus` 的 blocking `ray.get` 保证）。Gate 3 需验证此不变量。
4. **Cache owner `_cache_lock`**：`selective_sync_active_cache` 持有 `_cache_lock` 贯穿整个 "cache lookup → transport → NCCL teardown" 窗口（参照 `megatron_strategy.py:2095-2099`）。防止 `build_latest_bucket_cache` / `_cache_ready_step` 更新与正在进行的 transport 竞争。顺序契约（不变量 3）是正常路径保证；`_cache_lock` 是异常路径（timeout / error recovery）的安全网。
   - **必须写满整个临界区**：从“读取 active cache pointer / `_cache_ready_step` / bucket 列表”开始，到“最后一个 bucket 的 IPC/NCCL receiver barrier 完成 + 动态 NCCL group destroy 完成”为止，期间不得释放 `_cache_lock`。`build_cpu_bucket_cache()` 在“写入新 bucket 列表 + publish `_cache_ready_step`”时也必须持同一把锁。禁止只锁 cache lookup 或只锁 pointer swap 的半截实现。

**comm_plan 分类逻辑与 receiver-side 双路径 mask**

复用 ROLL 的 `_build_comm_plan_for_sender()`（`model_update_service.py:100-226`）。comm_plan 不仅决定"哪些 workers 参与 NCCL"，还携带 **per-dp_rank local-rank mask**：
- `ipc_local_ranks`：该 dp_rank 中与 sender 共 GPU 的 local ranks → 走 IPC path
- `broadcast_local_ranks`：该 dp_rank 中在不同 GPU 的 local ranks → 参与 NCCL broadcast

当 `tp > 1` 时，单个 vLLM worker 可能有部分 local ranks 走 IPC、部分走 broadcast（TP peers 跨 GPU）。Receiver-side 必须实现两个 mask guard（参照 `worker.py:757` 和 `worker.py:640`）：
- `update_parameter_in_bucket`：检查 `self.rank in ipc_local_ranks`，不在则 skip（该 rank 会通过 broadcast 接收）
- `destroy_collective_group`：检查 `is_group_exist(group_name)`，IPC-only ranks 从未 join group → no-op skip，避免 KeyError

改动量：~250 行（简化版 ModelUpdateService routing 层 + CPU bucket build + 动态 NCCL group 生命周期。transport 实现复用现有代码）

---

### Feature 5+6: Two-path weight refresh (active in-flight + expand sync) + version accounting

**作用：** 解决 partial overlap 下非重叠 active ranks 的权重更新问题。原 Feature 5/6 假设 expand 是唯一的 weight sync 路径 — 这对 ROLL（所有 ranks 都 shrink/expand）正确，但对 NeMo RL（部分 ranks 始终 active）是正确性 bug：非重叠 ranks 永远无法获得新权重。

#### 核心差异

| | ROLL | NeMo-RL |
|---|---|---|
| GPU 重叠 | `actor_train` = `actor_infer`（同 GPU） | `actor_train` ⊂ `actor_infer`（子集） |
| Shrink | 所有 inference DP ranks → zero | 仅重叠 DP ranks |
| 训练期间 | 无 inference（全部 sleeping） | 非重叠 ranks 继续 serving |
| 权重刷新 | Expand syncs all（全部被 shrunk） | 两条路径：training loop 刷新 active；expand 刷新 woken |

#### 方案：两条路径，两个 owner，一份 CPU cache

| 路径 | Shard 状态 | Owner | 机制 |
|---|---|---|---|
| **Active refresh** | 非重叠 active ranks | Training loop（`after_training` hook） | `coordinator.sync_base_weights_to_active()` → `model_update_service.sync_selected_workers(active_ranks)` — in-flight，无 drain |
| **Expand sync** | 重叠 slept/woken ranks | Scheduler（`resize_infer(add=...)`） | `_expand_workers()` → `model_update_service.sync_selected_workers(overlap_ranks)` — ranks 未进入 routing |

**不变量：** 所有已 active 的 rank 由 training loop 刷新。所有后续被激活的 rank 由 expand 刷新。因此没有 active rank 会保持 stale。

两条路径共享同一份 CPU bucket cache（单 cache owner pp0/dp0/tp0/cp0）。

#### 为什么不用 NeMo RL 原生 refit 路径

调查确认 `refit_policy_generation()` 无法作为 active-rank refresh 机制：

1. **无子集定向** — `run_all_workers_single_data()` 命中所有 DP ranks
2. **需要 GPU 张量** — IPC 路径用 `get_handle_from_tensor()`（CUDA IPC handles），NCCL 路径广播 CUDA 张量，无法从 CPU cache 读取
3. **全局 barrier** — `ray.get(all_futures)` 无逐 shard 完成信号

Feature 4 的 `ModelUpdateService.sync_selected_workers()` 已解决这三个问题。两条路径复用同一传输机制。

#### Active refresh 安全模型

Active refresh 在非重叠 ranks **继续 serving 的同时**推送权重。无 routing 移除，无 drain，无 idle 等待。

**使用与 NeMo 原生 refit 相同的原始更新风格，相同类别的可容忍过渡窗口。** 调查确认：

- NeMo 原生 `update_weights_via_ipc_zmq()` 同样原始：直接调用 `model_runner.model.load_weights()` → 逐参数 `param.data.copy_(loaded_weight)`，无引擎级暂停或锁定（`vllm_backend.py:164-255`，`weight_utils.py:1007`）
- `in_flight_weight_updates=True` 仅影响 trainer 端等待行为（跳过 `wait_for_pending_generations()`），不影响 vLLM 引擎行为（`async_utils.py:558-564`）
- ROLL 通过 `engine_core.collective_rpc_async()`（`async_llm.py:21`）路由更新，调度/扇出更协调，但最终仍通过 `load_weights()`（`worker.py:732`）应用权重 — 相对于 decode 并非更原子

in-flight refresh 期间的过渡窗口是**可容忍的，未消除的**（见下方 Version Accounting）。RLix selective sync 传输在生产负载下是否与 NeMo 原生路径行为一致，仅凭代码追踪未证明 — 必须在 Gate 3 验证。

Drain-then-sync **不在本移植方案范围内**。

#### Control-plane 不变量

**Pipeline 在 `sync_base_weights_to_active()` 完成且 version 发布之前，不得调用 `notify_release_cluster_gpus(actor_train)`。** GPU 释放信号表示”我的 active ranks 权重一致”，而非仅”训练完成”。此顺序由 pipeline 的 `after_training` hook 序列强制执行。

#### Actor call graph

`after_training` hook 在 pipeline actor 内运行。Coordinator 是独立 Ray actor。调用图必须避免对 pipeline actor 的 re-entrant self-call。

**遵循 `sync_lora_weights` 模式**（`coordinator.py:440-500`）：coordinator 直接调用 `ModelUpdateService.sync_selected_workers.remote()` — 不通过 pipeline actor 回路。

```
Pipeline actor (after_training):
  │
  ├── ray.get(coordinator.sync_base_weights_to_active.remote())
  │     │
  │     └── Coordinator actor:
  │           acquire _resize_sync_lock
  │           active_ranks = _active_infer_dp_ranks
  │           ray.get(model_update_service.sync_selected_workers.remote(active_ranks))
  │           release _resize_sync_lock
  │           return                         ← 不回调 pipeline actor
  │
  ├── _finalize_weight_update(active_non_overlap_ranks)  ← 一次性 post-load hooks
  ├── self._current_weight_version = self._cache_ready_step    ← 本地，无 remote call
  ├── ray.get(trajectory_collector.set_weight_version.remote(version))
  └── notify_release_cluster_gpus(actor_train)
```

无 re-entrant call。Coordinator 在锁下执行 sync 后返回。Pipeline 在 coordinator 返回后本地处理 version bookkeeping。

#### Training step 序列

```
1. train_step()
2. build_cpu_bucket_cache(step)          ← 所有 training ranks 参与 gather；
                                            单 cache owner 在 CPU 存储完整模型
3. _cache_ready_step = step
4. offload training GPU / destroy NCCL groups
5. coordinator.sync_base_weights_to_active()  ← coordinator 直接调用 ModelUpdateService
                                                 在 _resize_sync_lock 下（无回路到 pipeline）
5b. _finalize_weight_update(active_ranks)     ← process_weights_after_loading + FP8 hooks，每 worker 一次
6. pipeline 本地更新 version                   ← _current_weight_version = _cache_ready_step
                                                 set_weight_version on collector
7. notify_release_cluster_gpus(actor_train)   ← 在 active refresh + version 发布之后才释放 GPU
8. (later) scheduler resize_infer(add=...)    ← 从同一 cache expand woken overlap ranks
```

#### Hardening

Active refresh 在 **critical post-train path** 上（阻塞 GPU 释放），需要比 expand sync 更强的运维保障：

- **Sender-side `_cache_lock`**：已存在于 ROLL（`megatron_strategy.py:2095-2099`）。在 “cache lookup → transport → NCCL teardown” 窗口期间持有。防止 `build_cpu_bucket_cache` 与进行中的 sync 竞争。必须沿用。
- **Timeout / fail-fast**：如果 sync 挂起（active workers 繁忙，NCCL timeout），training GPU 将被无限期占用。`ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S`（150s）适用。超时时 crash pipeline（符合 “fail fast, no retry” 设计原则）。可能需要与 expand case 不同的调优，因为 active workers 有来自 inference 的 GPU 争用。
- **”因 slept 而安全” vs “在 serving 时安全”**：Expand sync 面向 idle workers（无并发 GPU 活动，无争用）。Active refresh 面向 serving workers（GPU 忙于 inference，并发 NCCL staging 可能造成显存压力）。传输相同但故障模式不同。

#### Version accounting

**问题：独立计数器导致 double-bump。** 如果两条路径各自递增 version 计数器：

```
sync_base_weights: version 2 → 3     (training step 3 的权重)
_expand_workers:   version 3 → 4     ← 错误：相同权重，不同 version
```

**修复：version = `_cache_ready_step`。** Version 绑定到产生 cache 的 training step，而非 sync 操作：

```python
# training step 3 之后：
self._cache_ready_step = 3

# sync_base_weights（active refresh）— pipeline 在 coordinator 返回后更新：
self._current_weight_version = self._cache_ready_step  # = 3
ray.get(self._trajectory_collector.set_weight_version.remote(self._current_weight_version))

# _expand_workers（later，同一 cache）：
# version 已经是 3，确保 collector 看到即可
ray.get(self._trajectory_collector.set_weight_version.remote(self._current_weight_version))
# 不 bump — 相同权重，相同 version
```

**Active in-flight refresh 期间的过渡窗口：**

```
dp2 以 v2 权重 serving
  ├── request A dispatched（v2 权重）
  │   sync_selected_workers 推送 v3 到 dp2
  │   collector version 设为 v3
  ├── request A 完成 → 用 v2 生成，标记 v3  ← 误标
  ├── request B dispatched（v3 权重）→ 正确标记 v3
```

**此过渡窗口可容忍，未消除。** 与 NeMo RL `in_flight_weight_updates=True` 相同类别的权衡。误标仅影响 **weight push 时刻已在 vLLM engine 中 in-flight 的请求** — 这些请求在推送开始前已 dispatched，在推送完成后才返回结果。误标数量受 in-flight batch size 和单次 decode step 延迟约束（通常为个位数请求），而非受 `max_trajectory_age_steps` 约束（后者是 replay buffer 的 lookahead/age window，语义不同，见本文档 Scope 注释）。Version 标签是 **best-effort**：反映 version 何时发布到 collector，不是每个 token 的精确权重状态。如需精确 per-turn version 保真度，需 per-request dispatch-time version tagging — 不在本方案范围内。

#### Path 1: `sync_base_weights_to_active()` — Training loop driven

```python
# Coordinator（与 sync_lora_weights 并行）：
def sync_base_weights_to_active(self) -> None:
    acquired = self._resize_sync_lock.acquire(timeout=_RESIZE_LOCK_TIMEOUT_S)
    if not acquired:
        raise RuntimeError(“sync_base_weights timed out on _resize_sync_lock”)
    try:
        active_ranks = sorted(self._active_infer_dp_ranks)
        if not active_ranks:
            return  # all sleeping, expand will sync on wake
        # 直接调用 ModelUpdateService — 不通过 pipeline actor 回路
        ray.get(self._model_update_service.sync_selected_workers.remote(
            tgt_dp_ranks=active_ranks,
        ))
    finally:
        self._resize_sync_lock.release()
```

Lock 保障：
- `_active_infer_dp_ranks` 快照在 sync 期间稳定
- Scheduler 无法在 mid-sync 执行 shrink/expand
- 不与并发 `resize_infer` 或 `sync_lora_weights` 冲突

#### Path 2: `_expand_workers()` — Scheduler driven

仅处理 overlap ranks 被唤醒的情况：

```python
def _expand_workers(self, *, dp_ranks_to_add: List[int]) -> None:
    # 1. 不把新增 ranks 暴露给 routing
    vllm_generation.mark_dp_ranks_inactive(dp_ranks_to_add)

    # 2. Wake overlap ranks（training 已 offload，GPU VRAM 空闲）
    vllm_generation.wake_up_partial(dp_ranks_to_add)

    # 3. 从同一 CPU bucket cache sync 权重到 overlap ranks
    model_update_service.sync_selected_workers(tgt_dp_ranks=dp_ranks_to_add)

    # 4. Finalize — process_weights_after_loading + FP8 hooks，每 woken worker 一次
    self._finalize_weight_update(dp_ranks_to_add)

    # 5. 发布 version 到 collector（与 active refresh 相同 version — 不 bump）
    ray.get(self._trajectory_collector.set_weight_version.remote(
        self._current_weight_version
    ))

    # 6. 激活 overlap ranks 进入 routing
    vllm_generation.activate_dp_ranks(dp_ranks_to_add)
```

在 `coordinator._resize_sync_lock` 下执行（由 `resize_infer` 持有）。

#### Receiver-side：target worker API surface

`ModelUpdateService.sync_selected_workers()` 和 pipeline 共调用 target inference workers 上的 **6 个方法**。前 5 个由 `ModelUpdateService` 在传输阶段调用（`model_update_service.py:297-397`，`megatron_strategy.py:2240-2370`），第 6 个由 pipeline 在所有 bucket 传输完成后调用。NeMo vLLM workers 必须全部实现：

| 方法 | 调用者 | 传输路径 | 调用时机 |
|---|---|---|---|
| `setup_collective_group(model_update_name, comm_plan, mode, timeout_s)` | ModelUpdateService | NCCL broadcast | Target 有跨 GPU 的 TP peers（tp > 1，非 colocated） |
| `update_parameter_in_bucket(payload_list, is_lora, ipc_local_ranks, model_update_transport)` | ModelUpdateService | IPC（colocated） | Target 与 sender 共享物理 GPU |
| `broadcast_parameter(group_name, names, dtypes, shapes, is_lora, broadcast_local_ranks)` | ModelUpdateService | NCCL broadcast | Target 通过动态 NCCL group 接收 |
| `destroy_collective_group(group_name)` | ModelUpdateService | NCCL broadcast | Sync 完成后；IPC-only ranks 必须有 no-op guard |
| `verify_model(expected_stats)` | ModelUpdateService | Both | Post-sync 验证（可选，由 `verify` flag 控制） |
| `finalize_weight_update()` | Pipeline | — | 所有 bucket 完成后，一次性执行 `process_weights_after_loading()` + FP8 hooks |

**`finalize_weight_update()` 必须在 vLLM worker/backend 上执行**，不在 pipeline actor 上。`process_weights_after_loading(model, model_config, device)`（`vllm_backend.py:181`）和 `_maybe_process_fp8_kv_cache()`（`vllm_backend.py:244`）需要访问 worker 本地的 `model_runner`、`model_config`、`device` — 这些对象不可序列化，无法通过 Ray 传到 pipeline actor。ROLL 的等价路径通过 `base_pipeline.py:89` + `executor/worker.py:215` 以 worker RPC 执行。NeMo 原生路径在 `vllm_backend.py` worker 侧完成 post-load。本方案必须遵循相同模式。

tp=1（Gate 1-4）仅使用 IPC 路径。tp > 1（Gate 2.5+）单个 target worker 可能混合 IPC + broadcast devices。

**Receiver 生命周期：apply many buckets, then finalize once。**

Sender 驱动序列：逐 bucket 调用 `update_parameter_in_bucket.remote()` 或 `broadcast_parameter.remote()`，每 bucket 后 barrier。所有 bucket 应用完成后，pipeline 调用 `finalize_weight_update.remote()` 到每个 target worker — **finalization 在 worker 上执行**，不在 pipeline 上。

```
Per-bucket（ModelUpdateService 调用 N 次）：
  update_parameter_in_bucket() → 反序列化 + load_weights()（仅此 bucket）
  broadcast_parameter()        → 接收 NCCL broadcast + load_weights()（仅此 bucket）

所有 bucket 完成后（pipeline 在 sync_selected_workers 返回后对每个 target worker 调用一次）：
  finalize_weight_update() → worker 内部执行：
    process_weights_after_loading(model, model_config, device)   ← vllm_backend.py:181
  _maybe_process_fp8_kv_cache()                                ← vllm_backend.py:244
```

匹配 ROLL expand 顺序（`gitignored/code_review/2026-03-01-multi-lora-eng123-review.md:834` 确认）：`sync_selected_workers → process_weights_after_loading → load_states_partial`。

**需修改的 NeMo 侧文件：**
- `nemo_rl/models/generation/vllm/vllm_backend.py` — 添加全部 6 个 target-worker 方法（5 个传输方法 + `finalize_weight_update()`）
- `nemo_rl/models/generation/vllm/vllm_generation.py` — 在 worker group 上暴露 receiver，供 `ModelUpdateService.sync_selected_workers()` 通过 `tgt_cluster.rank2worker[rank]` 调用

#### Sequence diagram

```
Scheduler              Coordinator              Pipeline                    vLLM Engines
   │                      │                        │                        dp0 dp1 dp2 dp3
   │                      │                        │                         ●   ●   ●   ●  (v2)
   │                      │                        │
   │ resize_infer(rm=[0,1])                        │
   │─────────────────────>│  lock                  │
   │                      │───────────────────────>│ _shrink_workers([0,1])
   │                      │                        │─────────────────────> 😴  😴  ●   ●
   │                      │  _active={2,3}         │
   │                      │  unlock                │
   │                      │                        │
   │              [ training on GPUs 0,1 ]         │                       😴  😴  ●   ●  (dp2,3 serve v2)
   │                      │                        │
   │                      │                        │ after_training(step=3):
   │                      │                        │   build_cpu_cache(step=3)
   │                      │                        │   _cache_ready_step = 3
   │                      │                        │   offload training GPU
   │                      │                        │
   │                      │← sync_base_weights ────│ ray.get(coordinator.sync_base_weights_to_active())
   │                      │  lock                  │
   │                      │  _active={2,3}         │
   │                      │── model_update_service.sync([2,3]) ──────────> 😴  😴  ●→v3 ●→v3 (in-flight)
   │                      │  unlock                │
   │                      │── return ──────────────│
   │                      │                        │ finalize_weight_update([2,3])
   │                      │                        │ version = 3 (local)
   │                      │                        │ set_weight_version(3) on collector
   │                      │                        │ notify_release(actor_train)
   │                      │                        │
   │ resize_infer(add=[0,1])                       │
   │─────────────────────>│  lock                  │
   │                      │───────────────────────>│ _expand_workers([0,1])
   │                      │                        │── wake([0,1])        ⏳  ⏳  ●v3 ●v3
   │                      │                        │── sync([0,1])        ✓v3 ✓v3 ●v3 ●v3
   │                      │                        │── finalize([0,1])
   │                      │                        │── publish version 3 (no bump)
   │                      │                        │── activate([0,1])    ●v3 ●v3 ●v3 ●v3
   │                      │  _active={0,1,2,3}     │
   │                      │  unlock                │
```

#### Edge cases

1. **全部 ranks 重叠（退化 = ROLL 拓扑）**：shrink 后 `_active_infer_dp_ranks` 为空。`sync_base_weights_to_active()` 立即返回。Expand sync 所有 ranks on wake。正确。
2. **无重叠（所有 ranks 非重叠）**：不发生 shrink/expand。`sync_base_weights_to_active()` in-flight sync 所有 ranks。正确。
3. **Init 后首步**：CPU cache 有 base weights（`_cache_ready_step = -1`）。Active refresh 推送 base weights 到 active ranks。Expand 推送到 woken ranks。全部 version -1。正确。

#### 改动量

- `rlix/protocol/coordinator.py` — 添加 `sync_base_weights_to_active()` 抽象方法
- `rlix/pipeline/coordinator.py` — 实现 `sync_base_weights_to_active()`，在 `_resize_sync_lock` 下直接调用 ModelUpdateService
- `rlix/pipeline/nemo_rl_pipeline.py`（新）— `_expand_workers()`、`_after_training` hook、`_finalize_weight_update()`
- `nemo_rl/models/generation/vllm/vllm_backend.py` — 实现全部 6 个 target-worker 方法（5 个传输方法 + `finalize_weight_update()`）
- `nemo_rl/models/generation/vllm/vllm_generation.py` — 在 worker group 上暴露 receiver

~200 行（两条路径 + version accounting + receiver API surface，不含 Feature 5 原有的 pipeline adapter / init bootstrap / config bridge 部分）

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
4. **审计所有 NeMo RL child actors** — NeMo RL 的 `AsyncTrajectoryCollector` 和 `ReplayBuffer` 是 Ray actors（`grpo.py:2496,2519`）。当前它们是匿名创建的（无 `name=` 参数），因此不会产生跨 pipeline 的命名冲突。但如果后续 NeMo 代码给这些 actors 加 `name=`（或者需要通过 `ray.get_actor()` 跨 actor 查找），匿名创建就不够了。**建议**：对这些 child actors 显式传入 `namespace=ray_namespace`（从 `runtime_env` 的 `ROLL_RAY_NAMESPACE` env var 读取），作为 consistency / future-proofing 措施，而非解决当前已知冲突
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
# Driver 脚本 — 声明式注册（匹配 RLix 实际 API）
from rlix.protocol.types import PipelineType, get_pipeline_namespace

cluster_device_mappings = {
    "actor_train": list(train_device_mapping),          # e.g. [0,1,2,3]
    "actor_infer": list(infer_device_mapping),          # e.g. [0,1,2,3,4,5,6,7]
}
cluster_tp_configs = {
    "actor_train": 1,                                    # Megatron: 固定 1 GPU/worker（bridge canonicalize）
    "actor_infer": vllm_cfg.get("tensor_parallel_size", 1),  # vLLM: tp
}

# Step 1: 分配 pipeline ID（需要 pipeline_type 参数）
pipeline_id = ray.get(
    orchestrator.allocate_pipeline_id.remote(PipelineType.FULL_FINETUNE)
)
ray_namespace = get_pipeline_namespace(pipeline_id)

# Step 2: 注册 GPU 拓扑
ray.get(
    orchestrator.register_pipeline.remote(
        pipeline_id=pipeline_id,
        ray_namespace=ray_namespace,
        cluster_tp_configs=cluster_tp_configs,
        cluster_device_mappings=cluster_device_mappings,
    )
)

# Step 3: 准入 — scheduler 开始为该 pipeline 分配 GPU
ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))

# Step 4: 创建 coordinator actor（需要 pipeline_id + pipeline_config）
coordinator = PipelineCoordinator.options(
    name=f"rlix:coordinator:{pipeline_id}",
    namespace=ray_namespace,
).remote(
    pipeline_id=pipeline_id,
    pipeline_config=nemo_config,
)

# Step 5: 创建 pipeline actor（内部 init bootstrap — 见 Feature 5+6）
ray.get(coordinator.create_pipeline_actor.remote())
```

`cluster_device_mappings` 必须来自 **配置中声明的实际 train/infer device_mapping**，不是 `list(range(n))` 这种连续 GPU toy 示例的泛化版。上面的示例仅用于单机连续编号说明；正式实现应直接读取 NeMo config / bridge canonicalize 后的 mapping，以支持非连续和多节点场景。

**顺序契约：** driver 必须先 `allocate_pipeline_id` → `register_pipeline` → `admit_pipeline`，再创建 coordinator actor。`PipelineCoordinator.__init__` 已把这视为前置条件（`coordinator.py:183`）。

Coordinator 保持不变 — 正常创建 `RollResourceManagerProxy` singleton。PG 共享和 worker bundle mapping 见 Feature 12。

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
| `metrics["collected"]` | `_local_intended_count`（raw，**不 clamp**） | reporter 端原始计数；`ReplayBuffer` 中 `target_weight_version == current_weight_version` 的可用条目数 |
| `metrics["completed"]`（仅 coordinator → scheduler 段） | `min(total_collected, total_required)` | coordinator `_aggregate_and_emit` 端做 clamp，对齐 ROLL 协议 |

**Wire 协议两层分工**（与 ROLL [rollout_scheduler.py:621-635](external/ROLL/roll/distributed/scheduler/rollout_scheduler.py#L621) 严格对齐）：

- **Reporter → coordinator**：`metrics["collected"]`（raw `_local_intended_count`，**不 clamp**）
- **Coordinator → scheduler**：`metrics["completed"]` = `min(total_collected, total_required)`

Reporter 端 clamp 在单 stream 下结果等价，但破坏多 stream（LoRA / val concurrent）聚合数学；NeMo 当前 scope 单 stream 不暴露 bug，但 wire 协议对齐 ROLL 是 forward-compat 必需。

**关键点：不能用 age-window `valid_count`。**
RLix scheduler 用 `completed` 推导 remaining demand（`scheduler.py:827`：`remaining = max(step_target - completed, 0)`）；而 NeMo training 真正等待的是"当前 step 的 intended trajectories 是否够数"（`ReplayBuffer.sample()` 显式过滤 `target_weight_version == current_weight_version`，见 `async_utils.py:102,167`）。如果把 future-targeted 或仅 age-valid 的轨迹也计入 completed，会错误低估 remaining demand，导致 scheduler 过早 shrink。`max_trajectory_age_steps` 保留在 sampling 逻辑中（它属于那里），但从 progress metric 中移除。

**Demand window = inter-training-step collection period：**
- expand + selective sync 完成后，新激活 shards 重新进入 routing，collector 为下一个 step 继续积累 buffer
- `intended_ready_count` 逐渐增长 → `completed` 从 0 趋近 `num_prompts_per_step`
- `replay_buffer.sample()` 成功时 training step 开始 → 下一轮 shrink

**上报时机：** 保持与 ROLL 一致的 lifecycle。ROLL 不是在 training start reset progress，而是在 `get_batch()` 开始时 `begin_progress_batch()` 激活/重置，在 `get_batch()` 返回后 `end_progress_batch()` 清除（`rollout_scheduler.py:672-698,1043-1086`）。NeMo RL 对应的 active-demand window 不是 train compute，而是 `replay_buffer.sample(...)` 的等待窗口（`grpo.py:2646` 一带）。

**因此 NeMo 侧要改成 batch-begin / batch-end 语义：**
- `begin_progress_batch(current_weight_version)`：在 training loop 进入 `replay_buffer.sample(...)` 等待前调用。作用是：
  1. 激活 progress stream
  2. 设置 `_progress_target_step = current_weight_version`
  3. 从 `ReplayBuffer` 一次性读取当前 step 已就绪的 intended count，作为初始 completed
  4. 计算 bucket 并立即发送 `new_batch=True` 的首个快照
- `end_progress_batch()`：放在包裹整个 `sample(...)` wait window 的 `finally` 中。无论成功拿到足量 trajectories，还是等待过程中抛异常，都必须清除 progress stream，防止 stale demand 残留到 scheduler。语义对齐 ROLL `get_batch()` 的 `finally: end_progress_batch()`，但作用域是 NeMo 的 sample-wait loop，而不是每次 `sample()==None`

**为什么必须有 batch-begin snapshot：** 不能像旧草案那样在“training step 开始时 reset local counter=0”。因为 NeMo collector 会持续 prefetch，当前 step 的一部分 intended trajectories 可能在进入 sample wait 之前就已经在 buffer 里了。若直接 reset 为 0，会低估 completed，和 ROLL 的 batch-open snapshot 语义不一致。

**避免 hot-path 阻塞：** `ReplayBuffer` 查询 intended count 的 `ray.get` 只放在 `begin_progress_batch()` 这一处，一次 batch 一次，不放在每次 push 的 hot path。push 成功后仍由 collector 维护本地增量计数器（只对 `target_weight_version == _progress_target_step` 的成功 push 做 `+1`），用来触发后续 2% bucket 上报。

**实现：**

```python
# AsyncTrajectoryCollector 构造时注入 rlix_hooks（不依赖全局单例）
class AsyncTrajectoryCollector:
    def __init__(self, ..., rlix_hooks=None):
        self._rlix_hooks = rlix_hooks or NoOpRLixHooks()
        self._progress_active = False
        self._last_progress_bucket = -1  # 2% granularity
        self._local_intended_count = 0   # batch-begin snapshot + local increments
        self._progress_target_step = -1  # current replay_buffer.sample target version

# grpo.py: before entering replay_buffer.sample(...) wait loop
def begin_progress_batch(self, current_weight_version):
    self._progress_active = True
    self._progress_target_step = current_weight_version
    self._local_intended_count = ray.get(
        self.replay_buffer.count_intended_for_step.remote(current_weight_version)
    )
    # Reporter 端只用 raw count 算 bucket（与 ROLL rollout_scheduler.py:605 同 pattern）；
    # 不在 reporter clamp completed — coordinator 端聚合后才 clamp。
    bucket = int(min(self._local_intended_count, self._num_prompts_per_step)
                 / self._num_prompts_per_step * 50)
    self._last_progress_bucket = bucket
    self._rlix_hooks.report_progress(
        step_target_trajectories=self._num_prompts_per_step,
        collected=self._local_intended_count,   # ← raw collected，不 clamp
        new_batch=True,
    )

# grpo.py: in finally wrapping the entire sample(...) wait window
def end_progress_batch(self):
    self._progress_active = False
    self._progress_target_step = -1
    self._local_intended_count = 0
    self._last_progress_bucket = -1
    self._rlix_hooks.clear_progress()

# _run_prompt_group_worker 中，push 成功后上报
def _run_prompt_group_worker(self, ...):
    ...
    # NeMo 的 push_with_wait_signal 签名：
    #   push_with_wait_signal(trajectory, weight_version, target_weight_version)
    # target_weight_version 是 _process_batch() 从 _calculate_target_weights() 预分配的，
    # 可能是当前 step 或 future step（async_utils.py:294,458,695）
    replay_buffer.push_with_wait_signal.remote(
        trajectory, weight_version, target_weight_version
    )

    # 上报 progress（fire-and-forget, 2% 变化阈值）
    # 只计 target_weight_version == 当前 training step 的轨迹。
    # NeMo collector 会为 future target weights 生成轨迹，这些不算当前 step 的 demand。
    #
    # _progress_target_step 定义：
    #   = 当前 training loop 正在等待 sample 的 weight_version
    #   = grpo.py 中 replay_buffer.sample(current_weight_version=weight_version) 的 weight_version
    #   由 begin_progress_batch(weight_version) 设置，end_progress_batch() 后 reset
    #   初始值 = _cache_ready_step（init bootstrap 的 base model version）
    #
    # 本地计数器避免 push hot-path 上的额外 ray.get：
    if self._progress_active and target_weight_version == self._progress_target_step:
        self._local_intended_count += 1
    # bucket 用 clamped 值算（百分比上限 100% = bucket 50），但发给 coordinator 的是
    # raw collected（不 clamp）— 与 ROLL rollout_scheduler.py:605,627 严格对齐。
    bucket = int(min(self._local_intended_count, self._num_prompts_per_step)
                 / self._num_prompts_per_step * 50)
    if bucket != self._last_progress_bucket:
        self._last_progress_bucket = bucket
        self._rlix_hooks.report_progress(
            step_target_trajectories=self._num_prompts_per_step,
            collected=self._local_intended_count,   # ← raw collected，coordinator 端聚合 + clamp
            new_batch=False,
        )
```

`NemoRLRLixHooks.report_progress(step_target_trajectories, collected, new_batch)` 构造 `ProgressReport(metrics={"collected": collected, "bucket": ..., "mode": ..., "new_batch": ..., ...})` 并 fire-and-forget 发给 coordinator。**`metrics["collected"]` 是 raw `_local_intended_count`，不 clamp**；coordinator `_aggregate_and_emit` 端做 `completed = min(total_collected, total_required)` 聚合 + clamp 给 scheduler（[rollout_scheduler.py:621-635](external/ROLL/roll/distributed/scheduler/rollout_scheduler.py#L621) 同协议）。首个 batch-begin 快照带 `new_batch=True`，后续 bucket 更新带 `new_batch=False`，与 ROLL 对齐。`clear_progress` 调用 `coordinator.clear_progress_stream(mode, adapter_id)`（`coordinator.py:326`）— 注意 API 是 `clear_progress_stream` 不是 `clear_progress`，coordinator 聚合层负责判断是否还有其他 active streams 并决定是否通知 scheduler。

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

**RLix resize safety 不依赖 `_refit_pause_cleared`。** NeMo 原生 `prepare_for_refit()` / `resume_after_refit()` 使用 `_refit_pause_cleared` Event 做 admission control（`async_utils.py:542,601`），该机制在 check-to-start 窗口存在已知 non-atomicity（archived plan `adaptation_nemo_rl.md` Section 1.1 记录）。RLix 模式下 resize safety 完全由 generation 层的 routing-state 变更（`_active_dp_ranks` / `_preempted_shards`）加 abort-drain-sleep 保证（Feature 2+3），不经过 `_refit_pause_cleared` 路径。

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
    """Local helper — 不修改上游 Megatron。不调用 destroy_model_parallel()。"""
    from megatron.core import parallel_state
    # 1. 从 parallel_state 收集所有非 None 的 process groups
    # 2. 过滤出 NCCL backend groups（排除 Gloo）
    # 3. 去重 handles
    # 4. 对每个调用 torch.distributed.destroy_process_group(pg)
    # 5. 清理本地 parallel_state cache / globals（使用本地 reset helper）
    #    暂不依赖 destroy_model_parallel() 作为 RLix offload 的唯一机制；
    #    虽然它是官方 cleanup API（NeMo 自身在 setup.py:108,1022 中调用），
    #    但针对长生命周期 worker 的反复 destroy/re-init + VRAM 回收语义，
    #    本方案尚未验证。Gate 2.5 之前不把它当作已证明可用的路径。
    # 下次训练 / checkpoint / eval 前：显式调用 initialize_model_parallel(...) 重建
```

**NCCL teardown 策略：manual PG destroy + local state reset + explicit re-init。** `destroy_model_parallel()` 是 Megatron 官方 cleanup API（reset parallel_state globals + destroy process groups），NeMo 自身已在 `setup.py:108,1022` 中调用。但针对 RLix 的特定生命周期（长生命周期 worker 中反复 destroy/re-init + 精确 VRAM 回收），本方案尚未验证其行为是否完全匹配需求。Gate 2.5 是验证点；如果 `destroy_model_parallel()` + `initialize_model_parallel()` 循环在 Gate 2.5 中被证明可靠，可简化为直接使用它。

已知风险：`parallel_state` 可能不是唯一 owner（其他 Megatron 模块可能缓存 group handle 引用）；反复 `destroy → initialize` 在长生命周期 worker 中需要 Gate 2.5 验证。如果 NeMo RL 在 offload 状态下做 checkpoint/export/eval，必须先 reload comm state。**这是 Feature 5+6 之外风险最高的项**。

**Gate 2.5 fallback rule（必须显式写明）：**
1. 默认实现先验证 `destroy_megatron_nccl_groups()`（manual PG destroy + local reset）。
2. 若 Gate 2.5 发现任一失败模式：VRAM 未释放到阈值、下一轮 `initialize_model_parallel()` 失败、出现 stale/dangling PG handle、或 3+ step destroy/re-init 不稳定，则**唯一允许的替代实现**是切换为 Megatron 官方 `destroy_model_parallel()` → `initialize_model_parallel()` 全量循环。
3. 若官方 cleanup 循环仍不能稳定通过 Gate 2.5，则 **tp>1 selective-sync / time-sharing 路径保持 blocked，不带条件地继续推进实现是不允许的**。此时应将 Feature 11 标记为未通过 gate，而不是保留一个“理论可用”的 manual teardown 方案。

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

**统一策略：RLix 模式下不创建 NeMo 自己的 placement groups；改为提供一个 `RayVirtualCluster`-compatible adapter，底层复用 `RollResourceManagerProxy` 的 shared PGs。**

原因：`VllmGeneration` 和 `RayWorkerGroup` 当前都要求 `cluster: RayVirtualCluster` 类型语义（`world_size()`, `get_placement_groups()`, bundle ordering 等，见 `vllm_generation.py:46`，`worker_groups.py:316`）。因此不能只传裸 `bundle_indices_list`，必须保留 cluster abstraction。

实现：

1. **新增 `RLixVirtualClusterAdapter`**：
   - 不创建 placement groups — 底层复用 `RollResourceManagerProxy.allocate_placement_group(...)` 返回的 shared PG allocation
   - 实现 NeMo 当前实际用到的最小接口：
     - `world_size()`
     - `get_placement_groups()`
     - bundle ordering / sorted bundle indices helpers
     - `num_gpus_per_node`
   - 未实现的 `RayVirtualCluster` 方法 raise `NotImplementedError`（fail fast 捕获意外调用）
2. `VllmGeneration` / `RayWorkerGroup` 继续接收 cluster-like object，不需要大改调用层
3. standalone 模式仍用原生 `RayVirtualCluster`；RLix 模式下注入 adapter

```python
# NeMo RL initialize_pipeline 内：
proxy = RollResourceManagerProxy(num_gpus_per_node=num_gpus_per_node)

# inference workers: 全部 GPU
infer_pg_alloc = proxy.allocate_placement_group(
    world_size=infer_dp_size, device_mapping=list(range(total_gpus))
)
infer_cluster = RLixVirtualClusterAdapter(pg_alloc=infer_pg_alloc, ...)

# training workers: overlap GPU 子集
# world_size = total Megatron workers (1 GPU each, worker_config.py:231), NOT dp_size
train_pg_alloc = proxy.allocate_placement_group(
    world_size=len(train_device_mapping), device_mapping=train_device_mapping
)
train_cluster = RLixVirtualClusterAdapter(pg_alloc=train_pg_alloc, ...)

# VllmGeneration / RayWorkerGroup 接收 adapter，接口兼容
policy_generation = VllmGeneration(cluster=infer_cluster, ...)
```

Coordinator 保持不变 — 所有 backend 都用 `RollResourceManagerProxy`。原生 `RayVirtualCluster` 仅用于 standalone 模式。

改动量：~120 行（`RLixVirtualClusterAdapter` + shared-PG 接入 + pipeline 内 adapter 注入）

**新增文件：** `nemo_rl/distributed/rlix_virtual_cluster.py`（adapter 实现）

---

## 测试策略

### 验证环境

vast.ai 2x 3060（$0.15/hr）= **2 GPU**。所有 Gate 按 2 GPU 设计。

### 测试工作负载

NeMo RL 自带的 **calculator multiturn async GRPO example**（简单、有单元测试可参考）。

### 分步验证

1. **先跑通单个 NeMo RL pipeline** — 验证 Feature 1-3 + Feature 6 + Feature 8-10 + Feature 12（partial sleep/wake + routing + selective sync + registration + progress + validation + shared PG）
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
8. 若 step 3-7 任一失败，按 Feature 11 fallback rule 切到 `destroy_model_parallel()` → `initialize_model_parallel()` 重试一次；若仍失败，则 Gate 2.5 判定失败，tp>1 路径不放行
预期：无 NCCL 错误，无 VRAM 泄漏（每轮 peak VRAM 稳定），权重正确
关键：这是唯一覆盖 NCCL broadcast transport 和 Megatron NCCL lifecycle 的 gate。
      dp=1 意味着没有 partial overlap（所有 GPU 都 overlap），但足以验证 transport 和 NCCL 生命周期。
```

### Gate 3: 单 pipeline 端到端 async GRPO

```
配置：2 GPU, dp=2, tp=1, async GRPO, calculator example
测试：完整 async training loop — `actor_infer` 持有长期 GENERATION allocation，
     generation 在后台持续，training 时 shrink dp[1]，
     after_training: active refresh (in-flight sync dp[0]) → version publish → GPU release，
     scheduler expand dp[1] → selective sync + finalize + 原子激活，
     非重叠 shard 无全局 pause，
     `after_training(step)` 完成前 `before_training(step+1)` 不进入下一轮 train，
     version 一致性：两条路径 publish 同一 `_cache_ready_step`（无 double-bump），
     过渡窗口：in-flight active refresh 期间少量请求可能误标（tolerated，非 eliminated），
     **此 gate 是 in-flight active refresh 在生产负载下的主要验证点**
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
| `nccl_offload.py` (**新增**) | F1, F11 | Megatron NCCL group 手动 destroy/reload（从 `parallel_state` 收集 NCCL groups + `torch.distributed.destroy_process_group` + re-init；暂不依赖 `destroy_model_parallel()` 作为唯一机制，Gate 2.5 验证后可简化） | +90 |
| `grpo.py` | F5, F11 | `async_grpo_train()` training hook 调用点 + `DO_TIME_SHARING` 行为分支 | +60 |
| `async_utils.py` | F9 | `AsyncTrajectoryCollector` 按 batch-begin / batch-end lifecycle 上报 progress（`begin_progress_batch` 初始 snapshot + 2% bucket 阈值 + 本地计数器避免 hot-path ray.get, 仅计 `target_weight_version == current` 的 push）+ `ReplayBuffer.count_intended_for_step(current_weight_version)`（`ReplayBuffer` 定义在同一文件 `async_utils.py:35`，不拆文件） | +60 |
| `rlix_hooks.py` (**新增**) | F5, F9 | `RLixHooks` protocol + `NoOpRLixHooks` 默认实现（NeMo/RLix 共享 import seam） | +30 |
| `rlix_virtual_cluster.py` (**新增**) | F12 | `RLixVirtualClusterAdapter`（`RayVirtualCluster`-compatible adapter，底层复用 shared PG allocation） | +80 |

### RLix 侧

| 文件 | Feature | 改动 | 行数 |
|------|---------|------|------|
| `nemo_rl_pipeline.py` (**新增**) | F5, F6, F8, F10, F12 | NemoRLFullFinetunePipeline（含 resize_infer, expand+selective sync, registration, validation, shared-PG `bundle_indices_list` helper） | +420 |
| `nemo_rl_model_update_service.py` (**新增**) | F4, F6 | 简化版 ModelUpdateService（selective sync, CUDA IPC + 动态 NCCL group 生命周期, 无 versioning） | +250 |
| `nemo_rl_config_bridge.py` (**新增**) | F5, F8 | ConfigBridge + 声明式 registration helper（见下方必须提供的属性清单） | +100 |

### 测试

| 文件 | 改动 | 行数 |
|------|------|------|
| `tests/test_partial_sleep_wake.py` (**新增**) | Feature 1-3 单元测试 | +150 |
| `tests/test_nemo_rl_pipeline.py` (**新增**) | Feature 5-6 集成测试 | +200 |

**总计：~1700 行**

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
  ├── Day 5-6: Feature 6 — expand + selective sync 原子操作
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

---

## 附录：术语映射（Archived Plan ↔ New Plan）

| Archived Plan (`adaptation_nemo_rl.md`) | New Plan / NeMo RL 实际代码 | 说明 |
|---|---|---|
| `active_checkpoint_version` | `_cache_ready_step` | 产生 CPU cache 的 training step |
| `generation_checkpoint_version` | `generation_weight_version` | 轨迹生成时的权重版本 |
| `SchedRLProxy` | `rlix_hooks` + `DO_TIME_SHARING` flag | 代理层改为 hooks + flag 模式 |
| `migration_policy=REQUEST_RETRY` | abort-drain-sleep + `ShardPreemptedError` retry | 不再需要 per-request tracking |
| `expand_workers(indices)` / `shrink_workers(indices)` | `wake_up_partial(dp_ranks)` / `sleep_partial(dp_ranks)` | DP-leader-only 调用，vLLM 内部传播 |
| `oldest_unfinished_creation_ts` | （未移植） | 当前 RLix scheduler 不消费此字段 |
| `queued_trajectories` / `inflight_trajectories` | `step_target_trajectories` + `completed` | 连续快照模型，非离散 batch |
