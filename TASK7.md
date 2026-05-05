# Task 7 — Scheduler-Driven Shrink/Expand + Atomic Expand

## 背景

NeMo RL 的 `async_grpo_train()` 是一个闭环训练主循环。RLix 调度器需要在训练步骤边界介入，在训练阶段回收 overlap GPU 给推理，在训练完成后把 GPU 还给推理并做选择性权重同步。

Task 7 在训练循环里打入 hook 接口，让调度器能在 before/after training 时驱动 shrink/expand；并保证 expand 操作的原子性：worker 被唤醒后，只有在权重同步和版本更新都成功后，才会被加入推理路由。

Task 3 依赖（NCCL）在 `after_training` 里用 TODO 注释标出，Task 4/11/12 的调度器 RPC 同样是 TODO 占位。

---

## 文件结构

```
RL/ (NeMo RL repo, branch: feat/nemo-rl-rlix-f5-f6)
└── nemo_rl/algorithms/
    ├── rlix_hooks.py                  # hook 接口定义（新增）
    └── grpo.py                        # async_grpo_train 插桩（修改）

rlix/ (RLix repo, branch: feat/nemo-rl-pipeline-adapter)
└── rlix/pipeline/
    ├── nemo_rl_pipeline.py            # NemoRLRLixHooks + NemoRLFullFinetunePipeline
    ├── nemo_rl_config_bridge.py       # 配置适配（已有）
    └── nemo_rl_model_update_service.py  # 权重同步 stub（待 Task 4 实现）

tests/
├── test_f6_expand_atomic.py           # atomic expand 单元测试（17 个）
└── test_nemo_rl_pipeline.py           # F5/F6 集成测试（31 个）

RL/examples/configs/
└── grpo_async_qwen0.5b.yaml           # VastAI 2-GPU 测试配置
```

---

## Hook 接口与插桩

### `RLixHooksProtocol`（`RL/nemo_rl/algorithms/rlix_hooks.py`）

```python
@runtime_checkable
class RLixHooksProtocol(Protocol):
    def before_training(self, step: int) -> None: ...
    # RLix 模式下阻塞，等调度器批准 actor_train GPU 分配
    # 调度器在批准前异步 shrink overlap 推理 workers

    def after_training(self, step: int) -> None: ...
    # 通知调度器 actor_train GPU 已释放
    # 调度器异步触发 coordinator.resize_infer(add=overlap_ranks) → _expand_workers

    def on_trajectory_collector_created(self, collector: Any) -> None: ...
    # 把 AsyncTrajectoryCollector handle 注册到 pipeline
    # _expand_workers 用它调用 set_weight_version
```

用 `@runtime_checkable` + `Protocol`，不需要继承，`isinstance()` 可做类型校验。

### `NoOpRLixHooks`

所有方法 `pass`。NeMo RL 单独运行时默认使用，零侵入原有行为。

### `NemoRLRLixHooks`

实际调度器集成，注入到 `async_grpo_train` 的 `rlix_hooks` 参数。持有同一 Ray actor 内的 pipeline 引用，无需 remote call。

```python
NemoRLRLixHooks(pipeline=<NemoRLFullFinetunePipeline self>)
```

### `grpo.py` 插桩（`DO_TIME_SHARING` 开关）

```python
DO_TIME_SHARING: bool = os.environ.get("RLIX_CONTROL_PLANE") == "rlix"

def async_grpo_train(config, ..., rlix_hooks=None):
    hooks = rlix_hooks if rlix_hooks is not None else NoOpRLixHooks()

    # 训练开始：注册 collector（_expand_workers 的前置依赖）
    hooks.on_trajectory_collector_created(trajectory_collector)

    for step in range(num_steps):
        # ...轨迹收集...

        hooks.before_training(step)          # Hook 1: 请求 training GPU
        # policy.logprob_inference → policy.train
        if DO_TIME_SHARING:
            # TODO(Task 4): policy.build_cpu_bucket_cache(step)
            # TODO(Task 11): policy.offload_training_gpu() + destroy_nccl_groups()
            hooks.after_training(step)       # Hook 2: 释放 training GPU，触发 expand
            weight_version += 1
        else:
            # 原有 refit 路径（standalone 模式不变）
            ...
```

---

## Atomic Expand（`_expand_workers`）

### 五步原子序列

```
Step 1  mark_dp_ranks_inactive(ranks)
        ↓  明确把 ranks 排出路由（幂等，sleeping ranks 本已不在路由里）
Step 2  wake_up_partial(ranks)
        ↓  唤醒 vLLM worker，GPU VRAM 恢复；ranks 进入 _pre_activation_ranks
        ──────────────── try/except 保护起点 ────────────────
Step 3  ray.get(model_update_service.sync_selected_workers.remote(tgt_dp_ranks=ranks))
        ↓  只同步唤醒的 shards，不暂停全局推理（Task 4 CPU bucket → GPU）
Step 4  ray.get(trajectory_collector.set_weight_version.remote(new_version))
        ↓  collector 先知道新版本，再让 shard 上线，防止新轨迹被打旧版本号
        _current_weight_version = new_version  ← 只有 remote call 成功后才更新
Step 5  activate_dp_ranks(ranks)
        ↓  ranks 加入推理路由；_pre_activation_ranks → _active_dp_ranks
```

**核心不变式：`activate_dp_ranks` 只有在步骤 3 和 4 都成功后才会执行。**

### 失败状态

步骤 3-5 任意一步抛异常：
- ranks 留在 `_pre_activation_ranks`（已唤醒但不在路由里，不会用旧权重服务请求）
- `_current_weight_version` 不变
- 调用方可检查 `pipeline._pre_activation_ranks` 诊断失败

### `_shrink_workers`

```python
asyncio.run(self._policy_generation.sleep_partial(dp_ranks_to_remove, level=2))
# Task 2: abort_all_requests → drain（等 engine idle）→ sleep（释放 VRAM）
```

### `resize_infer` 入口

```python
def resize_infer(*, dp_ranks_to_remove, dp_ranks_to_add) -> ActionResponse:
    validate_resize_params(...)          # exactly one of remove/add must be non-empty
    with self._infer_resize_lock:
        if dp_ranks_to_remove:
            self._shrink_workers(...)
        else:
            self._expand_workers(...)
    return ActionResponse(success=True)
```

---

## 测试覆盖（48 个，全部 pass，无 GPU / Ray / torch 依赖）

```bash
cd rlix/
python -m pytest tests/test_f6_expand_atomic.py tests/test_nemo_rl_pipeline.py -v
# 48 passed in 0.14s
```

### `test_f6_expand_atomic.py` — 17 个

| 测试类 | 测试数 | 验证内容 |
|--------|--------|---------|
| `TestF6ExpandAtomicHappyPath` | 5 | 五步顺序、版本递增、collector 版本、_active_dp_ranks 更新、_pre_activation 清空 |
| `TestF6ExpandAtomicSyncFailure` | 4 | sync 失败时 activate 不调用、版本不变、ranks 留在 pre_activation |
| `TestF6ExpandAtomicSetVersionFailure` | 3 | set_weight_version 失败时 activate 不调用、版本不变 |
| `TestF6ExpandAtomicMissingDeps` | 3 | policy_generation / model_update_service / trajectory_collector 为 None 时 raise |
| `TestF6ExpandMultipleSteps` | 2 | 多次 expand 版本累积正确、全局调用顺序 |

### `test_nemo_rl_pipeline.py` — 31 个

| 测试类 | 测试数 | 验证内容 |
|--------|--------|---------|
| `TestHookTiming` | 4 | before→train→after 顺序、step 号正确、真实 hooks 调用 scheduler RPC、collector 注册 |
| `TestResizeInferDispatch` | 5 | shrink/expand 路由、同时传 remove+add 报错、两者都空报错、返回 ActionResponse |
| `TestExpandWorkersAtomic` | 5 | 五步顺序、版本递增、collector 版本、_pre_activation 清空、_active_dp_ranks |
| `TestExpandWorkersSyncFailure` | 7 | sync 失败 / set_version 失败时各状态不变量 |
| `TestShrinkWorkers` | 4 | sleep_partial 被调用、_active_dp_ranks 更新、level=2、空 ranks 报错 |
| `TestMissingDependencies` | 3 | 三种 None dep 均 raise |
| `TestMinimalIntegrationFlow` | 3 | 完整 shrink/expand 生命周期、多 step、失败后二次 expand 恢复 |

---

## 状态字段

| 字段 | 含义 |
|------|------|
| `_active_dp_ranks` | 当前在推理路由中的 DP ranks |
| `_pre_activation_ranks` | 已唤醒但尚未进入路由的 ranks（步骤 2 之后、步骤 5 之前，或失败驻留） |
| `_current_weight_version` | 本地权重版本号，与 collector 版本严格同步 |

---

## 未实现（有意 TODO，等对应 Task 完成后填入）

| 位置 | 等待 | 内容 |
|------|------|------|
| `_expand_workers` Step 3 | Task 4 | `NemoRLModelUpdateService.sync_selected_workers` CPU bucket cache → GPU 传输 |
| `_expand_workers` Step 2 | Task 2 | `VllmGeneration.wake_up_partial` 真实 VRAM 恢复 |
| `_expand_workers` Step 1/5 | Task 2/3 | `mark_dp_ranks_inactive` / `activate_dp_ranks` 真实路由切换 |
| `_shrink_workers` | Task 2 | `VllmGeneration.sleep_partial` abort → drain → sleep |
| `after_training` | Task 11 | `policy.offload_training_gpu()` + `destroy_nccl_groups()` |
| `after_training` | Task 4 | `policy.build_cpu_bucket_cache(step)` |
| `initialize_pipeline` | Task 12 | 共享 PlacementGroup（`RollResourceManagerProxy`） |

---

## 运行配置（VastAI 2-GPU 验证）

`RL/examples/configs/grpo_async_qwen0.5b.yaml` — 关键参数：

| 参数 | 值 | 说明 |
|------|----|------|
| `policy.model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | 最小 Qwen2.5，~1GB 权重 |
| `grpo.async_grpo.enabled` | `true` | 启用异步 GRPO |
| `grpo.async_grpo.max_trajectory_age_steps` | `2` | 允许 2 步 off-policy |
| `loss_fn.use_importance_sampling_correction` | `true` | 异步模式必须开启 |
| `policy.generation.vllm_cfg.async_engine` | `true` | vLLM 异步引擎 |
| `policy.generation.colocated.enabled` | `false` | 非 colocated，async GRPO 要求 |
| `policy.max_total_sequence_length` | `256` | 小 VM 省显存 |
| `cluster.gpus_per_node` | `2` | 2× GPU VastAI 节点 |
