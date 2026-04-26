# Task 5 & 6 — RLix Hooks + Progress Reporting

## 背景

`grpo_train()` 是 NeMo RL 的训练主循环（~3700 行闭环函数）。RLix 调度器需要在 5 个关键时机介入（请求/释放 GPU、唤醒 sleeping workers），但原函数没有任何扩展点。

Task 5 在训练循环里打入 hook 接口；Task 6 在 hook 里实现进度上报，让调度器知道 rollout 进度以做 GPU 公平分配。

---

## 文件结构

```
nemo-rl/nemo_rl/algorithms/
├── rlix_hooks.py        # Task 5 + 6 核心实现
├── grpo.py              # grpo_train() stub，含 hook 插入点
└── TASK5_6_HOOKS.md     # 本文档

tests/
└── test_rlix_hooks.py   # 31 个单元测试，无 GPU/Ray 依赖
```

---

## Task 5 — Hook 接口与插桩

### `RLixHooks` Protocol

```python
@runtime_checkable
class RLixHooks(Protocol):
    def before_generation(self, step: int) -> None: ...    # Hook 1: 请求 inference GPU
    def after_generation(self, step: int) -> None: ...     # Hook 2: 释放 inference GPU
    def before_training(self, step: int) -> None: ...      # Hook 3: 请求 training GPU
    def after_training(self, step: int) -> None: ...       # Hook 4: 释放 training GPU
    def before_weight_sync(self, step: int) -> None: ...   # Hook 5: 唤醒 sleeping workers
    def begin_progress_batch(self, step: int, count_intended: int) -> None: ...  # Task 6
    def end_progress_batch(self, step: int, trajectories_collected: int) -> None: ...
```

用 `@runtime_checkable` + `Protocol`，不需要继承，`isinstance()` 可做类型校验。

### `NoOpRLixHooks`

所有方法 `pass`。NeMo RL 单独运行时 `grpo_train(hooks=None)` 自动使用，零侵入原有行为。

### `NemoRLRLixHooks`

实际调度器集成。GPU hook 为 TODO 占位（依赖 Task 7 pipeline actor）；`before_weight_sync` 内 NCCL 重建为 TODO（依赖 Task 3）。

```python
NemoRLRLixHooks(
    scheduler=<Ray actor handle>,      # rlix:scheduler
    pipeline_id="ft_000000000000",
    cluster_ids={
        "actor_train": "ft_xxx_actor_train",
        "actor_infer":  "ft_xxx_actor_infer",
    },
)
```

### `grpo.py` 中的完整调用顺序

```python
DO_TIME_SHARING: bool = False  # RLix 模式下设为 True

def grpo_train(config, *, hooks=None):
    if hooks is None:
        hooks = NoOpRLixHooks()
    for step in range(num_steps):
        # begin_progress_batch 必须在 before_generation 之前调用（见下方说明）
        hooks.begin_progress_batch(step, step_trajectory_target)
        hooks.before_generation(step)
        # ... prepare_for_generation → generate → finish_generation ...
        # hooks.end_progress_batch(step, n) 在 generation 循环内每批调用
        hooks.after_generation(step)
        # ... compute_advantages ...
        hooks.before_training(step)
        # ... policy.train ...
        hooks.after_training(step)
        hooks.before_weight_sync(step)
        # ... refit_policy_generation ...
```

---

## Task 6 — 进度上报（2% 桶粒度）

### 问题：0 trajectory 时 scheduler 看不到需求

generation 开始前 pipeline 还没有任何 trajectory，scheduler 看到的 demand = 0，gap-ratio 算法会忽略这个 pipeline，不分配 GPU。但没有 GPU 就没法收集 trajectory——chicken-and-egg。

### 解决方案：两套机制配合

| 机制 | 时机 | 作用 |
|------|------|------|
| `step_target_estimate`（在 GPU request 里传） | generation **开始前**（0 trajectories） | Bootstrap demand：告诉 scheduler "我将要有这么多需求" |
| `end_progress_batch` 桶上报 | generation **进行中** | 实时更新 remaining demand，驱动动态 rebalance |

`begin_progress_batch` 必须在 `before_generation` **之前**调用，原因是 Task 7 填入 `before_generation` 的 TODO 时，需要从 `hooks._count_intended_for_step` 读取 `step_target_estimate` 一起发给 scheduler：

```python
# Task 7 填入 before_generation 时的样子：
ray.get(self._scheduler.request_gpus.remote(
    cluster_id=self._cluster_ids["actor_infer"],
    priority=Priority.GENERATION,
    global_step=step,
    step_target_estimate=self._count_intended_for_step,  # ← begin_progress_batch 已设好
))
```

### 桶状态机

```
begin_progress_batch(step, count_intended)
    _current_step            = step
    _count_intended_for_step = count_intended   # 本 step 目标 trajectory 数
    _collected_so_far        = 0
    _last_emitted_bucket     = -1               # -1 表示本 step 尚未 emit

end_progress_batch(step, trajectories_collected)
    _collected_so_far += trajectories_collected
    bucket = min(floor(_collected_so_far / _count_intended_for_step * 50), 50)
    if bucket != _last_emitted_bucket:
        _last_emitted_bucket = bucket
        _emit_progress(step)                    # fire-and-forget RPC（Task 7 填入）
```

### 示例（count_intended=100，每次收集 1 条）

| collected | bucket | emit? |
|-----------|--------|-------|
| 1         | 0      | ✅ 首次 |
| 2         | 1      | ✅ 桶推进 |
| 3         | 1      | ❌ 同桶去重 |
| 4         | 2      | ✅ 桶推进 |
| 100       | 50     | ✅ 100% |

最多 51 次 emit（桶 0-50），而非 100 次。

### `_emit_progress()`

独立方法，当前为 TODO 注释，Task 7 填入真实内容：

```python
def _emit_progress(self, step: int) -> None:
    # TODO(Task 7):
    # self._scheduler.report_progress.remote(
    #     ProgressReport(
    #         pipeline_id=self._pipeline_id,
    #         step_target_trajectories=self._count_intended_for_step,
    #         fifo_timestamp=time.monotonic(),
    #         metrics={"completed": float(self._collected_so_far), "mode": "train"},
    #     )
    # )
    pass
```

独立成方法的原因：测试可以直接替换它来验证 emit 行为，而不需要真实调度器。

---

## 测试覆盖（31 个，全部 pass，无 GPU/Ray 依赖）

### Task 5 — Protocol & NoOp（5 个）

| 测试 | 验证内容 |
|------|---------|
| `test_all_methods_callable_without_error` | 7 个方法都可调用，不抛异常 |
| `test_satisfies_rlix_hooks_protocol` (NoOp) | `isinstance(NoOpRLixHooks(), RLixHooks)` 为 True |
| `test_returns_none_for_all_methods` | 所有方法返回 None |
| `test_satisfies_rlix_hooks_protocol` (NemoRL) | `NemoRLRLixHooks` 也满足 Protocol |
| `test_gpu_hooks_are_no_ops_until_task7` | GPU hook 为 placeholder，返回 None 不 crash |

### Task 5 — DO_TIME_SHARING & grpo_train 插桩（8 个）

| 测试 | 验证内容 |
|------|---------|
| `test_flag_exists_and_is_bool` | `DO_TIME_SHARING` 存在且类型为 bool |
| `test_flag_defaults_to_false` | 默认 False，不影响 NeMo RL 独立运行 |
| `test_all_five_hooks_called_each_step` | 5 个 hook 每 step 均被调用 |
| `test_hook_order_within_step` | 顺序：before_gen → after_gen → before_train → after_train → before_weight_sync |
| `test_hooks_called_once_per_step` | 3 steps → 每个 hook 恰好 3 次，无重复无遗漏 |
| `test_step_index_passed_correctly` | step=0,1 正确传入每个 hook |
| `test_noop_hooks_used_when_none_passed` | `hooks=None` 时使用 NoOp，不 crash |
| `test_begin_progress_batch_called_before_before_generation` | `begin_progress_batch` 在 `before_generation` 之前，保证 `step_target_estimate` 已就绪 |

### Task 6 — begin/end_progress_batch 状态机（10 个）

| 测试 | 验证内容 |
|------|---------|
| `test_resets_counter_and_bucket` | begin 正确初始化 5 个状态变量 |
| `test_re_init_between_steps` | 下一 step 的 begin 清零上一 step 的残留 |
| `test_raises_on_zero_count_intended` | `count_intended=0` 抛 ValueError |
| `test_raises_on_negative_count_intended` | `count_intended=-1` 抛 ValueError |
| `test_raises_without_begin` | 未调用 begin 就调用 end 抛 RuntimeError |
| `test_raises_on_step_mismatch` | step 不匹配抛 ValueError |
| `test_raises_on_negative_trajectories` | 负数 trajectories 抛 ValueError |
| `test_zero_trajectories_does_not_raise` | 0 条合法，不抛异常 |
| `test_accumulates_collected_count` | 多次 end 累加正确 |
| `test_bucket_does_not_exceed_max` | 超额收集时 bucket 钳位到 50 |

### Task 6 — 桶去重逻辑（8 个）

| 测试 | 验证内容 |
|------|---------|
| `test_single_full_batch_emits_once` | 一次收满 → 仅 emit 1 次 |
| `test_repeated_zero_batches_emit_once` | 连续 0 条 → 仅 emit 1 次（均在 bucket 0） |
| `test_two_percent_granularity` | 100 条各 1 个 → emit 次数 ≤ 51 |
| `test_bucket_advances_at_correct_threshold` | count=50 时桶严格递增，末尾必须到达 50 |
| `test_no_duplicate_emits_for_same_bucket` | 同桶不重复 emit，emitted_buckets 严格递增 |
| `test_complete_collection_reaches_max_bucket` | 100% 收集后 `_last_emitted_bucket == 50` |
| `test_overcollection_clamps_to_max_bucket` | 已到 bucket 50 后超额收集不再 emit |
| `test_emit_progress_called_with_correct_step` | emit 时 step 参数与 begin 一致 |

---

## 未实现（有意 TODO，等对应 Task 完成后填入）

| 位置 | 等待 | 内容 |
|------|------|------|
| `before_weight_sync` | Task 3 | NCCL communicator destroy/reload（TP>1） |
| `before/after_generation` | Task 7 | `scheduler.request_gpus.remote(step_target_estimate=...)` / `notify_release_gpus.remote()` |
| `before/after_training` | Task 7 | 同上，actor_train cluster |
| `_emit_progress` | Task 7 | `scheduler.report_progress.remote(ProgressReport(...))` |

---

## 运行测试

```bash
PYTHONPATH=nemo-rl python -m pytest tests/test_rlix_hooks.py -v
# 31 passed in 0.05s
```
