# F1–F12 代码审查 — M11 里程碑

> 审查日期 2026-06-28 起，基于分支 `zhenyu/miles-mvp-e2e`
> F1-F2 审查于 `da5aa3c`；2026-07-05 更新反映 PR #17 (Howard) 和 PR #30 (Kyle) 合并后状态 (`7463c16`)
> F3 审查于 `492a75c`
> F7-F12 复查于 `492a75c`（2026-07-05）：重读 rlix 侧 orchestrator / miles_coordinator / miles_pipeline / miles_hooks / protocol / utils 全文，及 miles 侧 `run_miles_rlix.py`、`run_miles_dual.py`、`placement_provider.py`、`rlix_validation.py` 全文
> Scheduler 核心 + rollout.py 补审于 `492a75c`（2026-07-05）：`scheduler.py`（1758 行）/ `planner.py` / `state.py` / `types.py` / `validation.py` 全文（rlix 4 标准全审，见 §12）；miles `rollout.py` 关键路径（EngineInfo / shrink / expand / finish_init_offload / activate_routing / shutdown_hard / abort-cache，只报 crash/泄漏级）
> 最终补齐（2026-07-05 同日）：rlix `miles_model_update_service.py`（473 行逐行重读）、`client/client.py`、`scheduler/resource_manager.py`；miles `rlix_train_loop.py`、`actor_group.py`、`rlix_hooks.py`、`router.py`（617 行全文）、`base_types.py`、`rollout.py` generate 链路 + offload/onload 子集原语、`sglang_engine.py` rlix 相关段（init/register/unregister/shutdown/release/is_idle/abort/pause/continue/update_weights_from_cpu_bucket/finalize/flush_cache）
> F4/F5 接收端纵深补齐（2026-07-05 同日）：miles `cpu_bucket_cache.py`（234 行全文）、`megatron_utils/actor.py` 相关段（init tms 模式 / sleep / wake_up / train / build_cpu_bucket_cache / run_sync_session / _dispatch_cpu_serialize_bucket / _dispatch_nccl_broadcast）、`fully_async_rollout.py`（452 行全文）、`multi_turn.py`（176 行全文）、`radix_tree_middleware.py`（C17 守卫段）。未读且明确排除在审查范围外：rlix `tracer.py`（纯 tracing）、ROLL 后端 pipeline 文件（非 M11 范围）；miles megatron 训练内部（train_actor/loss/checkpoint）、sglang_engine 非 rlix 段、rollout.py 数据转换段（不在 rlix 调用链）

---

## §1 F1 — SGLang 显存释放

### 功能说明

F1 管理 GPU 时分共享过程中推理引擎的显存（VRAM）释放与恢复。
当调度器从推理回收 GPU（分配给训练）时，SGLang 引擎必须物理释放 VRAM（模型权重 + KV cache）。训练完成后，引擎重新获取 VRAM 并恢复推理。

**核心流程：**

1. **缩容（Shrink）**: Scheduler → `resize_infer(dp_ranks_to_remove)` → coordinator → `rollout_manager.shrink_engines()` → SGLang HTTP `/release_memory_occupation` → 引擎状态转为 `offloaded`
2. **验证（Verify）**: `_wait_for_overlap_engines_offloaded()` 轮询引擎状态 + nvidia-smi 确认 VRAM 确实已释放（SGLang 返回 200 OK 并不足以证明）
3. **扩容（Expand）**: Scheduler → `resize_infer(dp_ranks_to_add)` → `expand_engines()`（HTTP `/resume_memory_occupation`）→ 权重同步 → `activate_routing` → 引擎状态恢复为 `active`

M11.2 Option Beta 初始化：引擎以 `loading` 状态启动 → `finish_init_offload()` → 立即进入 `offloaded`，后续的唤醒/卸载周期由 F40 Runtime 分支处理。

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `miles_pipeline.py` | 505-604 | `_wait_for_overlap_engines_offloaded()` — 两阶段验证（引擎状态轮询 + nvidia-smi 探测） |
| `miles_pipeline.py` | 607-644 | `_probe_max_used_gpu_mem_gb()` — 静态方法，解析 nvidia-smi 输出获取最大已用显存 |
| `miles_coordinator.py` | 431-465 | `_shrink_workers()` — 快照加锁 / RPC / 提交加锁模式 |
| `miles_coordinator.py` | 446-447 | `MILES_MAX_RESIDUAL_GPU_MEM_GB` 环境变量传递给 miles shrink_engines |

### Guide 已知 Bug（guide §3 F1 section，共 2 个）

| ID | 描述 | 修复位置 | 状态 |
|----|------|----------|------|
| F1-1 | M11.1 attempt 5 — `release_memory_occupation` 返回 200 OK 但 nvidia-smi 显示引擎仍占 ~30 GB VRAM。根因：miles `start_rollout_servers` 在 partial-overlap 拓扑下计算 `needs_offload=False`，导致 `enable_memory_saver=False`。 | miles `f58b365`（非 rlix 侧修复） | ✅ 已解决 |
| F1-2 | M11.1 attempt 1 — torch_memory_saver 默认 `hook_mode="preload"` 在 CUDA 12.9 / Blackwell 上段错误。修复：smoke 脚本设置 `MILES_TMS_HOOK_MODE=torch` 切换到 CUDAPluggableAllocator 路径。 | miles `f58b365`（非 rlix 侧修复） | ✅ 已解决 |

> **注意**: F1 的两个 guide bug 修复都在 miles 仓库。rlix 侧贡献的是验证层（`_wait_for_overlap_engines_offloaded` + nvidia-smi 探测），由 PR #17 (Howard) 重写。
>
> **2026-07-05 补审**: miles `rollout.py` 的 F1 相关路径（`start_rollout_servers` rlix-mode override、`shrink_engines` 的 `release_memory_occupation` 调用链）已逐行阅读，与 guide 描述一致，未发现 crash / 泄漏级新问题。

### F1 硬编码值

**✅ 已解决（PR #17 / Howard）：**

| 值 | 原位置 | 说明 |
|----|--------|------|
| `20.0` GB | `miles_pipeline.py:580` | `target_free_gb` — 替换为 `MILES_MAX_RESIDUAL_GPU_MEM_GB`（默认 13.0 GB） |
| `0.5` s | `miles_pipeline.py:609` | Phase 2 nvidia-smi 轮询间隔 — Phase 2 替换为整卡残留硬门控 |
| `3.0` s | `miles_pipeline.py:599` | nvidia-smi 不可用时宽限 sleep — 回退逻辑重写 |

**🆕 新发现（仍存在）：**

| 值 | 位置 | 风险 |
|----|------|------|
| `13.0` GB | `miles_pipeline.py:580` + `miles_coordinator.py:447` | PR #17 新增：`MILES_MAX_RESIDUAL_GPU_MEM_GB` 默认值。已有环境变量可配，低风险。⏳ rlix PR #31（OPEN）将默认值降为 7.0 GiB；7/5 周会结论：offload 后残留 ~5-7 GiB（Megatron ~3.0 + SGLang ~1.7），Zhenyu 排查中，最终目标 < 2 GiB。miles PR #21（OPEN，接手 Howard）将 post-sleep VRAM assert 改为无条件执行。 |

### F1 设计观察（🆕 新发现）

| ID | 内容 | 位置 | 说明 |
|----|------|------|------|
| F1-NVIDIA | nvidia-smi 不可用时验证完全跳过（fail-open） | `miles_pipeline.py:582-587` | 日志警告后直接 return，无重试、无升级。PR #17 改写后仍为 fail-open。生产环境如果 nvidia-smi 不可用，VRAM 验证被静默跳过。注意：6/28 周会已裁定 nvidia-smi 为显卡服务器标配、拒绝引入 pyNVML 依赖（miles PR #22 因此 CLOSED）——收紧此项只能走"告警升级/hard-fail"路线，不能换库。 |

---

## §2 F2 — 选择性引擎生命周期

### 功能说明

F2 通过 5 状态机管理单个推理引擎的生命周期：

```
shell → loading → active ⇄ disabling → offloaded
                    ↑__________________________|
```

**要点**: 状态机**没有**在 rlix 代码中定义为 enum。状态以字符串值（`"shell"`, `"loading"`, `"active"`, `"offloaded"`）存在于下游 MILES `RolloutManager` 中。rlix coordinator 通过 `rollout_manager.get_engine_states.remote()` 读取状态并分发行为。`"disabling"` 状态是 MILES 内部的瞬态 — rlix 代码中从未引用。

Coordinator 通过 `_active_engine_indices: Set[int]` 追踪自己的记账信息。

**由 rlix 管理的生命周期转换：**

| 转换 | 触发条件 | 代码路径 |
|------|----------|----------|
| `loading` → `offloaded` | 初始化（Option beta） | `finish_init_offload()`，pipeline 初始化时 |
| `loading/shell` → `active` | 初始化（standalone） | 引擎直接以 active 启动 |
| `offloaded` → `active` | 扩容 | `expand_engines` → 同步权重 → `activate_routing` |
| `active` → `offloaded` | 缩容 | `shrink_engines` → `release_memory_occupation` |

### 代码位置

| 文件 | 行号 | 作用 |
|------|------|------|
| `miles_coordinator.py` | 131-132 | `_active_engine_indices: Set[int]` + `_active_engines_bootstrapped` 标记 |
| `miles_coordinator.py` | 143 | `_resize_sync_lock` — 保护引擎状态变更 |
| `miles_coordinator.py` | 407-427 | `resize_infer()` — 调度器入口 |
| `miles_coordinator.py` | 431-465 | `_shrink_workers()` — 快照/RPC/提交模式 |
| `miles_coordinator.py` | 467-556 | `_expand_workers()` — 状态机分发（shell / active / offloaded 分支） |
| `miles_pipeline.py` | 505-604 | `_wait_for_overlap_engines_offloaded()` — 两阶段验证 |
| `miles_pipeline.py` | 607-644 | `_probe_max_used_gpu_mem_gb()` — nvidia-smi 探测 |
| `miles_pipeline.py` | 336-365 | Phase B 初始化 Option Beta: `finish_init_offload` |

### Guide 已知 Bug（guide §3 F2 section，共 2 个）

| ID | 描述 | 修复位置 | 状态 |
|----|------|----------|------|
| F2-1 | M11.1 attempts 5-7 — `release_memory_occupation` 失败，报 `Pointer argument cannot be accessed from Triton (cpu tensor?)`，然后 `flush_cache` 60s 超时。根因：`is_idle=True` 不代表 SGLang scheduler 线程已停止解码，Triton kernel 访问了已移到 CPU 的 buffer。修复：shrink 路径在 drain 和 release 之间插入 `pause_generation(mode="retract")`。 | miles `f58b365`（非 rlix 侧修复） | ✅ 已解决 |
| F2-2 | M11.2 attempt 4 — `_wait_for_overlap_engines_offloaded` 崩溃：`KeyError('unknown engine_index 2')`。根因：物理 GPU ID → 本地引擎索引映射错误，M11.2 P2 pool 从 GPU 2 开始，`2 // 1 = 2` 但本地引擎只有 `[0, 1]`。修复：用 `infer_first = min(infer_mapping)` 减去偏移。 | rlix `549cfbd` | ✅ 已解决 |

### F2 硬编码值（🆕 新发现）

| 值 | 位置 | 风险 |
|----|------|------|
| `60.0` s | `miles_pipeline.py:505` | `_wait_for_overlap_engines_offloaded` 默认超时（两阶段共用）。⏳ 相关：rlix PR #20（OPEN）将 Phase 1 超时从"只 warn 继续"改为 fail-fast。 |
| `0.1` s | `miles_pipeline.py:568` | Phase 1 引擎状态轮询间隔 |
| `5.0` s | `miles_pipeline.py:628` | nvidia-smi subprocess 超时 |
| `1024.0` | `miles_pipeline.py:644` | MiB 转 GB 除数（正确但隐式，未用 `1024` 常量命名） |

### F2 冗余代码（🆕 新发现）

| 内容 | 位置 | 说明 |
|------|------|------|
| `set(engine_indices)` 重复包裹 | `miles_coordinator.py:528, 556` | 参数注解已是 `Set[int]`，无需再 `set()` 包裹。 |
| 方法体内 `import os as _os` | `miles_coordinator.py:510` | 对 stdlib `os` 的懒导入 — 顶层已有 `import os`（L21），开销可忽略。⏳ rlix PR #25（OPEN，hoist function-local imports）覆盖此类。 |

### F2 设计观察（🆕 新发现）

1. **MilesCoordinator 缺少锁超时**: `PipelineCoordinator` 使用 `_RESIZE_LOCK_TIMEOUT_S = 180s` + `lock.acquire(timeout=...)` 并在超时时抛异常。`MilesCoordinator` 到处用裸 `with self._resize_sync_lock:` → **卡住时无限阻塞**。潜在死锁面。
2. **缺少 `validate_resize_params`**: `PipelineCoordinator.resize_infer` 调用 `validate_resize_params(to_remove, to_add)` 防止同时缩容+扩容。`MilesCoordinator.resize_infer` 没有校验 → 调用方可以同时传入两个非空列表。
3. **未处理 `"disabling"` 状态**: `_expand_workers` 处理 `{"shell"}`、`{"active"}`、`{"offloaded"}`，其他状态直接抛异常。如果引擎恰好处于瞬态 `"disabling"`，异常会触发。可能是正确的（调用方应重试），但没有文档。2026-07-05 补审确认：`disabling` 只存在于 `shrink_engines` 单次调用内部（`rollout.py:952-955` 置入 → L1070 置 `offloaded`；中途失败则停留在 `disabling` 并向上抛异常 → 触发 scheduler fail-fast），正常运行时 `_expand_workers` 不会观测到它。
4. **`"shell"` 被视为 offload 完成**: `miles_pipeline.py:562` 检查 `uniq.issubset({"offloaded", "shell"})`。shell 状态的引擎是否释放了 VRAM 没有文档说明。2026-07-05 补审确认：`rollout.py` 中 `shell` = 无 actor handle（从未创建，或被 `shutdown_hard` 杀死后置回 shell，L1218-1220），不占 VRAM，视作 offload 完成是正确的。

> **2026-07-05 补审**: miles `rollout.py` 的 F2 核心（`EngineInfo` 5 状态机、`shrink_engines` 的 unregister-router→abort→drain→pause→release 顺序、`expand_engines` / `finish_init_offload` / `activate_routing` 的状态前置校验、`_abort_engines` 幂等缓存及失败重置、`shutdown_hard` 的先杀 SGLang 进程树再 ray.kill）已逐行阅读。关键不变量在位：router 先关闭准入再释放显存（Codex Q5-b）、`register_with_router` 非 2xx 抛异常故不会对 router 不知情的引擎置 active、shrink 失败重置 abort 缓存使重试能重新 abort。未发现 crash / 泄漏级新问题。

---

## §3 F3 — 路由跳过睡眠引擎（router admission + 0-active suspend）

### 功能说明

F3 确保 router **不会**将推理请求分发到已被缩容的引擎。M11.2 增加了更难的场景：当所有引擎都被训练抢占时（active set 为空），router 必须**挂起等待**（`asyncio.Condition`），而不是抛 `RuntimeError`。

**核心机制：**

1. **Shrink** → `shrink_engines` 内部调用 router `/disable_worker`，将引擎从路由表移除
2. **0-active suspend** → 当所有引擎都被 disable，router 的 `_use_url_async` 阻塞在 `_workers_changed.wait_for(predicate)` 直到有引擎可用
3. **Expand** → `activate_routing` 调用 router `/add_worker`，唤醒挂起的 dispatch 协程

**注意**: F3 的核心实现在 `miles/` 仓库（`miles/miles/router/router.py`），不在 `rlix/` 中。rlix 侧只是调用方。

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `miles_coordinator.py` | 431-465 | `_shrink_workers()` — 调用 `shrink_engines.remote()`，触发 router disable |
| `miles_coordinator.py` | 544 | `expand_engines.remote()` — 唤醒引擎 |
| `miles_coordinator.py` | 552 | `activate_routing.remote()` — 将引擎重新注册到 router |
| `miles_pipeline.py` | 427-444 | Phase B init 断言 `get_router_enabled_workers` 为空（Option Beta 合规检查） |

> **参考**: F3 核心实现在 `miles/router/router.py`（`rlops/miles` @ `zhenyu/m11-mvp-test`），包括 `_use_url_async` 0-active 挂起、`disable_worker`/`enable_worker` 路由准入、`_health_check_loop` 健康检查。已阅读确认 rlix 的调用方式与 miles 的 API 契约一致。
>
> **2026-07-05 补审**: `router.py` 617 行全文重读。0-active suspend（`_workers_changed.wait_for`）、4-dict 准入状态、`_admission_declared` 兼容回退、`do_proxy` 的 try/finally 计数平衡（PR #16 附带修复）、`_disable/_enable` 的 F68 failure-count 重置均与 guide 描述一致。`_use_url_async` 无界等待是 guide §6 已记录的 F79（M11.5）延期项，不重复报。无 crash / 泄漏级新问题。
>
> **2026-07-05 纵深补审**（F3 消费端 miles 实现）：`multi_turn.py` 全文 — 快照/恢复式 turn-level redispatch、重试预算 = 总引擎数（非活跃数）、metadata 缺失时 `RLixRouterMetadataError` fail-fast（而非静默降级）✓。`fully_async_rollout.py` 全文 — `_FatalError` 单路径队列传播（F01 反回归不变量 #9）、raise 前 `end_progress_batch()` 收流 ✓。`radix_tree_middleware.py` — C17 双守卫成立（`rlix_validation` 启动检查 + middleware 构造函数在 rlix 模式直接 raise）✓。无 crash / 泄漏级新问题。

### Guide 已知 Bug（guide §3 F3 section，共 0 个）

Guide 记录 F3 在 M11.1/M11.2 happy path 上**没有发现独有 bug**。

Phase 7 修复了两个 driver 侧问题（均在 `miles/` 仓库，非 rlix）：
- **R04-F2**: Driver crash 未触发 `shutdown_hard` → 调度器分配泄漏（✅ 已修复，miles `79f2874`）
- **R04-F3**: dual driver `asyncio.gather` 未在失败时取消 peer → 孤儿 actor（✅ 已修复，miles `79f2874`）

### F3 正确性 / 设计观察（🆕 新发现）

| ID | 内容 | 位置 | 说明 |
|----|------|------|------|
| F3-TIMEOUT-1 | `shrink_engines.remote()` 的 `ray.get` 无超时 | `miles_coordinator.py:449` | rollout manager 卡住时 coordinator 永远阻塞。`shutdown_hard` 有 30s 超时，但正常 shrink 路径没有。 |
| F3-TIMEOUT-2 | `expand_engines.remote()` 的 `ray.get` 无超时 | `miles_coordinator.py:544` | 同上。三步 RPC 链（expand → sync → activate）任一步卡住都会阻塞。 |
| F3-TIMEOUT-3 | `activate_routing.remote()` 的 `ray.get` 无超时 | `miles_coordinator.py:552` | 同上。 |
| F3-PARTIAL | expand 三步 RPC 链无回滚 | `miles_coordinator.py:544-552` | 如果 `expand_engines` 成功但 `sync_selected_workers` 失败，引擎已唤醒但未同步权重且未路由。状态不一致，无回滚机制。 |
| F3-EXCEPT | 宽泛 `except Exception` 掩盖 API 违反 | `miles_coordinator.py` 多处 | 9 处 `except Exception` + `# noqa: BLE001`。R01 review 已标记 shrink 路径的 catch 可能吞掉 SGLang 契约错误。 |
| F3-ATOMIC | `resize_infer` 先 shrink 后 expand 无原子性 | `miles_coordinator.py:423-426` | shrink 成功但 expand 失败时，引擎被移除但未恢复，无事务回滚。 |
| F3-VERIFY | rlix 无法验证 router 状态变更是否生效 | `miles_coordinator.py` | 调用 `shrink_engines` / `activate_routing` 后信任 miles 侧正确更新 router 状态，无回读确认。唯一检查点是 Phase B init 的断言（L427-444），运行时无类似断言。 |

---

## §4 F4 — 训练侧权重缓存（CPU bucket cache + `_cache_ready_step`）

### 功能说明

F4 管理训练后权重的缓存和发布流程。每次训练步结束后，cache_owner Megatron actor（`pp0+dp0+tp0+cp0`）将新权重导出到 CPU bucket cache，后续 `sync_selected_workers` 从缓存读取并推送到 SGLang 引擎。

**两阶段模型：**
- **发送方**: `build_cpu_bucket_cache(step)` → `publish_cache_ready_step(step)` → `sync_base_weights_to_active(step)`
- **接收方**: transport（cpu_serialize tmpfs）→ `finalize_weight_update` → `set_weight_version`

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `miles_pipeline.py` | 163-189 | `_init_phase_a_train` — `RayTrainGroup` 构造，含 `with_ref` 推导 |
| `miles_pipeline.py` | 196-201 | Phase A step 3.5 `wake_up` + step 4 `build_cpu_bucket_cache(-1)` |
| `miles_pipeline.py` | 646-676 | `_before_training()` — 请求 actor_train GPU，等待 overlap offload |
| `miles_pipeline.py` | 678-701 | `_after_training()` — `build_cpu_bucket_cache(step)` → `offload()` → `sync_base_weights_to_active(step)` |
| `miles_coordinator.py` | 332-344 | `publish_cache_ready_step(step)` — 设置 `_cache_ready_step` |
| `miles_coordinator.py` | 372-401 | `sync_base_weights_to_active(step)` — 驱动 atomic-unit sync |

### Guide 已知 Bug（guide §3 F4 section，共 2 个）

| ID | 描述 | 修复位置 | 状态 |
|----|------|----------|------|
| F4-1 | M11.1 attempt 8 — `[torch_memory_saver.cpp] Cannot resume allocation that is not paused`。根因：`_before_training` 调用了 `train_group.onload()`，但 `MegatronTrainRayActor.train()` 内部也调用 `wake_up()`，导致双重 resume。修复：移除 `_before_training` 中显式的 `onload()`。 | rlix `e0a6b27`（`miles_pipeline.py:669-676` 注释说明） | ✅ 已解决 |
| F4-2 | M11.1 attempt 9 — `torch.cat(NoneType, dim=int)` 因为 `batch["ref_log_probs"]` 为 None。根因：`RayTrainGroup(..., with_ref=False)` 导致 ref model 未加载，`compute_log_prob` 被跳过。修复：从 args 推导 `with_ref`（`use_kl_loss` 或 `kl_coef != 0`）。 | rlix `e0a6b27`（`miles_pipeline.py:185-188`） | ✅ 已解决 |

### F4 硬编码值（🆕 新发现）

| 值 | 位置 | 风险 |
|----|------|------|
| `0` | `miles_pipeline.py:176` | `num_gpus_per_actor=0` — rlix 模式下训练 actor 不通过 Ray 预留 GPU，依赖手动 `CUDA_VISIBLE_DEVICES`。与 standalone 的 `0.01` 不同，有注释解释。低风险。 |
| `-1` | `miles_pipeline.py:152, 201, 226, 238` | 初始化哨兵 step 值，用于 `global_step=-1` / `build_cpu_bucket_cache(-1)` / `publish_cache_ready_step(-1)`。语义明确，低风险。 |

### F4 冗余代码（🆕 新发现）

| 内容 | 位置 | 说明 |
|------|------|------|
| `with_ref` 推导逻辑与 miles standalone 重复 | `miles_pipeline.py:185-188` vs `miles/ray/placement_group.py:192` | 两处独立推导 `with_ref = use_kl_loss or kl_coef != 0`。如果 miles 侧修改了条件（如新增 KL 相关参数），rlix 侧不会自动同步。存在静默分歧风险。 |

### F4 设计观察（🆕 新发现）

| ID | 内容 | 位置 | 说明 |
|----|------|------|------|
| F4-GETATTR | `with_ref` 推导使用 `getattr(miles_args, "use_kl_loss", False)` | `miles_pipeline.py:186-187` | `getattr` 默认值静默吞掉 miles_args 缺少该字段的情况。如果上游重命名了 `use_kl_loss` → `enable_kl_loss`，rlix 侧静默回退到 `False`，导致 ref model 不加载，复现 F4-2 bug。 |

> **2026-07-05 纵深补审**（F4 接收端 miles 实现，只报 crash/泄漏级）：`cpu_bucket_cache.py` 全文 — 单槽位 ready-step + 锁保护 + `put_step` 原子发布 + CUDA tensor 拒收，无问题。`megatron actor.py` — `build_cpu_bucket_cache` 全员参与 collective gather、仅 owner 存储、`_cache_lock` 单临界区（F20）✓；`train()` 内部 `wake_up()`（L360-362）确认 F4-1 双重 resume 修复的另一半；`MILES_SKIP_TMS_PAUSE` 的 sleep/wake 行为与 B-14 记录一致。`run_sync_session` — cache_owner 校验、plan key 校验、`master_port=0` 拒收（F26/C16）✓。接收端 `sglang_engine.update_weights_from_cpu_bucket` — tmpfs 写入 + POST + **finally 无条件 unlink**（F28/F66 契约成立，之前对 tmpfs 泄漏的疑虑排除）。无 crash / 泄漏级新问题。

---

## §5 F5+F6 — 双路径权重刷新 + 版本记账

### 功能说明

F5+F6 管理训练后权重推送到推理引擎的两条路径，共享同一个原子单元 `MilesModelUpdateService.sync_selected_workers(sync_id, targets, version)`：

- **Active in-flight refresh**（活跃刷新）: 引擎保持服务，`pause_generation` 包裹 `finalize_weight_update`；引擎上报新版本（即使在途请求可能用旧权重完成 — A8 有界误归因）
- **Expand sync**（扩容同步）: 之前 offloaded 的引擎唤醒后，接收 base sync（可能是 `version=-1` 的初始 CPU bucket），然后开放路由

版本号由 `manager.set_weight_version` 每次 sync 发布一次，**不是**每个 bucket 一次（F21）。

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `miles_model_update_service.py` | 32-77 | `SyncSessionPlan` frozen dataclass |
| `miles_model_update_service.py` | 101-225 | `MilesModelUpdateService.__init__` + `sync_selected_workers`（原子单元入口） |
| `miles_model_update_service.py` | 248-338 | `_run_atomic_unit` — transport + finalize + version publish |
| `miles_model_update_service.py` | 288-318 | pause_generation / finalize / continue_generation 包裹 |
| `miles_model_update_service.py` | 341-451 | `_build_plan` — wire-format plan 构建 + SharedStorage 端口声明 |
| `miles_coordinator.py` | 372-401 | `sync_base_weights_to_active(step)` — 驱动 atomic-unit sync |
| `miles_coordinator.py` | 467-556 | `_expand_workers` — 状态机分发 + expand RPC 链 |

### Guide 已知 Bug（guide §3 F5+F6 section，共 2 个）

| ID | 描述 | 修复位置 | 状态 |
|----|------|----------|------|
| F56-1 | M11.1 attempt 7 — 首次训练步后 `sync_base_weights_to_active(0)` → `finalize_weight_update` → `flush_cache` 60s 超时。根因：同 F2-1，全异步 rollout 函数不会同步 quiesce 引擎，pending decode batch 阻止 `flush_cache` 返回 200。修复：在 `finalize_weight_update` 前调用 `pause_generation(mode="retract")`，结束后 `continue_generation`。 | rlix `e0a6b27`（`miles_model_update_service.py:288-318`） | ✅ 已解决 |
| F56-2 | M11.1 attempt 4 — `_expand_workers` 在首次 INIT→GENERATION 转换时异常，因 miles `RolloutManager` 直接以 `state="active"` 启动引擎（无 shell→offloaded→active 路径）。修复：对已 active 引擎 no-op，仅更新 `_active_engine_indices` 记账。F22 严格状态机推迟到 M11.3。 | rlix `2b73aef`（`miles_coordinator.py:501-534`） | ✅ 已解决 |

### F5+F6 硬编码值（🆕 新发现）

| 值 | 位置 | 风险 |
|----|------|------|
| `150.0` s | `miles_model_update_service.py:122` | `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` 默认超时。已有环境变量可配。低风险。 |
| `8` | `miles_model_update_service.py:398` | SharedStorage 端口声明重试次数上限。固定值，无环境变量。 |
| `0.05` s | `miles_model_update_service.py:414` | 端口碰撞 backoff 延时。固定值。 |

### F5+F6 冗余代码（🆕 新发现）

| 内容 | 位置 | 说明 |
|------|------|------|
| `import asyncio` 重复导入 | `miles_model_update_service.py:158` | 函数体内 `import asyncio` 遮蔽模块级 L18 导入。冗余。⏳ rlix PR #25（OPEN）覆盖。 |
| `import asyncio` 死代码 | `miles_model_update_service.py:468` | `_await_ref` 函数体内导入 `asyncio` 但函数体只有 `await ref`，未使用 asyncio。`# noqa: F401` 抑制了 lint 警告。⏳ rlix PR #26（OPEN，inline await + drop `_await_ref`）整个移除该 helper。 |
| broadcast 路径完整死代码 | `miles_model_update_service.py:177-190` | `broadcast_set` 非空时直接 `raise NotImplementedError`。`_build_plan` 中 L424-436 的 `comm_ranks` 构建和 L436 的 `world_size` 计算永远走 fallback 分支。⏳ PR #24（OPEN，[need discussion]）计划删除。 |
| `import os as _os` 懒导入 | `miles_coordinator.py:510` | 顶层已有 `import os`（L21）。备注：此项当前也列在 §2 F2 冗余代码中，但 `_expand_workers` 代码归属 F5+F6。⏳ rlix PR #25 覆盖。 |
| `set(engine_indices)` 重复包裹 | `miles_coordinator.py:528, 556` | 参数注解已是 `Set[int]`，`\|=` 操作不需要再 `set()` 包裹。备注：同上，当前也列在 §2 F2 中。 |

### F5+F6 设计观察（🆕 新发现）

| ID | 内容 | 位置 | 说明 |
|----|------|------|------|
| F56-PAUSE-REF | `pause_refs` 和 `cont_refs` 未加入 `inflight_refs` | `miles_model_update_service.py:302, 315` | 超时/取消时 `_cancel_inflight` 不会 cancel 这些 Ray 引用 → actor 方法泄漏运行。已有 PR #23 待合并。（2026-07-05 全文重读复核：仍然成立。） |
| F56-PORT-LEAK | 端口声明**永不释放**（成功路径也累积），且当前服务于一条**死路径** | `miles_model_update_service.py:387-422` | 2026-07-05 全文重读后升级此发现：`SharedStorage.try_put` 声明的 `MASTER_ADDR_PORT:{addr}:{port}` key 在整个文件中**没有任何删除路径**——不仅 sync 失败不清理，**成功的 sync 也不清理**。每次 sync 用 `get_free_port` 拿新端口 → 新 key，声明表随训练步数无界增长，唯一清理点是 `orchestrator.kill_pipeline` 的 `delete_port_claims`。长时运行中 OS 端口复用会撞上陈旧声明 → 消耗重试预算（8 次），理论上可耗尽。**追加（纵深补审）**：`master_addr/port` 只被 NCCL broadcast 路径消费，而该路径两侧都是 `NotImplementedError` 自守卫（service L184-190 + miles `actor.py:_dispatch_nccl_broadcast` P1-8 自守卫）；cpu_serialize 完全不用端口 → 当前每次 sync 的整套 get_free_port + SharedStorage claim 机制是**为死路径付出的活泄漏**。PR #22 曾做部分修复（transport 成功后释放；失败/超时路径仍漏，已留言指出），7/12 被作者关闭、**7/19 重开并按我们 note 1 加固**（commit 612bdf7：try/finally 覆盖成功+抛错；`port_claim_holder` 递出原子单元供超时/取消 handler 释放；取消路径用裸 `.remote()` 规避 CancelledError 连带取消）——三条退出路径均闭环，**已于 8/1 复验并 approve**。作者对 note 2 的回应：broadcast 是将来要实现的能力，保留 `NotImplementedError` 占位不删 → #22 与 #24 互斥，团队投票保留 broadcast（合 #22、关 #24）。待 #22 merge 后此项标 ✅。 |
| F56-CONT-STICKY | `continue_generation` 失败仅 warn，引擎可能滞留 paused | `miles_model_update_service.py:313-320` | PR #14 把 `continue_generation` 移进 `finally` 保证总会尝试，但 continue 自身失败被 `except Exception` 吞掉只记 warning——引擎停留在 `pause_generation(retract)` 状态，后续 generate 503。与 PR #14 要解决的 sticky-pause 是同族的残余窗口（概率低：需要 pause 成功而 continue 失败）。 |
| F56-CANCEL-SYNC | **已证实（2026-07-06 查 Ray 2.55.1 官方文档）**：R06-F1 的 `ray.cancel(force=True)` 对 actor task 是**完全 no-op** | `miles_model_update_service.py:236-246` | Ray 文档原文："**Only `force=False` is allowed for an Actor Task. Otherwise, it raises `ValueError`.**"（[ray.cancel API](https://docs.ray.io/en/latest/ray-core/api/doc/ray.cancel.html)）。`inflight_refs` 里全部是 actor 方法引用 → 每个 `ray.cancel(ref, force=True)` 都抛 ValueError，被 `_cancel_inflight` 自己的 try/except 吞成 warning——**连排队中的任务都从未被取消过**，R06-F1 机制实为只打日志的 no-op。叠加：即使改 `force=False`，对同步 actor 执行中的任务也只是设置协作式取消标志（`is_canceled()`），miles/rlix 代码从未检查 → 执行中的 `run_sync_session` 无论如何不可中断，继续持有 `_cache_lock` 到自然结束。缓解：超时异常向上传播触发 fail-fast，通常观测不到。修法：⏳ 最小修已发 [rlops/rlix#39](https://github.com/rlops/rlix/pull/39)（2026-07-07，`force=False` + 修正注释 + 回归测试）；彻底解法（cache_owner 侧协作式检查或 async actor 化）仍为 M11.x 设计决策。 |
| F56-EXPAND-NORB | `_expand_workers` 三步 RPC 链无回滚 | `miles_coordinator.py:544-552` | `expand_engines` 成功但 `sync_selected_workers` 失败时：引擎已唤醒（active）但权重过期且未路由、不在 `_active_engine_indices` 中。若调度器重试 expand，`get_engine_states` 返回 `active` → 命中 "already-active" no-op 分支 → 跳过权重同步直接加入活跃集。此时引擎携带过期权重服务流量。2026-07-05 复核修正：expand 后引擎状态是 `loading` 而非 `active`（`rollout.py:1073-1112`），重试会命中 heterogeneous-states 异常而非 no-op 分支 → fail-fast 而非旧权重服务。降级为：无回滚导致引擎滞留 `loading` 持有 VRAM、需 fail-fast 重启恢复；不会静默用旧权重服务流量。 |

---

## §6 F7 — Per-pipeline Ray namespace isolation

### 功能说明

每条 pipeline 的 actor 运行在独立的 Ray namespace `pipeline_<pipeline_id>_NS` 中。调度器通过 `get_coordinator_actor_name(pipeline_id)` + `get_pipeline_namespace(pipeline_id)` 解析 coordinator handle。没有 namespace 隔离时，双 pipeline 运行会在 actor 名称上冲突。

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `protocol/types.py` | 8-15 | `RLIX_NAMESPACE`、`COORDINATOR_ACTOR_NAME_PREFIX`、`PIPELINE_ACTOR_NAME_PREFIX` 常量 |
| `protocol/types.py` | 42-49 | `get_pipeline_namespace(pipeline_id)` 和 `get_coordinator_actor_name(pipeline_id)` 辅助函数（PR #27 Tianye） |
| `scheduler/scheduler.py` | 1213-1238 | `_get_or_lookup_coordinator_handle_locked` — 调度器解析 coordinator 句柄 |
| `miles_coordinator.py` | 618 | `create_pipeline_actor` — 使用 `PIPELINE_ACTOR_NAME_PREFIX` + `self._ray_namespace` |
| `miles_coordinator.py` | 324 | `MilesCoordinator.__init__` — 使用 `get_pipeline_namespace(pipeline_id)` 设置 `_ray_namespace` |

### Guide 已知 Bug（guide §3 F7 section，共 1 个）

| ID | 描述 | 修复位置 | 状态 |
|----|------|----------|------|
| F7-1 | M11.1 attempt 4 — `resize_infer` / `shrink_engines` RPC 失败 `Failed to resolve actor`。根因：driver 将 coordinator 命名为 `f"miles_coordinator_{pipeline_id}"`（错误名称）且放在 `RLIX_NAMESPACE`（错误 namespace）。修复：使用规范名称 `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` + `get_pipeline_namespace(pipeline_id)` 作为 namespace。 | miles `7b83be5` + `992fa26`（driver 侧修复），rlix PR #27（Tianye `get_coordinator_actor_name` helper），PR #30（Kyle `GENERATION_CLUSTER_NAME` 一致性） | ✅ 已解决 |

> **注意**: 主修复在 miles driver 侧（命名约定纠正）。rlix 侧由 PR #27 提取 `get_coordinator_actor_name` helper 消除重复的 f-string 拼接，PR #30 统一 `GENERATION_CLUSTER_NAME` 引用。调度器解析路径 (`scheduler.py:1230`) 已使用 `get_coordinator_actor_name` helper。

### F7 正确性/设计观察（🆕 新发现）

无。F7 代码清晰，常量和辅助函数定义合理，调度器解析路径有缓存 + namespace 校验。

本次复查确认：两个 driver（`run_miles_rlix.py:195-205`、`run_miles_dual.py:302-325`）均通过 `get_coordinator_actor_name` + `get_pipeline_namespace` 创建 coordinator，与调度器解析路径（`scheduler.py:1230`）一致；`_coordinator_handle_cache` 在 pipeline 注销时正确清除（`scheduler.py:292`）。

---

## §7 F8 — Pipeline registration lifecycle（M11.1 single + M11.2 dual）

### 功能说明

F8 管理 pipeline 从创建到销毁的完整生命周期：

1. `rlix.init()` → 创建 Orchestrator / Scheduler / ResourceManager
2. `orchestrator.allocate_pipeline_id` → 分配唯一 ID（`ft_` / `lora_` + 12 hex）
3. `orchestrator.register_pipeline` → 注册拓扑到 scheduler
4. `orchestrator.admit_pipeline` → scheduler 开始调度
5. 创建 `MilesCoordinator` actor（per-pipeline namespace）
6. `coordinator.create_pipeline_actor` → `MilesPipeline` actor → `initialize_pipeline`（Phase A train + Phase B infer）

M11.2 对两条 pipeline 执行步骤 1-6，然后通过 `asyncio.gather` 并发训练。

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `orchestrator/orchestrator.py` | 115-188 | `_ensure_scheduler_singleton` — 创建/获取 scheduler + 初始化拓扑 |
| `orchestrator/orchestrator.py` | 216-276 | `allocate_pipeline_id` / `register_pipeline` / `admit_pipeline` |
| `orchestrator/orchestrator.py` | 291-420 | `kill_pipeline` — 5 步清理流程 |
| `orchestrator/orchestrator.py` | 428-457 | `shutdown` — 全集群强制关停 |
| `miles_pipeline.py` | 60-117 | `MilesPipeline.__init__` + `_validate_topology`（F10 前置校验） |
| `miles_pipeline.py` | 120-133 | `initialize_pipeline` — try/except + `shutdown_hard` 清理 |
| `miles_pipeline.py` | 135-243 | `_init_phase_a_train` — Step 1-7（request train → RayTrainGroup → bucket cache → offload → release） |
| `miles_pipeline.py` | 244-499 | `_init_phase_b_infer` — Step 1-8（request infer → placement → RolloutManager → bootstrap → INIT→GEN 转换） |
| `miles_pipeline.py` | 982-1033 | `_build_placement_provider` — F8-4 修复位置 |
| `miles_coordinator.py` | 590-633 | `create_pipeline_actor` — 构建 MilesPipeline Ray actor |

### Guide 已知 Bug（guide §3 F8 section，共 4 个）

| ID | 描述 | 修复位置 | 状态 |
|----|------|----------|------|
| F8-1 | M11.1 attempt 4 — Phase B step 8 INIT→GEN 转换后首次 ACTOR_TRAINING 抢占时挂起。根因：`step_target_estimate` 在 INIT→GEN re-request 时缺失，gap-ratio planner 无法规划。修复：Phase B step 8 从 `rollout_batch_size × n_samples_per_prompt` 推导 `gen_step_target_estimate` 并传入 `_request_cluster_gpus`。 | rlix `2b73aef` + `e0a6b27`（`miles_pipeline.py:484-494`） | ✅ 已解决 |
| F8-2 | M11.2 attempt 1 — raylet SIGABRT（`errno=24` EMFILE）。根因：默认 soft fd limit 1024 被 2 个 MilesPipeline + ~20 SGLang 子进程耗尽。修复：`scripts/run_smoke_dual.sh` 设置 `ulimit -n 65536`。 | 运维脚本（非 rlix 代码） | ✅ 已解决 |
| F8-3 | M11.2 attempt 1 — 两个 `MilesPipeline` 并发调用 `find_available_port` 均使用起始端口 15000 → 端口竞争碰撞。修复：miles 侧 `MILES_ROLLOUT_BASE_PORT` 环境变量，dual driver 为每条 pipeline 设置不同值（15000 / 16000）。 | miles `992fa26`（非 rlix 侧修复） | ✅ 已解决 |
| F8-4 | M11.2 attempt 4 — `_build_placement_provider` 硬编码 `train_device_mapping=range(actor_count), infer_device_mapping=range(rollout_num_gpus)`。M11.2 P2（物理池 [2,3]）仍传 [0,1]。修复：读取 `pipeline_config.cluster_device_mappings`，无时回退到 `range(...)`。 | rlix `d97178e`（`miles_pipeline.py:1007-1020`） | ✅ 已解决 |

### F8 硬编码值（🆕 新发现）

| 值 | 位置 | 风险 |
|----|------|------|
| `getattr(..., "rollout_batch_size", 1)` + `getattr(..., "n_samples_per_prompt", 1)` | `miles_pipeline.py:485-487` | 与 F4-GETATTR 同模式：上游重命名字段时静默回退到 1。影响 planner 效率（认为每步只需 1 sample），不影响正确性。低风险。 |
| `8` | `miles_coordinator.py:117` | `RollResourceManagerProxy(num_gpus_per_node=8)` 默认值。非 8-GPU 机器上如果 `pipeline_config.num_gpus_per_node` 缺失，proxy 拓扑错误。影响仅限 coordinator 的 node-PG 引用（用于 scheduling_strategy）；构造失败有 try/except 兜底（L120-126）。中风险。 |
| `4` | `orchestrator.py:156` | Scheduler actor `max_concurrency=4`。注释解释了推导（2 pipeline × fan-out），但无环境变量可配。低风险。 |
| `2` | `miles_coordinator.py:53` | `_MILES_PIPELINE_ACTOR_MAX_CONCURRENCY` — MilesPipeline actor 的 `max_concurrency`。注释解释了推导（init + 并发 resize），但 `before_training`（内含最长 60s 的 offload 等待）+ `signal_rollout_demand` 可同时占满 2 个槽位，此时 `shutdown_hard` 需排队。无环境变量可配。低风险。 |

### F8 冗余代码（🆕 新发现）

| 内容 | 位置 | 说明 |
|------|------|------|
| `import os as _os` | `miles_coordinator.py:606` | `create_pipeline_actor` 方法体内对 stdlib `os` 的懒导入 — 顶层已有 `import os`（L21）。与 F2/F5+F6 中 `_expand_workers:510` 的同一模式。⏳ rlix PR #25 覆盖。 |

### F8 设计观察（🆕 新发现）

| ID | 内容 | 位置 | 说明 |
|----|------|------|------|
| F8-SHUTDOWN-FLAGS | `shutdown_hard` 释放失败仍无条件清零 ledger 标志 | `miles_pipeline.py:748-770` | 循环内 `notify_release_gpus` 失败仅 warn，循环结束后 `_actor_train_allocated` / `_actor_infer_allocated` 无条件置 False（L769-770）。与 R11-F1 约定（"只在成功释放后翻转标志"）及 `_notify_release_cluster_gpus` docstring（"shutdown_hard will retry the release if the flag stays True"）自相矛盾——第二次 `shutdown_hard` / `dispose` 调用会跳过重试，调度器 ledger 泄漏该分配（peer pipeline 饿死），与 PR #14 修复的泄漏同族。触发需要首次释放 RPC 瞬时失败 + 二次调用，概率低但契约已破。 |
| F8-INIT-RELEASE-HANG | init 阶段释放失败后继续执行，后续无超时的 `request_gpus` 可能永久挂起 | `miles_pipeline.py:236-243, 254-259` | Phase A step 7 释放 actor_train 失败（返回 False）只记 warning 不中断 init；当 GPU 池按单角色 sizing（P1-7 注释描述的正是该场景）时，Phase B step 1 的 `_request_cluster_gpus`（`ray.get` 无超时，L904-906）等待的正是本 pipeline 尚未释放的 GPU → 静默死锁而非 fail-fast。Phase B step 8 的 release→re-request 是同一模式（L475-495）。 |
| F8-PIPELINE-TYPE | `allocate_pipeline_id` 类型契约过时且无运行时校验 | `orchestrator.py:30, 216-226` | `PipelineType = Literal["ft", "lora"]`（docstring 称前缀用于 Perfetto trace 可读性），但 miles driver 实际传 `"miles"`（`run_miles_rlix.py:113`、`run_miles_dual.py:224`）。Literal 在运行时不生效，任意字符串均被接受。低风险，但类型注解/docstring 与实际用法已脱节。 |
| F8-INIT-PEER-LEAK | dual driver 中 P2 init 失败导致 P1 全套 actor 泄漏 | `run_miles_dual.py:423-448`（miles 侧，资源泄漏级） | `_build_pipeline`（含 `initialize_pipeline`）在 `_async_main` 的 try/finally 之外顺序执行；P2 init 抛异常时 P2 自身会 `shutdown_hard`，但已初始化的 P1（coordinator 为 `lifetime="detached"`，引擎持有 VRAM）无人清理，直到人工 `ray stop`。与 guide §6.1 F8/F10 NOTE（detached coordinator persists；需 `orchestrator.cleanup_stale_pipelines()` RPC，M11.3）同一问题族。 |
| F8-SHUTDOWN-ROUTER-LEAK | `SGLangEngine.shutdown` 在 router 不可达时跳过 `kill_process_tree` → SGLang 进程树孤儿化持有 VRAM | `miles/backends/sglang_utils/sglang_engine.py:567-599`（miles 侧，资源泄漏级） | `shutdown()` 的 miles-router 分支（`use_miles_router` / 旧 router，L574-577）对 `/remove_worker` 的 `requests.post` **无 try/except**：router 已死时（恰是失败清理场景——router 崩溃导致训练失败 → driver finally → shutdown_hard）抛 `ConnectionError` → L599 `kill_process_tree(self.process.pid)` 被跳过。上层 `RolloutManager.shutdown_hard`（`rollout.py:1210-1218`）捕获异常后 `ray.kill(handle)`——但 ray.kill 只杀 Ray actor，**不杀其 spawn 的 SGLang server 子进程树**（rollout.py 的注释自己写明 "CUDA context lives in the SGLang server child processes"）。结果：孤儿 SGLang 进程持有 VRAM，`ray stop` 也未必回收（已脱离 Ray 进程树），需人工 pkill，后续 run OOM。≥0.3.0 router 分支（L582-593）有 try/except 保护，miles-router 分支没有——修法是对齐该保护或把 `kill_process_tree` 移入 finally。 |
| F8-INTERNAL-API | `kill_pipeline` 使用 Ray 内部 API 强杀未命名 actor | `orchestrator.py:386-413` | 当未命名 actor 在超时后仍存活时，使用 `ray._raylet.ActorID` + `core_worker.get_actor_handle` 强杀。代码已有 FIXME 标注。跨 Ray 版本脆弱（内部 API 可能变更）。 |
| F8-ADMIT-SOFT | `admit_pipeline` 对未注册 pipeline 返回 `scheduler=None` | `orchestrator.py:269-271` | 未注册 pipeline 调用 admit 时 warn + 返回 None scheduler（非 raise）。如果调用方未检查，后续 `scheduler.request_gpus.remote()` 触发 `AttributeError`。Fail-soft 设计在此场景不如 fail-fast。 |

---

## §8 F9 — Progress reporting

### 功能说明

F9 实现 per-rollout 的进度汇报，让调度器的 gap-ratio planner 获得实时的训练/推理节奏信号。hooks 协议（`MilesRLixHooks`）由 rollout function 调用，coordinator 聚合后按 2% bucket 粒度转发给 scheduler。

**三步生命周期：**
1. `begin_progress_batch(target_weight_version, step_target_groups, ...)` — 开始新 batch
2. `bump_completed(target_weight_version)` — 每完成一个 trajectory
3. `end_progress_batch()` → `clear_progress_stream` — 退出 batch

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `miles_hooks.py` | 32-209 | `MilesRLixHooks` — hooks 协议实现，fire-and-forget 向 coordinator 发送 `ProgressReport` |
| `miles_coordinator.py` | 175-261 | `report_progress_from_scheduler` — 接收报告 + `_aggregate_and_emit` 聚合 + 2% bucket 门控 + 转发 scheduler |
| `miles_coordinator.py` | 204-220 | `clear_progress_stream` — 退出 batch，清除 coordinator 侧状态 |

### Guide 已知 Bug（guide §3 F9 section，共 0 个）

Guide 记录 F9 在 M11.1/M11.2 happy path 上**没有发现 bug**。每次 rollout 的报告聚合在两次 smoke 运行中端到端验证通过。

### F9 硬编码值（🆕 新发现）

| 值 | 位置 | 风险 |
|----|------|------|
| `50` | `miles_coordinator.py:241` | `bucket = math.floor(percent_completed * 50)` — 2% bucket 门控（50 个 bucket = 100% / 2%）。符合 plan spec，但作为裸数字嵌入公式而非命名常量。低风险。 |
| `"actor_train"` | `miles_hooks.py:29` | `_DEFAULT_MODE` 默认模式标签。约定常量，coordinator 和 hooks 两侧一致。低风险。 |

### F9 正确性/设计观察（🆕 新发现）

| ID | 内容 | 位置 | 说明 |
|----|------|------|------|
| F9-SCHED-HANDLE | Coordinator 的 scheduler 句柄 fail-open 且从不重试 | `miles_coordinator.py:150-159` | `__init__` 解析 central scheduler 失败时仅 warn 并置 `_rlix_scheduler = None`；之后 `_aggregate_and_emit` / `clear_progress_stream` 的转发全部静默跳过（`if self._rlix_scheduler is not None`），actor 整个生命周期不再重试。对比 `MilesPipeline._get_scheduler_handle`（`miles_pipeline.py:965-980`）是懒加载 + 失败后下次重试的模式，两侧不一致。缓解：B-13 修复后 `signal_rollout_demand` 直连 scheduler + `step_target_estimate` 兜底，不会死锁，但 planner 永久失去该 pipeline 的实时进度信号且无告警升级。 |
| F9-REFACTOR-HOOKS | experimental refactor 路径不转发 `rlix_hooks`，且 rlix 模式无守卫 | `miles/ray/rollout.py:1271-1279`（miles 侧） | `_get_rollout_data` 只在 legacy 路径（`call_rollout_fn`）转发 `self._rlix_hooks`；`MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` 时走 `call_rollout_function(...)`，hooks 根本不传 → 整个 F9 进度通道（begin/bump/end + `clear_progress_stream`）静默失效，且 `rollout_open_pipelines` entry 因 `clear_progress` 永不触发而永久滞留（叠加 S12-INFLATE-WINDOW 的需求双计）。rlix 侧 F10 的 C1-C23 未校验该 env 组合。不会 crash（`signal_rollout_demand` 直连路径兜底防死锁），但建议在 `assert_rlix_topology` 加一条 C-check 禁止 rlix 模式 + experimental refactor 同开。 |

其余无重大发现。Progress RPCs 采用 fire-and-forget 设计（`ObjectRef` 不等待），符合高频热路径需求。Hooks 侧（`miles_hooks.py`）coordinator 不可达时进度信号静默丢失，但 scheduler 有 `step_target_estimate` 兜底，不会导致分配卡死。

⏳ **相关 in-flight**：rlix PR #28（Tianye，OPEN）正在统一 gen step_target 的计量单位为 group count——当前 rlix 侧 `gen_step_target_estimate = rollout_batch_size × n_samples_per_prompt`（trajectory 数，`miles_pipeline.py:484-488`），而 hooks 的 `step_target_groups` 是 group 数，两个口径并存。合入后 F8-1/F9 相关行号与推导描述需复核。

---

## §9 F10 — Partial GPU topology validation（启动 fail-fast C1–C23）

### 功能说明

F10 在启动时运行一系列拓扑和配置前置断言（partial-overlap 子集检查、推理引擎数 ≥ 2、`cpu_serialize` transport、无 MoE/EP、fullasync generation、Megatron 并行度整除等）。错误配置必须在初始化阶段 fail-fast，而非运行时静默 OOM。

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `miles_pipeline.py` | 89-114 | `_validate_topology` — 在 `__init__` 中调用 `assert_rlix_topology`（GPU 分配前执行） |

> **注意**: F10 核心实现在 `miles/miles/utils/rlix_validation.py`（`assert_rlix_topology` 现位于 L180-405）。rlix 侧只是调用入口。本次复查已完整阅读该文件（433 行），未发现 crash / 数据损坏 / 资源泄漏级问题（miles 侧按审查范围只报此三类）。

### Guide 已知 Bug（guide §3 F10 section，共 0 个）

Guide 记录 F10 在每次 smoke 运行中正确触发。M11.1 attempt 10 日志确认 `F10 startup validation passed`。

### F10 设计观察（🆕 新发现）

| ID | 内容 | 位置 | 说明 |
|----|------|------|------|
| F10-FAILOPEN | 验证模块不可用时静默跳过 | `miles_pipeline.py:106-112` | `from miles.utils.rlix_validation import assert_rlix_topology` 失败时 warn + return，不 raise。测试环境（miles 未安装）合理；生产环境如果 miles 安装损坏，所有 C1-C23 检查被静默绕过。⏳ rlix PR #32（Tianye，OPEN 2026-07-05）将 except 收窄为 `ModuleNotFoundError`——正是此 finding 的修复方向，合入后可标 ✅。 |

---

## §10 F11 — Conditional RLix behavior flag（`RLIX_CONTROL_PLANE=rlix`）

### 功能说明

单一环境变量 `RLIX_CONTROL_PLANE=rlix` 门控所有 rlix 模式特有行为。standalone `train_async.py` 在该变量未设置时保持原有行为不变（AC8）。

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `utils/env.py` | 24-35 | `pipeline_identity_env_vars` — 将 `RLIX_CONTROL_PLANE` 传播到子 actor 的 `runtime_env`，默认 `"rlix"` |
| `protocol/types.py` | 16 | `ROLL_RESOURCE_MANAGER_ACTOR_NAME` 注释引用 `RLIX_CONTROL_PLANE=rlix` |

> **注意**: F11 的所有门控调用点（`is_rlix_mode()` 判断、各 miles 模块的条件分支）都在 `miles/` 仓库。rlix 侧仅负责环境变量传播。

### Guide 已知 Bug（guide §3 F11 section，共 0 个）

Flag 在所有 smoke 运行中正常工作，未产生误判。

### F11 正确性/设计观察（🆕 新发现）

无。环境变量传播逻辑简洁正确。

---

## §11 F12 — Shared PG cluster（`MilesPlacementProvider`、`LOCAL_RANK=0`、SGLang `base_gpu_id=0`）

### 功能说明

F12 替换 MILES standalone 的 `_create_placement_group(rollout_num_gpus)` 为 rlix 驱动的 `MilesPlacementProvider` 适配器。Train 侧使用 `RayTrainGroup(num_gpus_per_actor=0)` + 手动 `CUDA_VISIBLE_DEVICES`；SGLang 使用 `base_gpu_id=0`（post-CVD 本地视角）；`LOCAL_RANK=0` 显式注入。

### 代码位置（rlix 侧）

| 文件 | 行号 | 作用 |
|------|------|------|
| `miles_pipeline.py` | 982-1033 | `_build_placement_provider` — 构建 `MilesPlacementProvider`，优先从 `cluster_device_mappings` 读取设备映射 |
| `miles_pipeline.py` | 303-316 | Phase B step 2 — `get_all_rollout_engine_placements()` → `legacy_pg` 元组转换 |
| `miles_pipeline.py` | 165-189 | Phase A — `RayTrainGroup(num_gpus_per_actor=0, worker_placements=...)` |

> **注意**: F12 核心实现在 `miles/miles/ray/placement_provider.py`（`MilesPlacementProvider` 类，325 行，本次已完整阅读）。rlix 侧是调用方和适配层。

### Guide 已知 Bug（guide §3 F12 section，共 0 个）

Guide 指出 F12 的唯一问题即 F8-4（`_build_placement_provider` 硬编码 `range(...)`），已在 §7 F8 中记录（✅ 已解决）。F12 section 无独立 bug。

### F12 正确性/设计观察（🆕 新发现）

无。`_build_placement_provider` 在 F8 中已充分审查。`legacy_pg` 元组转换逻辑正确（`assert_structural` 前置校验 + 扁平化 bundle_indices / gpu_ids）。miles 侧 provider 自身校验完备：构造期断言（tp 整除、sorted、gap-free per-engine slice）、`WorkerPlacement.__post_init__` 结构校验、跨节点 slice fail-fast、`get_active_engine_indices` 拒绝 partial-engine 分配 — 未发现 crash / 泄漏级问题。

⏳ **跟进项**：Tianye 的 miles PR #31（DRAFT，2026-07-05）指出 F33 "SGLang `base_gpu_id=0`" 存在 doc/code mismatch（需要方向决策）。本 review 未独立核查 `base_gpu_id` 的实际取值路径（sglang_engine 非 rlix 段未读），F12 该点以 Tianye 的 draft 为准跟进。

---

## §12 Extra — Scheduler 核心与 B-13（非 F1-F12，guide §3.5 域）

### 功能说明

> **Extra 分类说明**：本节内容不属于任何 F1-F12 移植 feature——scheduler 核心（`scheduler.py` / `planner.py` / `state.py`）是 rlix 原有的中央控制面，是 F1-F12 各 feature 插入的"底座"。guide 也将其修复（B-13 等）记录在 §3.5 而非 §3 的 F 编号下；本 review 对应地以 §12 平行放置，不跨归属到任何 F section。

Scheduler 是 rlix 的中央 GPU 仲裁器（单例 actor，async 模型，全部可变状态由 `asyncio.Lock` 保护）。核心循环 `_central_scheduling_loop` 由事件（request / progress / release）或 1s 后台轮询触发，每个 cycle 依次执行：planned release → 非 GEN 分配（可抢占 GEN donor）→ gap-ratio GEN 规划（`planner.py`）→ 11 条不变量校验（`validation.py`）→ 锁外执行 resize RPC → 锁内提交状态。

B-13 修复（Tianye PR #16）引入 durable 的 `rollout_open_pipelines` 注册表：`request_gpus(GENERATION)` 入队时写入 `pipeline_id → step_target_estimate`，跨 planning cycle 存活，直到 release / `clear_progress` / unregister / fail-fast 时清除。gap-ratio planner 用它做接收方资格判定和 bootstrap 需求注入，解决"双 pipeline 全缩容后无人唤醒"的死锁。

### 代码位置（本次全审范围）

| 文件 | 行号 | 作用 |
|------|------|------|
| `scheduler/state.py` | 32 | `rollout_open_pipelines: Dict[str, Optional[int]]` durable 注册表（B-13） |
| `scheduler/scheduler.py` | 588-651 | `request_gpus` — 短路 / 入队 / 写注册表（L637-638） |
| `scheduler/scheduler.py` | 653-669 | `notify_release_gpus` — 释放 + 清注册表（L666-668） |
| `scheduler/scheduler.py` | 671-755 | `notify_release_then_request_gpus` — 原子 release+request（同样写注册表 L737-739） |
| `scheduler/scheduler.py` | 779-824 | `_central_scheduling_loop` — 事件驱动 + 1s 后台轮询 |
| `scheduler/scheduler.py` | 874-939 | `_should_background_rebalance_locked` — Case 2.1/2.2 触发启发式 |
| `scheduler/scheduler.py` | 941-1211 | `scheduling_cycle` — 六阶段规划执行 |
| `scheduler/scheduler.py` | 1372-1420 | `_execute_resize_calls` — 锁外 shrink→expand RPC |
| `scheduler/scheduler.py` | 1437-1655 | `_apply_plan_and_signal` — 状态提交 + 唤醒 waiter |
| `scheduler/planner.py` | 137-457 | `plan_generation_gap_ratio` — gap-ratio DP 算法（消费注册表 L181/195-217） |
| `scheduler/validation.py` | 152-507 | `validate_execution_plan` — 11 条不变量 + 模拟执行 |

### Guide §3.5 修复验证（B-13 / PR #16）

| 项 | 验证结果 | 状态 |
|----|----------|------|
| `rollout_open_pipelines` durable 注册表 | `state.py:32` 存在，生命周期注释与实现一致：release 清除（`scheduler.py:666-668`）、`clear_progress` 清除（L582）、unregister 清除（L291）、fail-fast 清除（L777） | ✅ 已解决 |
| `request_gpus` 写注册表 + 唤醒循环 | `scheduler.py:637-638`（GENERATION 入队时写入）+ L645 `_wakeup_event.set()`；`notify_release_then_request_gpus` 同样处理（L737-739） | ✅ 已解决 |
| planner 消费 durable 注册表 | `planner.py:181`（`_receiver_eligible` 以注册表 membership 判定资格）、L195-208（无 progress 时用 estimate 做 bootstrap 需求）、L213-217（demand 注入） | ✅ 已解决 |
| `signal_rollout_demand` 调度器侧行为 | `miles_pipeline.py:829-870` → `request_gpus`：engines 仍 active 时走 L614-625 短路返回既有分配；全缩容时（`active_dp_ranks == ∅`，L620-621）入队 + 写注册表 + 阻塞至 gap-ratio 激活 ≥1 个 DP worker（wake-only signaling，L1132-1148）——与 docstring 描述一致 | ✅ 已解决 |
| guide §6.1 LOW1（`generate_rollout_fully_async` 是否接受 `rlix_hooks` kw） | **已验证可关闭**：`miles/examples/fully_async/fully_async_rollout.py:225` 声明 `rlix_hooks: RLixHooks | None = None` 并实际调用 begin/bump/end（L275/385/422）；`call_rollout_fn` 的 `inspect.signature` 转发（`base_types.py:97-98`）能命中，不会静默 no-op。注意：仅覆盖 legacy 路径，experimental refactor 路径见 §8 F9-REFACTOR-HOOKS | ✅ 已验证 |
| Phase 1（R04-F1）train 异常泄漏 actor_train 分配 | **代码级验证通过**：`miles/utils/rlix_train_loop.py:117-201` — `before_step` + `train` 同在 try 内（before_step 先于任何后续失败点 claim 了 actor_train，必须包含）；`except BaseException` 标记 `train_failed` 后 re-raise；`finally` 中失败路径走 `release_only`（跳过会在未 onload 权重上崩溃的 `after_step`），`release_only` 自身异常被捕获记录不遮蔽原始异常；成功路径 `after_step` 完整执行 | ✅ 已解决（guide §3.5 Phase 1） |
| guide §6 F19/F20（save 需要 onload→save→offload bracket）状态过时 | guide §6.1 标 "Unchanged"，但 `rlix_train_loop.py:203-218` **已实现**显式 `onload → save_model → offload`（try/finally）bracket；eval 仍按注释跳过（L220-224）。建议 guide 更新该行状态为"save 已解决 / eval 仍缺" | 🆕 文档过时 |

### §12 硬编码（🆕 新发现）

| 值 | 位置 | 风险 |
|----|------|------|
| `"actor_infer"` 字符串字面量 | `planner.py:96, 188, 239`；`miles_pipeline.py:537, 1011` | 应使用 `GENERATION_CLUSTER_NAME` 常量 — PR #30 (Kyle) 做过同类统一，这 5 处是残留。改名时静默失配。低风险（常量值稳定），但与 PR #30 的意图不一致。背书：7/5 周会上 Tao 点名 `actor_infer` 硬编码问题（该命名源自 ROLL 的 Rollout 阶段设计），PR #30 是会上直接合入的——此 5 处残留清理与团队共识一致。 |
| `1.0` s / `10.0` / `0.5` | `scheduler.py:794, 939, 933` | 后台轮询间隔、rebalance 触发的 10 百分点偏差阈值、worst-half 比例 — 启发式常量嵌入代码，无命名常量、无配置。调参需改源码。低风险。 |

### §12 冗余代码（🆕 新发现）

| 内容 | 位置 | 说明 |
|------|------|------|
| `notify_release_then_request_gpus` 三重存在性检查 | `scheduler.py:698-714` | L698 `in` 检查 → L700-703 `get` + None 检查（代码自注释 "Redundant guard"）→ L712-714 `pop` + None 检查，同一锁内同一 key 检查三次。 |
| `clusters_to_remove` 永不填充 | `types.py:97`（自带 TODO）| `ExecutionPlan.clusters_to_remove` 从未被写入，但 planner / scheduler / validation 中有 ~8 处针对它的分支和校验（如 `validation.py:247-252` condition 11、`_collect_shrink_trace_infos_locked` L1280-1294）。守护死路径的代码。 |
| `_prepare_resize_calls_locked` docstring 与实现矛盾 | `scheduler.py:1319-1323` | Docstring 声称 "exactly one of {dp_ranks_to_remove, dp_ranks_to_add} may be non-empty"，但实现（L1356-1365 行内注释）明确允许同 pipeline 同 cycle 一缩一扩（只禁同一 dp_rank 重叠）。文档过时。 |

### §12 设计观察（🆕 新发现）

| ID | 内容 | 位置 | 说明 |
|----|------|------|------|
| S12-RESIZE-NOTIMEOUT | `_execute_resize_calls` 无超时，单 pipeline 卡死拖垮全局调度 | `scheduler.py:1390-1408` | 中央循环锁外 `asyncio.gather` coordinator 的 `resize_infer` RPC，无超时、无 watchdog。与 F3-TIMEOUT-1/2/3（coordinator 侧 `ray.get` 也无超时）叠加：任何一条 pipeline 的 shrink 卡住（如 SGLang drain 挂起），整个 scheduling loop 停摆，**所有** pipeline 的 request/release 无人处理。fail-fast 机制只覆盖异常路径，不覆盖挂起路径。与 guide §6.1 MED1 同一问题族（M11.3 concurrent-resize stress test）。 |
| S12-EST-NONE-HANG | GENERATION 请求缺 `step_target_estimate` 且无 progress 时静默永久阻塞 | `scheduler.py:637-638` + `planner.py:199-208` | `request_gpus(GENERATION, step_target_estimate=None)` 被接受并写入注册表值 None；planner 侧 estimate None 且 progress=0 → `continue`（pipeline 整体跳过）→ pending 请求永不 signal，调用方 `ray.get` 无限阻塞。该坑仅记录在 `miles_pipeline._request_cluster_gpus` 的 docstring 里；scheduler ingress 不校验也不告警。建议：入队时 warn 或直接拒绝无 estimate 且无 progress 的 GEN 请求。 |
| S12-INFLATE-WINDOW | demand 膨胀不限于 bootstrap 窗口 | `planner.py:213-217` | 注释说 "Inflate demand in the bootstrap window (request sent, no progress yet)"，但条件只检查注册表 membership——entry 存活至 release/`clear_progress`，rollout 进行中（progress > 0）仍然 `remaining += step_target`，需求双计。双 pipeline 同时在 rollout 时近似对称，影响小；单侧持有 entry 时会偏斜 gap-ratio 分配。低风险。✅ **已确认并修复**（2026-07 中）：issue [#36](https://github.com/rlops/rlix/issues/36) Tianye 确认应门控于 no-progress，其 PR [#38](https://github.com/rlops/rlix/pull/38)（OPEN）在 estimate 分支覆盖前捕获 `in_bootstrap_window = step_target <= 0.0` 并改门控为 `has_pending and in_bootstrap_window`——bootstrap 路径不变、只消除双计；diff 已核对与确认语义一致。 |
| S12-REREGISTER | 重复注册重置 scheduler 侧 admitted，与 orchestrator 状态失配 | `scheduler.py:489-496` + `orchestrator.py:257-262, 272-273` | `register_pipeline_topology` 无条件覆盖 registry entry（`"admitted": False`），而 orchestrator 侧 re-register 保留 `admitted=True` 且 `admit_pipeline` 对已 admit 的 pipeline 提前返回（不再调 scheduler.admit）→ re-register 后该 pipeline 所有 `request_gpus` 永远报 "not admitted"。注意：scheduler/orchestrator 重启（fail-fast 场景）时两边状态都清空，无失配；失配只发生在 orchestrator 存活期间对同一 pipeline_id 重复 register。当前流程没有该场景，低风险。 |


