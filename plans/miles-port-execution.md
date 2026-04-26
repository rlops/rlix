# MILES Port Execution Plan

日期：2026-04-26

本执行计划按阶段拆分，目的是让实现能并行推进，同时避免把 Phase 3 的复杂问题提前绑死在 Phase 1 里。

阶段定义：

- **Phase 1**：验证 RLix-managed barrier async，并以最小 framework 改动把 scheduler 跑通
- **Phase 2**：补 subset lifecycle 和 admission close
- **Phase 3**：补 true request migration 与 turn-scoped retry

当前 milestone 只把 TASK 0-6 作为 release critical path；TASK 7-13 保留为后续与 NeMo/ROLL 更强对齐的 deferred work。

## TASK 0：分支、基线、文档

Owner：RLix docs / tech lead

写入范围：

- `plans/miles-port-plan.md`
- `plans/miles-port-execution.md`

依赖：

- 无

动作：

- 保持 `_rlix` 本地 `miles` 分支基于 `origin/release`
- 在该分支上推进 RLix 计划与 adapter 工作
- 明确后续是否创建 `rlops/miles` 作为 framework patch 载体

验收：

- 计划通过 review
- 工作流 owner 明确
- 文档里没有把 current-turn retry 写成当前已有能力

## TASK 1：MILES Topology 与 Placement-Group Bridge

Owner：MILES topology team

写入范围：

- `miles/ray/placement_group.py`
- `miles/utils/arguments.py`
- 可选新增 `miles/ray/rlix_placement.py`
- RLix 侧的 config / example 如有需要

依赖：

- TASK 0

动作：

- 增加 RLix placement provider 模式，允许接收已有 Ray placement group 或显式 bundle map
- 保留 `create_placement_groups(args)` 的当前默认行为，当 RLix mode 关闭时完全不变
- 导出 actor / rollout 的 GPU 与 bundle 映射，供 RLix 验证
- Phase 1 约束：`sglang_data_parallel_size == 1`
- infer DP rank 初始定义为 MILES rollout engine index

验收：

- 非 RLix 的 MILES 运行仍自己创建 PG
- RLix mode dry run 不会创建冲突的 placement group
- actor / rollout 映射与 RLix scheduler cluster config 一致

## TASK 2：RLix MILES Pipeline Runtime Skeleton

Owner：RLix control-plane team

写入范围：

- 优先：`rlix/pipeline/miles_pipeline.py`
- `rlix/pipeline/base_coordinator.py` 或等价 backend-neutral coordinator 抽象
- 或先补：`rlix/pipeline/miles_coordinator.py`
- 可选：`rlix/conf/*` 或 example config

依赖：

- TASK 1 的 topology contract；若未完成，可先用 stub mapping 开发

动作：

- 复用现有 `PipelineCoordinator` 的设计模式，而不是直接复用当前实现
- 当前 `PipelineCoordinator` 仍强耦合 ROLL / vLLM / offload-NCCL / `RollResourceManagerProxy`
- 选择实现路径：
  - 方案 A：先抽 backend-neutral base coordinator / resource-provider seam
  - 方案 B：先做 `miles_coordinator.py`，后续再把公共逻辑抽回 base
- 为 MILES 增加 per-pipeline pipeline actor / runtime
- 注册 `actor_train` 与 `actor_infer` cluster config
- 第一版 request/release 可以先做 no-op 或 logging-only
- 给所有会碰同一个 MILES runtime 的调度入口加串行化边界

验收：

- 一条 MILES pipeline 能注册到 RLix scheduler
- 在不启动完整训练的情况下，scheduler 日志里能看到 train / infer cluster 的 allocate / release
- 现有 ROLL coordinator / ROLL pipeline 行为不受影响
- 文档和实现不得再把当前 `PipelineCoordinator` 称为可直接复用的 backend-neutral coordinator

## TASK 3：Phase 1 生命周期接入

Owner：RLix control-plane team + MILES training-loop team

写入范围：

- `train_async.py` 或新的 `train_rlix_async.py`
- 如需要同步 smoke test，也可先接 `train.py`
- TASK 2 中的 RLix MILES pipeline runtime

依赖：

- TASK 2

动作：

- 在 MILES safe point 周围增加 RLix request/release
- 当前 milestone 明确 port 主线 `train_async.py` 的 one-step-off 语义，不把 `examples/fully_async` 当作 baseline
- Phase 1 async path 仅覆盖 non-colocate，因为当前 `train_async.py` 明确不支持 colocate
- colocate 配置只能先走 sync-safe smoke / fallback
- 第一批 smoke test 可先用 sync-safe 顺序：
  1. request infer
  2. generate
  3. release infer
  4. request train
  5. train/update
  6. release train
- 随后保留 `train_async.py` 的 one-rollout prefetch：
  - rollout/train overlap 只在 scheduler 已分配 infer/train 资源时保留
  - 如果 scheduler 不能同时提供两类 allocation，则自动退化到 sync-safe ordering
  - 如果配置是 colocate，则自动退化到 sync-safe ordering
  - 在 Phase 1 中，不允许 resize / offload 一个 live rollout future

验收：

- 单条 MILES pipeline 在 RLix scheduler 下至少能跑完两轮 rollout/train
- 任何 `actor_model.update_weights()` 执行时，都不存在 tracked rollout future in flight
- non-colocate async overlap 和 colocate sync-safe fallback 的路径边界明确
- 配置关闭 RLix mode 后，运行行为退回普通 MILES

## TASK 4：Progress Accounting 与 Heartbeats

Owner：progress/accounting team

写入范围：

- `miles/ray/rollout.py`
- 如 timestamps / counters 需要绑在 buffer 上，可改 `miles/rollout/data_source.py`
- RLix MILES runtime 的 progress reporting 逻辑

依赖：

- TASK 2

动作：

- 在 MILES 本地维护 batch-window counters：
  - groups released for generation
  - groups queued
  - groups in flight
  - groups completed
  - groups dropped
  - groups consumed by training
- 用 `n_samples_per_prompt` 将 group 转成 trajectory
- 跟踪 `oldest_unfinished_creation_ts`
- 在 batch start、分母变化、2% 完成度 band 上报 heartbeat
- 在 batch 被训练消费后清除 progress stream

关键协议约束：

- 本地可以维护 `queued/inflight/collected/consumed/version` 等 richer counters
- 但上报给 RLix scheduler 的 wire-level contract 仍应遵守：
  - `step_target_trajectories`
  - `metrics["completed"]`
- scheduler 不应直接以 `queued/inflight` 作为 canonical remaining 入口
- MILES runtime 不应绕过 coordinator 直接向 scheduler 上报 `queued/inflight/remaining`

验收：

- `percent_completed >= 1.0` 仅在下一训练 batch 真正 ready 时成立
- scheduler 收到的是 trajectory counts，而不是原始 group counts
- scheduler ingress 的 canonical demand 只由 `metrics["completed"]` 推导
- progress 在单 batch window 内单调前进

## TASK 5：Full Sync 路径与 Weight Boundary

Owner：weight-sync team

写入范围：

- `miles/backends/megatron_utils/actor.py`
- 如仅缺 metadata 传播，再触碰 update_weight 模块
- RLix MILES runtime 的 logs / metrics

依赖：

- TASK 3
- TASK 4

动作：

- 把 MILES 的 `update_weights_interval` 当作 active checkpoint activation boundary
- 先把当前 full/global sync 路径在 RLix 下跑通
- 把 `Sample.weight_versions` 与 `oldest_weight_version` 传播到 RLix metrics
- 记录 rollout batch start version 与 train-data handoff version
- 当前 milestone 保持现有 global weight update 语义，不尝试 selective sync

验收：

- 日志可显示每个 train batch 对应的 generation weight version
- Phase 1 不触发 selective sync
- stale rollout 的程度在 metrics 中可见

## TASK 6：当前 Milestone 集成测试（先 1 个，再 2 个）

Owner：integration team

写入范围：

- test configs
- example scripts
- 可选 trace / log helper

依赖：

- TASK 3
- TASK 4
- TASK 5

动作：

- 先跑一个 simple single-turn / 低副作用示例，验证 scheduler、registration、progress、full/global sync 都正常
- 再让两条 pipeline 在 RLix scheduler 下运行，其中至少一条是 MILES
- 两条 pipeline 的目标是验证 scheduler 是否能正常完成 resource handoff / overlap，而不是先做复杂 multi-turn preemption
- 记录 scheduler 日志或 Perfetto trace，展示 request/release 顺序
- `retool_v2` 作为第一批 multi-turn 集成目标，SWE-agent 后置

验收：

- 单 pipeline 简单示例能稳定跑通
- 各 pipeline 不会越过 scheduler allocation 重叠占用 GPU
- 两条 pipeline 在同一测试中完成真实的 release/request handoff
- MILES 在声明的 safe point 上释放 infer/train 资源
- 当前 milestone 不使用任何 mid-flight subset preemption

## 未来对齐任务（不属于当前 milestone critical path）

以下 TASK 7-13 保留在总计划中，用于后续与 NeMo/ROLL 的更强语义对齐；它们当前不阻塞 first release。

## TASK 7：Router Admission API

Owner：MILES rollout lifecycle team

写入范围：

- `miles/router/router.py`
- `miles/backends/sglang_utils/sglang_engine.py`
- router tests

依赖：

- 可在 TASK 0 后并行启动

动作：

- 增加 `/disable_worker`、`/enable_worker`、`/remove_worker`
- 扩展 `/list_workers`，返回 worker state 与 inflight metadata
- 确保 routing 会跳过 disabled / dead worker
- 让 `SGLangEngine.shutdown()` 与新的 router API 对齐

验收：

- disabled worker 不再接收新请求
- disabled worker 上的 in-flight request 可以自然跑完
- `/remove_worker` 不再指向不存在的 endpoint

## TASK 8：Subset Rollout Lifecycle

Owner：MILES rollout lifecycle team

写入范围：

- `miles/ray/rollout.py`
- 必要时增加 engine-index metadata helper

依赖：

- TASK 7

动作：

- 增加稳定 rollout engine index
- 给 rollout lifecycle 调用增加 `indices=None`
- 把 index 映射到 engine handle / worker URL
- 保留 `indices is None` 时的 all-engine 旧语义

Phase 2 scope 限制：

- 先只支持单 updatable rollout server / 单 updateable model
- 不覆盖 PD / 多 server_group 的复杂拓扑

验收：

- `offload(indices=[i])` 只影响 engine `i`
- `onload(indices=[i])` 只影响 engine `i`
- 其他 engine 继续 routable 且健康

## TASK 9：Phase 2 Close-And-Drain Shrink（deferred）

Owner：RLix control-plane team + MILES lifecycle team

写入范围：

- RLix MILES runtime 的 shrink/expand handler
- TASK 8 的 subset lifecycle 实现

依赖：

- TASK 7
- TASK 8

动作：

- 这一步当前明确是 deferred，不进入 first release critical path
- shrink 顺序：
  1. disable router admission for target indices
  2. wait for drain 或 timeout
  3. flush target engines
  4. release memory on target engines
  5. 向 scheduler ACK release
- expand 顺序：
  1. onload target engines
  2. sync/load weights as needed
  3. enable router admission
  4. 在 scheduler 中标记 active

验收：

- 在 admission close 后，持续新请求不会再饿死 shrink
- shrink 一个 engine 不会停掉其他 active engines
- stateful multi-turn path 默认使用 drain，不使用 abort

## TASK 10：Sync-On-Expand 与 Selective Sync 准备

Owner：weight-sync team

写入范围：

- `miles/backends/megatron_utils/actor.py`
- `miles/backends/megatron_utils/update_weight/*`
- `miles/ray/rollout.py`

依赖：

- TASK 8
- TASK 9

动作：

- 让 updatable engine discovery 支持 subset-aware
- 确保 expanded engine 在 router enable 之前拿到最新 active version
- 先做 tensor 或 broadcast subset sync，再考虑 P2P subset sync
- 如 transfer plan 仍静态，P2P selective update 后置

验收：

- newly expanded engine 提供服务时使用的是预期的最新 weight version
- scheduler 能区分 loaded-but-disabled 与 enabled-and-serving
- global sync path 仍保留可用

## TASK 11：Deterministic Request IDs 与 Targeted Abort（deferred，当前 milestone 不做）

Owner：migration team

写入范围：

- `miles/rollout/generate_hub/single_turn.py`
- `miles/rollout/generate_hub/multi_turn.py`
- `miles/rollout/sglang_rollout.py`
- `miles/rollout/inference_rollout/*`
- 必要时 router / engine 集成代码

依赖：

- TASK 7
- TASK 8

动作：

- 这一步当前不进入 first release critical path，目的是尽量减少对 MILES 原有框架的改动
- 只有当后续明确要做 true targeted abort + current-turn retry 时，再启用这一组工作
- 生成 RLix request id：
  - `pipeline_id`
  - `rollout_id`
  - `group_id`
  - `sample_id`
  - `turn_id`
  - `attempt`
- 把 request id 透传到 `/generate`
- 跟踪 `rid -> worker_url -> sample/turn`
- 实现 targeted abort
- 在 retry/offload 前等待 abort ACK

验收：

- abort 一个 request 不会把同 engine 上其他无关请求一起杀掉
- 缺失 ACK 时会受控失败或标 unhealthy，而不是 silent offload
- retry 前后 request id 只允许 `attempt` 改变

## TASK 12：Retool V2 Turn-Scoped Retry（deferred，作为第一批 multi-turn 目标）

Owner：migration + examples team

写入范围：

- `miles/rollout/generate_hub/multi_turn.py`
- `examples/retool_v2/*`
- 必要的新 retry-state helper

依赖：

- TASK 11

动作：

- 每次 model call 前 capture turn state
- 保留 completed earlier turns
- 收到 abort ACK 后，仅把当前 turn 重派到 active engine
- 当前路径不允许使用 `Sample.reset_for_retry()`
- 只有 non-abort generation 才提交 tool/env side effect

验收：

- Retool 当前 turn 被强制 abort 后，只重试该 turn，不重启整条 sample
- earlier completed turns 保持 intact
- metrics 能区分 preemption retry 与普通 engine error retry

## TASK 13：SWE-Agent Idempotency 与 Resume（deferred，晚于 Retool）

Owner：agentic examples team

写入范围：

- `examples/experimental/swe-agent/*`
- 可选新增 `examples/experimental/swe-agent-v2/*`
- 如需要，也包括 session server 集成

依赖：

- TASK 11
- TASK 12

动作：

- 给 tool action 定义 idempotency key
- 定义 tool action 之间的 resume point
- 确保 retry 不会重复执行同一个 side-effecting tool action
- 只有在 Retool 语义验证通过后，才把 targeted abort 扩展到 SWE-agent

验收：

- SWE-agent trajectory retry 不会重复执行同一个 side-effecting tool action
- abort/retry 在 logs / metrics 中可观察
- 不安全示例可以 opt out，继续只用 close-and-drain

## TASK 14：发布分支

Owner：release owner

写入范围：

- `_rlix` 分支 `miles`
- 如果已创建，则包括 MILES framework branch

依赖：

- 当前 milestone release 依赖 TASK 6
- 后续 subset lifecycle / admission-close release 依赖 TASK 9 与 TASK 10
- 后续 true migration / multi-turn retry release 依赖 TASK 12 或 TASK 13

动作：

- review 后把 `_rlix` 本地 `miles` push 成 `origin/miles`
- 如果有 framework patch，则把 MILES 分支 push 到约定 remote
- 附上对应 phase 的 validation 日志

验收：

- phase 对应 gate 全部通过
- 分支名、remote、发布范围已记录清楚
- 文档与实现状态一致
