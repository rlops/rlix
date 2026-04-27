# MILES Port Plan

日期：2026-04-26

本次分析基于以下本地输入：

- RLix 工作树：`_rlix` 仓库，本地分支 `miles`，当前基线为 `origin/release`
- 当前 MILES 源码：`miles` 仓库，分支 `main`，当前提交 `a772c33`（`Add zero_std all_zero_ratio and all_one_ratio metrics (#1034)`）
- NeMo 参考：`plans/nemorl-port-plan.md`、`plans/nemorl-port-execution.md`
- 旧 MILES 适配文档：`design_doc/archive/adaptation_miles.md`

## 1. 执行摘要

之前的 MILES 适配分析方向是对的，但新计划必须比旧文档更严格，不能把“理想目标”写成“当前框架已有能力”。

MILES 当前已经具备一批对接 RLix 很有价值的基础能力：

- `train_async.py` 中已经有异步 rollout / train 重叠循环
- rollout engine 已有显式 `offload/onload` 生命周期 API
- 训练侧已有显式的权重更新 barrier
- 生成样本已经携带 `weight_versions`
- 已集成 SGLang router 和 rollout engine 生命周期 hook

但当前 MILES 还不具备把 NeMo 那种 partial-overlap 直接当作第一阶段交付的基础：

- rollout lifecycle 现在本质上仍是“全 engine 操作”
- Miles Router 没有 scheduler-owned 的 admission-close API
- abort 仍以 `abort_all` 或样本终止为主，不是 targeted request migration
- 当前还没有 request-level migration 或更细粒度恢复语义
- placement group 由 MILES 内部创建和切分，不能直接复用 RLix 拥有的 shared PG
- 权重更新路径会 pause / flush / update / continue 全部 rollout engines

因此推荐采用三阶段策略：

- **Phase 1：RLix-managed bounded-staleness fullasync baseline**
  - 先把 MILES 变成 RLix 可统一调度的 fullasync backend
  - baseline 明确对齐 `examples/fully_async` 这条 bounded-staleness 路线，当前以 `fully_async_rollout.py` + `run-qwen3-4b-fully_async.sh` 承载的 math example 作为第一版载体
  - handoff 语义先落在 engine-granularity abort + requeue，复用现有 `fully_async` 的 group-level recycle 行为
  - Phase 1 暂沿用当前 global sync / `actor_model.update_weights()` 作为 functional baseline，但明确它不是最终形态
- **Phase 2：router admission、subset lifecycle、sync-on-expand 准备**
  - 增加 engine indexing、router disable/remove、subset `onload/offload` 与 sync-on-expand 所需基础设施
- **Phase 3：selective sync 与更细粒度恢复语义（如确有必要）**
  - 只有在 Phase 1/2 证明当前 baseline 不足时，再补 subset-aware selective sync 或更细粒度 recovery hardening

当前 milestone 仍然先追一个最小可跑通的版本，但目标语义已经切到基于 `examples/fully_async` 的 bounded-staleness fullasync baseline。重点是先让 scheduler、refresh、engine-level abort+requeue 这条链路在 math example 上跑通，再逐步加后续基础设施。

## 2. 分支与仓库策略

### 2.1 RLix 仓库

- 在 `_rlix` 仓库的 `miles` 分支上推进 plan 和 RLix 接入层工作
- 该分支当前基于 `origin/release`，这是正确的干净基线，因为当前不存在 `origin/miles`
- 计划文件和 RLix 侧 adapter / pipeline / scheduler 接入先落在这条分支上
- 评审通过后，再把本地 `miles` 分支 push 成远程 `origin/miles`

### 2.2 MILES 框架仓库

- 当前 MILES framework 源码位于 `miles` 仓库
- 当前 remote 还是个人 fork；如果确认要做框架级修改，建议创建或对齐社区仓库 `rlops/miles`
- Phase 2 和 Phase 3 的 framework patch 应放在 `rlops/miles` 上的单独集成分支，例如 `rlix-integration` 或 `rlix-miles`
- Phase 1 仍尽量少改 framework，但 placement-group bridge 和 progress hook 大概率仍然需要落到 MILES 源码里

### 2.3 NeMo 仓库的角色

- `nemo-rl` 仓库适合作为对照样例
- 但 MILES 不应机械复制 NeMo 的 partial-overlap 计划，因为 MILES runtime 当前缺少若干 scheduler-safe primitive

## 3. 当前 MILES Reality Check

### 3.1 `train_async.py` 当前仍是 one-step-off，对照用而非本计划 baseline

文件：`train_async.py`（MILES 仓库根目录）

当前行为：

- MILES 会在训练当前 rollout 时提前启动下一个 rollout future
- 到 `update_weights_interval` 时，会先 drain `rollout_data_next_future`，然后才执行 `actor_model.update_weights()`
- 这保证了权重更新不会发生在 rollout future 仍在运行时

含义：

- 当前主线 async 是 **one-step-off**：`rollout N+1` overlap `train N`
- 这条路径不是本计划要对接的 baseline，只作为对照与必要时的 smoke fallback
- 本计划的 baseline 明确建立在 `examples/fully_async` 的 bounded-staleness fullasync 路线上

### 3.2 `examples/fully_async` 是当前 baseline

文件：

- `examples/fully_async/README.md`
- `examples/fully_async/fully_async_rollout.py`
- `examples/fully_async/run-qwen3-4b-fully_async.sh`

当前行为：

- generation 由 persistent background worker 持续拉取数据并执行
- `train_async.py` 每次只是从输出队列 drain 已完成结果
- aborted group 会被 reset 后重新塞回 data buffer
- 已有 `max_weight_staleness`、`weight_version`、`oldest_weight_version` 等 fullasync / staleness 相关机制
- 当前最直接的 baseline 载体就是 `fully_async_rollout.py` + `run-qwen3-4b-fully_async.sh` 这一组 math example

含义：

- 这条路径本身就是 bounded-staleness fullasync rollout pool，本计划明确 built against 这条路径
- 第一版不要求 request-level migration，但 handoff 语义要复用这里已有的 background generation + aborted-group recycle

### 3.3 同步训练循环已经有完整的 safe boundary

文件：`train.py`（MILES 仓库根目录）

当前行为：

- 同步 loop 会按顺序执行 generation、可选 rollout offload、training、可选 train offload、weight sync、rollout KV onload
- 这个结构天然适合作为 scheduler request/release 的第一批接入点

含义：

- `train.py` 适合作为最早的 control-plane smoke test 目标
- 但当前 milestone baseline 仍是 `examples/fully_async` 路径；这里的 `train_async.py` 只作为承载 fullasync rollout function 的 training driver，不应再被表述成独立的 Phase 1 主目标

### 3.4 Placement Group 仍由 MILES 内部创建

文件：`miles/ray/placement_group.py`

当前行为：

- `create_placement_groups(args)` 直接创建 Ray PG
- 然后在内部切 actor / critic / rollout 的 bundle 范围
- colocate 模式下 rollout 和 actor 还会重叠

含义：

- RLix 如果不能把 shared PG 提供给 MILES，MILES 就不可能被 RLix 统一调度
- Phase 1 必须补一个 placement-group provider / bridge
- 第一版 bridge 应尽量保守：
  - 允许外部传入已有 PG
  - 允许显式指定 actor/rollout bundle index 范围
  - 不改变 RLix 之外的普通 MILES 运行方式

### 3.5 Rollout lifecycle 现在是 all-engine 级别

文件：

- `miles/ray/rollout.py`
- `miles/backends/sglang_utils/sglang_engine.py`

当前行为：

- `RolloutManager.offload/onload/onload_weights/onload_kv` 都会 fan-out 到所有 rollout server / engine
- `RolloutServer` 和 `ServerGroup` 还没有对 scheduler 暴露 `indices` 过滤接口
- `SGLangEngine.release_memory_occupation()` 之前会先 `flush_cache()`

含义：

- Phase 1 只能在 safe boundary 上调用当前全量 lifecycle
- Phase 2 才能增加 engine index map 和 `indices=` 过滤
- 如果未来要 shrink subset，必须先 close admission，再 flush/offload；否则新请求会继续进来，drain 可能永远不会完成

### 3.6 Router 还没有 scheduler-owned 的 admission close

文件：`miles/router/router.py`

当前行为：

- Miles Router 只有 `/add_worker` 和 `/list_workers`
- 没有 `/remove_worker`、`/disable_worker`、`/enable_worker`
- `SGLangEngine.shutdown()` 仍在尝试调 `/remove_worker`
- router middleware 里对 partial rollout 仍有限制，尤其 radix-tree 路径下会显式 reject

含义：

- 旧文档里“shrink 时 remove worker”的说法目前不是现有能力
- Phase 2 需要给 router 增加 worker state：至少 `enabled / disabled / dead`，最好还有 inflight 统计
- 后续 subset lifecycle 至少要保证：
  - 先 close admission
  - 再走 engine-granularity abort + requeue 或明确的安全释放点
  - 在确认安全释放点之前不能释放显存

### 3.7 权重更新当前是 global sync，Phase 1 先沿用其 functional baseline

文件：

- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

当前行为：

- tensor / distributed / broadcast / P2P 更新模式都默认操作当前全量 rollout engines
- 更新流程固定是：pause generation -> flush cache -> transfer weights -> post-process -> continue generation
- P2P 是性能优化，但当前仍假设一个静态的全-engine 更新计划

含义：

- Phase 1 暂沿用这条 global sync / `actor_model.update_weights()` 路径作为 functional baseline
- 但它不是最终形态；subset sync / sync-on-expand 仍后置到后续阶段
- 文档必须明确这一点，避免把“Phase 1 继续沿用 global sync”误写成“全计划最终保留 global sync”

### 3.8 当前现成的 retry 语义是 fully_async 的 group-level recycle

文件：

- `miles/rollout/generate_hub/multi_turn.py`
- `miles/utils/types.py`
- `examples/fully_async/fully_async_rollout.py`

当前行为：

- 除 `fully_async` 之外的通用 generate path 在遇到 `abort` 或 `length` 时会直接终止当前路径
- `Sample.reset_for_retry()` 会清空整条 sample 的输出状态，不是 request-level migration primitive
- `examples/fully_async` 会在 group 层面把 aborted 样本 reset 后重新塞回 buffer

含义：

- 当前 milestone 不再追 request-level deterministic migration，也不定义更细粒度的 request/turn-level recovery
- handoff 语义直接建立在 engine-granularity abort + group-level requeue 之上
- 这正好匹配 `examples/fully_async` 已有的 recycle 行为

### 3.9 Weight version metadata 已经有用

文件：

- `miles/utils/types.py`
- `miles/rollout/generate_utils/generate_endpoint_utils.py`
- `miles/ray/rollout.py`

当前行为：

- sample 已携带 `weight_versions`
- train data 已可带 `weight_versions`
- `oldest_weight_version` 已经能在 sample 上计算

含义：

- RLix 不需要从零重新发明 generation version tracking
- Phase 1 只要把这些版本信息正确接到 RLix progress / metrics / validation 就够了

## 4. 架构映射

### 4.1 Cluster 概念映射

| RLix 概念 | MILES 组件 | Phase 1 映射 |
| --- | --- | --- |
| `actor_train` | `RayTrainGroup` / `actor_model` | 整体训练 allocation |
| `actor_infer` | `RolloutManager` / SGLang engines | 整体 rollout allocation |
| Pipeline runtime | MILES 训练入口 / 新 wrapper | 以 fullasync / background generation 语义接 RLix |
| Progress stream | rollout batch/group accounting | batch 开始 + band 更新 |
| Weight activation | `actor_model.update_weights()` | 沿用 MILES barrier |
| Rollout version | `Sample.weight_versions` | 上报到 metrics / validation |

### 4.2 初始 DP 映射

Phase 1 和 Phase 2 的 infer DP 单位定义为：

- `infer_dp_rank = rollout_engine_index`
- `infer_tp_size = args.rollout_num_gpus_per_engine`
- 初始限制：`args.sglang_data_parallel_size == 1`

原因：

- SGLang 内部 DP 会引入另一层 rank 轴
- 当前 MILES 的 server group / internal DP 还没有暴露成 RLix scheduler 可直接操作的 worker 集
- 把一个 MILES rollout engine 视为一个 RLix infer DP worker，是当前最简单也最真实的 Phase 2 子集生命周期单元

### 4.3 Phase 1 资源语义

这是本计划最重要的硬约束之一。

#### Phase 1 允许的情况

- 同一个 MILES pipeline 可以同时持有 `actor_infer` 和 `actor_train` allocation
- generation 应持续在后台推进，trainer 消费已完成结果；这更接近 `fully_async` 的 persistent worker 语义
- 第一版仍只覆盖 non-colocate；colocate 不作为 fullasync baseline
- scheduler 可以在目标 subset 上触发 preempt，但预期语义应是 abort + requeue / retry，而不是以 wait-for-drain 为主

#### Phase 1 的 fallback

- 如果 scheduler 或环境限制导致 fullasync 路径暂时跑不起来，允许用更保守的 safe ordering 做临时 smoke test
- 但这只能作为 bring-up / debugging fallback，不能写成当前 milestone 的目标形态

#### Phase 1 明确不做的事

- 不承诺第一版就有 request-level deterministic migration
- 不承诺第一版就有最细粒度 current-turn targeted retry
- 不要求第一版就完成 selective sync

### 4.4 与 NeMo F4/F7/F10 的对齐说明

- **F4（CPU bucket cache）**：MILES 当前路径不假设必须复刻 NeMo 的 CPU bucket cache 方案。Phase 1 先沿用现有 global sync baseline；只有在后续 subset expand / sync-on-expand 场景证明会遇到 OOM 或 staging 约束时，再显式补 CPU-side cache / staging 设计。
- **F7（namespace isolation）**：当前 milestone 的最终 gate 要跑 2 个 MILES pipelines，因此必须显式隔离 per-pipeline Ray namespace、actor 命名、progress stream 与临时状态。
- **F10（topology validation）**：Phase 1 需要 fail-fast validation；至少覆盖 non-colocate、`sglang_data_parallel_size == 1`、单 updateable model/server、以及 actor/infer device mapping 约束。

### 4.5 Safe Phase 1 Timeline

#### fullasync baseline 顺序

1. request `actor_infer`
2. 启动 persistent background generation
3. request `actor_train`
4. trainer 持续消费已完成 rollout 结果
5. scheduler 需要训练资源或做 handoff 时，对目标 subset 触发 abort
6. aborted group / request 重新回到 data buffer 或 active worker 集合继续生成
7. 执行改造后的 refresh/sync 路径，继续 generation

#### bring-up fallback

- 如果 fullasync 路径在最早调试阶段还不稳定，可以短暂用 safe ordering 验证 PG / registration / scheduler hook 是否正常
- 但该 fallback 不应成为最终 gate，也不应写成当前 milestone 的目标语义

## 5. Phase 1：RLix-Managed Fullasync Baseline

目标：

- 让 MILES 成为 RLix 可统一调度的 fullasync backend
- baseline 对齐 `examples/fully_async` 的 persistent generation + abort 后 requeue / retry 语义
- 第一版先不要求最细粒度 current-turn targeted retry
- 但不再把 current global sync / drain-only 路径当成目标形态

### 5.1 RLix 侧工作

默认方案：

- **复用现有 `PipelineCoordinator` 的设计，而不是直接复用当前实现**
- 当前 `PipelineCoordinator` 里有值得复用的 scheduler/progress/resize-lock/per-pipeline actor delegation 设计
- 但当前实现仍带有 ROLL / vLLM / offload-NCCL / `RollResourceManagerProxy` 假设，不是 backend-neutral generic coordinator
- MILES Phase 1 应先抽出 backend-neutral base coordinator / resource-provider seam，或先新增专用 `miles_coordinator.py`
- 新增 MILES-specific pipeline runtime / pipeline actor，并在这里落下 F7 对应的 per-pipeline namespace isolation

原因：

- 当前 `PipelineCoordinator` 的设计已经证明了几个关键模式：
  - progress aggregation
  - resize / sync serialization
  - per-pipeline actor delegation
  - scheduler actor lookup 与 request/release 封装
- 这些模式应该沿用，避免为 MILES 重新发明一套控制面
- 但当前类初始化时会校验 ROLL/vLLM 相关配置，并直接依赖 ROLL resource manager
- 因此 MILES 不应把它当作可直接复用的 generic coordinator

可能涉及文件：

- `rlix/pipeline/miles_pipeline.py`
- `rlix/pipeline/base_coordinator.py` 或等价 backend-neutral 抽象
- 或者先补 `rlix/pipeline/miles_coordinator.py`，后续再把公共逻辑抽回 base
- 可选配置辅助：`rlix/conf/` 或 `examples/`

职责：

- 将 MILES pipeline id 和 cluster config 注册到 RLix scheduler
- 把 MILES GPU topology 翻译为 RLix 的 `actor_train` / `actor_infer` cluster config
- 让 scheduler 能驱动 fullasync generation / train / refresh 生命周期
- 串行化所有会触碰同一个 MILES runtime 的 scheduler 动作
- 把 MILES 本地 progress accounting 聚合成 RLix 接受的 wire-level progress report

### 5.2 Progress 协议约束

这里必须区分两层，不可混写。

#### MILES 本地 / coordinator 内部可以维护的计数器

- `target_trajectories`
- `queued_trajectories`
- `inflight_trajectories`
- `completed_trajectories`
- `collected_trajectories`
- `consumed_trajectories`
- `oldest_unfinished_creation_ts`
- `active_weight_version`
- `oldest_generation_weight_version`

#### RLix scheduler ingress 真实接受的协议

当前 RLix `ProgressReport` 的 wire-level contract 是：

- `step_target_trajectories`
- `metrics["completed"]`

其余字段可以作为辅助 metadata，但 scheduler 的 remaining demand 推导仍应以 `completed` 为唯一 canonical 入口。

因此 MILES port 里应这样处理：

- 本地先维护 richer counters
- coordinator 聚合后上报：
  - `step_target_trajectories = 当前 batch window 的目标 trajectory 数`
  - `metrics["completed"] = min(collected, target)`
- 允许附带 `collected`、`bucket`、`mode`、版本信息等辅助字段
- 不应假设 scheduler 直接消费 `queued/inflight`
- 不应让 MILES runtime 绕过 coordinator 直接把 `queued/inflight/remaining` 当作 scheduler canonical demand 上报

### 5.3 MILES 侧工作

最小 framework 变更：

- 给 `create_placement_groups(args)` 增加 placement-group provider hook
- 允许外部提供 PG 与 actor/rollout bundle indices
- 在 rollout batch start、group/sample completion 位置加入可选 progress callback hook
- 增加可选参数：
  - `rlix_enabled`
  - `rlix_pipeline_id`
  - RLix topology / placement provider 相关参数
- 确保 RLix mode 关闭时，普通 MILES 路径完全不变

Phase 1 推荐首个目标：

- 非 PD 的 SGLang rollout
- Megatron actor
- non-colocate fullasync path
- `sglang_data_parallel_size = 1`
- 单 updateable model
- **当前 milestone baseline 直接绑定 `examples/fully_async` 的 math example（由 `fully_async_rollout.py` + `run-qwen3-4b-fully_async.sh` 承载）**
- 不再额外设计 single-turn smoke path；当前第一批验证直接从这组 `fully_async` baseline 起步

Phase 1 明确排除：

- PD disaggregation
- first milestone 中的 request-level deterministic migration
- colocate async overlap path；colocate 只作为 bring-up fallback
- selective P2P update

### 5.4 Phase 1 Validation Gates

Gate P1-A：topology bridge

- MILES 能接受 RLix-owned placement info
- actor 与 rollout bundle map 和 RLix cluster config 一致
- 普通 non-RLix MILES 路径不变

Gate P1-B：single-pipeline scheduler control

- 一个 MILES pipeline 能成功注册到 RLix
- generation / training phase 会发起 request / release
- 不会在 rollout future live 时尝试 resize

Gate P1-C：fullasync correctness

- baseline 走 `fully_async` 风格的 persistent background generation
- generation 与 trainer consume 并行存在，scheduler 不会把 pipeline 错误降级成单纯 one-step-off smoke
- abort 后的 group 能回到 data buffer / active worker 集合继续生成
- generated train data 中带有 `weight_versions`

Gate P1-D：progress correctness

- scheduler 最终收到的是 trajectory-count progress
- `completed` 只在 train batch 真正 ready 时达到 target
- 旧 progress stream 会在 batch 被消费后清除

Gate P1-E：two-pipeline scheduler integration

- 先通过 `examples/fully_async` 的 math example gate，证明 scheduler、registration、progress、refresh path 都正常
- 最后再跑两个 MILES pipelines
- 目标是验证 RLix scheduler 的真实 resource handoff / overlap 是否成立
- 该 gate 允许通过 engine-granularity abort + requeue 实现 handoff
- Perfetto 或 scheduler logs 能显示 overlap、handoff、requeue

## 6. Phase 2：Router Admission 与 Subset Lifecycle Preparation

目标：

- 让后续 subset lifecycle 和 sync-on-expand 拥有明确的 router / engine / index 基础设施
- 但不把 request-level migration 或更细粒度恢复语义带回当前计划

### 6.1 Phase 2 scope 限制

Phase 2 默认只覆盖：

- 单 updateable rollout server
- 单 updateable model
- 非 PD
- 非多 server_group 的复杂拓扑
- `sglang_data_parallel_size = 1`

原因：

- 当前 `RolloutManager.get_updatable_engines_and_lock()` 还是基于一个 updatable server 组织
- 如果不先限定 scope，subset lifecycle / sync-on-expand 的复杂度会被文档提前低估

### 6.2 Rollout engine indexing

增加稳定索引映射：

- `engine_index -> SGLangEngine handle`
- `engine_index -> worker_url`
- `engine_index -> server_group`
- `engine_index -> placement bundles`
- `engine_index -> active/enabled/offloaded state`

扩展接口：

- `ServerGroup.offload(indices=None)`
- `ServerGroup.onload(indices=None)`
- `RolloutServer.offload(indices=None)`
- `RolloutServer.onload(indices=None)`
- `RolloutManager.offload(indices=None, tags=...)`
- `RolloutManager.onload(indices=None, tags=...)`
- `RolloutManager.onload_weights(indices=None)`
- `RolloutManager.onload_kv(indices=None)`

### 6.3 Router admission API

增加 router endpoint：

- `POST /disable_worker?url=...`
- `POST /enable_worker?url=...`
- `POST /remove_worker?url=...`
- `GET /list_workers` 返回 `enabled/dead/inflight` 等 metadata

router 选择规则：

- 仅把新请求派发到 `enabled == true && dead == false` 的 worker
- disabled worker 允许把 in-flight 工作跑完，但不得再接新请求
- baseline handoff 继续沿用 Phase 1 的 engine-granularity abort + requeue，不在这里引入 request-level migration

### 6.4 Phase 2 Validation Gates

Gate P2-A：router disable/remove

- disabled worker 不再接收新 routed request
- `/remove_worker` 与 `SGLangEngine.shutdown()` 语义对齐

Gate P2-B：subset lifecycle

- `offload(indices=[i])` 只释放目标 engine `i`
- 其他 engine 继续 serving
- `onload(indices=[i])` 只恢复目标 engine `i`

Gate P2-C：expand readiness

- newly expanded engine 在重新 enable 前已经拿到预期的 active weight version
- router 会把新工作发到恢复后的 active set

## 7. Phase 3：Selective Sync（如确有必要）

目标：

- 在已有 bounded-staleness fullasync baseline 之上，补 subset-aware sync-on-expand
- 不默认引入 request-level migration；只有在 Phase 1/2 被证明不够时，才继续细化恢复语义

### 7.1 Selective sync

Selective sync 与 lifecycle 是独立问题。

需要的改动：

- 让 `get_updatable_engines_and_lock(indices=None)` 变成 subset-aware
- 让 tensor / broadcast / P2P weight update 路径接受目标 engine 子集
- 为 active subset 重建或缓存 transfer plan
- 确保 newly expanded engine 在 router enable 前已经拿到最新 active weight version

务实顺序：

1. Phase 1 继续沿用 global sync / `actor_model.update_weights()` 作为 functional baseline
2. Phase 2/3 再增加 subset sync-on-expand
3. P2P selective update 放在 broadcast/tensor subset sync 正确之后

## 8. 建议的人力拆分

### Team A：RLix control plane

- MILES pipeline actor / adapter
- scheduler request/release integration
- progress stream integration
- multi-pipeline Phase 1 E2E

### Team B：MILES topology bridge

- external placement-group provider
- config / argument bridge
- device mapping validation
- 普通 MILES 运行的 backward compatibility

### Team C：MILES rollout lifecycle

- engine index map
- subset `onload/offload/onload_weights/onload_kv`
- router worker disable/enable/remove
- router admission + subset lifecycle glue

### Team D：progress / accounting

- group -> trajectory accounting
- creation timestamps
- in-flight/completed/consumed counters
- 2% band reporting

### Team E：lifecycle / retry glue

- group-level abort/requeue 集成
- `fully_async` baseline 与 RLix handoff 的粘合逻辑
- 如后续证明必要，再单独补更细粒度 recovery hardening

### Team F：weight sync

- version propagation
- sync-on-expand
- 先做 tensor/broadcast subset sync
- P2P selective update 后置

## 9. 关键风险与待定决策

### 9.1 待定决策

- Phase 1 是否先拿 `train.py` 做同步 smoke test，再切到 `examples/fully_async` baseline（由 `train_async.py` + fullasync rollout function 承载）
- RLix 应以 library 方式调用 MILES，还是由 MILES 新建 RLix-aware training entrypoint
- 是先抽 backend-neutral base coordinator，还是先实现 `miles_coordinator.py` 再回收公共逻辑
- 是否现在就创建 `rlops/miles` 社区 fork
- Phase 2 是否统一基于 Miles Router，而不是 SGLang Model Gateway
- 第一版 fullasync baseline 明确使用 `examples/fully_async/fully_async_rollout.py` + `examples/fully_async/run-qwen3-4b-fully_async.sh` 这组 math example

### 9.2 主要风险

- `sglang_data_parallel_size > 1` 会引入一层当前 RLix 还未建模的 DP 轴
- PD disaggregation 的 engine role 比“一个 rollout engine = 一个 infer DP worker”复杂得多
- P2P weight transfer 对性能重要，但它当前仍偏静态；不要让 selective P2P 阻塞 Phase 1
- 如果没有 admission close，global `flush_cache()` 在 shrink 时会被持续新请求饿死

### 9.3 推荐的立即决策

- 批准 Phase 1 采用 **RLix-managed bounded-staleness fullasync baseline**
- 让 engine-granularity abort + requeue 成为当前 milestone 的 handoff 主方向
- 第一批 scheduler 集成验证目标直接绑定 `examples/fully_async` 的 math example
- 最终 gate 收窄为 2 个 MILES pipelines under RLix scheduler
