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
- 多 turn 路径还没有 current-turn retry / turn-scoped retry
- placement group 由 MILES 内部创建和切分，不能直接复用 RLix 拥有的 shared PG
- 权重更新路径会 pause / flush / update / continue 全部 rollout engines

因此推荐采用三阶段策略：

- **Phase 1：RLix-managed barrier async**
  - 先把 MILES 变成 RLix 可统一调度的 async backend
  - 只在 MILES 已有 safe boundary 上做 request / release / progress / shared-PG 接入
  - 不承诺 mid-flight subset preemption
- **Phase 2：subset lifecycle + admission close**
  - 增加 subset shrink/expand 所需的 engine indexing、router disable/remove、close-and-drain
  - 支持 DP 粒度 rollout subset 生命周期
- **Phase 3：true mid-flight migration**
  - 增加 deterministic request id、targeted abort ACK、turn-scoped retry、selective sync

这是当前最可信、也最快能做出端到端 demo 的路线。重点是先让计划和 MILES 当前真实状态一致，而不是提前承诺框架还没有的能力。

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

### 3.1 Async Loop 已存在，但权重激活边界是 barrier

文件：`train_async.py`（MILES 仓库根目录）

当前行为：

- MILES 会在训练当前 rollout 时提前启动下一个 rollout future
- 到 `update_weights_interval` 时，会先 drain `rollout_data_next_future`，然后才执行 `actor_model.update_weights()`
- 这保证了权重更新不会发生在 rollout future 仍在运行时

含义：

- Phase 1 必须保留这个 barrier
- RLix 应把 MILES async 理解为：
  - “一个 rollout 可以和一个 train step overlap”
  - “active checkpoint 只在 MILES 自己的安全更新边界上前进”
- 这不是 hot weight update，也不是任意 mid-turn 的模型替换

### 3.2 同步训练循环已经有完整的 safe boundary

文件：`train.py`（MILES 仓库根目录）

当前行为：

- 同步 loop 会按顺序执行 generation、可选 rollout offload、training、可选 train offload、weight sync、rollout KV onload
- 这个结构天然适合作为 scheduler request/release 的第一批接入点

含义：

- `train.py` 适合作为最早的 control-plane smoke test 目标
- `train_async.py` 是真正的 Phase 1 主目标，但在此之前可以先拿 `train.py` 打通最简单链路

### 3.3 Placement Group 仍由 MILES 内部创建

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

### 3.4 Rollout lifecycle 现在是 all-engine 级别

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

### 3.5 Router 还没有 scheduler-owned 的 admission close

文件：`miles/router/router.py`

当前行为：

- Miles Router 只有 `/add_worker` 和 `/list_workers`
- 没有 `/remove_worker`、`/disable_worker`、`/enable_worker`
- `SGLangEngine.shutdown()` 仍在尝试调 `/remove_worker`
- router middleware 里对 partial rollout 仍有限制，尤其 radix-tree 路径下会显式 reject

含义：

- 旧文档里“shrink 时 remove worker”的说法目前不是现有能力
- Phase 2 需要给 router 增加 worker state：至少 `enabled / disabled / dead`，最好还有 inflight 统计
- shrink 顺序必须是：
  - close admission
  - drain 或 abort
  - 等 drain / ACK
  - 再释放显存

### 3.6 权重更新当前是 global sync

文件：

- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

当前行为：

- tensor / distributed / broadcast / P2P 更新模式都默认操作当前全量 rollout engines
- 更新流程固定是：pause generation -> flush cache -> transfer weights -> post-process -> continue generation
- P2P 是性能优化，但当前仍假设一个静态的全-engine 更新计划

含义：

- Phase 1 直接复用当前 global sync
- Phase 2 如果 expand 仍然只发生在 batch boundary，可以先用 global sync
- 真正的 selective sync 应放到 Phase 3；因为 process group、transfer plan、active engine set 都要变成 subset-aware

### 3.7 Multi-turn abort 当前仍是 terminal 语义

文件：

- `miles/rollout/generate_hub/multi_turn.py`
- `examples/retool/generate_with_retool.py`
- `examples/retool_v2/*`
- `examples/experimental/swe-agent/generate_with_swe_agent.py`
- `miles/utils/types.py`
- `examples/fully_async/fully_async_rollout.py`

当前行为：

- 通用 multi-turn generate 在遇到 `abort` 或 `length` 时会终止当前路径
- Retool v1 直接把 abort 标成 `Sample.Status.ABORTED`
- SWE-agent v1 把非成功退出当作 aborted / truncated
- `Sample.reset_for_retry()` 会清空整条 sample 的输出状态，不是 current-turn continuation primitive
- 只有 fully-async 示例会在 group 层面把 aborted 样本 reset 后重新塞回 buffer

含义：

- 旧计划里“preserve token/history state，然后 retry 当前 turn”不是当前现成能力
- Phase 1 绝不能依赖这个语义
- Phase 2 默认只做 close-and-drain，或对明确无 side effect 的 stateless path 做 whole-sample retry
- Phase 3 才能显式增加：
  - per-turn state capture
  - deterministic request id
  - targeted abort ACK
  - side-effect commit 规则

### 3.8 Weight version metadata 已经有用

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
| Pipeline runtime | MILES 训练入口 / 新 wrapper | 在 safe boundary 上调 RLix |
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
- 这样才保留 MILES 当前 `train_async.py` 的 rollout / train overlap
- Phase 1 async path 仅覆盖 non-colocate，因为当前 `train_async.py` 明确 `assert not args.colocate`
- 但 RLix 不得在 rollout future 仍 live 时对该 pipeline 执行 infer shrink/offload

#### Phase 1 的 fallback

- 如果 scheduler 在某一时刻不能同时满足 infer/train 两类 allocation，或者 MILES 配置为 colocate，pipeline 必须退化成 sync-safe ordering：
  1. request infer
  2. generate
  3. release infer
  4. request train
  5. train + update
  6. release train
- 也就是说，Phase 1 允许 overlap，但不保证永远能 overlap

#### Phase 1 明确不做的事

- 不对 live rollout future 做 shrink
- 不做 mid-flight subset preemption
- 不在 weight update 期间尝试 selective sync

### 4.4 Safe Phase 1 Timeline

#### 同步安全顺序

1. request `actor_infer`
2. generate rollout batch
3. release `actor_infer`
4. request `actor_train`
5. train rollout batch
6. 在 MILES update boundary 上执行 `actor_model.update_weights()`
7. release `actor_train`

#### barrier-async 顺序

1. request `actor_infer`
2. 启动 rollout `N`
3. 在 train allocation 已就绪时，训练 rollout `N-1`
4. 到 MILES 的 weight update boundary 前，先 drain in-flight rollout future
5. 只有 future drain 完成后，才允许 release / resize `actor_infer`
6. 执行 MILES 现有 global update
7. 以新的 active version 恢复下一轮 rollout

这个语义和当前 MILES 源码一致，也避免了 unsafe mid-flight preemption。

## 5. Phase 1：RLix-Managed Barrier Async

目标：

- 让 MILES 成为 RLix 可统一调度的 async backend
- 只在 MILES 已有 safe boundary 上接入 scheduler
- 不要求 subset shrink/expand
- 不要求 current-turn retry
- 保留 MILES 现有 weight update barrier

### 5.1 RLix 侧工作

默认方案：

- **复用现有 `PipelineCoordinator` 的设计，而不是直接复用当前实现**
- 当前 `PipelineCoordinator` 里有值得复用的 scheduler/progress/resize-lock/per-pipeline actor delegation 设计
- 但当前实现仍带有 ROLL / vLLM / offload-NCCL / `RollResourceManagerProxy` 假设，不是 backend-neutral generic coordinator
- MILES Phase 1 应先抽出 backend-neutral base coordinator / resource-provider seam，或先新增专用 `miles_coordinator.py`
- 新增 MILES-specific pipeline runtime / pipeline actor

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
- 在 generation / train / update safe point 周围发起 request/release
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
- non-colocate async path
- `sglang_data_parallel_size = 1`
- 单 updateable model
- `retool_v2` 或一个更简单的 single-turn 示例

Phase 1 明确排除：

- PD disaggregation
- SWE-agent external gym
- session-server agentic tool loops
- colocate async overlap path；colocate 只作为 sync-safe smoke / fallback
- Miles router + partial rollout 混合路径
- selective P2P update
- mid-flight shrink

### 5.4 Phase 1 Validation Gates

Gate P1-A：topology bridge

- MILES 能接受 RLix-owned placement info
- actor 与 rollout bundle map 和 RLix cluster config 一致
- 普通 non-RLix MILES 路径不变

Gate P1-B：single-pipeline scheduler control

- 一个 MILES pipeline 能成功注册到 RLix
- generation / training phase 会发起 request / release
- 不会在 rollout future live 时尝试 resize

Gate P1-C：barrier async correctness

- `train_async.py` 在 weight update 前仍会 drain next rollout future
- 不会在 tracked rollout future live 时执行 `actor_model.update_weights()`
- async overlap 验证仅覆盖 non-colocate；colocate 不进入该 path
- generated train data 中带有 `weight_versions`

Gate P1-D：progress correctness

- scheduler 最终收到的是 trajectory-count progress
- `completed` 只在 train batch 真正 ready 时达到 target
- 旧 progress stream 会在 batch 被消费后清除

Gate P1-E：two-pipeline time sharing

- 两个 MILES pipeline，或 MILES + 另一个 RLix pipeline，能在 batch/update boundary 上交替运行
- Perfetto 或 scheduler logs 能显示 release/request 顺序

## 6. Phase 2：Subset Lifecycle + Admission Close

目标：

- 支持 DP 粒度 rollout shrink/expand
- 引入 close-and-drain 语义
- 仍默认避免对 stateful multi-turn 做 abort+retry

### 6.1 Phase 2 scope 限制

Phase 2 默认只覆盖：

- 单 updateable rollout server
- 单 updateable model
- 非 PD
- 非多 server_group 的复杂拓扑
- `sglang_data_parallel_size = 1`

原因：

- 当前 `RolloutManager.get_updatable_engines_and_lock()` 还是基于一个 updatable server 组织
- 如果不先限定 scope，subset lifecycle / selective sync 的复杂度会被文档提前低估

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
- remove 只能在 worker drain/shutdown 后执行

### 6.4 Close-and-drain shrink

对 `indices=P` 的 shrink 顺序：

1. 在 router 中 disable 目标 worker URL
2. 向 scheduler 报告目标 subset 进入 draining 状态
3. 等待目标 engine drain 或超时
4. 仅对目标 engine 调 `flush_cache()`
5. 仅对目标 engine 调 `release_memory_occupation(tags=...)`
6. 在 RLix 里把目标 infer DP rank 标为 inactive
7. 向 scheduler 发送 release ACK

默认策略：

- 对 stateful multi-turn path，默认使用 drain，不使用 abort
- abort 仅在明确无 side effect 的 stateless 示例或专项测试中启用

### 6.5 Phase 2 Validation Gates

Gate P2-A：router disable/remove

- disabled worker 不再接收新 routed request
- 其上的 in-flight request 可以自然完成
- `/remove_worker` 与 `SGLangEngine.shutdown()` 语义对齐

Gate P2-B：subset lifecycle

- `offload(indices=[i])` 只释放目标 engine `i`
- 其他 engine 继续 serving
- `onload(indices=[i])` 只恢复目标 engine `i`

Gate P2-C：shrink without starvation

- scheduler 会在 flush/offload 之前先 close admission
- 持续外部请求不会让 target engine 永远 drain 不完

Gate P2-D：expand

- 只有在 weights/KV 已加载完成后，expanded engine 才重新 enable
- router 会把新工作发到恢复后的 active set

## 7. Phase 3：True Mid-Flight Migration

目标：

- 支持 scheduler 触发的 mid-flight shrink/expand
- 要求 request-level abort ACK + safe retry
- 第一批仅针对受控的 multi-turn 示例，例如 Retool v2

### 7.1 Deterministic request IDs

每次模型生成请求都需要 RLix-owned request id：

`rid = "{pipeline_id}:{rollout_id}:{group_id}:{sample_id}:{turn_id}:{attempt}"`

需要维护的状态：

- `trajectory_id`
- `turn_id`
- `attempt`
- 当前分配到的 engine
- worker URL
- creation timestamp
- active weight version
- retry cause

### 7.2 Targeted abort

需要补齐或验证 SGLang 端支持：

- `POST /abort_request` 支持按 request id 定向 abort
- 仅 abort 指定 worker 上的指定 request
- 能明确返回或暴露 finish reason 为 `abort` 的 ACK

shrink + abort 顺序：

1. disable target worker admission
2. 找出 target worker 上的 in-flight rid
3. 对选中的 rid 发送 targeted abort
4. 等 abort ACK
5. 把可重试的 current turn 重新分派到 active workers
6. 只有 ACK 或安全 drain 完成后，才能 offload target engines

硬规则：

- 如果 ACK 超时，正常模式下不得 silent offload
- 必须 crash 或把 pipeline 标成 unhealthy，而不是默默丢工作

### 7.3 Turn-scoped retry

这是当前 MILES 没有的能力，必须新增。

需要的语义：

- 保留已完成 earlier turns
- 保留直到最后一个 committed non-abort turn 为止的环境 / tool 状态
- 只重试当前 aborted 的 model call / 当前 turn
- request id 中仅 `attempt` 增长
- 当前 turn retry 不得复用 `Sample.reset_for_retry()`，因为那是 whole-sample reset

side-effect commit 规则：

- tool / env side effect 只有在该 turn 的模型生成结果被确认为 non-abort 后才算 committed
- 如果外部工具已经执行，则 example 必须提供 idempotency key 或 resume protocol

推荐第一目标：

- `examples/retool_v2/*`
- 具体实现落点优先考虑 `miles.rollout.generate_hub.multi_turn.generate`

SWE-agent：

- 晚于 Retool
- 因为 external gym 持有 tool 执行语义，需要更明确的 idempotency / resume 设计

### 7.4 Selective sync

Selective sync 与 lifecycle 是独立问题。

需要的改动：

- 让 `get_updatable_engines_and_lock(indices=None)` 变成 subset-aware
- 让 tensor / broadcast / P2P weight update 路径接受目标 engine 子集
- 为 active subset 重建或缓存 transfer plan
- 确保 newly expanded engine 在 router enable 前已经拿到最新 active weight version

务实顺序：

1. Phase 2 expand 先继续使用 global batch-boundary sync
2. Phase 3 再增加 subset sync-on-expand
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
- close-and-drain shrink 实现

### Team D：progress / accounting

- group -> trajectory accounting
- creation timestamps
- in-flight/completed/consumed counters
- 2% band reporting

### Team E：migration / multi-turn retry

- deterministic request ids
- targeted abort + ACK tracking
- Retool v2 的 turn-scoped retry
- 后续 SWE-agent idempotency / resume

### Team F：weight sync

- version propagation
- sync-on-expand
- 先做 tensor/broadcast subset sync
- P2P selective update 后置

## 9. 关键风险与待定决策

### 9.1 待定决策

- Phase 1 是否先拿 `train.py` 做同步 smoke test，再切到 `train_async.py`
- RLix 应以 library 方式调用 MILES，还是由 MILES 新建 RLix-aware training entrypoint
- 是先抽 backend-neutral base coordinator，还是先实现 `miles_coordinator.py` 再回收公共逻辑
- 是否现在就创建 `rlops/miles` 社区 fork
- Phase 2 是否统一基于 Miles Router，而不是 SGLang Model Gateway

### 9.2 主要风险

- `sglang_data_parallel_size > 1` 会引入一层当前 RLix 还未建模的 DP 轴
- PD disaggregation 的 engine role 比“一个 rollout engine = 一个 infer DP worker”复杂得多
- SWE-agent / session-server 例子会产生外部 side effect，在 idempotency 明确之前，abort+retry 是不安全的
- P2P weight transfer 对性能重要，但它当前仍偏静态；不要让 selective P2P 阻塞 Phase 1
- 如果没有 admission close，global `flush_cache()` 在 shrink 时会被持续新请求饿死

### 9.3 推荐的立即决策

- 批准 Phase 1 采用 **RLix-managed barrier async**
- 把 subset shrink/expand 与 current-turn retry 明确当作 Phase 2 / Phase 3 的独立工作流
- 第一批 multi-turn 验证目标选 `Retool v2`
- 在 retry 语义跑通前，不把 SWE-agent 当作早期主验证路径
