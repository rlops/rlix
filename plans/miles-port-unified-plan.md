# MILES 整合进 RLix — 方案

---

## GOAL

**一句话：让 MILES 的 fullasync GRPO 训练可以被 RLix 调度器管理，实现 partial overlap
下的 GPU 时分复用与多 pipeline 共享。**

核心原语与 ROLL 一致：**partial overlapping** — `actor_train` GPU 是 `actor_infer` GPU
的子集。需要 training 时，重叠部分的 inference engine "sleep" 把 GPU 让给 training；
非重叠 GPU 上的 inference 继续 generation（async 核心价值）。training 完成后
"wake_up" 恢复 inference。多 pipeline 在 RLix scheduler 调度下共享同一组 GPU。

---

## 范围

### In Scope

- **推理引擎：仅 SGLang**（非 PD）
- **训练后端：仅 Megatron**
- **算法：fullasync GRPO + multi-turn custom generate path**。外层绑定
  `examples/fully_async/fully_async_rollout.py`；内层必须通过
  `--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate`
  走 multi-turn path，并显式启用 `--max-weight-staleness`。`run-qwen3-4b-fully_async.sh`
  的 math example 只作为 fully_async launcher baseline，默认不覆盖 multi-turn。
- **资源模式：partial overlap**（`actor_train ⊂ actor_infer`，重叠 GPU 时分复用）
- **DP 映射：** `infer_dp_rank = rollout_engine_index`，`infer_tp_size =
  args.rollout_num_gpus_per_engine`，`args.sglang_data_parallel_size == 1`。**当前
  milestone 只支持排序连续的 `infer_device_mapping`**（如 `[0,1,2,3]` 按
  `rollout_num_gpus_per_engine` 切 engine）；非连续 / 自定义顺序 mapping 先 fail fast，
  不在第一版做显式 `scheduler_dp_rank -> engine_index` adapter。
- **Sleep/wake 粒度：** subset (per-engine indices)
- **Routing 粒度：** subset (跳过 sleeping engines)
- **Sync 粒度：** subset (sync-on-expand + active in-flight refresh)
- **权重路径：** CPU bucket cache + selective sync（不再用 MILES 既有 standalone
  weight update path: `RolloutManager.update_weights_from_distributed/tensor` 全量 broadcast / IPC,
  作为 RLix 模式 baseline）
- **Handoff 语义：** turn-level redispatch（与 ROLL / NeMo F3 对齐）。MILES 现有
  group-level recycle 路径处理的是 tool error 等**非 RLix** abort 来源，与本 port
  正交，不改
- **参考 commit：** maocheng23 的 staleness control 以 squash commit
  `41615af98ef921e3d48dc23a11dcabbf4c1e2ea0` 为基线；若精确测试
  `multi_turn.generate` custom path，还需要包含后续
  `9a0036447 Use load_generate_function in legacy sglang_rollout path (#1016)`，
  否则 `sglang_rollout.py` 的 legacy custom-generate 调用签名与
  `GenerateFnInput` 版 `multi_turn.generate` 不匹配。

### Out of Scope

- ❌ PD disaggregation
- ❌ `sglang_data_parallel_size > 1`
- ❌ Selective P2P weight transfer（先做 broadcast / tensor subset sync）
- ❌ Multi-LoRA / DPO / SFT
- ❌ vLLM backend
- ❌ Request-level deterministic migration（ROLL 的 `RequestScheduler` 路径不复刻 —
  用 turn-level redispatch 替代，与 NeMo 同形态）
- ❌ 非连续 / 自定义顺序 `infer_device_mapping` — 当前 milestone 强制排序连续 mapping，
  让 RLix scheduler DP rank 与 MILES rollout engine index 天然一致；通用 mapping
  adapter 留作 follow-up
- ❌ RLix mode 下的 radix-tree middleware / prefix-cache routing — first build 直接
  禁用 `RadixTreeMiddleware`，不做 `partial_rollout + radix_tree` 兼容
- ❌ MoE / EP（expert parallel）— 若启用，F4 CPU bucket cache 的 collective gather 需
  补 `get_expert_tensor_parallel_group()` + `get_expert_model_parallel_group()` 路径；
  当前 scope 不启用，留作 follow-up。它不是简单 flag：需要重新审计 expert / non-expert
  参数分组、bucket layout、cache owner 唯一性、version accounting 与 receiver load
  顺序。

#### MoE / EP note（为什么不在当前 scope）

ROLL **有 MoE 支持**，但不是 ModelUpdateService / RLix scheduler 层“天然免费支持”。
ROLL 在 Megatron export、metadata、broadcast group 与 receiver loader 多处都有专门逻辑：

- `roll/third_party/megatron/model_update.py:_gather_hf_weights()` 先用
  `dist_converter.is_expert_parallel_weight(name)` 把参数分成 expert / non-expert：
  expert 参数走 `mpu.get_expert_tensor_parallel_group()`，并在
  `expert_model_parallel_size > 1` 时额外走 `mpu.get_expert_model_parallel_group()`；
  non-expert 参数才走普通 `mpu.get_tensor_model_parallel_group()`。
- 同文件的 metadata path 对 expert 参数按
  `expert_model_parallel_size * expert_tensor_parallel_size` 计算完整权重大小，和普通
  TP 参数不同。
- ROLL 的 Megatron update broadcast group name 带 `ep_rank`
  (`..._pp{pp_rank}_ep{ep_rank}`)，避免不同 EP shard 的动态 NCCL group 混在一起。
- vLLM receiver 侧还有 `patch_vllm_moe_model_weight_loader()`，给 expert weights 补
  loader 语义；这说明“能传过去”不等于“能正确 load”。

因此 MILES 当前 F4 CPU bucket cache 只覆盖 dense Megatron 参数。要支持 MoE / EP，需要
把 expert / non-expert 参数分类、expert TP/EP gather、EP-aware bucket metadata、
EP-aware dynamic NCCL group、receiver-side MoE loader 全部补齐，并增加专门 parity gate。

旧文档 `plans/adaptation_miles.md` 的具体实现路径偏 overengineered（per-request migration、
P2P selective、复杂 versioning 双槽）；本方案保留 partial overlap 的 **必要** 能力，
但实现路径向 NeMo RL port plan 对齐：单槽 `_cache_ready_step`、broadcast/tensor subset
sync 优先、turn-level retry。

### Parity targets and non-negotiable semantics

本 plan 按**阶段化 milestone (M11.1-M11.5)** 交付 ROLL/NeMo port parity, 而不是一步到位
全部实现. M11.1 first build (vast.ai single-pipeline) 保留所有 load-bearing parity
semantics; M11.2-M11.6 按 milestone 完成 production parity (见各 feature 段 milestone 标签
+ Implementation follow-up appendix). 不设置 tp=1 MVP, 不用简化 workload 作为第一阶段
替代品 — M11.1 已要求 `tp>1` mixed receiver mask:

- F1-F3 (M11.1)：必须实现 subset sleep/wake、router admission、scheduler-preempt retry。
  parity workload 绑定 `multi_turn.generate + --max-weight-staleness`，因此 turn-level
  redispatch 是主路径；single-turn generation-call retry / group recycle 只能作为兼容
  路径，不能替代 multi-turn retry。
- F4-F6 (M11.1 主路径, cuda_ipc adapter → M11.6)：必须覆盖 CPU bucket cache、active
  in-flight refresh、expand selective sync、version accounting。M11.1 端到端验证必须覆盖
  `tp>1` mixed receiver mask: colocated `cpu_serialize` (M11.1 唯一 colocate transport) +
  non-colocated dynamic NCCL broadcast (partial overlap 必需) + receiver-side dual-mask
  (`cpu_serialize_local_ranks` + `broadcast_local_ranks`) / `is_group_exist` hardening。
  cuda_ipc colocate adapter (`ipc_local_ranks` mask) → M11.6.
- F7-F8-F11 (M11.1)：implementation 可合并到一个 `MilesRLixAdapter` / `rlix_hooks.py`
  以减少分散 glue，但三个 feature spec 必须独立保留；namespace isolation、registration
  lifecycle、conditional standalone/RLix behavior 都必须保留。
- F9 (M11.1)：progress reporting 必须保留 ROLL/NeMo 的 batch-begin snapshot 语义、2%
  bucket gating、`target_weight_version` filtering、`new_batch=True` 首个快照 — 不能
  退化成只报累计 completed (nemorl correctness analysis: 错误低估 demand → scheduler
  过早 shrink). multi-stream aggregation impl → M11.4 (M11.1 hook signature 保留
  `mode/adapter_id` nullable 字段做 forward-compat).
- F10 (M11.1)：topology validation 是 parity 必需项，不是 nice-to-have。必须 fail fast
  覆盖 `train_devices ⊂ infer_devices`、`infer_engine_count >= 2`、fullasync enabled、
  `sglang_data_parallel_size == 1`、Megatron 并行度 divisibility、单 updateable
  model/server、**M11.1 强制 `model_update_transport == "cpu_serialize"`** + 跨节点
  rollout engine 拒收 (`rollout_num_gpus_per_engine <= num_gpus_per_node`, 按 engine 总
  GPU 数计, 与 sharding 形式 TP/EP 无关).
- F12 (M11.1)：shared PG 不是可选 glue。MILES 必须接受 RLix/ROLL shared PG，并显式完成
  ROLL-style device allocation 到 MILES `(pg, bundle_indices, gpu_ids)` 的 adapter.
  **M11.1 multi-node DP capability** (rollout engines distributed across nodes, each engine
  node-local; cross-node rollout engine forbidden via existing M3 assert `rollout_num_gpus_per_engine
  <= num_gpus_per_node`, 与 sharding 形式无关) — `WorkerPlacement` 必须 multi-node-compatible (node-local
  gpu_ids + node_rank + bundle_index, **不假设 global GPU id == local id**, 见 Cut 1'
  段). Dev gate 跑单机 4 卡, 但架构层面不堵死多机部署.

**M11.2-M11.6 后续 milestone** (见各 feature 段 milestone 标签):
- M11.2 unified-plan scope (Gate 4 happy path):
    • shell partial GENERATION allocation
    • RolloutManager shell → loading → active lazy ctor path
    • coordinator expand sync with base version -1 (v=-1 short-circuit, Fix #13)
      and later step versions
    • Fix #14 F10 startup fail-fast on resume args (M11.2 不支持 resume; CPU-only
      base sync 是 M11.2 follow-up)
    • Gate 4 happy-path dual MILES pipeline acceptance:
        (c) Pipeline B init under contention gets partial allocation
        (d) expand-before-first-after_training uses base version -1
            (no transport, no finalize, just set_weight_version)
        (e) donor shrink before receiver expand/add ordering is verified

  M11.2-tagged but **NOT in unified-plan implementation scope** (各自 follow-up
  deliverables, 同 milestone hardening 层, 不是 Gate 4 landing criteria;
  crash 时手动 `ray stop` 是 accepted recovery):
    • admission_epoch race defense (follow-up L3618)
    • multi-pipeline orchestrator-driven cleanup (follow-up L3618)
    • graceful actor drain replacing ray.kill (follow-up L3631)
- M11.3 (skipped — cross-node rollout engine not supported; multi-node DP capability rolled into M11.1)
- M11.4: LoRA + multi-stream aggregation impl
- M11.5: ingress 503 + 5xx synthesis (preempt sentinel) + cleanup daemon + NCCL port hardening (port cooldown queue, receiver crash tolerance / conditional port leak / periodic port GC)
- M11.6: cuda_ipc colocate adapter + smoke-test capability check (former M11.2; pushed after M11.5 production hardening)

**验证 caveat**：阶段化不等于线性串行落地。F1-F3 可以先用 unit / mock integration
覆盖 router metadata、preempted-engine 判定、snapshot/restore 与 retry exhaustion；但
M11.1 最终 e2e gate 只在 F4-F6 weight refresh、F7-F11 control plane、F12 placement 全部
到位后，按 M11.1 `tp>1` parity 配置判定通过 (Gate 1, 2, 2.5, 3 — 不含 Gate 4).

#### Prior simplification notes under final parity

以下判定覆盖早期 MVP / bring-up 讨论，防止后续实现回退到简化语义：

- **F12 Shared PG adapter 仍成立，而且是 parity 必需项**：不能假设
  `RollResourceManagerProxy.allocate_placement_group()` 与 MILES
  `RolloutManager(pg, reordered_bundle_indices, reordered_gpu_ids)` shape 天然对齐。
  必须通过 `MilesPlacementProvider` 显式把 RLix/ROLL per-worker device mapping
  materialize 成 MILES 所需三元组；禁止用隐式/伪造三元组掩盖约束。
- **F2 routing lock 仍成立，但只锁短 critical section**：lock 保护 dispatch 选择与
  `active → disabling/loading/offloaded` 状态转换；abort / drain / sleep / wake 等慢操作
  必须在 lock 外执行。`EngineInfo.state` 是 source of truth，dispatch 只能选择
  `state == "active"` 的 engine。
- **F7/F8/F11 可以合并实现，但能力不能合并掉**：允许落成一个
  `MilesRLixAdapter` / `rlix_hooks.py`，但 namespace isolation、registration lifecycle、
  `DO_TIME_SHARING`、actor naming、skip `ray.shutdown()` 都是必需能力。
- **F9 可以薄实现，但不能退化成只累计 completed**：不复制 ROLL `GroupQueueManager`
  的全部内部状态，也不维护 RLix scheduler 不消费的 `queued / inflight / consumed`；
  但必须保留 `begin_progress_batch()` batch-open snapshot、`new_batch=True/False`、2%
  bucket、`target_weight_version` 过滤。
- **F5 不能按 tp=1 / MVP 降级**：首个 parity gate 就是 `tp>1`。active refresh 必须覆盖
  mixed receiver mask：同 GPU receiver 走 Ray ObjectRef `cpu_serialize`，non-colocated
  receiver 走 dynamic NCCL broadcast。receiver-side hardening、version publish、
  timeout crash、sync barrier 不能后置。本 milestone 不引入 `verify_model` debug
  validation；传输正确性由 warmup allreduce + per-bucket barrier 保证。
- **F5 不是整体 best-effort**：只允许 active refresh 期间存在短暂 request-level version
  attribution 过渡窗口；refresh 完成后 engine 必须发布
  `_current_weight_version == _cache_ready_step`，trajectory `weight_versions` 必须能被
  `--max-weight-staleness` 可靠消费。

---

## 方案：逐 Feature 从 ROLL 移植到 MILES

以下每个 Feature 是 ROLL + RLix 所需的所有独立能力。对每个 Feature 说明：ROLL 怎么做
的 → MILES 现状 → 移植方案。

---

### Feature 1: SGLang sleep/wake with sleep_level

**作用：** 释放 inference engine 的 GPU VRAM（weights + KV cache），腾给 training
worker 使用。

#### ROLL 怎么做的

- vLLM 引擎创建时 `enable_sleep_mode=True`，sleep_level 从 config 传入（RLix 模式下
  level=2，weights + KV 都释放）
- `roll/distributed/strategy/vllm_strategy.py:582` — `offload_states(level)` →
  `self.model.offload_states(self.sleep_level)`
- `roll/distributed/strategy/vllm_strategy.py:569` — `load_states()`

#### MILES 现状

- [miles/ray/rollout.py](external/miles/miles/ray/rollout.py) — `RolloutManager.offload/onload/onload_weights/onload_kv`
  已存在，**fan-out 到所有 rollout server / engine**
- [miles/backends/sglang_utils/sglang_engine.py](external/miles/miles/backends/megatron_utils/actor.py) —
  `SGLangEngine.release_memory_occupation()` / `resume_memory_occupation()` 已实现
  SGLang 端 sleep/wake，但 sleep_level 行为绑定在 `flush_cache` + memory release
- 缺: SGLang server-side VRAM assertion (Ray actor 自己的 torch.cuda 不可信, 显存在 SGLang 子进程)
- 缺：scheduler-driven 调用入口（当前仅训练 loop 自己调）

#### 移植方案

1. 把现有 all-engine `RolloutManager.offload/onload` 暴露成 RLix coordinator 可调用
   的 RPC（关闭时不影响普通运行）
2. SGLang 端复用 `release_memory_occupation(tags=None)` / `resume_memory_occupation(tags=None)`。
   **SGLang 没有 vLLM 的 `level` integer 概念**；改用 tag-based API（更灵活）：

   | vLLM | SGLang 等价 |
   |---|---|
   | `sleep(level=1)`（仅 KV） | `release_memory_occupation(tags=["kv_cache"])` |
   | `sleep(level=2)`（KV + weights） | `release_memory_occupation(tags=None)` 默认 = `GPU_MEMORY_ALL_TYPES` = `["kv_cache", "weights", "cuda_graph"]`（[constants.py:1-11](external/sglang/python/sglang/srt/constants.py#L1)） |
   | `wake_up()` | `resume_memory_occupation(tags=None)` |

   RLix 模式默认 `tags=None`（释放全部三类，与 vLLM level=2 等价）。MILES wrapper
   在 [sglang_engine.py:431,439](external/miles/miles/backends/sglang_utils/sglang_engine.py#L431)
   已暴露 `tags=` 参数；coordinator-driven offload 默认走 `tags=None` 全释放。
   语义见 [scheduler_update_weights_mixin.py:136-167](external/sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py#L136)：
   `kv_cache` → `flush_cache()` + pause，`weights` → `_export_static_state` stash +
   pause，`cuda_graph` → pause。**调用前 SGLang 强制 `assert is_fully_idle()`** —
   与 Feature 2 abort-drain-sleep 顺序对齐：必须先 abort + drain 才能 release
3. **Post-sleep VRAM assertion（必须读 SGLang server 内部, 不能读 Ray actor）**：MILES
   `SGLangEngine` Ray actor 通过 `subprocess.Popen` 启动独立 SGLang server process
   ([sglang_engine.py:75](external/miles/miles/backends/sglang_utils/sglang_engine.py#L75))，
   显存全在子进程；Ray actor 自己的 `torch.cuda.memory_allocated()` 永远接近 0，**不能用
   作 sleep 验证**。改用 SGLang `/server_info` response 中已有的 `memory_usage` 字段
   （见 [scheduler.py:3338-3343](external/sglang/python/sglang/srt/managers/scheduler.py#L3338)
   返回 round 后的 GB 数, **单位是 GB 不是 MB**）。
   `assert_post_sleep_memory_below_threshold()` 实现:
   ```python
   resp = requests.get(f"http://{host}:{port}/server_info").json()
   total_gb = sum(s["memory_usage"][k] for s in resp["internal_states"]
                  for k in ("weight", "kvcache", "graph"))
   assert total_gb < args.miles_post_sleep_vram_threshold_gb
   ```
   阈值字段统一成 `args.miles_post_sleep_vram_threshold_gb` (默认 1.0 GB; NCCL
   communicator 残留 ~50-200 MB 由阈值兜底, 不在 `memory_usage` 三项里).
4. **Idempotency guard**：仅作为 RolloutManager 层 `EngineInfo.state` 的快路径。manager
   层 `EngineInfo.state ∈ {shell, active, disabling, offloaded, loading}` 是唯一 source of
   truth（详见 Feature 2 第 5 条）；engine actor 内部不再单独维护 `is_engine_resident_on_gpu`
   bool flag —— 重复 sleep / wake 在 manager 层根据 `state` 短路即可，避免 manager
   `state` 与 worker flag 双源漂移
5. **NCCL teardown 已无条件挂载**: MILES [actor.py:58](external/miles/miles/backends/megatron_utils/actor.py#L58)
   actor init 时无条件调 `monkey_patch_torch_dist()`, 不存在 "未 apply" 的情况.
   ROLL `coordinator.py:136 _validate_offload_nccl` 等价 ack verification 在 MILES
   是 dead check, **不引入**.

**SGLang TP NCCL communicator 状态说明：** SGLang `release_memory_occupation()` 仅释放
weights + KV cache 的 CuMem，TP NCCL communicator 保留有效（与 vLLM `LLM.sleep()` 同形态；
SGLang 内部 release 路径用 `barrier(self.tp_cpu_group)`，本身就依赖 TP group 保留 —
[scheduler_update_weights_mixin.py:159](external/sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py#L159)）。
Gate 2 是回归确认，不是真正风险点 — 强制 destroy SGLang TP NCCL 会 break SGLang 自己
的 release 路径，**不应作为 fallback 设计**。NCCL communicator buffer 残留（~50-200 MB）
由 `args.miles_post_sleep_vram_threshold_gb` 兜底，partial overlap 可接受。

改动量：~25 行 (砍掉 coordinator-side validate hook, 见 §5)

**前置依赖（不在本 feature 内）：** Megatron 侧 actor offload 时的 NCCL communicator
buffer 释放由 [miles/utils/reloadable_process_group.py](external/miles/miles/utils/reloadable_process_group.py)
（`monkey_patch_torch_dist` + `destroy_process_groups` + `reload_process_groups`）提供；
[miles/backends/megatron_utils/actor.py:58](external/miles/miles/backends/megatron_utils/actor.py#L58)
已在 actor init 时挂上 monkey patch。这是 ROLL `offload_nccl=True` 的等价物，**MILES
已具备**，本 port 不重新实现，但要在 RLix coordinator 释放 `actor_train` 路径上确保
触发 `destroy_process_groups()`，并在 reacquire 时触发 `reload_process_groups()`。

---

### Feature 2: Selective engine sleep/wake (partial sleep/wake)

**作用：** 只 sleep 重叠 GPU 上的 engine，非重叠 GPU 继续 generation。

#### ROLL 怎么做的

- `roll/pipeline/base_worker.py:527` — `InferWorker.offload_states_partial(target_dp_ranks)`
- `roll/pipeline/base_worker.py:494` — `InferWorker.load_states_partial(target_dp_ranks)`
- `roll/distributed/scheduler/generate_scheduler.py:1885` — `shrink_workers(dp_ranks)`：
  从 routing 移除 + abort in-flight + offload
- `roll/distributed/scheduler/generate_scheduler.py:1973` — `expand_workers(dp_ranks)`：
  load + 恢复 routing

#### MILES 现状

- [miles/ray/rollout.py:63-330](external/miles/miles/ray/rollout.py#L63) — `ServerGroup`
  / `RolloutServer` 抽象已存在，但 `offload/onload` 全部对全量 engines 操作
- 无 engine index map（`engine_index → SGLangEngine handle / worker_url / state`）
- `RolloutServer.engines`、`ServerGroup.all_engines` 已是 list 形式，索引访问可行，
  只是没有暴露 subset API
- 没有 `_active_dp_ranks` / `_preempted_shards` 之类的 routing state

#### 移植方案

1. **Engine indexing：单一 `EngineInfo` dataclass，不是 5 个并行 map**：
   ```python
   @dataclass
   class EngineInfo:
       handle: SGLangEngineActor | None  # None when state == "shell" (no Ray actor yet)
       worker_url: str | None            # None for shell (no SGLang server)
       server_group: ServerGroup | None  # None for shell
       bundle_index: int                 # placement bundle (always populated; declared at init)
       gpu_ids: list[int]                # declared GPU ids (always populated)
       placement: WorkerPlacement        # declared placement (always populated; expand reads this)
       state: Literal["shell", "active", "disabling", "offloaded", "loading"]
   ```
   `RolloutManager._engines: Dict[int, EngineInfo]`（单一 dict，`engine_index` 为 key,
   长度恒等 `engine_count`，shell 也占 slot）。文档可保留字段说明，但实现层不要拆 5 个
   并行 map（容易状态漂移）。

   **`shell` 语义 (M11.2 Gate 4 partial GENERATION allocation)**: slot 已声明拓扑
   (engine_idx + declared `WorkerPlacement` + declared gpu_ids), 但 **没有 Ray actor,
   没有 SGLang server process, 没有 GPU**. 唯一合法转换:
   - `shell → loading → active` — 第一次 expand 时 `_create_sglang_actor(engine_idx, placement)`
     lazy 创建 + `onload()` + 由 coordinator 调 service.sync_selected_workers + activate_routing
   - `shell` 是 shrink 的终点: 不允许 `active → shell` (shrink 只到 offloaded; offloaded
     engine 仍持 Ray actor + GPU)

   **SGLang TP fan-out 行为**：MILES 的 1 个 SGLang engine = 1 个 head HTTP server
   process（[sglang_engine.py:60-68](external/miles/miles/backends/sglang_utils/sglang_engine.py#L60)），
   占 `num_gpus_per_engine` 张 GPU。TP 内部进程间通信由 SGLang 内部 collective 处理；
   外部对 `engine_index=k` 的一次 RPC（HTTP request）只命中 head，head 自动 fan-out 到
   TP peers。**因此 engine_index 是单一寻址维度，不需要 `(engine_index, tp_rank)`
   二维**。Gate 2 仍验证 sleep/wake 在 TP=2 上正确传播。

2. **Subset lifecycle 接口：仅在 `RolloutManager` 层接受全局 `engine_indices`**：
   ```python
   RolloutManager.offload(engine_indices=None, tags=...)   # None = 全量
   RolloutManager.onload(engine_indices=None, tags=...)
   RolloutManager.onload_weights(engine_indices=None)
   RolloutManager.onload_kv(engine_indices=None)
   ```
   Manager 内部通过 `_engines[idx]` 找到 `(server, server_group, local_idx)` 后 dispatch。
   `RolloutServer.offload()` / `ServerGroup.offload()` 不暴露 `indices` 参数（避免三层
   重复路由 / TOCTOU）。
3. **`sleep_partial` 必须 admission-close → abort-drain-sleep**：
   ```python
   async def sleep_partial(self, engine_indices, tags=None):
       # tags=None → SGLang GPU_MEMORY_ALL_TYPES（vLLM level=2 等价；释放
       # weights + kv_cache + cuda_graph）。Feature 1 已锁定默认行为
       async with self._routing_lock:
           for idx in engine_indices:
               assert self._engines[idx].state == "active"
               self._engines[idx].state = "disabling"
           self._active_engine_indices -= set(engine_indices)
           self._preempted_engines |= set(engine_indices)
       # lock 外执行慢操作；dispatch 只允许选择 state == "active" 的 engine
       # 每个 engine 内部 abort all running requests，走 SGLang /abort_request
       # endpoint with abort_all=True（[http_server.py:1402](external/sglang/python/sglang/srt/entrypoints/http_server.py#L1402)）
       # 实现：新增 SGLangEngine.abort_all_requests() 方法 wrap POST /abort_request {"abort_all": true}
       await self._abort_engines(engine_indices)
       # Drain：轮询 engine 直到 idle（详见下方）
       for idx in engine_indices:
           await self._wait_engine_idle(idx)
       # Engine idle，安全 sleep（SGLang 强制 assert is_fully_idle()）
       await self.run_on_engines(engine_indices, "release_memory_occupation", tags=tags)
       # Post-sleep VRAM assertion (Feature 1)
       await self.run_on_engines(engine_indices, "assert_post_sleep_memory_below_threshold")
       async with self._routing_lock:
           for idx in engine_indices:
               self._engines[idx].state = "offloaded"
   ```

   **`_preempted_engines` 不是 `EngineInfo.state` 的冗余缓存——它是 preempt 归因窗口**：

   - `EngineInfo.state` 反映 engine 当前生命周期阶段（shell / active / disabling / offloaded /
     loading），是**瞬时**值
   - `_preempted_engines` 标记"该 engine 在本次 generation 周期内被 scheduler preempt
     过"，跨越 disabling → offloaded → loading → active 多个状态阶段都保留
   - 仅靠 `state` 在 wake/activate 边界会误判：engine 已切回 `active` 但实际处于
     "刚 wake 完，前一轮 abort 的尾部异常还在 caller 侧未消费"窗口；此时 `state == "active"`
     但异常应归类为 preempt（让 multi_turn turn-level redispatch 触发），不是普通 abort
   - **set/clear 时机**：`sleep_partial` 第 1 步 set（admission close 前置）；
     `wake_up_partial` 完成后 clear（[miles/ray/rollout.py](external/miles/miles/ray/rollout.py)
     的 partial wake 路径返回前）。clear 时机晚于 `state = "active"` 的恢复，覆盖刚 wake
     的过渡窗口
   - 维护成本是单个 `set`，换错误分类语义稳定；不与 `state` 合并

4. **Drain 机制：worker-side `is_idle()` API（必须读 `/v1/loads`，不是 `/server_info`）**：
   - **重要**：SGLang `/server_info` 的 `internal_states` 来自
     [scheduler.py:3335 `get_internal_state()`](external/sglang/python/sglang/srt/managers/scheduler.py#L3335)，
     字段为 `last_gen_throughput / memory_usage / token_capacity / graph / step_time_dict`，
     **没有 `num_running_reqs`**。running req 数量在
     [http_server.py:648 `/get_load`](external/sglang/python/sglang/srt/entrypoints/http_server.py#L648)
     (deprecated shim) 或 `/v1/loads` 的 `GetLoadsReqOutput`
     ([io_struct.py:2013-2016](external/sglang/python/sglang/srt/managers/io_struct.py#L2013))，
     fields 为 `num_running_reqs / num_waiting_reqs`, endpoint 派生 `num_total_reqs =
     running + waiting` ([v1_loads.py:170](external/sglang/python/sglang/srt/entrypoints/v1_loads.py#L170)).
   - 新增 `SGLangEngine.is_idle() -> bool` 方法（[sglang_engine.py](external/miles/miles/backends/sglang_utils/sglang_engine.py)），
     带 timeout + raise_for_status (sleep/drain 主路径必须 fail-fast, 不能静默挂):
     ```python
     def is_idle(self) -> bool:
         resp = requests.get(
             f"http://{self.server_host}:{self.server_port}/v1/loads",
             timeout=5.0,  # drain hot path, server hung 必须 fail-fast 不能阻塞 sleep
         )
         resp.raise_for_status()  # 4xx/5xx 直接 raise, 不假装 idle
         data = resp.json()
         return all(slot["num_total_reqs"] == 0 for slot in data["loads"])
     ```
   - `RolloutManager._wait_engine_idle(idx)` 轮询直至 `is_idle() == True` 或 timeout
   - **不引入 per-request tracking** — 不维护 `_inflight_requests: Dict[int, Set[str]]`，
     不修改 request ID 生成路径。所有 in-flight ID 由 SGLang 内部追踪，外部只问 idle bool（与 NeMo F2 同形态）
5. **`_routing_lock` + explicit engine state（强制不变量）**：
   - **字段位置**：`RolloutManager._routing_lock: asyncio.Lock` 实例字段（与
     `RolloutManager._engines: Dict[int, EngineInfo]` 同对象；与现有 distributed Ray
     actor lock `RolloutManager.rollout_engine_lock` ([rollout.py:374](external/miles/miles/ray/rollout.py#L374))
     **不冲突也不复用** — 后者是跨进程 distributed lock，前者是 manager 进程内单线程
     `asyncio.Lock`）。`ServerGroup` / `RolloutServer` **不持有 `_routing_lock`**；它们
     通过 `engine_indices` 接收已加锁过的请求，不重复加锁
   - 用 `asyncio.Lock` 保护两类短 critical section：dispatch 的 "读 active state +
     选择 engine + 提交到 router admission 前的状态确认"，以及 shrink/expand 的
     `active → disabling/loading/offloaded` 状态切换
   - **不**把 abort / drain / sleep / wake 这些慢操作放进 lock；慢操作在 lock 外执行，
     但 engine 已处于 `disabling` 或 `loading`，dispatch 不会再选中
   - engine state 用 `shell / active / disabling / offloaded / loading`，不要只靠
     `_active_engine_indices` set。set 可作为派生缓存；source of truth 是
     `EngineInfo.state`
   - dispatch 只允许选择 `state == "active"` 的 engine；若已选 engine 在提交前状态
     变化，必须重新选择或抛 `EnginePreemptedError`
   - **与 ROLL/NeMo lock 语义的差异**：ROLL/NeMo 文档里常把 routing lock 描述为覆盖
     整段 compound operation。MILES 这里用 `EngineInfo.state` 显式状态机替代长 lock：
     lock 不覆盖 abort/drain/sleep/wake 慢操作，但在慢操作开始前 engine 已从
     `active` 切走，因此与 ROLL/NeMo 保持同一 invariant：dispatch 不会选中正在被
     disable/offload 的 engine。
   - **反向引用：F11 resize safety 自述（Feature 11）应用此 invariant**

6. **复合操作命名对齐 ROLL：`shrink_engines` / `expand_engines` / `activate_routing`**（参 ROLL
   `generate_scheduler.py:1885,1973`）：
   - 在 `RolloutManager` 上加 `shrink_engines(engine_indices)` / `expand_engines(engine_indices)`
     高层方法
   - `shrink_engines` 封装 `admission_close + _abort_engines + drain + offload`
     (`active → disabling → offloaded`)
   - `expand_engines` 内部 dispatch on entry state:
     - `shell → loading`: lazy `_create_sglang_actor(engine_idx, placement)` +
       `onload()` (M11.2 partial GENERATION allocation 路径; 见 F4 + Gate 4 (c))
     - `offloaded → loading`: existing `wake_up()`
     **不开 routing**; 返回 engines 在 `loading` 状态
   - 新增 `activate_routing(engine_indices)`: `loading → active` + open router;
     由 coordinator 在 `service.sync_selected_workers` 完成后调用 (Fix #3 ownership
     split: manager 不调 service, service 不开 routing)
   - Coordinator 编排顺序 (`_expand_workers`):
     `manager.expand_engines(target)` → `service.sync_selected_workers(sync_id, target, version)`
     → `manager.activate_routing(target)`. 不直接拼装 abort/drain/offload 序列
     (避免漏 step 或顺序错位)

改动量：~180 行（engine index map + subset API + abort-drain-sleep + shrink/expand_engines + activate_routing + shell lazy ctor 复合）

---

### Feature 3: Generation routing skip sleeping engines

**作用：** Generation 只分发到 active engines，跳过 sleeping。

#### ROLL 怎么做的

- `generate_scheduler.py` 的 `active_dp_ranks` set 控制 routing — shrink 移除，
  expand 添加
- `RequestScheduler._select_dp_rank()` 只从 active_dp_ranks 中选择
- shrink 时 abort sleeping shard 上的 in-flight，caller 侧 retry 到其他 active shard

#### MILES 现状

- [miles/router/router.py:73-74](external/miles/miles/router/router.py#L73) — Miles
  Router 仅有 `/add_worker` 和 `/list_workers`
- 缺 `/disable_worker`、`/enable_worker`、`/remove_worker`
- [miles/backends/sglang_utils/sglang_engine.py:389](external/miles/miles/backends/sglang_utils/sglang_engine.py#L389)
  — `SGLangEngine.shutdown()` 仍尝试调 `/remove_worker`，与 router 实际 endpoint 不
  一致
- [miles/router/middleware_hub/radix_tree_middleware.py:180](external/miles/miles/router/middleware_hub/radix_tree_middleware.py#L180)
  — radix tree path 显式 reject `partial_rollout`
- `examples/fully_async/fully_async_rollout.py` 在 group 层面把 aborted 样本 reset
  后塞回 buffer；这是 fully_async 外层已有兜底路径，不是 multi-turn preempt 的
  主路径
- `fully_async_rollout.py` 通过 `generate_and_rm_group()` 进入
  `sglang_rollout.generate_and_rm()`；因此只要设置
  `--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate`，
  fully_async 外层可以跑 multi-turn trajectory。注意
  `run-qwen3-4b-fully_async.sh` 默认没有设置该参数，maocheng23 的
  `update test sh` 也只添加了注释版 `#--max-weight-staleness 2`

#### 移植方案

**1. Router admission API**

给 [miles/router/router.py](external/miles/miles/router/router.py) 增加 endpoint：

- `POST /disable_worker?url=...` — worker 仍 alive，但不再接新 request
- `POST /enable_worker?url=...` — 恢复接收
- `POST /remove_worker?url=...` — 与 `SGLangEngine.shutdown()` 当前调用对齐
- `GET /list_workers` 扩展返回 `enabled/dead/inflight` metadata

Router 选择规则：仅把新请求派发到 `enabled == true && dead == false`；disabled
worker 允许 in-flight 跑完，但不再接新请求。

**2. RLix mode 禁用 radix-tree middleware；只解禁 multi_turn partial path**

为了 first build 简单可靠，当前 milestone **不支持 `partial_rollout + radix_tree` 组合**。
RLix entry / topology validation 必须 fail fast：

```python
assert "RadixTreeMiddleware" not in (args.miles_router_middleware_paths or []), (
    "RLix mode currently disables radix-tree middleware; partial_rollout + radix_tree "
    "is a follow-up after turn-level redispatch is stable"
)
```

原因：radix middleware 当前会在 `/generate` abort response 上做内部 retry / sleep，并且
`postprocess_sample_with_radix_tree()` 会重写 sample token/logprob 状态。这会隐藏
scheduler-preempt abort，使 `multi_turn.py` 收不到用于 snapshot/restore 的明确信号。

当前 milestone 只保留一处必要解禁：

- [miles/rollout/generate_hub/multi_turn.py:29](external/miles/miles/rollout/generate_hub/multi_turn.py#L29)
  `assert not args.partial_rollout, "Partial rollout is not supported"` — 删除该 assert，
  让 turn-level redispatch / partial trajectory 主路径走通

[miles/router/middleware_hub/radix_tree_middleware.py:180](external/miles/miles/router/middleware_hub/radix_tree_middleware.py#L180)
的 assert **本 milestone 不改成 pass-through**；RLix mode 会在启动时禁止加载该 middleware。
`partial_rollout + radix_tree` 的 abort 透传、prefix-cache 污染防护与 sample 状态回滚留作
follow-up。

**3. Targeted retry：multi_turn.py 内 turn-level redispatch（主路径）+ group recycle 兜底**

**适用路径必须显式配置：**

```bash
--rollout-function-path fully_async_rollout.generate_rollout_fully_async
--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
--max-weight-staleness 2
```

其中 `--custom-generate-function-path` 只替换单条 trajectory 的 generation 函数；
fully_async 的 worker / queue / staleness collection 仍由
`fully_async_rollout.generate_rollout_fully_async` 负责。默认 math single-turn path
不会进入 `multi_turn.py`，不能作为 F3 turn-level redispatch 的覆盖测试。

主路径**与 ROLL agentic env-manager / NeMo F3 对齐**：abort 当前 turn → snapshot 回滚
preempted partial token → 同 turn 重新 dispatch（Miles Router 已把 sleeping engine 排
除）→ 完成的 turn 全保留。**不**在 reset_for_retry 后从 turn 0 重做。

**Router-side preempt classification 闭环（router metadata + multi_turn 判定 + fully_async fatal sentinel）：**

**先前版本依赖 `GenerateFnInput.preempt_state` snapshot 已废弃** — snapshot 仅反映
dispatch 时刻, 不能覆盖 sleep_partial 在 request 飞行期间发生的 race window. 改用
**router 端 admission state classification**, 闭环全部由 router metadata + multi_turn
判定承担, **不扩 `GenerateFnInput`**.

(a) 新增 `class EnginePreemptedError(Exception)` + `class RLixRouterMetadataError(Exception)`，
**位置**: [miles/rollout/base_types.py](external/miles/miles/rollout/base_types.py)
(与 `GenerateFnInput / GenerateFnOutput` 同模块; multi_turn.py 已 import 这个模块, 零
额外 import seam).

(b) **Router 注入 admission state 与 engine_index** (3 处都要改):

(b.1) **`do_proxy` 必须把 `worker_url` 传给 `build_proxy_response`** —
现有 `do_proxy` ([router.py:158-166](external/miles/miles/router/router.py#L158))
返回 `{request_body, response_body, status_code, headers}`, 不带 `worker_url`.
推荐**直接在 `do_proxy` 内 mutate content** (一次解析一次序列化). **必须只对
`/generate` 注入** —— 其它 endpoint (`/model_info` / `/v1/loads` / `/health` 等) 的
schema 不应被修改:
```python
# router.py do_proxy(request, path, ...)
response = await self.client.request(...)
content = await response.aread()
upstream_headers = dict(response.headers)

if path == "generate":  # 仅 multi_turn /generate path 注入 metadata
    # 硬化: 改 body 必须配套 header 处理, 否则下游解析崩
    #   - Content-Length: 旧值已 stale; build_proxy_response 会 strip + JSONResponse
    #     重算, 这里不需要手动改
    #   - Content-Encoding: 上游 (SGLang) 默认不 gzip, 但若启用, content 是 gzip bytes,
    #     json.loads 会 fail → injection 跳过 → multi_turn 在 RLix mode 触发
    #     RLixRouterMetadataError fail-fast (符合"router 升级遗漏"的失败模型, 不静默)
    #   - 主动 strip Content-Encoding (假设无 gzip; 若实际 gzip 应 fail fast 而不是
    #     返回未压缩 body 配 gzip header, 那会让客户端解码崩)
    upstream_headers.pop("content-encoding", None)
    upstream_headers.pop("Content-Encoding", None)

    try:
        data = json.loads(content)
        meta = data.setdefault("meta_info", {})
        meta["miles_engine_index"] = self.worker_engine_index_map.get(worker_url, -1)
        # response 时刻读 enabled_workers (与 request 生命周期不绑定); 见下方 race window
        # 边界说明. turn retry 兜底 false positive; false negative 实际拓扑不发生
        meta["miles_admission_disabled"] = (worker_url not in self.enabled_workers)
        content = json.dumps(data).encode()
    except (json.JSONDecodeError, KeyError):
        pass  # non-JSON body (例如 gzip 压缩) 不注入; multi_turn RLix mode 缺 metadata
              # 会 raise RLixRouterMetadataError → driver process exit → ray stop 收尾.
              # RLix mode 禁用 radix middleware; 流式由 multi_turn 强制 payload["stream"]
              # = False + F10 validation 阻挡

return {"request_body": body, "response_body": content,
        "status_code": response.status_code, "headers": upstream_headers}
# build_proxy_response 进一步 strip Content-Length / Transfer-Encoding (既有逻辑
# [router.py:174](external/miles/miles/router/router.py#L174)); JSONResponse 重算
# Content-Length. 因此完整 header hardening 是: do_proxy strip Content-Encoding +
# build_proxy_response strip Content-Length / Transfer-Encoding.
```

**`miles_admission_disabled` 语义边界 (Critical Invariant — 简化代价显式 surface)**:

response 时刻读 `enabled_workers`, 与 request 生命周期不绑定. 两条 race:

| Race | 触发 | 影响 | 处理 |
|---|---|---|---|
| False positive (request 时 enabled, response 前被 disable) | scheduler shrink mid-flight | 浪费 1 次 turn retry | acceptable |
| False negative (request 时 disabled, response 前被 enable) | scheduler shrink + expand 在 in-flight RTT 内 | 当前 turn 走 group recycle, 丢已完成 turn | **理论 race, 实际拓扑不发生** (shrink/expand 周期秒级, RTT ms 级, 错开 3 个数量级) |

production 多 pipeline 高频 shrink/expand 触发 false negative 时, 引入 admission_epoch
race 防御 (留 follow-up).

(b.2) **`enabled_workers: set[str]` 必须真正影响 `_use_url` 选择, 不只是 metadata**.
现有 `_use_url` ([router.py:217-225](external/miles/miles/router/router.py#L217))
只过滤 `dead_workers`. 必须改成只从 `enabled_workers - dead_workers` 选, 否则
`/disable_worker` 仅影响 admission metadata, **新请求仍会被路由到 disabled engine**:
```python
def _use_url(self):
    valid = set(self.worker_request_counts) & self.enabled_workers - self.dead_workers
    if not valid:
        raise RuntimeError("no enabled live workers")
    return min(valid, key=self.worker_request_counts.get)
```

(b.3) `worker_engine_index_map: dict[str, int]` 由 `/add_worker?engine_index=...`
维护; remove_worker 时清掉. 加 `engine_index` 是 query param 必填 (RLix mode);
standalone 模式可选, 缺则 `-1`.

(b.4) **`/add_worker` / `/remove_worker` 完整 lifecycle 维护 4 个 dict/set**.
伪代码用提取后的 `url` / `engine_index` 表达, **真实签名仍是
`async def add_worker(self, request: Request)`** ([router.py:182](external/miles/miles/router/router.py#L182))
从 `request.query_params` / JSON body 提取参数后调内部 helper:
```python
# router.py — 概念逻辑, 实际 method 仍接 Request
def _add_worker_internal(self, url: str, engine_index: int = -1):
    # setdefault 避免 re-add 把 inflight counter 清零 (worker wake/re-register 时
    # 仍可能有 in-flight proxy 请求, 直接清零会污染 load accounting)
    self.worker_request_counts.setdefault(url, 0)
    self.worker_failure_counts.setdefault(url, 0)
    self.enabled_workers.add(url)        # admission default ON
    self.worker_engine_index_map[url] = engine_index
    self.dead_workers.discard(url)       # wake/re-register 同 URL 不被旧 dead 状态污染

def _remove_worker_internal(self, url: str):
    self.worker_request_counts.pop(url, None)
    self.worker_failure_counts.pop(url, None)
    self.enabled_workers.discard(url)
    self.dead_workers.discard(url)
    self.worker_engine_index_map.pop(url, None)

def _disable_worker_internal(self, url: str):
    self.enabled_workers.discard(url)    # 仅 admission close, 不动 mapping/health
    self.worker_failure_counts[url] = 0  # reset; 防止 sleep 期间 health probe 累计 dead

def _enable_worker_internal(self, url: str):
    if url in self.worker_request_counts:  # 仍在 router 注册表内
        self.enabled_workers.add(url)

async def add_worker(self, request: Request):
    url = request.query_params.get("url") or ...  # 既有解析逻辑保留
    engine_index = int(request.query_params.get("engine_index", "-1"))
    self._add_worker_internal(url, engine_index)
    return JSONResponse(...)
# disable_worker / enable_worker / remove_worker 同模式
```
`enable_worker` 不会复活 dead worker (health check 仍会重新探测); add 与 remove
是注册生命周期, disable 与 enable 是 admission state.

(b.5) **Health monitor 必须只 probe `enabled_workers - dead_workers`**, 不能 probe
disabled (sleeping) workers — 否则 sleep 期间 `/health` 失败 → worker 被错标 dead →
`_enable_worker_internal` 又显式 "不复活 dead worker" → expand 后永远进不了 routing pool.
修复 [router.py:101](external/miles/miles/router/router.py#L101) `_health_check_loop`:
```python
urls = [u for u in self.worker_request_counts
        if u in self.enabled_workers and u not in self.dead_workers]
```

(c) **`_is_scheduler_preempt(output)` 判定** (RLix mode 缺 metadata 必须 fail fast,
不能静默降级成普通 abort). **RLix mode 缺 metadata raise 专用
`RLixRouterMetadataError`**, 不要 raise generic `RuntimeError` —— fully_async
callback 仅 catch `(EnginePreemptedError, RLixRouterMetadataError)` 走 fatal
sentinel, generic Exception 仍走既有 group recycle 路径 (避免扩大 fatal 面到
tool error / OOM):
```python
def _is_scheduler_preempt(output, *, rlix_mode: bool) -> bool:
    finish = output["meta_info"]["finish_reason"]["type"]
    if finish != "abort":
        return False
    admission_disabled = output["meta_info"].get("miles_admission_disabled")
    if admission_disabled is None:
        if rlix_mode:
            # router metadata 注入失效 = 闭环断了, 不能假装 abort 是 non-preempt;
            # 必须暴露 (router 升级遗漏 / proxy bypass router 等真实 bug).
            # 用 RLixRouterMetadataError (不是 RuntimeError) 让 fatal sentinel
            # 能精准识别并 crash pipeline, 同时不扩大 fatal 面.
            raise RLixRouterMetadataError(
                "RLix mode: router metadata missing miles_admission_disabled; "
                "router upgrade incomplete or response bypassed router"
            )
        return False  # standalone 兼容旧 router (不会 trigger turn redispatch)
    return bool(admission_disabled)
```
`rlix_mode = DO_TIME_SHARING`. multi_turn.py 在 entry 拿一次 flag 闭包传入.

(d) **强制 `stream=False`**: multi_turn.py 在 `compute_request_payload(...)` 返回后
**强制覆盖** `payload["stream"] = False`. F10 topology validation 同步加 assert (见 §F10).
理由: streaming response (SSE) 无 `meta_info`, router 无法注入 admission_disabled,
classification 闭环断.

(e) **抛出位置**: [miles/rollout/generate_hub/multi_turn.py](external/miles/miles/rollout/generate_hub/multi_turn.py)
中 `output = await post(url, payload)` 返回后立刻检查 (在 `update_sample_from_response`
调用之前), 若 `_is_scheduler_preempt(output, rlix_mode=DO_TIME_SHARING)` 返回 True 则
raise `EnginePreemptedError`; 函数本身在 RLix mode 缺 metadata 时 raise
`RLixRouterMetadataError`.

(f) **捕获位置**: multi_turn.py snapshot/restore retry loop 内 (包住 `await post()` 那
段), catch `EnginePreemptedError` 后调 `_restore_turn_state(sample, snapshot)` 然后
`continue` 重 dispatch. `RLixRouterMetadataError` **不 catch, 上抛 fully_async fatal
sentinel** (router 闭环断不能 retry).

(g) `_preempted_engines` set/clear 时机不再用作 multi_turn 判定 — 仅供
`RolloutManager._abort_engines` 等内部 manager 路径使用. 实际 attribution 全由 router
admission state (response 时刻 `worker_url not in self.enabled_workers`) 承担.

**Custom generate 签名前提：** `multi_turn.generate` 是
`generate(input: GenerateFnInput) -> GenerateFnOutput` 新接口。若基于
`41615af98` squash commit 单独测试，legacy `sglang_rollout.py` 仍按
`func(args, sample, sampling_params)` 调 custom generate，会不匹配。测试该 scope
必须基于包含 `9a0036447` 的 main，或 cherry-pick 等价的 `load_generate_function`
compatibility shim。

**多 turn 跨 weight_version 显式声明：** 多 turn 轨迹允许跨 weight_version。约束只需要
做到"单个 turn 的 generate 调用是 pure 的"；resize 在 turn 边界之间发生时，后续 turn
落到新版本权重是允许的（与 NeMo turn-level retry 同形态）。

**改造点：[miles/rollout/generate_hub/multi_turn.py:46-66](external/miles/miles/rollout/generate_hub/multi_turn.py#L46-L66)**

当前结构：

```python
for _turn in range(max_turns):
    payload, halt_status = compute_request_payload(args, sample.tokens, ...)
    if payload is None:
        sample.status = halt_status; break
    output = await post(url, payload)                         # ← preempt 发生处
    await update_sample_from_response(args, sample, ...)      # ← 即便 preempt 也已 mutate sample
    if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
        break                                                  # ← 直接上抛 group recycle
    # ... tool_calls / env step（commit point）...
```

两个问题：

1. `update_sample_from_response()` 在 abort 检查之前调用 — preempted partial token 已被
   append 到 `sample.tokens` / `weight_versions` / `loss_mask` / `prefix_cache_info`
2. `break` 把 abort 一路抛到 fully_async 的 group recycle 路径

改造为 snapshot-then-retry：

```python
# multi_turn.generate 内, 不依赖 GenerateFnInput 新增字段 (不扩 GenerateFnInput).
# total_engine_count = dispatch 时刻总 engine 数 (含 sleeping); 直接 args 派生.
args = input.args  # 既有 GenerateFnInput.args property
MAX_TURN_REDISPATCH_ATTEMPTS = args.rollout_num_gpus // args.rollout_num_gpus_per_engine
RLIX_MODE = os.environ.get("RLIX_CONTROL_PLANE") == "rlix"  # = DO_TIME_SHARING

for _turn in range(max_turns):
    payload, halt_status = compute_request_payload(args, sample.tokens, ...)
    if payload is None:
        sample.status = halt_status; break
    payload["stream"] = False  # F3 (d) 强制非流式, 确保 router metadata 可注入

    # ─── TURN-LEVEL REDISPATCH (主路径) ───
    snapshot = _snapshot_turn_state(sample)   # tokens / weight_versions / loss_mask / spec_info / prefix_cache_info 长度
    output = None
    for attempt in range(MAX_TURN_REDISPATCH_ATTEMPTS):
        output = await post(url, payload)
        # _is_scheduler_preempt 在 RLix mode 缺 metadata 会 raise RLixRouterMetadataError —
        # 不 catch, 直接上抛 fully_async fatal sentinel (router 闭环断, 不能 retry)
        if _is_scheduler_preempt(output, rlix_mode=RLIX_MODE):
            _restore_turn_state(sample, snapshot)             # 丢弃 preempted partial 增量
            if attempt == MAX_TURN_REDISPATCH_ATTEMPTS - 1:
                # 用尽 = 拓扑/admission/scheduler bug, fail fast 暴露问题
                raise EnginePreemptedError(
                    f"turn redispatch exhausted after {MAX_TURN_REDISPATCH_ATTEMPTS} attempts"
                )
            continue                                          # router admission 已排除 preempted engines, 自动选新 engine
        break                                                  # 成功 / length / 非 preempt abort

    await update_sample_from_response(args, sample, payload=payload, output=output, ...)

    if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
        break

    # ... tool_calls, env step (commit point — turn generation 成功后才执行) ...
```

**关键不变量：**

- **Commit point 在 generation 成功之后**：tool_call / env step 只在 turn generation
  非 abort 时执行（[multi_turn.py:70-75](external/miles/miles/rollout/generate_hub/multi_turn.py#L70-L75)
  当前已是这个顺序）。Aborted turn retry 不会重复 commit env side effect。这是
  ROLL agentic env-manager 与 NeMo F3 retry safety invariant 的同形态。
- **完成 turn 全保留**：snapshot 只覆盖当前 turn 增量；turn N-1 及之前的 `tokens` /
  tool_responses 不动。
- **结构性不会失败**：`MAX_TURN_REDISPATCH_ATTEMPTS = total_engine_count`（dispatch 时刻
  的总 engine 数, 含 sleeping; 不是 active 数）。**先前版本错用 `active_engine_count`,
  在 active=1 时 retry 上限=1 等于没 retry**。配合 Feature 10 拓扑验证（非重叠 engine
  至少 1 个常驻 active）+ Feature 3 router admission（新 dispatch 跳过 sleeping engines），
  turn retry 在拓扑合法时不会用尽。**不加 safety margin** — 如果 retry 用尽, 是
  routing/admission/scheduler 真 bug, 而不是边界 race。
- **真用尽就 fail fast**：如果 turn retry 仍用尽，那是拓扑违规或 scheduler bug，**直接
  raise `EnginePreemptedError`**，不静默落到 group recycle — 让上层暴露问题，而不是静默
  吞掉一条 trajectory。fully_async outer 可以选择把 `EnginePreemptedError` 上报为
  pipeline crash 而不是 reset_for_retry。

**辅助函数（新增到 [miles/rollout/generate_utils/generate_endpoint_utils.py](external/miles/miles/rollout/generate_utils/generate_endpoint_utils.py)）：**

`update_sample_from_response()` 当前对多个字段做累加 / append（`tokens`,
`weight_versions`, `loss_mask`, `prefix_cache_info`, `spec_info`）。需要拆出可逆的一对：

- `_snapshot_turn_state(sample) → TurnSnapshot` — 记录上述 5 个字段的当前长度 / 状态
- `_restore_turn_state(sample, snapshot)` — truncate / pop 回 snapshot 长度

Snapshot 不深拷整条 sample，只记 trailing-edge 长度，回滚是 `O(增量长度)`。

**`single_turn.py` 路径：preempt 时 status=ABORTED → group recycle 处理**

[miles/rollout/generate_hub/single_turn.py:43-44](external/miles/miles/rollout/generate_hub/single_turn.py#L43-L44)
是单次 `await post(url, payload) → update_sample_from_response → return`，**没有内部
循环**。preempt 时 finish_reason=abort，处理路径选 (a)：

- single_turn 内不引入 turn-level retry（没有 turn 概念）
- preempt 后 `sample.status = Sample.Status.ABORTED`，由 fully_async group-level
  recycle 路径 reset_for_retry 后塞回 buffer 重新 dispatch
- **不丢失 multi-turn 已完成 turn 的工作**，因为 single_turn 本身没有 turn 序列；
  整 sample 重做的代价就是该 sample 一次完整 generation
- 这与 multi_turn 的 turn-level retry 机制不矛盾 — 两者按 generation pattern 不同分别
  处理 preempt

**Fatal sentinel propagation (fully_async outer 必须能让 EnginePreemptedError /
RLixRouterMetadataError crash pipeline, 不静默吞掉)**:

queue sentinel 单路径 (queue FIFO 已经够; 不引入 `worker._fatal_error` flag).
**First build 决策: 仅这两类异常 fatal**, 其它 Exception 沿 MILES 既有 group recycle
路径:
```python
# examples/fully_async/fully_async_rollout.py
from miles.rollout.base_types import EnginePreemptedError, RLixRouterMetadataError

class _FatalError:  # sentinel, 不继承 Exception (避免被 outer try/except 误吞)
    def __init__(self, exc): self.exc = exc

class AsyncRolloutWorker:
    # 不新增 _fatal_error 字段 — queue sentinel 单路径已足够

    def make_callback(self, gid):
        def task_done_callback(done_task):
            try:
                result = done_task.result()
                self.output_queue.put((gid, result))
            except (EnginePreemptedError, RLixRouterMetadataError) as exc:
                # 仅 RLix mode 闭环失败两种异常 fatal → pipeline crash:
                #   - EnginePreemptedError: turn redispatch 用尽 = 拓扑 / admission /
                #     scheduler bug, 不能静默 retry
                #   - RLixRouterMetadataError: router metadata 注入失效 = preempt
                #     classification 失效, 不能静默降级成普通 abort
                # 其它 Exception 让 done_task.result() 抛, 走 MILES 既有 callback
                # 异常路径 (current behavior) — 对应 tool error / framework error,
                # 不一刀切 fatal
                self.output_queue.put((gid, _FatalError(exc)))
        return task_done_callback

# generate_rollout_async free function 主循环 — dequeue 时检测 sentinel
async def generate_rollout_async(args, rollout_id, data_buffer):
    worker = get_global_worker(args, data_buffer)  # 既有
    ...
    while len(data) < target_data_size:
        completed = worker.get_completed_groups()
        for gid, result in completed:
            if isinstance(result, _FatalError):
                raise result.exc  # → pipeline crash
            # ... 既有 logic ...
```
`_FatalError` 是新增 sentinel class. 文件改动表加这条.

**4. Group recycle 与本 port 正交，不作为 preempt 主路径**

[examples/fully_async/fully_async_rollout.py:247-261](external/miles/examples/fully_async/fully_async_rollout.py#L247-L261)
现有的 `if any_aborted: reset_for_retry → 回 buffer` 路径处理的是 **MILES 既有的非 RLix
abort 来源**（[agentic_tool_call.py:96,121](external/miles/miles/rollout/generate_hub/agentic_tool_call.py#L96)
tool error 等），与本 port 完全正交：

| 路径 | 触发来源 | 设计归属 |
|---|---|---|
| **Turn-level redispatch**（新增） | RLix scheduler preempt（`abort_partial`） | 本 port，Feature 3 |
| **Group-level recycle**（已有） | tool error 等 MILES 既有 ABORTED 路径；以及 single-turn preempt 的兜底重做 | MILES 原行为，本 port 不改 |

它们不是 primary / fallback 关系 — 处理不同根因，互不重叠。本 port **不**让 turn retry
失败时落到 group recycle，因为：

1. 拓扑验证 + router admission 已结构性保证 turn retry 不会用尽（见上方第 3 条不变量）
2. 真用尽时是拓扑违规或 scheduler bug，静默 group recycle 只会让 bug 静默重试 → 隐藏
   问题；fail fast 才能暴露
3. 非 preempt 的 SGLang abort（engine crash / OOM / 内部错误）也不该 retry，应该按
   framework error 上抛

**5. Routing lock 必要性**

引入 `_routing_lock`（与 Feature 2 共享）保护 dispatch 决策原子性。否则会出现
"dispatch 选了 engine X → shrink 并发把 X 标 disabled → request 仍发到 disabling
engine"的 TOCTOU。详见 Feature 2 第 5 条 compound operation 不变量。

**Shrink-to-zero 由 F10 拓扑保证：** MILES 不需要 NeMo 的 `_wait_for_active_dp_shards()`
collector 阻塞，因为 Feature 10 验证 `len(infer_devices - train_devices) >=
rollout_num_gpus_per_engine` 已保证 ≥1 非重叠 engine 常驻 active。所有 shrink 操作只
针对 overlap subset，永远不会出现 "全部 engines sleeping → collector 无处 dispatch" 的
状态。

**6. Retry safety 适用范围说明**

turn-level retry 的安全性依赖 commit point 在 generation 成功之后。当前 scope 内的
`multi_turn.py` tool-call path 满足（tool_call → env step 在 generation
非 abort 之后）。**对齐 ROLL agentic env-manager**：`GenerateStopReason.ABORT` 仅增加
attempt counter，`env.step()` 只在 `FINISH` / `MAX_LENGTH` 后执行 — 同一 commit-after-
success 模式。如未来引入 stateful tool / NeMo-Gym 风格的 mid-turn env effect，需要显式
idempotency key 才能启用 turn retry。这与 NeMo plan F3 的 retry safety invariant 是同
一约束。

改动量：~180 行（router 4-state lifecycle + metadata 注入 ~50, routing lock + abort
触发口 ~20, turn snapshot/restore + redispatch loop + raise EnginePreemptedError
~80, _is_scheduler_preempt + RLixRouterMetadataError + fully_async fatal sentinel
queue 单路径 ~30）

---

### Feature 4: Training-side weight caching (CPU bucket cache + `_cache_ready_step`)

**作用：** Training 完成后，把权重缓存到 CPU，供 expand 时 selective sync 到刚 woken
inference engine。

#### 原理简述（两阶段心智模型）

入门高层流程，**正确性由后续 9 条 invariant 强制约束，不是 invariant 的替代品**：

- **Phase 1（build cache）**：所有 PP / TP / CP rank 参与 collective gather；仅 cache
  owner（pp0+dp0+tp0+cp0）落到 pinned CPU `List[BucketRecord]`，其他 rank drain
  generator 推 collective 但丢弃结果。
- **Phase 2（send dual-path）**：cache owner 单线发送：
  - **Colocate engines**（同 GPU receiver, M11.1 = cpu_serialize 唯一）→ Ray ObjectRef
    bytes → tmpfs file path → SGLang HTTP route. M11.6 加 cuda_ipc 作为 production
    colocate transport, **必须用新 adapter** (CPU cache → per-bucket H2D staging → IPC
    handle serialize), **不复用既有 `_send_to_colocated_engine`** (后者依赖 live
    `dist.gather_object` 与 F4 destroy NCCL 顺序冲突). wire format 详见本 feature 段
    下方 "Bucket payload 格式" Case B (M11.1) / Case A (M11.6) 两节
  - **Non-colocate engines**（跨 GPU receiver, partial overlap 必需路径）→ 逐 bucket
    H2D staging + 临时 NCCL group broadcast + bucket-level barrier + group destroy
- **两阶段间由 invariant 3（顺序契约）保证不并发**；invariant 4（`_cache_lock`）作为
  异常路径 safety net。
- **内存账**：cluster 总 pinned host = `pipeline_count × model_size`；GPU 峰值（sender
  端 staging）= `bucket_size_bytes`，与模型大小无关。

#### ROLL 怎么做的

- `roll/distributed/executor/worker.py:363` — `build_latest_bucket_cache(checkpoint_version)`
  把模型参数序列化为 CPU bucket cache（raw bytes + metadata），存在 training worker
- `roll/distributed/executor/worker.py:387` — `promote_active_checkpoint(checkpoint_version)`
- `rlix/pipeline/full_finetune_pipeline.py` 中：
  - Init: `build_latest_bucket_cache(-1)` → `promote_active_checkpoint(-1)` 缓存 base model
  - 每次 train_step 后：`promote_active_checkpoint(checkpoint_version)` 标记最新权重
- `roll/distributed/strategy/megatron_strategy.py:1994` — `promote_active_checkpoint`
  切换 active pointer

#### MILES 现状

- [miles/backends/megatron_utils/update_weight/update_weight_from_distributed/](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/) —
  仅有 `update_weights_from_distributed`（broadcast）+ `update_weights_from_tensor`
  （local CUDA tensor）+ P2P 三条路径，全部假设权重在 GPU 上
- **不存在** CPU bucket cache 概念 — MILES 既有 weight update path
  (`RolloutManager.update_weights_from_distributed` broadcast 或 `_send_to_colocated_engine`
  cuda_ipc) 直接从 GPU 张量发出
- 流程固定：pause generation → flush cache → transfer weights → continue generation

#### 移植方案

**问题：refit 时训练权重在哪里？**

partial overlap 中，训练 GPU = 重叠 GPU = inference 需要 wake_up 的 GPU。要求权重在
GPU 上的 refit 路径（MILES 现有 distributed/tensor 路径）会与 inference wake 抢同一份
VRAM → OOM。**这是必须引入 CPU bucket cache 的核心原因，与 NeMo RL F4 同因。**

**方案：参照 ROLL `ModelUpdateService` 的简化版（与 NeMo RL port plan F4 同形态）**

**Step 0（Init bootstrap base model cache）：**

⚠️ **partial overlap 死锁约束**：`actor_train ⊂ actor_infer`，scheduler 不可能在 actor_train
仍持有 overlap GPU 的同时再分配 actor_infer (overlap GPU 已 busy)。先前版本把 actor_infer
request 嵌在 actor_train try 块内会**直接死锁**或**长期占住 train GPU 等不到 infer**。
必须按 ROLL [full_finetune_pipeline.py:280-310](rlix/pipeline/full_finetune_pipeline.py#L280)
的实际顺序: train init → build base CPU cache → **显式 offload_states + destroy NCCL** →
release train → 然后 request/init actor_infer.

⚠️ **时间窗硬约束**：cache build 必须落在 `actor_train.initialize()` 返回**之后**（model 已
load 到 GPU）且 `actor_train.offload_states(blocking=True)` **之前**（GPU 仍占用，gather 才能
cross-rank 推进）。

**精确落点（与 [full_finetune_pipeline.py:114](rlix/pipeline/full_finetune_pipeline.py#L114)
`initialize_pipeline()` 同形态，inline 单 method 不抽 hook）**：

`MilesPipeline.initialize_pipeline(self) -> ActionResponse:` 单一 method，用
`self._init_lock: threading.Lock` 守卫. **保持 sync def + threading.Lock**, 不改成
`async def + asyncio.Lock` —— RLix `_request_cluster_gpus` /
`_notify_release_cluster_gpus` 是 sync wrapper, 内部 `ray.get(scheduler.request_gpus.remote(...))`
([full_finetune_pipeline.py:471](rlix/pipeline/full_finetune_pipeline.py#L471)). 半 async
半 sync 会阻塞 event loop. `RayTrainGroup.{init,onload,offload}` 是 `async def` 但用
`miles.utils.async_utils.run` 同步等待 ([async_utils.py:43](external/miles/miles/utils/async_utils.py#L43)).

```python
from miles.utils.async_utils import run  # sync wrapper for async coroutines

def initialize_pipeline(self) -> ActionResponse:
    with self._init_lock:
        if self._initialized:
            return ActionResponse(success=True)

        init_global_step = -1
        # Step 1: actor_train allocation (sync helper, 内部 ray.get)
        #   必须在创建 RayTrainGroup 之前申请 GPU; 否则 Ray actor 会在 placement
        #   group ready 之前提前占 fractional GPU resource → 双 pipeline 共节点
        #   时另一 pipeline 抢不到
        self._request_cluster_gpus(
            cluster_id=self._actor_train_cluster_id,
            priority=Priority.INITIALIZATION,
            global_step=init_global_step,
        )

        train_init_succeeded = False  # M4: gate hard cleanup on failure only
        try:
            # Step 1b: 创建 RayTrainGroup actor (持引用, 后续 train_step 反复用)
            #   `RayTrainGroup.__init__` 现有签名 (actor_group.py:25-35):
            #     (args, num_nodes, num_gpus_per_node, pg, *,
            #      num_gpus_per_actor=1, role, with_ref)
            #   F12 真实 adapter 扩展为接受 `worker_placements` (替代 pg=三元组),
            #   pg 改为 Optional. 完整新签名:
            #     (args, num_nodes, num_gpus_per_node, *, pg=None,
            #      worker_placements=None, num_gpus_per_actor=0.4,
            #      role, with_ref)
            #   必须 `pg is not None` xor `worker_placements is not None` (二选一)
            train_workers = self._placement_provider.get_train_workers()
            self.actor_train = RayTrainGroup(
                args=args,
                num_nodes=args.actor_num_nodes,
                num_gpus_per_node=args.actor_num_gpus_per_node,
                pg=None,
                worker_placements=train_workers,    # F12 真实 adapter 路径
                # RLix retained-offload path 必须 tiny Ray GPU reservation (Critical Invariant):
                # offload 后 scheduler 视为 GPU 空闲, 但 Ray actor 仍 reserve 这份
                # fractional 会与下个 pipeline 冲突. 实际隔离靠 placement_group +
                # CUDA_VISIBLE_DEVICES, 见 §F12 (b) 路径.
                num_gpus_per_actor=0.01,
                role="actor",
                with_ref=(args.kl_coef != 0 or args.use_kl_loss),
            )

            # Step 2: RayTrainGroup.init() — async def, run() 同步等待
            #   actor.py:181 `if args.offload_train: self.sleep()` — init 末尾自动 sleep
            run(self.actor_train.init())

            # Step 3: 显式 wake (build cache 需要 GPU 上的权重)
            #   等价 ROLL `actor_train.load_states(blocking=True)`
            #   ([full_finetune_pipeline.py:286](rlix/pipeline/full_finetune_pipeline.py#L286))
            #   MILES 对应: RayTrainGroup.onload() → 每 actor wake_up() (actor.py:213).
            #   F10 已 assert args.offload_train, init 末尾必然 sleep, 此 if 实际恒成立
            if args.offload_train:
                run(self.actor_train.onload())

            # Step 4: build base CPU bucket cache (GPU 上有权重时)
            # M2: pipeline driver 进程没有 Megatron model state, 不能本地 build cache.
            # 必须 fan-out 到 train actors 进程 (RayTrainGroup wraps fan-out 到所有 worker).
            run(self.actor_train.build_cpu_bucket_cache(step=-1))
            self._cache_ready_step = -1

            # Step 5: offload (释放 overlap GPU + 内部 destroy NCCL)
            #   RayTrainGroup.offload() → 每 actor sleep() (actor.py:197), sleep()
            #   内部经 ReloadableProcessGroup destroy_process_groups (F1 已说明).
            #   注: actor.sleep() 内 `assert self.args.offload_train` (actor.py:198),
            #   所以 F10 必须 fail-fast `assert args.offload_train` (见 F10).
            run(self.actor_train.offload())
            train_init_succeeded = True  # M4: 成功 → 保留 self.actor_train 供 main loop

        finally:
            # Step 6: release scheduler allocation. RLix protocol 强制要求 — 不释放
            # 别的 pipeline 拿不到 GPU. INITIALIZATION priority allocation 总要释放;
            # 后续 train_step 用 ACTOR_TRAINING priority 重新申请.
            # M4 — Hard cleanup on failure only: 失败路径必须先 kill train actors 再
            # release scheduler. 只 release 不 cleanup → scheduler 视 GPU 空闲, Ray
            # actor 仍占 CUDA context → 下个 pipeline 调度到同 GPU OOM/卡死.
            # `RayTrainGroup.__init__` 内部 self-cleanup (见 §J) 已处理 ctor 半构造
            # 失败 (init.remote() 之前); 这里覆盖 init.remote() 之后失败 (Step 2-5
            # 任一) 的场景, 显式 kill 一次 (idempotent). 成功路径保留 actor handles.
            if not train_init_succeeded and getattr(self, "actor_train", None) is not None:
                for h in self.actor_train._actor_handles:
                    try: ray.kill(h, no_restart=True)
                    except Exception: pass
                self.actor_train = None
            # release 用 try/log/swallow 包裹, 避免 finally 内 raise 替换 try block 原异常
            # (Python finally 语义: 若 release 失败原 init error 会被掩盖, 严重退化诊断).
            try:
                self._notify_release_cluster_gpus(
                    cluster_id=self._actor_train_cluster_id,
                    global_step=init_global_step,
                )
            except Exception:
                logger.exception("scheduler release failed; original init error takes precedence")
                # swallow: finally 内不 raise, 让 try block 原异常透传 (如有)

        # ====================================================================
        # M4 — Phase 2 cleanup scope (Step 6.5 → 6.6 → 7 → 7.1-7.3).
        # ====================================================================
        # 这个外层 try/except 必须包住 Step 6.5 (cache_owner collection), Step 6.6
        # (manager pre-spawn + resource registration + base version publish +
        # empty-set bootstrap), Step 7 (`_request_cluster_gpus` for infer; scheduler
        # resize_infer(add) 在此期间 fire 到 coordinator), Step 7.1-7.3 (M1 subset
        # assert + active subset compute + sanity check). 若不包住:
        #   - Step 6.5 RPC raise → actor_train 僵尸泄漏 (Phase 1 finally 已设
        #     `train_init_succeeded=True`, 没 kill train)
        #   - Step 6.6b RolloutManager ctor raise → actor_infer 半构造泄漏
        #   - Step 6.6d/e/f register/publish/bootstrap raise → coordinator state 不一致
        #   - Step 7 `_request_cluster_gpus` raise → scheduler 视 train GPU 已释放
        #     但 train actor 还在; 同时 coordinator._expand_workers 可能在请求中途
        #     raise (now harmless because manager+service are registered)
        #   - M1 subset assert fire (allocated 不是 declared 的 subset) → 同上 +
        #     infer scheduler 已记账但 actor_infer 未活跃 → infer scheduler
        #     allocation 永远不释放. (注意: M11.2 partial allocation 是 valid 路径,
        #     不会触发 subset assert; assert fire 只在 scheduler 返回未声明 GPU 时)
        #   - half-engine allocation 在 provider.get_active_engine_indices raise → 同上
        #   - Step 7.3 consistency assert fire → 极少见, 只在 coordinator/pipeline
        #     active set 派生不一致时触发, 同上清理
        actor_infer_allocated = False
        try:
            # ----- Step 6.5 (M2): collect cache owner Ray actor handle ------
            # `MilesModelUpdateService` 是独立 RLix actor, sender 路径 (cpu_serialize
            # 与 NCCL broadcast) 必须直接 RPC 到 cache owner Megatron worker actor.
            # RayTrainGroup 内部把 worker 上报的 (global_rank, is_owner) 与自己持有的
            # actor handle 配对.
            roles = run(self.actor_train.collect_cache_owner_roles())
            # roles: list[(global_rank, is_owner, actor_handle)]
            cache_owner_actor = next(h for r, owner, h in roles if owner)
            assert cache_owner_actor is not None, (
                "no cache owner reported; expect exactly one (pp0+dp0+tp0+cp0) rank"
            )
            self._cache_owner_actor = cache_owner_actor

            # ----- Step 6.6: pre-register everything _expand_workers will need ------
            # CRITICAL: RLix scheduler executes `coordinator.resize_infer(add=…)` RPC
            # in Phase 5 BEFORE signaling the pending `_request_cluster_gpus` waiter
            # in Phase 6 (rlix/scheduler/scheduler.py:1178-1187, planner.py:418-425).
            # Even the very first single-pipeline GEN allocation triggers a
            # `SchedGuidedAllocationOp` → resize_infer RPC. So the coordinator MUST
            # be able to process resize_infer(add=…) BEFORE Step 7 returns.
            # That requires: rollout_manager registered, model-update service
            # resources registered, _cache_ready_step published, _active_engine_indices
            # bootstrap'd. All must happen here in 6.6 — not at Steps 8-10.

            # 6.6a: declared full placement table (provider call, no scheduler interaction).
            all_rollout_placements = (
                self._placement_provider.get_all_rollout_engine_placements()
            )

            # 6.6b: create all-shell RolloutManager. ALL engines start in `shell`
            # state; NO Ray actor / SGLang server / GPU is spawned at construction.
            # Manager ctor MUST also start an empty Miles router immediately and
            # bind `args.sglang_router_ip` / `args.sglang_router_port` (per-pipeline
            # isolation: each pipeline allocates its own router port via the placement
            # provider's port pool; do NOT inherit a peer pipeline's router port).
            # `manager.activate_routing(engine_indices)` is the SOLE add_worker entry;
            # `manager.shrink_engines(engine_indices)` is the SOLE remove/disable entry.
            # ctor is fast — just dict population + empty router bind.
            self.actor_infer = RolloutManager.options(
                namespace=self._ray_namespace,
                name=f"rlix:rollout_manager:{self._pipeline_id}",
            ).remote(
                args,
                all_engine_placements=all_rollout_placements,
                active_engine_indices=frozenset(),  # all-shell at construction
            )

            # 6.6c: engine_count sanity (== declared count, includes shell slots).
            engine_count = ray.get(self.actor_infer.get_engine_count.remote())
            expected = args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            assert engine_count == expected, (
                f"RolloutManager engine_count mismatch: got {engine_count}, expected {expected}"
            )

            # 6.6d: register cache_owner_actor + rollout_manager handles on coordinator
            # so MilesModelUpdateService can be lazy-created on first sync. Coordinator
            # holds these as ctor args for the service named actor.
            coordinator = self._get_coordinator_handle()
            ray.get(coordinator.register_model_update_resources.remote(
                cache_owner_actor=self._cache_owner_actor,
                rollout_manager=self.actor_infer,
            ))

            # 6.6e: publish base-version `_cache_ready_step = -1`. Step 4 already
            # built base CPU bucket cache (step=-1) on cache_owner_actor. With
            # version published BEFORE Step 7, scheduler-driven _expand_workers
            # can resolve `_cache_ready_step` (passes -1 to service.sync). The
            # `version == -1` path is the no-transport short-circuit (Fix #13 +
            # #15) — service skips run_sync_session, skips finalize, only calls
            # `manager.set_weight_version(-1, target_engines)`. cache_owner GPU
            # is not touched (Step 6 already released it to scheduler).
            ray.get(coordinator.publish_cache_ready_step.remote(-1))

            # 6.6f: bootstrap coordinator's _active_engine_indices to the empty
            # frozenset. Scheduler-driven _expand_workers will populate it during
            # Step 7. The coordinator's `_active_engines_bootstrapped: bool` flag
            # is set True by this call (regardless of set contents); empty bootstrap
            # is the documented default and is distinguishable from "not yet
            # bootstrapped" via the explicit flag (NOT via set truthiness).
            ray.get(coordinator.bootstrap_active_engines.remote(frozenset()))

            # ----- Step 7: actor_infer GENERATION allocation (long-lived) ------
            # During this blocking call, scheduler executes
            #   coordinator.resize_infer(dp_ranks_to_add=[<allocated dps>])
            # in Phase 5; coordinator's _expand_workers acquires _resize_sync_lock
            # and:
            #   - reads _cache_ready_step = -1 ✓ (6.6e)
            #   - reads _model_update_resources ✓ (6.6d)
            #   - manager.expand_engines(target): shell → loading lazy
            #     `_create_sglang_actor(engine_idx, placement)` + `onload()`
            #   - service.sync_selected_workers(sync_id, target, version=-1):
            #     v=-1 short-circuit — only `manager.set_weight_version(-1, target)`
            #   - manager.activate_routing(target): loading → active + add_worker
            #   - _active_engine_indices |= set(target)
            # Phase 6 then signals the waiter; Step 7 returns with allocated_gpus.
            # M1 (relaxed for M11.2 Gate 4): allocated 必须是 declared 的 subset
            # (`⊆`, 不是 `==`). RLix scheduler 在 contention 下返回 partial allocation
            # (full_finetune_pipeline.py:712-726 已有同样 pattern). Pipeline B 在
            # Gate 4 双 pipeline 下可能拿 2/4 GPU; 余下 engine 维持 shell, 等后续
            # expand 时 lazy 创建.
            allocated_actor_infer_gpus = self._request_cluster_gpus(
                cluster_id=self._actor_infer_cluster_id,
                priority=Priority.GENERATION,
                global_step=init_global_step,
            )
            actor_infer_allocated = True  # M4: gate scheduler release on alloc success

            # ----- Step 7.1: M1 subset assert + log partial flag ------
            assert set(allocated_actor_infer_gpus).issubset(set(self._infer_device_mapping)), (
                f"allocated actor_infer GPUs must be subset of declared infer_device_mapping; "
                f"got {allocated_actor_infer_gpus}, declared {self._infer_device_mapping}"
            )
            is_partial_allocation = (
                len(allocated_actor_infer_gpus) < len(self._infer_device_mapping)
            )
            if is_partial_allocation:
                logger.warning(
                    "PARTIAL allocation detected for actor_infer — got %d/%d GPUs. "
                    "Pipeline starts with active subset; remaining engines stay in "
                    "shell state until later scheduler-driven expand.",
                    len(allocated_actor_infer_gpus),
                    len(self._infer_device_mapping),
                )

            # ----- Step 7.2: pipeline-side derive active_engine_indices ------
            # Validates half-engine allocation (raises if any engine has only some
            # of its tp_size GPUs allocated).
            active_engine_indices: frozenset[int] = (
                self._placement_provider.get_active_engine_indices(
                    allocated_gpus=allocated_actor_infer_gpus,
                    tp_size=args.rollout_num_gpus_per_engine,
                )
            )
            self._actor_infer_gpu_ids = allocated_actor_infer_gpus
            self._active_engine_indices = active_engine_indices

            # ----- Step 7.3: consistency check vs coordinator state ------
            # _expand_workers populated coordinator._active_engine_indices during
            # Step 7. Pipeline-side computation must match.
            coord_active = ray.get(coordinator.get_active_engines.remote())
            assert coord_active == active_engine_indices, (
                f"coordinator/_expand_workers populated {coord_active}, but "
                f"pipeline derived {active_engine_indices} from allocated GPUs "
                f"{allocated_actor_infer_gpus} + declared {self._infer_device_mapping}; "
                f"this indicates a wiring bug between scheduler dp_ranks and "
                f"MILES engine indices (M11.2 first build assumes identity)"
            )
        except Exception:
            # M4 — Phase 2 hard cleanup on failure (kill order: train actors first,
            # then infer, then release infer scheduler). Train scheduler 已在 Phase 1
            # finally 释放 (无需再调 `_notify_release_cluster_gpus`), 但 train actor
            # handles 仍持 GPU + CUDA context, 必须 kill, 否则 scheduler 视 GPU 空闲
            # vs Ray actor 仍占的分裂 → 下个 pipeline 调度同 GPU OOM.
            if getattr(self, "actor_train", None) is not None:
                for h in self.actor_train._actor_handles:
                    try: ray.kill(h, no_restart=True)
                    except Exception: pass
                self.actor_train = None
            # Phase 2 actor_infer cleanup (`RolloutManager` ctor 内 self-cleanup 已
            # 处理 ctor 半构造失败; 这里覆盖 RolloutManager 已构造完整但 Step 9/10
            # 失败的场景).
            if getattr(self, "actor_infer", None) is not None:
                # best-effort hard shutdown: stop monitors + kill engine actors,
                # 给 10s timeout 兜底 RPC 卡死. 不做 graceful drain (follow-up).
                try:
                    ray.get(self.actor_infer.shutdown_hard.remote(), timeout=10)
                except Exception:
                    logger.exception("RolloutManager.shutdown_hard failed; falling back to ray.kill")
                try: ray.kill(self.actor_infer, no_restart=True)
                except Exception: pass
                self.actor_infer = None
            # release infer scheduler allocation only if we got past Step 7's request.
            # Step 6.5 raise / Step 7 request raise → actor_infer_allocated=False, no release.
            # Step 7 assert fire / Step 8-10 raise → actor_infer_allocated=True, release.
            if actor_infer_allocated:
                try:
                    self._notify_release_cluster_gpus(
                        cluster_id=self._actor_infer_cluster_id,
                        global_step=init_global_step,
                    )
                except Exception:
                    logger.exception("scheduler release failed; original init error takes precedence")
            raise  # 原 init 异常透传

        self._initialized = True
        return ActionResponse(success=True)
```

**放错位置失败模式**：

- (a) build cache 放到 Step 2（model load）之前 → 拿到未初始化权重 → 首次 expand 把垃圾 weights 推给 inference
- (b) build cache 放到 Step 5 (offload) 之后 → gather 死锁 (rank 已离开 NCCL group)
  或读到空 cache → 首次 expand 拿不到任何 weights
- (c) **[已修正] Step 7 request actor_infer 嵌在 actor_train try 块内** → partial overlap
  拓扑 (`train ⊂ infer`) 下死锁: scheduler 拿不出第二份 overlap GPU 来满足 actor_infer 申请
- (d) 漏掉 Step 5 `actor_train.offload()` → 仅靠 finally `notify_release` 不释放 GPU VRAM
  与 NCCL buffer, actor_infer 起来后 OOM
- (e) 漏掉 Step 10 active set bootstrap → 首次 `sync_base_weights_to_active()` 读到空 set →
  短路 → 非重叠 active engine 永远停在 base weight (Feature 5+6 不变量失效)
- (f) 用 `ray.get(actor_train.init())` 而不是 `run(...)` → init 是 async coroutine, 不是
  ObjectRef, `ray.get` 直接 `TypeError`
- (g) `initialize_pipeline` 改成 `async def` → 与 sync `_request_cluster_gpus` 半 async
  半 sync 混用阻塞 event loop
- (h) `RayTrainGroup` 用 `num_gpus_per_actor=0.4` (standalone fractional) → offload 后
  scheduler 视 GPU 空闲, Ray actor 仍 reserve 0.4, 下个 pipeline 调度卡死. 必须
  RLix mode 用 `0.01` (实际 GPU 隔离靠 PG + CUDA_VISIBLE_DEVICES)

第一次 expand（来自 runtime 后续 `_request_cluster_gpus(cluster_id="actor_train",
priority=Priority.ACTOR_TRAINING, global_step=N)` 重新分配）直接读这个 cache。对齐 ROLL
[full_finetune_pipeline.py:289-308](rlix/pipeline/full_finetune_pipeline.py#L289) 的 `setup()`
阶段时机。

**注意**：admit 仅注册拓扑，不创建 worker；worker 创建发生在第一次
`_request_cluster_gpus` 时。`_request_cluster_gpus` / `_notify_release_cluster_gpus`
是 RLix `FullFinetunePipeline` 已有 pipeline-side 方法（kwargs 签名，下划线前缀；MILES
`MilesPipeline` 应继承或参照该模式调用 `self._rlix_scheduler.request_gpus.remote(...)`）。
`Priority` 来自 [rlix/protocol/types.py:52](rlix/protocol/types.py#L52)（`enum.IntEnum`，
7 级；MILES 用 `INITIALIZATION` / `ACTOR_TRAINING` / `GENERATION`，CRITIC / OLD_LOG_PROBS /
REF_LOG_PROBS / VALUE_COMPUTE 暂不用）。

1. 每次 `train_step` 后构建 CPU bucket cache（**单 cache owner，单 cache slot，覆盖式
   写入**）：
   - 所有 TP/PP/CP ranks 参与 collective gather（PP collectives 汇聚所有 stage）
   - **EP-aware gather scope**：本 port scope **不启用 MoE/EP**（已在 Out of Scope 显式
     排除）；如未来支持，需补 expert 参数走 `get_expert_tensor_parallel_group()` +
     `get_expert_model_parallel_group()` 路径，non-expert 走 `get_tensor_model_parallel_group()`
   - 仅 cache owner 持有**唯一一份** CPU bucket（`_bucket_cache`），新 step 直接 in-place
     覆盖前一 step；其他 rank drain generator 让 collective 推进，但丢弃结果
   - **Cache owner = MILES 既有 `_is_distributed_src_rank`**（[update_weight_from_tensor.py:117-121](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L117)，
     `intra_dp_cp.rank == 0 AND tp_rank == 0 AND pp_rank == 0`，已覆盖 pp+dp+tp+cp 四维）
     — 与 ROLL `_select_global_sender_rank` (`model_update_service.py:90`) 语义等价。
     **不新建 helper**。注意 [broadcast.py:77-79](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py#L77)
     的 `_is_source` 只覆盖三维（缺 PP），仅供 broadcast 路径使用，**不能作 cache
     owner**。这是**唯一** cache owner，**唯一** transport sender
   - **Cache owner discovery — service 不能直接读 worker 内部状态**：`MilesModelUpdateService`
     是独立 RLix actor，不在 Megatron worker 进程内，无法读 `_is_distributed_src_rank()`
     这种依赖 mpu state 的函数。必须 **init 阶段 worker 主动上报**:
     `actor_train.initialize()` 完成后，每个 Megatron worker 通过
     `worker.report_cache_owner_role.remote(global_rank=..., is_cache_owner=bool)` 向
     coordinator 上报；coordinator 收齐后筛出 `is_cache_owner=True` 那个 rank 的
     `(global_rank, ray_actor_handle)`，作为 `MilesModelUpdateService.__init__(cache_owner_rank,
     cache_owner_actor_handle)` 的构造参数。**不要在 sync 路径运行时再 RPC 查询**——
     cache owner rank 在整个 pipeline lifetime 不变 (Megatron rank 不漂移)，init 一次即可
   - Bucket layout：`List[BucketRecord]`，每条至少含 `param_names`, `shapes`,
     `dtypes`, `used_bytes`, `cpu_uint8_bucket`（contiguous CPU tensor / bytes）
   - **HF-format invariant (M5)**：`build_cpu_bucket_cache` cache 的是 **HF-converted
     `named_tensors`** (经 `_gather_hf_weights()` 等价 ROLL 路径转换;
     [external/ROLL/roll/third_party/megatron/model_update.py](external/ROLL/roll/third_party/megatron/model_update.py)
     `_gather_hf_weights()`), **不是 raw Megatron state_dict**. 否则 SGLang receiver
     `load_weights` 找不到匹配的 HF 参数名 / shape, 直接 crash. MILES 既有
     `_send_to_colocated_engine` 调
     [`FlattenedTensorBucket(named_tensors=hf_named_tensors)`](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L311)
     时 `named_tensors` 已经是 HF 格式 (经 bridge 转换), `build_cpu_bucket_cache`
     复用同一转换路径
   - **首次 `build_cpu_bucket_cache()` 调用时分配 pinned CPU buffer 并保留供后续 step
     复写**（不是启动期分配 — 启动期 host RAM 估算偏差会触发不必要的 fail fast）。
     Host RAM budget check 在首次 build 时执行；超 budget 时 fail fast 仍然适用
2. **不做 ROLL 风格的 cache versioning**：单 cache slot 覆盖式写入；`_cache_ready_step`
   仅是版本标签（不是 slot selector）。顺序契约（不变量 3）保证 build 与 broadcast 永不
   并发，因此不需要 `promote_active_checkpoint` 双槽语义。**互斥责任由 invariant 3
   （顺序契约）+ invariant 4（`_cache_lock`）共同承担；`_cache_ready_step` 字段本身不
   承担互斥责任，仅供 metric / log / `_current_weight_version` propagation 使用**。
   读取 `_cache_ready_step` 不需要持锁；写入它必须在 `_cache_lock` 内、且与 bucket
   写入是同一原子操作。
3. Offload training GPU（释放全部 VRAM，借 `ReloadableProcessGroup` 销毁 NCCL group）
4. Expand 时：wake_up target inference engines（仅 overlap）
5. **`MilesModelUpdateService.sync_selected_workers(sync_id, tgt_engine_indices)`** —
  单 sender（cache owner）推到 woken engines。**cpu_serialize 不做 H2D
  staging, NCCL broadcast 才做**:
  - **同 GPU receiver (cpu_serialize)**: sender 直接 `ray.put(torch.save(cached_cpu_bucket,
    BytesIO).getvalue()) → ObjectRef`, 调 `engine.update_weights_from_cpu_bucket.remote(ref, ...)`.
    无 GPU 操作.
  - **跨 GPU receiver (NCCL broadcast)**: NCCL 不支持 CPU 张量, sender 必须 per-bucket
    H2D staging 到 cache_owner GPU → `dist.broadcast(staging_tensor, src, group)` → free
    staging. 每 bucket 立即 free, 严格逐 bucket 滚动, 不整模型一次性回 GPU.

  锁语义为"整段持锁": `acquire _cache_lock → 读取 cache pointer + 版本 → for bucket in
  buckets: (cpu_serialize: ray.put → invoke; NCCL: 分配 GPU staging → H2D → broadcast →
  等 bucket receivers ack → free staging) → (全部完成) destroy 动态 NCCL group →
  release _cache_lock`. cpu_serialize 与 NCCL receivers 可对同一 bucket 共存
  (dual-mask), 但 H2D staging 只对 NCCL 必要.

   `sync_id` 由 coordinator 传入，格式 `f"{pipeline_id}_step{step}_{uuid4().hex[:8]}"`，
   group_name 从 sync_id 派生，便于 cross-actor log correlation。

   **关键 strict invariants（受 `_cache_lock` 保护，压短 pseudo-code 时不能损失）**：

   - **Per-bucket barrier 必需**：每个 bucket 在 sender 端 free staging buffer 之前，
     必须等待该 bucket 的所有 receivers ack（barrier on this bucket）；否则 NCCL
     staging buffer 或 Ray ObjectRef payload lifecycle 可能早于 receiver load 完成。
     **术语注**：这是 per-bucket receiver ack barrier，**不是** NeMo 风格的 per-engine
     finalize RPC。MILES 不引入"每个 engine sync 完成后的 finalize hook"概念 — 所有
     ack 在 bucket 粒度完成。如未来要新增 engine 级 finalize，须单独提案，不要混淆为
     F4 既有约束
   - **NCCL broadcast 必须从 GPU 出去** — NCCL 不支持 CPU 张量。所以 broadcast path
     的 H2D staging 是必需步骤，不是优化；同 GPU receiver 不使用 CUDA IPC，走
     Ray ObjectRef `cpu_serialize`
   - **不能"整模型 H2D staging 到 sender GPU 再发"** — peak VRAM = `bucket_size_bytes +
     transport scratch`，与模型大小无关
   - **`args.miles_model_update_bucket_size_mb` 新增 arg**（默认 512 MB；不复用
     既有 `refit_buffer_size_in_mb` 之类 standalone-only arg, 因为 selective sync 有不同的
     peak VRAM 约束）；**仅当存在 NCCL broadcast receivers 时**, 启动时用"wake_up 后剩余 VRAM"
     做上界 check，确保 `bucket_size_bytes + transport scratch < overlap GPU 可用余量`。
     F10 topology validation 做这个 check (而不是 lazy 在首次 sync 时报错). 全 colocate
     cpu_serialize 拓扑下 (first build 默认), 此 GPU check 不触发 — 但 RAM check (CPU
     cache + N receivers `torch.load` 拷贝预算) 仍生效
   - **同 GPU receiver**（overlap GPU）→ `cpu_serialize` Ray ObjectRef bytes（portable
     path，不依赖 CUDA IPC / `--ipc=host` / CAP_SYS_PTRACE）, **sender 不 H2D**
   - **跨 GPU receiver**（target 有 TP 跨 GPU 的 rank）→ sender per-bucket H2D staging →
     动态 NCCL group broadcast → free staging
   - 同一 bucket 可能 **同时**有 cpu_serialize receivers 与 broadcast receivers
     （receiver-side dual-mask）；此时 H2D staging 只为 NCCL 路径做一次, cpu_serialize
     receivers 不依赖该 staging buffer (各自走 `ray.put(cpu_bucket)` 路径)

**Receiver-side dual-mask + `is_group_exist` no-op guard（tp>1 必须正确, 三种 transport
都要 mask）：**

`comm_plan` 按 transport 携带 per-engine receiver mask:

| Transport | mask 字段 | colocate? | Milestone |
|---|---|---|---|
| `cpu_serialize` | **`cpu_serialize_local_ranks`** | colocate | **M11.1** (vast.ai colocate) |
| NCCL broadcast | **`broadcast_local_ranks`** | non-colocate | **M11.1** (partial overlap 必需) |
| `cuda_ipc` | **`ipc_local_ranks`** | colocate | **M11.6** (production cluster, 加 `update_weights_from_tensor` 扩参) |

receiver-side guard:

- (**M11.6**) `update_weights_from_tensor(serialized_named_tensors, ipc_local_ranks, ...)`
  (cuda_ipc 路径): M11.6 时**既有 method 必须扩展 `ipc_local_ranks` 参数**, 检查
  `self.rank in ipc_local_ranks`, 不在则 skip. 对齐 ROLL [vllm_strategy.py:685](external/ROLL/roll/distributed/strategy/vllm_strategy.py#L685)
  签名 `update_weights_from_tensor(... ipc_local_ranks=None, model_update_transport="cuda_ipc")`.
  MILES 现状 [sglang_engine.py:282](external/miles/miles/backends/sglang_utils/sglang_engine.py#L282)
  `update_weights_from_tensor` 没有该参数, M11.6 加. **M11.1 不动这个 method** (M11.1
  cuda_ipc 不在 scope, 既有 `update_weights_from_tensor` standalone path 仍可用).
- (**M11.1**) cpu_serialize 路径有**两层** receiver, 都要 guard:
  - **Layer 1 — Ray wrapper actor method** `SGLangEngine.update_weights_from_cpu_bucket(payload_bytes, cpu_serialize_local_ranks, ...)`:
    检查 `self.rank in cpu_serialize_local_ranks`, 不在则 skip. **`payload_bytes` 不是
    `payload_ref`** — Ray 自动 deref remote method 的 top-level ObjectRef 参数,
    wrapper 实际收到 `bytes` (auto-deref 是预期行为). 不要 `ray.get(payload_bytes)`
    (会 raise `TypeError`). Sender 仍 `ray.put(...) → ref`, 调用
    `engine.update_weights_from_cpu_bucket.remote(ref, ...)`, Ray runtime 在 method
    边界自动解引用. Wrapper 收到 bytes 后**不在 wrapper 进程内 `torch.load`** —
    写到 `/dev/shm/miles_cpu_bucket_{uuid}.pt`, HTTP body 只携带 `payload_path`
    (见上 "payload bytes 跨进程方式"段) + `try/finally os.unlink(path)` 收尾.
  - **Layer 2 — SGLang server-side HTTP handler** `POST /update_weights_from_cpu_bucket`
    (vendored `cpu_serialize_http_route.patch`): 收到 `payload_path` (str) +
    metadata, **同步** `payload_bytes = pathlib.Path(req["payload_path"]).read_bytes()`
    + `torch.load(BytesIO(payload_bytes), weights_only=True)` → dispatch to
    scheduler/tp_worker (synchronous, route 返回时所有 TP workers 已 ack). HTTP
    handler **不持有** `payload_path` beyond request lifetime — wrapper finally
    内 unlink, 不入队列, 不交后台线程.
- `broadcast_parameter(group_name, broadcast_local_ranks, ...)` (NCCL 路径): 检查
  `self.rank in broadcast_local_ranks`, 不在则 skip
- `destroy_collective_group(group_name)` 必须用 `is_group_exist(group_name)` no-op guard
  ([roll/utils/collective/collective.py](external/ROLL/roll/utils/collective/collective.py)) —
  colocate-only ranks (cuda_ipc 或 cpu_serialize, 不论哪种) 从未 join NCCL group, 没有
  guard 会 KeyError
- **M11.1** parity gate (Gate 1, 2, 2.5, 3) 按 tp>1 运行, 必须覆盖 `cpu_serialize-only
  ranks` + `broadcast ranks` + 混合 (cpu_serialize + broadcast) receiver mask; 不能
  依赖 tp=1 绕过该 bug. **M11.6** cuda_ipc Gate 加回 `cuda_ipc-only ranks` +
  `ipc_local_ranks` 覆盖.

**Cache 安全性 4 个不变量：**

1. **单 writer**：training hook 是唯一调用 `build_cpu_bucket_cache(step)` 的入口；
   写完即原子更新 `_cache_ready_step`
2. **单 reader 路径**：active refresh + expand sync 都通过 `MilesModelUpdateService.
   sync_selected_workers(...)` 读同一份 `_bucket_cache`
3. **顺序契约**：`before_training(step+1)` 阻塞到前一个 `after_training(step)` 触发的
   active refresh + expand 完成后才返回（由 RLix `request_cluster_gpus` 的 blocking
   `ray.get` 保证）。这是覆盖式 cache 安全的**主要**保证 — 因为有这条契约，build
   永不会发生在 broadcast 进行时
4. **Cache owner `_cache_lock` (单 method 单临界区，Fix #6)**: 必须覆盖完整
   "读 cache pointer / `_cache_ready_step` / bucket 列表 → 逐 bucket transport
   (cpu_serialize per-engine RPC + NCCL broadcast) → receiver barrier → 动态
   NCCL group destroy" 整段。**实现形态**: 锁由 `cache_owner_actor.run_sync_session(plan)`
   单 Ray method body 内 `with self._cache_lock` 持有, **不跨 RPC**. service 端
   只发一次 RPC, cache_owner 内部顺序跑 (snapshot → per-bucket transport →
   group destroy) 全在锁内. 之前的 5 个 split RPCs (get_bucket_count /
   serialize_bucket_to_objref / setup_collective_group / broadcast_bucket /
   destroy_collective_group) 不再是 top-level Ray methods, 全部降级为
   `run_sync_session` 内部 helpers — 因此跨 RPC 持锁的不可实现矛盾消失.
   `build_cpu_bucket_cache()` 写入新 buckets + publish `_cache_ready_step`
   时也必须在自己的 method body 内持同一把锁 (与 `run_sync_session` 互斥).
   **禁止只锁 cache lookup 或只锁 pointer swap 的半截实现** (对齐 ROLL
   `megatron_strategy.py:2095-2099`). 正常路径不变量 3 已经够; `_cache_lock`
   是抗超时与异常恢复的最后一道. **`version == -1` 路径不进 `run_sync_session`,
   不持 `_cache_lock`** (Fix #13 — service 短路, 只 publish version label).

**Bucket payload 格式：colocate 双 transport + non-colocate broadcast**

**Colocate（同 GPU receiver）按 milestone 分阶段交付** (`args.model_update_transport`,
命名与 ROLL 一致):

| 配置 | 路径 | 容器要求 | 性能 | Milestone |
|---|---|---|---|---|
| `"cpu_serialize"` | wrapper auto-deref `bytes` → `/dev/shm` tmpfs file → HTTP route | 无特殊要求 (vast.ai 受限容器可用) | D2H + 序列化 + plasma + tmpfs (~1 次额外 copy) | **M11.1 first build (唯一 colocate transport)** |
| `"cuda_ipc"` | **新 RLix adapter** (CPU cache → per-bucket H2D staging → IPC handle serialize, ~50-80 行); **不复用既有 `_send_to_colocated_engine`** (后者依赖 live `dist.gather_object` 与 F4 destroy NCCL 顺序冲突) | CAP_SYS_PTRACE 或 `--ipc=host` (production cluster / VM) | 基线 zero-copy GPU IPC handle | **M11.6 production cluster** + 启动期 smoke test capability check |

未识别值 fail-fast。不引入 env var；env var 仅在临时调试时通过 launcher 注入到 args
（避免两套 source-of-truth）。

##### Case A：CUDA IPC (M11.6 later milestone — first build vast.ai 不走)

**M11.1 first-build deployment target = vast.ai 受限容器** (无 `--ipc=host` /
`CAP_SYS_PTRACE`), 因此 colocate transport 选 `cpu_serialize` (Case B). cuda_ipc
adapter 推迟到 M11.6 production cluster milestone.

**为什么 M11.2 必须新写 adapter, 不能复用既有
[`_send_to_colocated_engine`](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L280)**:
既有路径依赖 live `dist.gather_object(..., group=ipc_gather_group)`
([update_weight_from_tensor.py:319](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L319))
+ 当前 GPU tensor 打包 IPC handle, 与 F4 sequence (build cache → offload → destroy
NCCL → sync) 顺序不兼容. M11.6 时新写 "CPU cache → per-bucket H2D staging → IPC handle
serialize" adapter (~50-80 行, 因 single cache_owner 不需要 cross-peer gather).
Standalone mode (非 RLix) 仍可走既有 `_send_to_colocated_engine` cuda_ipc 路径.

**M11.6 capability check (实施时再加)**: 不写脆弱环境探测器 (CAP_SYS_PTRACE / `--ipc=host`
/ VM heuristics). 改用启动期 smoke test:
`MultiprocessingSerializer.serialize(small_tensor, output_str=True)` →
`MultiprocessingSerializer.deserialize(handle)` 跨 process round-trip; 失败即提示用户切
`args.model_update_transport='cpu_serialize'`.

##### Case B：cpu_serialize (M11.1 default colocate transport, vast.ai 受限容器)

M11.1 RLix mode 唯一 colocate transport (F10 强制 `args.model_update_transport ==
"cpu_serialize"`). 所有以下新增代码**仅服务 M11.1 cpu_serialize 分支**; M11.6 加 cuda_ipc
adapter (Case A) 是独立分支, 不影响本段.

Wire payload（与 ROLL `roll/utils/send_recv_utils.py` 同 schema，**不复用 SGLang
`FlattenedTensorBucket` 嵌套结构**）：

  ```python
  {"bucket": pinned_cpu_uint8_tensor, "tensors_meta": list[dict]}
  ```

    Sender 路径（cache owner 端，per bucket — cpu_serialize 直接读 CPU
    cache, 不做 H2D staging）：

    F4 sequence 是 build CPU cache (Step 4) → offload + destroy NCCL (Step 5) → 后续
    sync. 到 sync 时 GPU 上已经没有 `hf_named_tensors` (offload 释放), 也不能依赖 live
    NCCL group. 因此 cpu_serialize sender **直接从 cached CPU bucket 序列化**, 跳过
    "H2D staging → D2H" 这条与 F4 顺序冲突的旧描述. H2D staging **只**在 non-colocate
    NCCL broadcast 路径需要 (NCCL 不支持 CPU tensor — 见 "Non-colocate" 段).

    1. 读 cache (cache_owner 已 build):
       `cached: BucketRecord = self._bucket_cache[bucket_idx]` (`cpu_uint8_bucket`
       已是 pinned, contiguous; `tensors_meta` 含 names / shapes / dtypes).
    2. 序列化:
       `payload_bytes = torch.save({"bucket": cached.cpu_uint8_bucket, "tensors_meta":
       cached.tensors_meta}, BytesIO).getvalue()` — payload size bounded by
       `args.miles_model_update_bucket_size_mb` (~512 MB-1 GB), receiver 侧 `torch.load`
       per-bucket 拷贝预算.
    3. Ray put: `payload_ref = ray.put(payload_bytes)` → ObjectRef 入 plasma store.
    4. Per-target receiver invocation:
       `SGLangEngine.update_weights_from_cpu_bucket.remote(payload_ref, load_format=...,
       flush_cache=..., cpu_serialize_local_ranks=...)`.
       **`weight_version` 不在 per-bucket 调用中** — 单次 sync 的 version publish
       由 atomic unit 的 step (c) `manager.set_weight_version(version, target_engines)`
       一次性完成 (在所有 bucket transport 完成 + per-engine finalize fan-out 之后).
       per-bucket payload 携带 version 会让 SGLang 既有 update endpoint 在收到第一个
       bucket 后就 publish version label, 而其他 bucket 还没 load 完, 出现 version
       inversion (受 stale weights 的请求被标成新 version).
       Ray runtime 在 method 边界 auto-deref ObjectRef → wrapper 收到 `bytes` (预期
       行为, 不是 bug — wrapper 落 `/dev/shm/miles_cpu_bucket_{uuid}.pt`, HTTP body 仅传 path; 见
       receiver 段 tmpfs flow). Sender 持 `payload_ref` 直到 sync barrier
       `ray.get([...])` 返回, 期间 plasma 不 GC.

    **没有** H2D staging / GPU OOM 风险 — sender 只触碰 CPU cache, 无 GPU 操作. (cuda_ipc
    follow-up 路径才需要 CPU cache → H2D staging → IPC handle, 那是 follow-up adapter,
    不是 cpu_serialize first build.)

  **payload bytes 跨进程方式 — tmpfs file path**: HTTP body 不能直接携带 GB-级 bytes
  (base64 膨胀 + HTTP buffer pressure). 也不能用 Ray ObjectRef ID 走 HTTP — SGLang
  child subprocess 不是 Ray actor, 没有 cluster 连接, `ray.get(ObjectRef(id))` 需要
  `ray.init(address="auto")`, 增 startup latency 且环境 fragile. 而且原始
  `engine.update_weights_from_cpu_bucket.remote(ref, ...)` 在 Ray boundary 处会自动
  deref top-level ObjectRef, 到 wrapper 时已是 `bytes`, 无法回头取 ObjectRef ID 转给
  HTTP. 综合最简方案: **wrapper 把 deref'd bytes 写到 `/dev/shm/miles_cpu_bucket_{uuid}.pt`,
  HTTP body 只携带 file path; SGLang server 同步 read 后立即返回 (不删, 见 lifecycle
  invariant: cleanup owner = wrapper `try/finally os.unlink`, single source of truth).**

  Trade-off: 多一次 copy (wrapper heap → tmpfs), 但 tmpfs 是 RAM-backed mmap, 实际
  开销小 (~1GB/s memcpy). 比 base64 HTTP 体面, 比 RefWrapper + subprocess `ray.init`
  简单. cpu_serialize 路径已是 colocate (sender + wrapper + SGLang 同节点), `/dev/shm`
  本就同节点共享, 无 NFS / 跨节点路径假设.

  Receiver 路径 (SGLang Ray actor method, 新增 + 配套 SGLang fork patch):

  ```python
  # miles/backends/sglang_utils/sglang_engine.py — Ray actor wrapper method
  def update_weights_from_cpu_bucket(self, payload_bytes, load_format=..., flush_cache=False, cpu_serialize_local_ranks=None):
      # NOTE: NO `weight_version` parameter. Per-sync version publish is the
      # atomic unit's step (c) on the manager, NOT per-bucket on each engine.
      # Sender 端 `engine.update_weights_from_cpu_bucket.remote(payload_ref, ...)`
      # Ray runtime 在 method 边界自动 deref top-level ObjectRef → 我们这里 payload_bytes
      # 已经是 1GB bytes (期望行为, 不是 bug). 通过 tmpfs 把 bytes 落到 SGLang subprocess
      # 可访问的 path, HTTP 只传 path string.
      uid = uuid.uuid4().hex
      path = f"/dev/shm/miles_cpu_bucket_{uid}.pt"  # 统一前缀 (见 lifecycle invariant)
      try:
          with open(path, "wb") as f:
              f.write(payload_bytes)
          payload = {
              "payload_path": path,
              "load_format": load_format,
              "flush_cache": flush_cache,
              "cpu_serialize_local_ranks": cpu_serialize_local_ranks,
          }
          # NO weight_version field — Fix #3 invariant: per-bucket payload
          # carries no version label.  Version publish happens once per sync
          # via manager.set_weight_version(version, target_engines).
          # 阻塞等 SGLang server load 完, 然后下面 finally 删 file (server 端不需要保留)
          return self._make_request("update_weights_from_cpu_bucket", payload)
      finally:
          try: os.unlink(path)
          except FileNotFoundError: pass

  # SGLang server side (vendored fork patch cpu_serialize_http_route.patch + scheduler_dispatch):
  # @app.post("/update_weights_from_cpu_bucket")
  # async def update_weights_from_cpu_bucket(req):
  #     # mmap 读 (Linux tmpfs zero-copy):
  #     payload_bytes = pathlib.Path(req["payload_path"]).read_bytes()
  #     # dispatch 到 scheduler.update_weights_from_cpu_bucket(payload_bytes, ...) →
  #     # 各 TP worker `weight_sync_utils.in_process_load_cpu_bucket(payload_bytes, ...)`
  #     # (cpu_serialize_weight_sync_utils.patch 装的 in-process path)
  #     # server 不删 file, 由 wrapper finally 收尾 (single source of truth)
  ```
  Sender 调用方仍 `ray.put(payload_bytes) → ref → engine.update_weights_from_cpu_bucket.remote(ref, ...)`;
  Ray runtime 在 method 边界 auto-deref → wrapper 收到 bytes; wrapper 落 tmpfs + HTTP path
  到 SGLang. Sender 的 `payload_ref` 在 sync barrier `ray.get` 等所有 engines ack 期间
  保持 alive (plasma 不 GC), barrier 完成后释放.

  **Ray ObjectRef GC 不变量**: per-bucket sync 全程 sender 持 `payload_ref` (sync barrier
  `ray.get([engine.X.remote(ref) for engine in ...])` 阻塞 → ref 不出 sender 作用域).
  Wrapper 自身不持 ref (auto-deref 已发生), 但持 tmpfs file 直到 HTTP response 返回 →
  SGLang load 已完成. 失败模式: sender crash mid-bucket → ref + tmpfs 都泄露, 由 pipeline
  cleanup (M4) 兜底; tmpfs `/dev/shm` 在节点重启时自动清.

  **tmpfs file lifecycle (Critical Invariant)**:
  - **Cleanup owner = wrapper Ray actor method** (single source of truth, **不是 SGLang
    server side**); SGLang server 只读 path, 不负责删
  - 必须 `try: ... finally: try os.unlink(path) except FileNotFoundError: pass` 包裹
    HTTP request, 任何异常路径 (HTTP 5xx / receiver crash / network timeout) 都保证
    unlink. SGLang server 已读完后, file 立即释放 RAM (tmpfs unlink 是 O(1))
  - File path 用 `f"/dev/shm/miles_cpu_bucket_{uuid.uuid4().hex}.pt"` 命名 (统一前缀
    `miles_cpu_bucket_` 便于运维 `ls /dev/shm/miles_cpu_bucket_*` 查泄漏)
  - 启动期 fail-fast check: `os.path.isdir("/dev/shm")` + writable; 不在则 raise
    (容器场景 `/dev/shm` 可能未挂载或太小)
  - 节点级 cleanup 兜底: `tmpfs` 在节点重启时自动清空; pipeline crash 后用户 `ray stop`
    + 重启即可
  - **不引入** 周期性 GC daemon / 跨 sync 持久 cache — single-bucket lifetime 严格
    与 HTTP request lifetime 1:1 绑定

  **tmpfs 并发上界 (Critical Invariant)**:
  - **Per-bucket receiver 串行**: `MilesModelUpdateService.sync_selected_workers` 对同
    bucket 的多 target engines 调用 **串行**, 不并发. 顺序调用 `ray.get(...)` 保证 wrapper
    N+1 启动前 wrapper N 已 unlink. 同节点 tmpfs peak = 1× bucket_size, 不是
    N× bucket_size. 跨 bucket 仍按 F4 既有 "逐 bucket 滚动" 语义 (per-bucket barrier).
    串行成本: 单 sync 内 RPC 序列化, 但 cpu_serialize 路径 colocate engines 通常
    N≤4 同节点, RPC 串行加 N× HTTP latency (~10ms × 4 = 40ms), vs tmpfs 爆掉的 hard
    fail tradeoff 完全可接受
  - **启动期 fail-fast** (见 §F10 args validation `shutil.disk_usage("/dev/shm")` 检查):
    确保 `/dev/shm` 容量 ≥ `bucket_size + 256MB safety margin`
  - **写失败 fail-fast**: wrapper 内 `open(path, "wb").write(...)` 若 raise OSError
    (No space left on device 28), **不 retry**, 直接 raise → MilesModelUpdateService.
    sync_selected_workers raise → coordinator → pipeline crash. 与 F5+6 hardening
    策略一致 (fail-fast > silent retry).

  TP fan-out 由 SGLang scheduler 内部按 `tp_rank` 分片处理, 不在 plan scope。

  **SGLang vendored fork patch 范围**（patches 落在新 feature 目录
  `external/sglang/3rdparty/cpu_serialize/`，与现有 `external/sglang/3rdparty/amd/profiling/*.patch`
  并列；不复用 amd 目录，不创建 `vX_patch` 版本目录）。patch 文件命名按 SGLang 上游
  目标文件分组：

  - `cpu_serialize_weight_sync_utils.patch` → 修改
    `external/sglang/python/sglang/srt/weight_sync/utils.py`：新增 in-process
    cpu_serialize load 路径，跳过 `MultiprocessingSerializer + monkey_patch_torch_reductions`，
    直接走 `torch.load(BytesIO, weights_only=True)` → `pin_memory().to(device,
    non_blocking=True)` → `named_tensors_from_bucket(...)` → 现有 `load_weights` /
    weight_loader
  - `cpu_serialize_scheduler_dispatch.patch` → 修改
    `managers/scheduler_update_weights_mixin.py` + `managers/tp_worker.py`：增加 dispatch
  - `cpu_serialize_engine_base.patch` → 修改 `entrypoints/EngineBase.py`：补充
    `update_weights_from_cpu_bucket` 方法签名
  - **`cpu_serialize_http_route.patch`** → 修改 `entrypoints/http_server.py`：新增
    `POST /update_weights_from_cpu_bucket` admin route, body 携带 `payload_path`
    (str, tmpfs `/dev/shm/miles_cpu_bucket_{uuid}.pt`) + metadata (`load_format`,
    `flush_cache`, `cpu_serialize_local_ranks`); **NO `weight_version` field** —
    per-sync version publish is the atomic unit's step (c) (Fix #3).
    **handler 进入 既有 `tokenizer_manager.model_update_lock.writer_lock`**
    (mirrors `update_weights_from_tensor` 路径; in-flight request admission 在
    weight copy 期间暂停, 避免 active-refresh 下半新半旧权重被 serving). handler
    内 `pathlib.Path(payload_path).read_bytes()` 同步读 tmpfs file 后 dispatch
    到 scheduler. patch scope 必须包括 lock 获取 + try/finally release, NOT
    just an HTTP entry point that bypasses it. (P0-2: MILES wrapper 经 HTTP 调
    子进程 SGLang server, 与既有 `update_weights_from_tensor` 同架构; subprocess
    不持有 Ray context)

    **HTTP route 同步语义 + 写者锁 (Critical Invariant)**:
    `POST /update_weights_from_cpu_bucket` route handler **必须**:
    0. **进入 SGLang 既有 `tokenizer_manager.model_update_lock.writer_lock`**
       (mirrors `update_weights_from_tensor` 路径, 既有 SGLang `tokenizer_control_mixin.py`
       已实现该锁 — 不要重发明). active refresh 语义 (Fix #3 from prior round)
       允许 engine 边 serving 边 load, 但 in-flight request admission 在
       weight copy 期间必须暂停, 否则部分请求会观察到半新半旧权重 → 不可接受.
       writer_lock 是这条暂停边界的唯一 source of truth. handler `try/finally`
       保证 unlock (含异常路径 / receiver crash / network timeout).
    1. 进锁后, 同步 `pathlib.Path(payload_path).read_bytes()` 完整读完文件到
       in-process bytes, **再** dispatch 到 scheduler/tp_worker.
    2. **route 返回点 = 所有 TP workers 完成 `update_weights_from_cpu_bucket` for this
       bucket**. 即 `scheduler_update_weights_mixin.py` patch 内的 dispatch 必须 block
       等待 `collective_rpc(...)` ack 全 TP rank, 才让 HTTP handler return 200
       (writer_lock 在 return 前才 release).
    3. Server 端**不持有** `payload_path` beyond request lifetime — 不入队列, 不交后台
       线程, 不缓存. read_bytes 后 path 可立即被 wrapper unlink (wrapper finally 内执行).
    4. Receiver 端 (TP workers) 读 bytes 后, 若发现已 corrupted / size mismatch (e.g.
       wrapper 在 race condition 下 unlink 早), 立即 raise → propagate 到 HTTP 5xx →
       wrapper raise → sync_selected_workers raise → pipeline crash (fail-fast).
       writer_lock 在 finally 路径仍 release.

    **禁止**:
    - async return + 后台 task pattern (Celery / asyncio.create_task / Thread executor 都不行)
    - HTTP timeout 中断后续 sync (timeout 必须 ≥ 单 bucket worst-case load 时间; F5+6 既有
      `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` 兜底)

  bump SGLang 版本时，patches 与上游 source 不兼容则**重新 rebase**单个 patch 文件，
  不需要新建版本目录。

  **Receiver ingress (必须走 HTTP route, 与 MILES 既有 wrapper 架构对齐)**：

  MILES `SGLangEngine` Ray actor 是 SGLang HTTP server (subprocess) 的 wrapper, **没有
  in-process 的 SGLang Engine**. 既有 `update_weights_from_tensor` 既走 HTTP POST 到
  `/update_weights_from_tensor` ([sglang_engine.py:282-305](external/miles/miles/backends/sglang_utils/sglang_engine.py#L282-L305)).
  cpu_serialize 必须沿用同一架构: 新增 SGLang admin route
  `/update_weights_from_cpu_bucket` (vendored fork patch
  `cpu_serialize_http_route.patch` 加), `SGLangEngine.update_weights_from_cpu_bucket`
  Ray actor method 通过 `self._make_request("update_weights_from_cpu_bucket", payload)`
  POST 到 child SGLang server, server side dispatch 到 scheduler/tp_worker 走
  `cpu_serialize_scheduler_dispatch.patch` 装的 in-process load 路径.

  payload 跨进程方式 + lifecycle 见上文 "payload bytes 跨进程方式 — tmpfs file path"
  段 + Receiver 路径 pseudo-code + Ray ObjectRef GC 不变量 + tmpfs file lifecycle
  (Critical Invariant) + tmpfs 并发上界 (Critical Invariant). 此处不重复, 仅列**明确禁止**:
  - 不传 ObjectRef ID 到 HTTP body / 不让 SGLang server `ray.get(ObjectRef)`
  - 不要求 SGLang server process 是 Ray-aware (subprocess 不调 `ray.init`)
  - 不引入 SHM ZMQ frame fallback (单一 tmpfs 路径足够)
  - 不扩展 `/update_weights_from_tensor` 的 `transport` 字段 (cuda_ipc / cpu_serialize
    走分开 HTTP route, 协议解耦)
  - 不保留 HTTP fallback / debug path (失败直接 raise)
  - cpu_serialize 路径不复用 `load_format="flattened_bucket"`（那是 SGLang
    `FlattenedTensorBucket` 内部协议，与 ROLL wire format 不兼容）

  **背景（仅适用 Case B）**：MILES 是首个把 cpu_serialize 引入 SGLang 链路的实现。
  ROLL 的 cpu_serialize 仅覆盖 vLLM（`infer_strategy == "sglang"` 分支恒走
  `MultiprocessingSerializer` + CUDA IPC）；SGLang 上游 `weight_sync/utils.py` 也只
  支持 `MultiprocessingSerializer`。CUDA IPC 等价路径在受限容器（vast.ai 等无
  CAP_SYS_PTRACE / `--ipc=host`）下不可用. 因此 **M11.1 first build target = vast.ai
  受限容器, F10 强制 `args.model_update_transport == "cpu_serialize"`** (本 Case B 是
  M11.1 唯一 colocate transport). M11.6 production cluster 加 cuda_ipc 作为 colocate
  替代, F10 relax 为 `("cuda_ipc", "cpu_serialize")` 二选一 + cuda_ipc 启动期 smoke
  test capability check (不写脆弱环境探测器). M11.6 cuda_ipc 用新 adapter (见 Case A),
  **不复用既有 `_send_to_colocated_engine`** (后者依赖 live `dist.gather_object` 与
  F4 destroy NCCL 顺序冲突). standalone mode 仍可走既有 `_send_to_colocated_engine`
  cuda_ipc 路径 (无 destroy NCCL 依赖).

  **LoRA → M11.4 (out of M11.1 scope)**: M11.1 仅做 base model weight sync.
  cuda_ipc 既有 `load_lora_adapter_from_tensors` 不动 (standalone path 用); RLix mode
  cpu_serialize / cuda_ipc LoRA adapter 推迟到 M11.4 (与 multi-stream aggregation impl 同
  hardening milestone).

##### 两条 colocate 路径共享的属性

无论选 cuda_ipc 还是 cpu_serialize：

- F4 §5 per-bucket barrier / `_cache_lock` invariant 同样适用（MilesModelUpdateService
  在外层 wrap，wire format 透明）
- Cache owner 由 `_is_distributed_src_rank` 决定（F4 §1）
- F5+6 receiver-side `finalize_weight_update` 由 `MilesModelUpdateService.sync_selected_workers`
  内部 fan-out (atomic unit), 与 transport 无关; pipeline 不直接调
- `update_weight_from_distributed/broadcast.py` 与 `p2p.py` 走 NCCL，不依赖任一种
  colocate transport，**两条 colocate transport 都不影响这条路径**
- **Non-colocate（NCCL broadcast 路径）**：bucket 在 cache owner 上 H2D staging 后，
  逐 bucket `dist.broadcast(staging_tensor, src=cache_owner_global_rank,
  group=temp_nccl_group)`。这里的 `cache_owner_global_rank` 是 `_is_distributed_src_rank`
  对应的全局 rank id，不是 bool flag。
  **直接用 raw `dist.broadcast`，不引入命名 helper**。先前提到的
  `packed_broadcast_producer/consumer` 仅存在于 [nemo-rl/utils/packed_tensor.py:39,98](external/nemo-rl/nemo_rl/utils/packed_tensor.py#L39)，
  **不可直接 import 到 MILES**（跨 framework dep）；如未来需要抽象成命名 helper，再单独
  提案。bucket 内 packing 已经在 build_cpu_bucket_cache 阶段完成（contiguous CPU
  tensor），broadcast 阶段不再额外 packing。

**Comm plan 与动态 NCCL group (TCP rendezvous, 跨进程必需 — Fix #7)**：

- 参照 ROLL `_build_comm_plan_for_sender` 分类逻辑（cpu_serialize vs broadcast，
  per-engine `cpu_serialize_local_ranks` / `broadcast_local_ranks` mask）
- **不能用 `dist.new_group`**: cache_owner Megatron actor 与 SGLang TP workers
  是独立进程, 不共享 default process group. `dist.new_group` 只能在已 init
  default PG 的 ranks 内 carve subgroup, 跨进程不可用. 必须 TCP rendezvous.
- **Sender 侧 (cache_owner)**: 在 `run_sync_session` 内, 通过 MILES 既有 helper
  `miles/utils/distributed_utils.py` 起 TCP-init group:
  `init_process_group(backend='nccl', init_method=f'tcp://{master_addr}:{master_port}',
   world_size=N, rank=0, ...)`. 这与 MILES 既有
  [broadcast.py:137](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py#L137)
  的 TCP init pattern 完全一致, 直接复用 helper.
- **Receiver 侧 (每个 non-colocate target SGLang server)**: 通过 SGLang 既有
  `init_custom_process_group`
  ([model_runner.py:1783](external/sglang/python/sglang/srt/model_executor/model_runner.py#L1783))
  以 rank ≥ 1 加入同一 TCP group. service 在 SyncSessionPlan 里给每个 broadcast
  receiver 分配 `comm_ranks[engine_idx]`, sender 是 rank 0, receivers ≥ 1.
- **`master_addr` / `master_port`**: service 在 sync entry pre-claim
  (`get_free_port()` + SharedStorage `MASTER_ADDR_PORT:*` claim) 后填入
  `SyncSessionPlan.master_addr` / `.master_port`. sender 与所有 receiver
  共享同一对 (addr, port).
- `group_name = f"miles_model_update_{sync_id}"`，sync_id 含 pipeline_id +
  step + uuid，避免跨 pipeline / 跨调用冲突
- 临时 group 每次 sync 后 destroy，4 步生命周期 CLASSIFY/CREATE/USE/DESTROY：
  - **CREATE 步骤补 warmup allreduce**: TCP group rendezvous 完成后立即发
    `dist.all_reduce(torch.zeros(1, device='cuda'), group=tmp)` 验证 group
    健康; 失败立即 `dist.destroy_process_group(tmp)` + raise，不进入 USE 阶段
  - **DESTROY 步骤**: sender + 各 receiver 都调 `dist.destroy_process_group(tmp)`;
    colocate-only ranks (cpu_serialize 路径, 没 join broadcast group) 必须用
    既有 `is_group_exist` no-op guard, 走错路径不能炸
  - **port claim 释放**: sync 完成（含成功 / sync 内部 fail-fast crash）后
    `shared_storage.release_port_claim(master_port)`，port 回 pool 供下次
    sync 复用。**本 milestone 不处理 receiver crash 容错**——sync 路径上的
    fail-fast 由 F5+6 `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` (复用 RLix 既有
    env var, 不再造 MILES_*) 触发整 pipeline crash，pipeline restart 时
    SharedStorage 整体清理 port pool；不需要 conditional leak 设计。如未来生产
    观察到 receiver crash 导致的 port collision，再单独立 plan 加 fault-tolerant
    port 管理

**`master_port` 选择 (NCCL master_port invariant)**:
service 在 setup 阶段调 `get_free_port()` (PyTorch helper / ROLL 等价) 获取确定 concrete
port, 通过 SharedStorage `try_put("MASTER_ADDR_PORT:{addr}:{port}", pipeline_id)` 原子
claim, 然后**同一 port 值传给 sender 与所有 receiver** 的 `setup_collective_group(...)`.
**禁止 `master_port=0`** — `tcp://addr:0` 对 multi-rank rendezvous 不成立: rank0 OS-bind
ephemeral port 后, 其它 rank 只看到 "port 0" 无法连. EADDRINUSE 时 retry 新 free port
即可; cooldown queue / port pool 的 TIME_WAIT 对策是 follow-up.

改动量：~290 行（CPU bucket build + ModelUpdateService routing 层 + 动态 NCCL group
生命周期 + transport 适配 + warmup allreduce + sync_id；不需要重新发明 bucket format；
**不含 receiver crash 容错 / 条件 port leak**）

**新增文件：**
- `rlix/pipeline/miles_model_update_service.py`（简化版 ModelUpdateService）
- `miles/backends/megatron_utils/update_weight/cpu_bucket_cache.py`（CPU cache build + lookup）

---

### Feature 5+6: Two-path weight refresh (active in-flight + expand sync) + version accounting

**作用：** 解决 partial overlap 下非重叠 active engine 的权重更新问题。

#### 核心差异（与 NeMo RL F5+F6 一致）

| 路径 | Engine 状态 | Trigger | 内部机制 |
|---|---|---|---|
| **Active refresh** | 非重叠 active engines | training loop（`after_training` hook） | pipeline → `coordinator.sync_base_weights_to_active(step)` (**显式传 step**, coordinator 在 `_resize_sync_lock` 内同步更新 `_cache_ready_step`, 避免 stale version label inversion) → `service.sync_selected_workers(sync_id, active_engines, step)` |
| **Expand sync** | 重叠 slept/woken engines（含 shell→active lazy ctor; 见 F4） | scheduler（`resize_infer(add=...)`） | coordinator `_expand_workers()` → manager `expand_engines` (lazy ctor + onload) → `service.sync_selected_workers(sync_id, woken_engines, version)` → manager `activate_routing` |

**不变量：** 所有已 active 的 engine 由 training loop 刷新；所有后续被激活的 engine 由
expand 刷新。两条路径共享同一份 CPU bucket cache（单 cache owner）。

**Atomic unit (两条路径共用):** `MilesModelUpdateService.sync_selected_workers(sync_id, target_engines, version)`
内部对 `version >= 0` 完成: (a) per-bucket transport (per-bucket payload
**不携带 weight_version** — version inversion guard, Fix #3) + (b) per-engine
`finalize_weight_update.remote()` fan-out + (c)
`rollout_manager.set_weight_version(version, engine_indices=target_engines)`
一次性 publish version label. 返回 version.

**`version == -1` short-circuit (Fix #13 + #15)**: 跳过 (a) per-bucket
transport, 跳过 (b) finalize fan-out, 跳过 `cache_owner_actor.run_sync_session`.
**只跑 (c) `manager.set_weight_version(-1, target_engines)`**. 因为 v=-1 的
target engines 已经在 `manager.expand_engines` lazy ctor 里通过 SGLang 自身的
`onload()` 从 `args.hf_checkpoint` 加载了 base weights (post-load hook
`process_weights_after_loading` 已经跑过), 不需要再 transport 字节, 也不需要
再 finalize. 跳过 transport 是必需的, 因为 base sync 可能在 `actor_train`
allocation 已被 release 给 scheduler 之后跑 (Fix #1+#2 init Step 6 Phase 1
finally), cache_owner GPU 可能已被别的 pipeline / actor_infer 占用 → broadcast
H2D staging 会 OOM 或非法并发. 跳过 finalize 是因为没有 pending bucket update
要 finalize (Fix #15).

Pipeline 不直接调用 `finalize_weight_update` 或 manager-level `set_weight_version`.
Coordinator 是 expand 的唯一 orchestrator. `RolloutManager` 不调 service.
service 调 manager 限于: 两个 read-only entries (`get_engine_handles`,
`get_engine_count`) at sync entry + 一次 write (`set_weight_version`) at sync exit
(Fix #5).

#### 为什么不复用 MILES 现有 standalone weight update path (`RolloutManager.update_weights_from_distributed/tensor`)

[miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py:157](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py#L157)
`update_weights_from_distributed`：

1. **无子集定向** — 当前路径假设全量 engine 参与
2. **需要 GPU 张量** — 直接从 GPU 张量 broadcast，无法读 CPU cache
3. **全局 barrier** — 一次性 fan-out，无逐 engine 完成信号

Feature 4 的 `ModelUpdateService.sync_selected_workers()` 已解决这三个问题。两条路径
复用同一传输机制。

#### Active refresh 安全模型

非重叠 engine 在**继续 serving 的同时**接收权重。无 routing 移除，无 drain，无 idle
等待。

这不是整体 best-effort 语义。与 NeMo RL `in_flight_weight_updates=True` 同类，只允许
active refresh 边界存在短暂 request-level version attribution 过渡窗口：weight push
时刻已在 SGLang engine 内 in-flight 的请求可能仍以旧权重生成，但外层 trajectory
bookkeeping 已接近新 version。**这类边界误标可容忍，未消除**。SGLang 端权重 load
路径与 NeMo vLLM 的 `load_weights()` 同样原始（逐参数 `param.data.copy_(...)`），不做
引擎级暂停。

**`tokenizer_manager.model_update_lock.writer_lock` 是 in-flight 边界的硬约束**
(Fix #4): 新增 `update_weights_from_cpu_bucket` HTTP route 必须 acquire 该锁,
mirrors 既有 `update_weights_from_tensor` 路径. 锁的语义不是"暂停所有请求", 而是
"暂停新请求 admission, 让正在 decode 的请求自然完成 (个位数 ms 级)". 没有这个锁,
load 期间新进来的请求会观察到半新半旧权重 — 不是过渡窗口的"边界误标", 而是结构性
错误. 不可接受. 锁的获取在 SGLang `tokenizer_control_mixin.py` 已实现 — 不要重发明,
patch scope 仅做 lock 获取 + dispatch + try/finally release.

误标数量受 in-flight batch size 与单次 decode step 延迟约束（个位数请求量级），不与
fully_async 的 staleness window 混淆。

refresh 完成后的语义必须是 exact-at-engine/trajectory-granularity：

- refreshed engine 必须完成 receiver-side barrier / finalize，并发布
  `_current_weight_version == _cache_ready_step`
- 后续 trajectory 的 `weight_versions` 必须能被 `--max-weight-staleness` 可靠消费
- staleness cutoff 允许丢弃旧 trajectory，但不能替代 refresh 完成后的 version publish

Drain-then-sync **不在本移植方案范围内**。

#### Hardening：timeout + fail-fast

- **直接复用 RLix `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` env var (default 150 秒,
  [rlix/pipeline/model_update_service.py:55](rlix/pipeline/model_update_service.py#L55))**。
  **不再造 `MILES_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` 同名常量** —— `MilesModelUpdateService`
  在 RLix process 内运行, 直接读 RLix 既有 env var, 同 process 两套同义命名只会埋雷。
  `MilesModelUpdateService` 在 `sync_selected_workers()` 整体外加 `asyncio.wait_for` /
  `ray.get(timeout=)`. 超时 → raise → coordinator → pipeline crash (**fail fast, no
  retry**). Sender-side `_cache_lock` 持有期间也受 timeout 约束.
- **不预先拆 active refresh / expand 两个常量**：active refresh 面向 serving workers
  （GPU 争用），expand 面向 idle workers（无争用），理论上 active 可能需要更宽 timeout，
  但**没有实测证据**支持预先拆分。先用单 const 跑 Gate 3，如真观察到 active refresh
  逼近 150s 上限再拆，YAGNI。

#### Control-plane 不变量

**Pipeline 在 `sync_base_weights_to_active(step)` 返回之前，不得调用
`self._notify_release_cluster_gpus(cluster_id="actor_train", ...)`
（[full_finetune_pipeline.py:497](rlix/pipeline/full_finetune_pipeline.py#L497) 已有
方法）。** sync_base_weights_to_active 已在 `_resize_sync_lock` 内同步: 更新
`coordinator._cache_ready_step` + service.sync_selected_workers atomic unit
(per-bucket transport + finalize fan-out + per-engine set_weight_version);
返回时 active engines 权重已对齐, coordinator state 已发布。GPU 释放信号表示
"我的 active engines 权重一致 + coordinator 已发布最新 version"，不是"训练完成"。
由 `after_training` hook 序列强制。

#### Active set bootstrap (Critical Invariant — 否则首次 active refresh 短路)

`MilesCoordinator._active_engine_indices: Set[int]` 默认空集 (对齐 [coordinator.py:218
`_active_infer_dp_ranks: Set[int] = set()`](rlix/pipeline/coordinator.py#L218))。**只有
`resize_infer()` 会写它**；初始 `request_cluster_gpus(actor_infer, GENERATION)` 不会
roundtrip 到 coordinator。如果不显式 bootstrap, 首次 `sync_base_weights_to_active()` 读到
空集 → 立即短路返回 → 非重叠 active engine 永远停在 base weight, F5+6 不变量直接失效。

**新增 `MilesCoordinator.bootstrap_active_engines(engine_indices: Set[int]) -> None` RPC**:

```python
# rlix/pipeline/miles_coordinator.py
def bootstrap_active_engines(self, engine_indices: Set[int]) -> None:
    """Set initial active engine set after actor_infer GENERATION allocation.

    Called exactly once by MilesPipeline.initialize_pipeline() Step 10 after
    actor_infer creation succeeds. `engine_indices` is the *initially-active*
    set: under M11.2 partial GENERATION allocation (Gate 4 contention),
    this is a strict subset of `range(engine_count)` — remaining engines are
    in `shell` state on `RolloutManager` and will activate via subsequent
    `resize_infer(add=...)` flowing through `_expand_workers`. Subsequent
    updates flow through `resize_infer`.
    """
    with self._resize_sync_lock:
        if self._active_engines_bootstrapped:
            raise RuntimeError(
                f"bootstrap_active_engines called twice: "
                f"existing={self._active_engine_indices}, new={engine_indices}"
            )
        self._active_engine_indices = set(engine_indices)
        self._active_engines_bootstrapped = True
```

**`_active_engines_bootstrapped: bool` flag (硬约束)**: 必须用显式 flag 而非
`if self._active_engine_indices:` 真值检查. 原因: M11.2 init 流程下
bootstrap 入参可能是 `frozenset()` (空集) — coordinator 等 `_expand_workers`
在 Step 7 期间填充. 若用真值检查, 空集 bootstrap 与 "未 bootstrap" 在状态
上不可区分, 第二次 bootstrap 会静默覆盖空集 (真实的 ordering bug 漏报).

按 F4 init bootstrap Step 6.6f 调用 — 在 Step 7 之前. 重复调用 fail fast
(避免静默覆盖, 包括空集场景).

#### Actor call graph

避免 pipeline actor re-entrant self-call（参照 NeMo `sync_lora_weights` 模式）：

```
Pipeline init (initialize_pipeline Step 6.6 - 7.3):
  # CRITICAL: scheduler executes coordinator.resize_infer(add) RPC in
  # Phase 5 BEFORE signaling pending GEN waiter in Phase 6. Therefore
  # everything _expand_workers needs (manager, service resources, base
  # cache_ready_step, bootstrap'd active set, empty router) MUST be in
  # place at coordinator BEFORE Step 7 fires.

  # ----- Step 6.6: pre-register everything _expand_workers needs -----
  ├── all_placements = provider.get_all_rollout_engine_placements()
  │     # length == engine_count, declared, no scheduler call
  ├── self.actor_infer = RolloutManager.options(...).remote(
  │       args, all_engine_placements=all_placements,
  │       active_engine_indices=frozenset())
  │     # all-shell ctor; empty Miles router started; per-pipeline router
  │     # port bound on self._args; no Ray/SGLang actors spawned yet
  ├── ray.get(self.actor_infer.get_engine_count.remote())  # sanity
  ├── ray.get(coordinator.register_model_update_resources.remote(
  │       cache_owner_actor=self._cache_owner_actor,
  │       rollout_manager=self.actor_infer))
  │     # service ctor args cached; lazy-create on first sync
  ├── ray.get(coordinator.publish_cache_ready_step.remote(-1))
  │     # base version label; v=-1 short-circuit (Fix #13/#15) won't actually
  │     # touch cache bytes, but version label MUST be available before
  │     # _expand_workers reads it
  └── ray.get(coordinator.bootstrap_active_engines.remote(frozenset()))
        # _active_engines_bootstrapped flag flips True; _active_engine_indices
        # remains frozenset() until _expand_workers populates during Step 7.

  # ----- Step 7: actor_infer GENERATION request -----
  └── allocated = self._request_cluster_gpus(actor_infer, GENERATION, ...)
        # During this blocking call:
        # scheduler.run_phase5: coordinator.resize_infer(add=[<dps>]) →
        #   coordinator._expand_workers([<dps>]):
        #     - reads _cache_ready_step = -1 ✓
        #     - reads _model_update_resources ✓
        #     - manager.expand_engines: shell→loading lazy ctor + onload
        #     - service.sync_selected_workers(sync_id, dps, version=-1):
        #         v=-1 short-circuit — only manager.set_weight_version(-1, dps)
        #         (NO run_sync_session, NO finalize, NO cache_owner GPU touch)
        #     - manager.activate_routing(dps): loading→active + add_worker to router
        #     - _active_engine_indices |= set(dps)
        # scheduler.run_phase6: signal waiter; allocated returns to pipeline.

  # ----- Step 7.1-7.3: post-request validation -----
  ├── M1 subset assert: set(allocated) ⊆ set(declared); log is_partial
  ├── active = provider.get_active_engine_indices(allocated, tp_size)
  │     # half-engine raise; pipeline-side derivation
  └── coord_active = ray.get(coordinator.get_active_engines.remote())
        assert coord_active == active   # consistency: _expand_workers ↔ provider

Pipeline actor (after_training, step=N):
  ├── # build_cpu_bucket_cache(N) 已 overwrite cache, 但 coordinator 仍持旧 version.
  │   # **必须把新 version 显式传进 sync 调用**, 不能让 coordinator 读 stale state.
  │   version = ray.get(coordinator.sync_base_weights_to_active.remote(N))
  │     └── Coordinator actor:
  │           acquire _resize_sync_lock
  │           # 在 lock 内同步更新 coordinator-side version label,
  │           # 与 service.sync_selected_workers 是同一原子操作:
  │           # 这样 sync 进行时 _cache_ready_step 已是新值,
  │           # 任何并发 _expand_workers 看到的也是新值.
  │           self._cache_ready_step = N
  │           active_engines = _active_engine_indices   # bootstrap 后非空
  │           if not active_engines:
  │               release _resize_sync_lock
  │               return N   # 全重叠退化拓扑 (Edge case 1); version 仍发布
  │           sync_id = make_sync_id(pipeline_id, step=N)
  │           ray.get(model_update_service.sync_selected_workers.remote(
  │               sync_id, active_engines, N))
  │             # service 内部: per-bucket transport + per-engine finalize fan-out
  │             # + rollout_manager.set_weight_version(N, engine_indices=active_engines)
  │             # 是单一 atomic unit; ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S 包整个调用.
  │           release _resize_sync_lock
  │           return N    ← 不回调 pipeline; pipeline 用 return value 更新 local
  ├── self._current_weight_version = version      ← 本地, 无 remote call
  └── notify_release_cluster_gpus(actor_train)
        # 不再额外调 publish_cache_ready_step — sync_base_weights_to_active 已在
        # _resize_sync_lock 内更新了 coordinator state. publish_cache_ready_step
        # 仍保留, 仅用于 init bootstrap 发布 -1 (Pipeline B 在 first after_training 前
        # 就可能被 expand). 退化拓扑下 sync 短路返回也已更新 coordinator state.

Coordinator actor (resize_infer, scheduler-driven, no pipeline callback):
  └── _expand_workers(target_engines):
        acquire _resize_sync_lock
        if self._cache_ready_step is None:
            raise RuntimeError("expand before publish_cache_ready_step")
            # init bootstrap 已发布 -1, 不该到这里; 防御 ordering bug.
        ray.get(rollout_manager.expand_engines.remote(target_engines))
            # manager: shell→loading (lazy ctor + onload) 或 offloaded→loading (wake_up).
            # 不调 service.
        sync_id = make_sync_id(pipeline_id, "expand", step)
        ray.get(model_update_service.sync_selected_workers.remote(
            sync_id, target_engines, self._cache_ready_step))
            # 同 atomic unit; finalize + set_weight_version 内含.
        ray.get(rollout_manager.activate_routing.remote(target_engines))
            # manager: loading→active + open routing.
        self._active_engine_indices |= set(target_engines)
        release _resize_sync_lock
```

#### Training step 序列（finalize + set_weight_version 由 `MilesModelUpdateService.sync_selected_workers` 内部驱动，atomic unit）

```
0a. _request_cluster_gpus(cluster_id="actor_train",
                          priority=Priority.ACTOR_TRAINING,
                          global_step=N)        ← scheduler may block; 对 actor_train
                                                   partial allocation **INVALID** —
                                                   assert set(allocated)==set(declared)
                                                   fail-fast (training 是单一 Megatron
                                                   world, no half-train semantics)
0b. actor_train.onload()                        ← wake_up() on every Megatron actor +
                                                   ReloadableProcessGroup.reload_process_groups();
                                                   mirrors init Step 3 (§J Step 3,
                                                   等价 ROLL load_states(blocking=True)).
                                                   ROLL 的 strategy 在 acquire 时
                                                   auto-wake，MILES `RayTrainGroup` 没
                                                   auto-wake hook，所以这一步必须显式调
1. train_step()
2. build_cpu_bucket_cache(step)             ← 所有 training rank gather；cache owner 存
                                               (cache bytes 翻新, 但 version label 还没发布)
3. (本地) _cache_ready_step = step          ← 仅 cache_owner actor 自身的 version 标签;
                                               coordinator 仍是上一 step 的旧值,
                                               必须由 Step 5 显式传 step 进去更新
4. offload training GPU / destroy NCCL groups（ReloadableProcessGroup）
5. version = coordinator.sync_base_weights_to_active(step)   ← **必须传 step**, 否则
                                               coordinator 用 stale `_cache_ready_step`
                                               把刚翻新的 cache bytes 标成旧 version 推下去
                                               (version inversion; service 写新权重但
                                               receiver 看到旧 label).
   ← coordinator 内部 (单 _resize_sync_lock 临界区):
       coordinator._cache_ready_step = step       ← 与 service 调用同一原子操作; 之后
                                                    任何并发 _expand_workers 看到的也是
                                                    新 version
       service.sync_selected_workers(sync_id, active_engines, step) atomic unit:
         (a) per-bucket transport (update_parameter_in_bucket / broadcast_parameter)
         (b) per-engine finalize_weight_update.remote() fan-out
             (一次性 post-load hook on each target engine; 不在 receiver 自驱,
              不在 sender 内嵌; 对齐 ROLL sync_selected_workers → process_weights_after_loading
              → load_states_partial 顺序)
         (c) rollout_manager.set_weight_version(step, engine_indices=active_engines)
       整个 (a)+(b)+(c) 包在单次 ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S 内
   coordinator 返回 step (作为 published version).
6. self._current_weight_version = version (本地)        ← 不再单独发 publish_cache_ready_step;
                                                          Step 5 已在 lock 内更新 coordinator state
7. notify_release_cluster_gpus(actor_train)
8. (later) scheduler resize_infer(add=...)  ← 同一 cache 推 woken overlap engines；
                                                 service.sync_selected_workers atomic unit
                                                 同样覆盖 sync + finalize + set_weight_version,
                                                 由 coordinator._expand_workers 驱动 (no
                                                 pipeline callback; 见 actor call graph).
                                                 expand 读 coordinator._cache_ready_step,
                                                 因 Step 5 已发布所以读到的是当前 step.

下一个 step 从 0a 重新进入. 第一个 step (N=0) 的 0a/0b 由 main loop 在 init 完成后
显式调用 — init 末尾 Step 5 已 offload，所以 step 0 也需要 0a/0b 完整序列.
```

#### Sequence diagram

与 NeMo plan F5+6 同形态的 ASCII 时序图（4 列：Scheduler / Coordinator / Pipeline /
SGLang Engines dp0..dp3，覆盖 partial overlap 场景 dp0/dp1 重叠、dp2/dp3 非重叠）。

**API 注**：`resize_infer(dp_ranks_to_remove, dp_ranks_to_add)` 是单方法签名，约定
exactly one non-empty（[coordinator.py:502](rlix/pipeline/coordinator.py#L502)）。
下方 `rm=` / `add=` 仅是表达哪一参数非空的简写。

```
Scheduler         Coordinator       Pipeline                    SGLang Engines
                                                                dp0 dp1 dp2 dp3
                                                                ●   ●   ●   ●  (v2)

resize_infer(dp_ranks_to_remove=[0,1], dp_ranks_to_add=[])
─────────────────>   lock
                     ───────────>   _shrink_workers([0,1])
                                    ───────────────────────> 😴  😴  ●   ●
                     _active={2,3}
                     unlock

                                    before_training(step=3):
                                      _request_cluster_gpus(actor_train,
                                                            ACTOR_TRAINING)  ← may block;
                                                                                partial INVALID
                                      actor_train.onload()                  ← wake + reload NCCL

         [training on overlap GPUs 0,1]                       😴  😴  ●   ●  (dp2,3 serve v2)

                                    after_training(step=3):
                                      build_cpu_cache(step=3)
                                      _cache_ready_step = 3
                                      offload training GPU
                                      destroy NCCL groups

                     <─ sync_base_weights(step=3) ─── ray.get(coordinator.sync_base_weights_to_active(3))
                     lock
                     coordinator._cache_ready_step = 3   ← lock 内同步发布, 避免 stale label
                     _active={2,3}
                     ── service.sync_selected_workers(sync_id, [2,3], v=3) ──> 😴  😴 ●→v3 ●→v3
                     │     atomic: per-bucket transport
                     │           + finalize_weight_update.remote(2,3) fan-out
                     │           + rollout_manager.set_weight_version(3, [2,3])
                     unlock
                     ── return v=3 ─>
                                    self._current_weight_version = 3   (本地)
                                    self._notify_release_cluster_gpus(cluster_id="actor_train", ...)

resize_infer(dp_ranks_to_remove=[], dp_ranks_to_add=[0,1])
─────────────────>   lock
                     guard: _cache_ready_step is not None  (init 已发布 -1)
                     ───────────>   _expand_workers([0,1]):
                                    ── manager.expand_engines([0,1])
                                       ├── shell→loading: lazy ctor + onload (Fix #4)
                                       └── offloaded→loading: wake_up         ⏳  ⏳  ●v3 ●v3
                                    ── service.sync_selected_workers(sync_id, [0,1],
                                                                     v=_cache_ready_step=3)
                                       atomic: sync + finalize_weight_update.remote(0,1)
                                               + rollout_manager.set_weight_version(3, [0,1])
                                                                          ✓v3 ✓v3 ●v3 ●v3
                                    ── manager.activate_routing([0,1])  ●v3 ●v3 ●v3 ●v3
                     _active={0,1,2,3}
                     unlock
```

#### Edge cases

1. **全部 engines 重叠（退化 = ROLL 拓扑）**：shrink 后 `_active_engine_indices` 为空。
   `sync_base_weights_to_active()` 立即短路返回（active set 为空）。Expand sync 所有
   engines on wake。**正确**。
2. **无重叠（所有 engines 非重叠）**：不发生 shrink/expand。`sync_base_weights_to_active()`
   in-flight sync 所有 engines。**正确**。
3. **Init 后首步 (v=-1 short-circuit)**：CPU cache 含 base weights
   （`_cache_ready_step = -1`，由 init Step 6.6e bootstrap 提供）。Active refresh /
   expand 路径 **不推送 bytes** (Fix #13+#15) — target SGLang engines 已通过
   `manager.expand_engines` lazy ctor 内的 `onload()` 从 `args.hf_checkpoint` 加载
   base, 仅需 `manager.set_weight_version(-1, target_engines)` publish version 标签.
   service `sync_selected_workers(version=-1)` 跳过 `cache_owner_actor.run_sync_session`
   + 跳过 finalize fan-out, 不触碰 cache_owner GPU (init Step 6 已 release train).
   **正确**。

#### Version accounting（修复 double-bump）

Version = `_cache_ready_step`（绑定到产生 cache 的 training step，不是 sync 操作）：

**注: `RolloutManager.set_weight_version()` 是新增方法** (MILES `RolloutManager` 当前
没有这个方法; 只有 [SGLangEngine.update_weight_version](external/miles/miles/backends/sglang_utils/sglang_engine.py#L538)).
新增 manager-level fan-out **由 `MilesModelUpdateService.sync_selected_workers` 内部
调用**, pipeline 与 coordinator 都不直接调:
```python
# miles/ray/rollout.py RolloutManager
def set_weight_version(self, version: int, engine_indices: list[int] | None = None):
    """Fan-out version publish to target engines (or all if None).

    调用方仅 MilesModelUpdateService.sync_selected_workers (atomic unit 的 (c) 步).
    Pipeline `_after_training` / Coordinator `_expand_workers` 都不直接调用 —
    都经 service.sync_selected_workers(sync_id, target_engines, version) 入口.
    """
    targets = engine_indices if engine_indices is not None else list(self._engines.keys())
    refs = [self._engines[idx].handle.update_weight_version.remote(str(version)) for idx in targets]
    ray.get(refs)  # 等所有 engine 确认 publish, version 才能被 trajectory 消费
```

Active refresh path (training step 3 之后) 的 version 流转 (与 [actor call graph](#feature-56-two-path-weight-refresh-active-in-flight--expand-sync--version-accounting)
+ [Sequence diagram](#sequence-diagram) 一致):
```python
# Pipeline _after_training(step=3):
version = ray.get(coordinator.sync_base_weights_to_active.remote(3))
# coordinator (under _resize_sync_lock):
#   self._cache_ready_step = 3        ← lock 内同步发布, 杜绝 stale label
#   service.sync_selected_workers(sync_id, _active_engine_indices, 3)
#     atomic: (a) per-bucket transport
#             (b) per-engine finalize_weight_update.remote() fan-out
#             (c) rollout_manager.set_weight_version(3, engine_indices=_active_engine_indices)
#   return 3
self._current_weight_version = version   # = 3, pipeline 本地

# Expand path (scheduler-driven, 无 pipeline callback):
# Coordinator._expand_workers(target_engines):
#   under _resize_sync_lock, 用 self._cache_ready_step (= 3, 因 active refresh
#   已发布) 调 service.sync_selected_workers(sync_id, target_engines, 3).
#   service 内部同样跑 (a)+(b)+(c) atomic unit, version 不 bump.
```

MILES 已在 [miles/utils/types.py](external/miles/miles/utils/types.py) `Sample` 上有
`weight_versions`，[miles/rollout/generate_utils/generate_endpoint_utils.py](external/miles/miles/rollout/generate_utils/generate_endpoint_utils.py)
在 train data 上传播 version。直接复用，无需新发明 version tracking。

#### Sender-side：cache_owner Megatron actor API surface (M2 — 跨进程必须暴露)

`MilesModelUpdateService` 是独立 RLix actor (driver-namespace), 不在 Megatron worker
进程内. 所有 sender 路径 (cpu_serialize 序列化 + NCCL broadcast) 必须 RPC 到 cache
owner Megatron worker. Service 在 `__init__` 时通过 `cache_owner_actor` 参数收 actor
handle (init bootstrap Step 6.5 已 collect, 见 §F4 init bootstrap), 后续 sync 路径全部
走 `cache_owner_actor.X.remote(...)`.

| 方法 | 调用者 | 路径 | 时机 |
|---|---|---|---|
| `build_cpu_bucket_cache(step) -> None` | RayTrainGroup fan-out | — | init bootstrap Step 4 + 每 train_step 后; cache owner rank 真存, 其它 rank 参与 collective gather 但丢弃结果 |
| `report_cache_owner_role() -> tuple[int, bool]` | RayTrainGroup `collect_cache_owner_roles()` | — | init bootstrap Step 6.5; 返回 `(global_rank, is_cache_owner)` (基于 `_is_distributed_src_rank`) |
| **`run_sync_session(plan: SyncSessionPlan) -> None`** (Fix #6) | MilesModelUpdateService | composite (cpu_serialize per-engine RPC + NCCL broadcast group setup/per-bucket/teardown all-in-one) | per-sync, only when `version >= 0` (v=-1 short-circuit skips this entirely per Fix #13). cache_owner 在该 method 内 acquire `_cache_lock` (单 method 单临界区, 不跨 RPC), snapshot bucket list + `_cache_ready_step`, dispatch per-bucket transport (cpu_serialize 走 `engine.update_weights_from_cpu_bucket.remote(payload_ref, ...)` per target via `plan.target_handles`; NCCL broadcast 走 TCP-rendezvous group setup + per-bucket `dist.broadcast(src=cache_owner_global_rank, group=tmp_group)` + group destroy), 最后释放 `_cache_lock`. **不返回 ObjectRef bytes; 调用结束 = 所有 buckets 已被所有 targets load 完成.** |
| ~~`get_bucket_count`~~ / ~~`serialize_bucket_to_objref`~~ / ~~`setup_collective_group`~~ / ~~`broadcast_bucket`~~ / ~~`destroy_collective_group`~~ | (旧 split RPCs) | — | **Fix #6 降级为 cache_owner 内部 helpers**, 不再是 top-level Ray methods. `run_sync_session` 在自己 method body 内顺序调用. 这样消除 "service 跨 5 个 RPC 持 actor 锁" 的不可实现矛盾. |
| ~~`serialize_bucket_cuda_ipc(bucket_idx) -> list[str]`~~ | M11.6 milestone | M11.1 不实现 (vast.ai 受限容器无 IPC 能力). M11.6 production cluster 加. |

非 cache_owner rank 的实现策略: `build_cpu_bucket_cache` 参与 gather; `run_sync_session`
和其它 sender 方法 raise (cache owner = pp0+dp0+tp0+cp0, 唯一; service 不应路由到
其它 rank — 走错路径直接 RuntimeError 暴露 bug).

**`SyncSessionPlan` dataclass (frozen)**:

```python
@dataclass(frozen=True)
class SyncSessionPlan:
    sync_id: str
    version: int                      # >= 0 (v=-1 short-circuit doesn't reach here)
    group_name: str                   # NCCL / TCP rendezvous group name
    master_addr: str
    master_port: int
    timeout_s: float
    # Per-sync receiver handles snapshot (Fix #5: provided by service via
    # rollout_manager.get_engine_handles(target_engines) at sync entry).
    target_handles: dict[int, RayActorHandle]   # engine_idx → SGLangEngineActor
    # Per-engine TP-local rank dual-mask (mirrors ROLL classification).
    cpu_serialize_local_ranks: dict[int, list[int]]
    broadcast_local_ranks: dict[int, list[int]]
    # NCCL group rank table for broadcast subset (sender = rank 0; each broadcast
    # receiver gets a stable rank ≥ 1).  cache_owner uses these to drive
    # `dist.broadcast(..., src=0, group=tmp)` per bucket.
    comm_ranks: dict[int, int]        # engine_idx → NCCL rank in this group
                                       # (broadcast receivers only;
                                       #  cpu_serialize-only engines absent)
```

Service builds the plan ONCE per sync (at sync entry, after
`get_engine_handles` snapshot + transport classification). Plan is small
(handles + ints + lists of ints), trivially serializable, frozen so it
can't be mutated mid-RPC. cache_owner has everything needed to drive both
cpu_serialize per-engine RPC AND broadcast group setup/per-bucket/teardown
**without calling back to service or manager**.

Why receiver handles must travel in the plan, not be re-fetched on
cache_owner side: cache_owner is a Megatron worker actor — it has no
reference to `RolloutManager` and shouldn't acquire one (would re-introduce
the manager↔service bidirectional dependency). Service is the only actor
holding both `cache_owner_actor` and `rollout_manager` handles, so service
must build the full plan upfront.

**M11.2 Gate 4 happy path 范围**: Gate 4 (c)/(d)/(e) acceptance 在
**cpu_serialize-only 拓扑**下验证 (4 GPU + tp=2, 所有 receiver 都是 colocate
or走 cpu_serialize). M11.2 不新增 transport 工作 — NCCL broadcast 路径
(`broadcast_local_ranks` + `comm_ranks` 字段 + `dist.broadcast` per-bucket)
是 M11.1 已 land 的 load-bearing 工作 (Gate 2.5 验证), M11.2 透传不动.
SyncSessionPlan 为 future-proof 保留全部字段; 但 **M11.2 acceptance 的
shell engine sync 默认 cpu_serialize**, broadcast-path partial allocation
扩展是 follow-up (only 在 partial allocation × non-colocate 同时出现且实测
吞吐成瓶颈时才考虑). M11.2 实现者可以认为: 新代码路径不接触 broadcast 字段;
所有 cache_owner ↔ shell engine sync 走 cpu_serialize.

#### Receiver-side：target engine API surface

SGLang engine 必须实现（参照 NeMo `vllm_backend.py`）。首个 parity gate 就是 `tp>1`，
因此 receiver API 必须覆盖 mixed receiver mask：同 GPU receiver 按
`args.model_update_transport` 走 cuda_ipc（既有 `update_weights_from_tensor`）或
cpu_serialize（新增 `update_weights_from_cpu_bucket`）；non-colocate receiver 走
dynamic NCCL broadcast；不能先降级成 tp=1-only 路径。

| 方法 | 调用者 | 路径 | 时机 |
|---|---|---|---|
| `setup_collective_group(name, comm_plan, mode, master_addr, master_port, comm_rank, world_size, timeout_s)` | cache_owner via SyncSessionPlan (NOT directly by service) | NCCL via TCP rendezvous (Fix #7) | non-colocate engines join via SGLang `init_custom_process_group` ([model_runner.py:1783](external/sglang/python/sglang/srt/model_executor/model_runner.py#L1783)). cache_owner 是 rank 0; receivers ≥ 1, rank by `plan.comm_ranks[engine_idx]`. **不能用 `dist.new_group`** (跨进程无 default PG); 必须 TCP-init |
| `update_weights_from_tensor(serialized_named_tensors, ipc_local_ranks, ...)` | ModelUpdateService | CUDA IPC | colocate, **签名扩展** ([sglang_engine.py:282](external/miles/miles/backends/sglang_utils/sglang_engine.py#L282) 现有 method 必须新增 `ipc_local_ranks` 参数, tp>1 dual-mask 不可绕过) |
| `update_weights_from_cpu_bucket(payload_bytes, load_format, flush_cache, cpu_serialize_local_ranks)` | ModelUpdateService | Ray ObjectRef bytes (top-level auto-deref → `bytes`) | colocate **新增**，仅 cpu_serialize 路径调用. **NO `weight_version` parameter** — per-sync version publish is the atomic unit's step (c) `manager.set_weight_version(version, target_engines)`, NOT per-bucket per-engine (Fix #3) |
| `broadcast_parameter(group_name, names, dtypes, shapes, broadcast_local_ranks)` | ModelUpdateService | NCCL | 动态 group |
| `destroy_collective_group(group_name)` | ModelUpdateService | NCCL | sync 完后；colocate-only ranks（不论 cuda_ipc 还是 cpu_serialize）必须有 `is_group_exist` no-op guard |
| `finalize_weight_update()` | ModelUpdateService (per-engine fan-out, atomic unit 内部) | — | Called by service ONLY when version >= 0 (i.e., after bucket transport). **Skipped for v=-1 path** (Fix #13 + Fix #15) — SGLang's own onload-time post-load hook (`process_weights_after_loading`) covers initial activation; calling finalize again on already-loaded weights is redundant. pipeline 不直接调用 |

**`verify_model` 不在本 milestone receiver API surface**：传输健壮性由 per-bucket
barrier + warmup allreduce 保证；权重错误更可能从 trajectory reward 异常 / 模型行为
退化更早暴露而不是 hash 比对。如未来需要 debug-time validation，再单独加。

`finalize_weight_update()` 必须在 **engine 进程内**执行（涉及 SGLang model_runner 本地
对象，不可序列化）。

receiver-side hardening、version publish、timeout crash、sync barrier 都是 parity
必需项。

改动量：~230 行（两条路径 + version accounting + receiver API surface；不含 verify_model）

**修改/新增文件：**
- `rlix/pipeline/miles_pipeline.py` — `_before_training` hook、`_after_training` hook（service.sync_selected_workers 经 coordinator.sync_base_weights_to_active；不直接 `_finalize_weight_update`）
- `rlix/pipeline/coordinator.py`（或 `miles_coordinator.py`）— 添加 `sync_base_weights_to_active()`
- `miles/backends/sglang_utils/sglang_engine.py` — 实现全部 6 个 receiver 方法

---

### Feature 7: Per-pipeline Ray namespace isolation

**作用：** 多 pipeline 共存时 Ray actor 命名隔离，防止冲突。

#### ROLL 怎么做的

- 每个 pipeline 独立 Ray namespace（env var `ROLL_RAY_NAMESPACE` → 派生 `RAY_NAMESPACE`）
- Actor 名称带 `pipeline_id` 前缀
- `rlix/utils/env.py:24` `pipeline_identity_env_vars()`（`PIPELINE_ID` +
  `ROLL_RAY_NAMESPACE` + `RLIX_CONTROL_PLANE`）通过 `runtime_env` 传给所有 actor
- `full_finetune_pipeline.py:376-390` init 时校验 namespace 与 pipeline_id 匹配

#### MILES 现状

- 无 namespace 隔离概念
- `miles/ray/...` 下所有 Ray actor 在默认 namespace
- 命名（rollout server / actor model 等）不带 pipeline 前缀

#### 移植方案

1. **MilesCoordinator** 在 `get_pipeline_namespace(pipeline_id)` 中创建
2. **MilesPipeline** actor 同 namespace 创建
3. **`MilesModelUpdateService`** 同 namespace 创建（**显式列出**：F4 新增的
   ModelUpdateService actor 由 coordinator 创建，是新增 RLix 侧 named actor 的第三个；
   coordinator 创建它时容易漏掉 `namespace=` — 必须传 `namespace=ray_namespace +
   name=f"rlix:miles_model_update_service:{pipeline_id}"`）
4. **审计 MILES 内所有 named Ray actor**：grep `name=` 在 `miles/ray/` /
   `miles/backends/` 下 — 当前结果：`RolloutRayActor.options(...)` ([rollout.py:144](external/miles/miles/ray/rollout.py#L144))、
   `Lock.options(...)` ([rollout.py:374](external/miles/miles/ray/rollout.py#L374))、
   `@ray.remote` decorator ([rollout.py:334](external/miles/miles/ray/rollout.py#L334))
   全部匿名。**当前不需改 MILES framework**；只需 RLix 侧新增的三个 actor 显式带
   namespace + name
5. 通过 `runtime_env` 传 `pipeline_identity_env_vars()` 给所有 actor。这个 runtime_env
   是 `RolloutRayActor` / `Lock` 等匿名 MILES child actor 继承 namespace 与
   `RLIX_CONTROL_PLANE` 的传播路径；第 1-3 项的 named RLix actors 仍必须显式传
   `namespace=`，因为它们由 driver/coordinator 直接创建，不能假设继承调用方环境
6. **`ROLL_RAY_NAMESPACE` import-time fail-fast**：如继承 ROLL utils 自动获得（ROLL 代码
   在 import time 读取 `ROLL_RAY_NAMESPACE`，再导出内部 `RAY_NAMESPACE`，缺失会 fail
   fast）；如 MILES 侧新写需复制 fail-fast 行为（缺失环境变量直接 raise）

改动量：~60 行

---

### Feature 8: Pipeline registration lifecycle

**作用：** Pipeline 向 RLix orchestrator 注册 GPU 拓扑，scheduler 才能调度。

#### ROLL 怎么做的

- `rlix/orchestrator/orchestrator.py:195-253` 三步：
  1. `allocate_pipeline_id(pipeline_type)`
  2. `register_pipeline(pipeline_id, ray_namespace, cluster_tp_configs, cluster_device_mappings)`
  3. `admit_pipeline(pipeline_id)`
- `cluster_device_mappings`: `{"actor_train": [0,1,2,3], "actor_infer": [0,1,2,3,4,5,6,7]}`
- `cluster_tp_configs`: `{"actor_train": 1, "actor_infer": rollout_num_gpus_per_engine}`

#### MILES 现状

- 无注册概念。`train.py` / `train_async.py` 直接 `import` 并 `main()`
- [miles/ray/placement_group.py](external/miles/miles/ray/placement_group.py)
  `create_placement_groups(args)` 进程内创建 PG，无外部入口
- 无 RLix-aware actor / coordinator 概念

#### 移植方案

driver 脚本按 ROLL 三步流程注册（与 NeMo F8 同形态）：

```python
# RLix `PipelineType` 是 `Literal["ft", "lora"]`（[orchestrator.py:30](rlix/orchestrator/orchestrator.py#L30)），
# 不是 enum；用字符串字面量调用
from rlix.protocol.types import get_pipeline_namespace

# MILES 没有显式 device_mapping args (grep 不到 args.actor_train_device_mapping 等);
# first build 用既有 args 派生. 解禁非连续 mapping 时再考虑新增 args (留 follow-up).
train_devices = list(range(args.actor_num_nodes * args.actor_num_gpus_per_node))
infer_devices = list(range(args.rollout_num_gpus))
cluster_device_mappings = {
    "actor_train": train_devices,
    "actor_infer": infer_devices,
}
cluster_tp_configs = {
    "actor_train": 1,                                      # Megatron: 固定 1 GPU/worker
    "actor_infer": args.rollout_num_gpus_per_engine,
}

pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote("ft"))  # full_finetune
ray_namespace = get_pipeline_namespace(pipeline_id)

ray.get(orchestrator.register_pipeline.remote(
    pipeline_id=pipeline_id,
    ray_namespace=ray_namespace,
    cluster_tp_configs=cluster_tp_configs,
    cluster_device_mappings=cluster_device_mappings,
))
ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))

# MilesCoordinator __init__ 是 keyword-only (对齐 PipelineCoordinator [coordinator.py:184](rlix/pipeline/coordinator.py#L184))
coordinator = MilesCoordinator.options(
    name=f"rlix:coordinator:{pipeline_id}", namespace=ray_namespace,
).remote(pipeline_id=pipeline_id, pipeline_config=miles_args)

# create_pipeline_actor 是 keyword-only ([coordinator.py:242](rlix/pipeline/coordinator.py#L242)
# `def create_pipeline_actor(self, *, pipeline_config: Any) -> Any:`)
# **不能** 写成 `coordinator.create_pipeline_actor.remote()` (会 TypeError: missing kwarg)
ray.get(coordinator.create_pipeline_actor.remote(pipeline_config=miles_args))
```

**顺序契约**：driver 必须先 `allocate_pipeline_id` → `register_pipeline` →
`admit_pipeline`，再创建 coordinator actor。

改动量：~80 行（merged config/registration helper + pipeline 内部 bundle mapping helper）

---

### Feature 9: Progress reporting

**作用：** Pipeline 向 RLix scheduler 报告 generation demand，scheduler 据此 gap-ratio
planning 决定何时 shrink。

#### ROLL / RLix 怎么做的

- `RolloutScheduler` 每 2% 进度变化时发 `ProgressReport`（`rollout_scheduler.py:601-635`）
- ROLL-internal scheduler → coordinator aggregation ingress 使用 `metrics["collected"]`
- RLix central scheduler ingress 使用 `metrics["completed"]`，并拒绝 `remaining`
- Fire-and-forget → coordinator → central scheduler
- scheduler `remaining = max(step_target - completed, 0)` 决定 shrink

#### MILES 现状

- [miles/utils/types.py](external/miles/miles/utils/types.py) — sample 已携带
  `weight_versions`
- [miles/rollout/generate_utils/generate_endpoint_utils.py](external/miles/miles/rollout/generate_utils/generate_endpoint_utils.py)
  — train data 已可带 `weight_versions`
- group/sample 计数器在 fully_async path 内已存在
- 缺：对外 progress callback hook；group → trajectory 聚合；2% bucket 上报；
  与 `ProgressReport` wire-level 字段对齐

#### 移植方案

**本地最小 counter 集（删除 dead state）**：本地只维护与 RLix wire 对齐的最小集。

**单位必须是 group, 不是 trajectory** — fully_async 实际 wait window 在
[fully_async_rollout.py:198 `target_data_size = args.rollout_batch_size`](external/miles/examples/fully_async/fully_async_rollout.py#L198)
+ `while len(data) < target_data_size` 的 `data` 是 group list, 一条 group 对应一个 prompt
的 N 个 trajectory。target 用 trajectory (rollout_batch_size × n_samples_per_prompt) 而
counter 用 group, 比例差 n_samples_per_prompt × → 报给 scheduler 的 progress 永远低估
n× → 过早 shrink。**target 与 counter 都用 group**:

- `step_target_groups = args.rollout_batch_size`（来自 config; MILES 现有 fully_async outer
  loop wait 的就是 `rollout_batch_size` group, 与 [fully_async_rollout.py:198](external/miles/examples/fully_async/fully_async_rollout.py#L198)
  保持一致）。MILES 没有 NeMo / ROLL 那种 `num_groups_per_step` / `num_prompts_per_step`
  独立 arg, 直接复用 `rollout_batch_size`
- `_local_completed`（本地 group 计数器，仅累加 `target_weight_version ==
  _progress_target_step` 的 group; 一个 group push 进 `data` 时 `+= 1`）
- `_last_progress_bucket`（2% 阈值跟踪，0..50）
- `_progress_target_step`（当前 wait 的 weight version）

**显式不维护**（RLix scheduler 不消费，纯 dead state）：`queued / inflight / consumed
/ oldest_unfinished_creation_ts / active_weight_version / oldest_generation_weight_version`。
sample 自身已带的 `weight_versions` 沿用，但**不**作为 progress 上报字段。

**ProgressReport wire 字段表（MILES hook → MilesCoordinator aggregation ingress）**：

| 字段 | 值 | 来源 |
|---|---|---|
| `pipeline_id` | 注册时分配 | Feature 8 |
| `step_target_trajectories` | RLix wire 字段名沿用, 但语义实际为 group count | `args.rollout_batch_size`（**单位 group, 不是 trajectory**, 与 fully_async wait condition 对齐）|
| `metrics["collected"]` | local counter | `_local_completed` group 数（reporter 端，**raw 不 clamp**；coordinator 端聚合 + clamp 才得到 scheduler-side `completed`） |
| `bucket` | 0..50 | `math.floor(_local_completed / step_target_groups * 50)`（reporter 端计算；只在 `bucket != _progress_last_bucket` 时才发 RPC，**2% gate 在 reporter 层**） |
| `current_train_step` | 当前 step (= `_progress_target_step`, 因为 `target_weight_version` 在 fully_async 即为 step 编号) | `_progress_target_step` |
| `mode` | `"train"` | full-finetune 单 stream |
| `new_batch` | True/False | begin_progress_batch 直接发 True 首个 RPC; bump_completed 之后只发 False |

**两层职责分工（对齐 ROLL [rollout_scheduler.py:601-635](external/ROLL/roll/distributed/scheduler/rollout_scheduler.py#L601)）**：

| 层 | 字段 | 职责 |
|---|---|---|
| **Reporter（MILES rollout hook, 实现 `RLixHooks` protocol）** | `_progress_last_bucket: int`、`_local_completed: int`、`step_target_groups: int`、`_progress_target_step: int` | 持有 target context；每 increment 算 `bucket = floor(collected/target * 50)`；**只在 `bucket != _progress_last_bucket` / `remaining == 0` 时**才发 hook RPC（**2% bucket gate 在这一层**）|
| **Coordinator（`MilesCoordinator`, 不 subclass `PipelineCoordinator`；详见文件改动表）** | streams keyed by `(mode, adapter_id)` | 聚合跨 stream 的 `collected`；`_aggregate_and_emit` 算 `completed = min(total_collected, total_required)`；fire-and-forget 给 RLix central scheduler |

**Hooks 解耦不变量（Critical Invariant — 严守 import seam）**: `examples/fully_async/fully_async_rollout.py`
**只能调** `self._rlix_hooks.report_progress(...)` / `begin_progress_batch(...)` /
`end_progress_batch()`. **不能 import `ProgressReport`, 不能 `ray.get_actor("rlix:coordinator:...")`,
不能拿 coordinator handle**. 所有 RLix wire 类型构造与 RPC 路由在 RLix-side
`MilesRLixHooks` 实现里完成 (位于 `rlix/pipeline/miles_hooks.py` 或 inline 在
`run_miles_rlix.py`). standalone 模式 `NoOpRLixHooks` 全 no-op, fully_async 不感知
RLix 存在.

1. **Reporter 实现**（`miles/utils/rlix_hooks.py` 提供 protocol + NoOp; RLix-side 提供真实
   实现）：
   - 持有 `_local_completed`（仅累加 `target_weight_version == _progress_target_step` 的 group）
   - 持有 `_progress_last_bucket`（per-stream，初始 -1）
   - 删除 `_progress_new_batch` flag — `begin_progress_batch` 直接发首个 RPC, bump 不需要
     额外 flag (避免 P1 set/clear 不闭合)
   - `bump_completed(group, target_weight_version)` (在 reporter 实例上, 不直接 hook 调用):
     ```python
     def bump_completed(self, group, target_weight_version):
         if target_weight_version != self._progress_target_step:
             return  # 跨 step / age window 不一致的 group 不计
         self._local_completed += 1
         bucket = math.floor(self._local_completed / self._step_target_groups * 50)
         remaining = max(self._step_target_groups - self._local_completed, 0)
         should_emit = (bucket != self._progress_last_bucket or remaining == 0)
         if not should_emit:
             return  # ← 2% gate 在 reporter 这一层
         self._progress_last_bucket = bucket
         # 通过 hook 抽象上报, 不直接持 coordinator handle
         self._rlix_hooks.report_progress(
             collected=self._local_completed,
             bucket=bucket,
             current_train_step=self._progress_target_step,
             new_batch=False,
         )
     ```
   - `RLixHooks.report_progress(...)` 签名是 plain kwargs (无 RLix 类型), RLix-side 实现
     才把它打包成 `ProgressReport(metrics={...})` fire-and-forget 给 coordinator
2. **Coordinator ingress**：`MilesCoordinator.report_progress_from_scheduler(report)`
   （[coordinator.py:299](rlix/pipeline/coordinator.py#L299) 复制并适配, **不继承**
   `PipelineCoordinator` —— 详见文件改动表 MilesCoordinator 段说明）。
   入站 report 必须带 `metrics["collected"]`，**不能直接带 scheduler-level `completed`**
3. **Coordinator → central scheduler**：`_aggregate_and_emit`
   ([coordinator.py:359](rlix/pipeline/coordinator.py#L359) 复制并适配, **不继承**)
   聚合跨 stream `collected`，算 `completed = min(total_collected, total_required)`，
   fire-and-forget 到 [scheduler.py:840](rlix/scheduler/scheduler.py#L840)。
   **MILES 单 stream 时聚合退化为 pass-through，但走标准协议**
4. 不让 MILES runtime 绕过 coordinator 把 `queued / inflight / remaining` 当 scheduler
   canonical demand 上报
5. **API 名澄清**：coordinator 用 `clear_progress_stream(...)` 不是 `clear_progress`
   （聚合层判断是否还有其他 active streams 决定是否通知 scheduler，对齐 ROLL
   `coordinator.py:326`）

**Demand window 必须 begin/end 包裹（对齐 ROLL `RolloutScheduler` 1043-1086）：**

不能只发增量 bucket — scheduler 看到的 demand 必须有明确的开始与结束，否则上一步的
stale demand 会泄漏到下一步：

- `begin_progress_batch(target_weight_version, step_target_groups, initial_completed)`
  （**硬不变量** — M11 hook signature 加 `initial_completed`）：
  进入"等待当前 step 凑齐 group"之前调用。reporter 端**必须**：
  - 持有 `step_target_groups`（从外部传入；reporter 没有 RLix wire 上下文时无法
    自己算 target，必须由 caller 传）
  - **`_local_completed = initial_completed`** — caller (fully_async) 在 wait window
    入口处传 `initial_completed = 0`. 此时 `data = []` (caller-local);
    output_queue 内任何已 queued 的 group 会在 while 循环第一次 iter 通过
    `worker.get_completed_groups()` drain → `bump_completed` 流转, 与新 batch 内
    完成的 group 走同一路径. **reporter 不读 worker 内部** — `AsyncRolloutWorker` 只暴露
    `output_queue` + `get_completed_groups()`, 没有 `worker.buffer` / `g.is_ready` /
    `g.target_weight_version` 字段; 任何 worker-side scan 路径都是 fake. 多发一次
    bucket-0 emit 的代价远小于 worker scan 的不变量复杂度
  - reset `_progress_last_bucket = -1`（强制首个 emit 触发，因为初始 bucket 必然 ≠ -1）
  - 直接通过 hook 上报首个快照: `self._rlix_hooks.report_progress(collected=self._local_completed,
    bucket=initial_bucket, current_train_step=target_weight_version, new_batch=True)` —
    `new_batch=True` 仅出现在这一处, 不需要 reporter 维护 `_progress_new_batch` flag
- `end_progress_batch()`：放在 wait window 的 `finally` 中，无论成功或异常都通过 hook 清除
  progress stream（`self._rlix_hooks.clear_progress()`, RLix-side 实现转 RPC 到
  `coordinator.clear_progress_stream(...)`）
- 后续 group completion hook（`bump_completed`）仅在 `_progress_active and
  target_weight_version == _progress_target_step` 时累加 `_local_completed` + 算 bucket，
  本地 2% gate 决定是否调 hook，避免 hot path 上每次 push 都 `ray.get`；hook 的报文一律
  `new_batch=False`，对齐 ROLL `RolloutScheduler` wire-level 行为

**MILES 的 wait window 精确锚点**（不是 `train_async.py:37`）：

`train_async.py:37` `rollout_data_curr_ref = await rollout_data_next_future` 是 await
Ray ObjectRef，从 trainer 视角看不到 generation 内部进度事件，**不能在这里 hook**。

实际 demand window 在 [examples/fully_async/fully_async_rollout.py:218-220](external/miles/examples/fully_async/fully_async_rollout.py#L218)
`generate_rollout_async()` 内部的 `while len(data) < target_data_size:` 主循环。这是
真正轮询完成 group 的地方，是 begin/end_progress_batch 的目标位置：

```python
# examples/fully_async/fully_async_rollout.py:generate_rollout_async()
# rlix_hooks 是 RLixHooks protocol 实例; standalone 走 NoOpRLixHooks (全 no-op),
# RLix mode 由 driver 注入 MilesRLixHooks (RLix-side 实现, fully_async 不 import RLix 类型)
async def generate_rollout_async(args, rollout_id, data_buffer, rlix_hooks=None):
    rlix_hooks = rlix_hooks or NoOpRLixHooks()  # standalone fallback
    ...
    target_data_size = args.rollout_batch_size  # group 单位, 与下面 wait condition 一致

    # data 在 begin_progress_batch 入口处恒为 [] (caller-local).  output_queue 内
    # 已 queued 的 group 会在 while 循环首次 iter 通过 worker.get_completed_groups()
    # → bump_completed 流转 (与新完成的 group 同一路径).  AsyncRolloutWorker 暴露的
    # 是 output_queue + get_completed_groups(), **没有** worker.buffer / g.is_ready /
    # g.target_weight_version 字段; 任何 worker-side scan 路径都是 fake.  多一次
    # bucket-0 emit 的代价远小于 worker scan 的不变量复杂度.
    initial_completed = 0
    rlix_hooks.begin_progress_batch(
        target_weight_version=current_weight_version,
        step_target_groups=target_data_size,
        initial_completed=initial_completed,
        # forward-compat (M11.4 multi-stream hook): nullable mode/adapter_id 字段保留
        mode=None,           # M11.1 单 stream, M11.4 LoRA 多 stream 时填
        adapter_id=None,     # 同上
    )
    try:
        while len(data) < target_data_size:                    # ← 实际 wait window (group)
            completed = worker.get_completed_groups()
            for group_id, group in completed:
                # ... existing logic, 一个 group push 进 data 时 +1 ...
                rlix_hooks.bump_completed(
                    target_weight_version=current_weight_version,
                )  # 2% bucket trigger; counter 单位 = group.  group 没有
                   # target_weight_version 字段 (group = list[Sample], Sample.weight_versions
                   # 才是 per-decision); 当前 step 视角下 group 的 target =
                   # caller-local current_weight_version, reporter filter 检查
                   # `target_weight_version == _progress_target_step` 仍然有效
            ...
    finally:
        rlix_hooks.end_progress_batch()                        # ← 必须 finally，异常也清

    return data
```

`begin_progress_batch` 在 while 循环**之前**调（紧靠循环 entry）；`end_progress_batch`
在外层 `finally` 中包住整个 while；`bump_completed` 在每个成功 push 的 sample 上调
（hot path，但只是本地 increment + 2% bucket 阈值检查，无 ray.get）。

改动量：~70 行

---

### Feature 10: Partial GPU topology validation

**作用：** 验证 GPU 拓扑满足 partial overlap 要求，启动时 fail fast。

#### ROLL 怎么做的

- `_validate_partial_gpu_config()` (`agentic_pipeline.py:770-894`) 检查：
  1. `train_devices ⊂ infer_devices`
  2. `infer_dp_size >= 2`（无法 partial）
  3. `async_generation_ratio > 0`
  4. TP/PP/EP compatibility
  5. 至少 1 DP rank 在 shrink 后保持 active
- `coordinator.py:136` `_validate_offload_nccl` 强制 `offload_nccl=True`

#### MILES 现状

- 无 RLix-aware 拓扑验证
- `train.py` 启动时只检查 MILES 内部一致性
- `ReloadableProcessGroup` 已强制 NCCL teardown 路径，等价于 ROLL 的
  `offload_nccl=True`

#### 移植方案

在 `MilesPipeline.initialize_pipeline()` 中加 fail-fast 验证：

**派生量定义**（在 validation 入口处一次性算出, 全 plan 复用同一定义; **不新增
device_mapping args**, 全部从既有 `actor_num_nodes / actor_num_gpus_per_node /
rollout_num_gpus` 派生）:

```python
train_devices = set(range(args.actor_num_nodes * args.actor_num_gpus_per_node))
infer_devices = set(range(args.rollout_num_gpus))
infer_engine_count = args.rollout_num_gpus // args.rollout_num_gpus_per_engine
async_generation_enabled = bool(getattr(args, "rollout_function_path", "").endswith(
    "fully_async_rollout.generate_rollout_fully_async"
))

def train_devices_subset_of_infer(args) -> bool:
    """供 F11 standalone fail-fast 复用 (检测用户用 partial overlap 拓扑但忘开 RLix mode)"""
    train = set(range(args.actor_num_nodes * args.actor_num_gpus_per_node))
    infer = set(range(args.rollout_num_gpus))
    return train.issubset(infer) and train != infer
```

**`single_updateable_model_and_server(args)` 定义**（MILES 现状: 仅 1 个 actor_model 注册到
weight update path, 仅 1 个 SGLang server group）:

```python
def single_updateable_model_and_server(args) -> bool:
    """Reject configs that drive weight updates to multiple model/server slots.

    本 milestone 不支持 critic / reward / RM 等 secondary updateable model;
    fully_async + multi_turn + GRPO 单 actor + 单 SGLang server group 才允许.
    若未来引入 critic refit / multi-server, 必须扩 ModelUpdateService 的 sender
    side; 当前 fail fast.
    """
    return (
        getattr(args, "critic_model_path", None) is None
        and getattr(args, "reward_model_path", None) is None
        and getattr(args, "sglang_secondary_server_count", 0) == 0
    )
```

**完整 validation**:

```python
assert args.sglang_data_parallel_size == 1, "RLix mode requires sglang dp == 1"
# PD disaggregation 走 SglangConfig property，不是 args field
# （[sglang_config.py:95,189](external/miles/miles/backends/sglang_utils/sglang_config.py#L95)）
assert not sglang_config.has_pd_disaggregation, "PD disaggregation out of scope"
assert train_devices.issubset(infer_devices), \
    "partial overlap requires train ⊂ infer"
assert infer_engine_count >= 2, (
    f"partial overlap requires >= 2 engines (got {infer_engine_count}; "
    f"derived from len(infer_device_mapping)/{args.rollout_num_gpus_per_engine})"
)
assert async_generation_enabled, "partial overlap requires fullasync"
assert len(infer_devices - train_devices) >= args.rollout_num_gpus_per_engine, \
    "at least 1 full inference engine must stay active after shrink"
assert single_updateable_model_and_server(args), \
    "RLix MILES first build requires single updateable model + single SGLang server group"

# offload_train must be True (Critical Invariant): MILES MegatronTrainRayActor.sleep() (actor.py:198)
# 内 assert self.args.offload_train. F4 init bootstrap Step 5 + 后续 each train step
# offload 都依赖 actor.sleep() 释放 GPU. 不开 → AssertionError + partial overlap 不能
# 释放 train GPU → train/infer 抢同一 GPU OOM
assert args.offload_train, (
    "RLix mode partial overlap requires offload_train=True; "
    "without it, actor_train cannot release overlap GPU after each step → OOM at infer wake_up"
)

# M7: async_save not supported: MILES `args.async_save` enabled
# (external/miles/miles/backends/megatron_utils/actor.py:479-487) 时,
# `save_model()` 起后台 ckpt flush; 紧接着的 `actor_train.offload() → sleep() →
# torch_memory_saver.pause()` 会与 flush 线程读 GPU optimizer state 竞争 → segfault.
# First build fail-fast 比实现 flush + per-actor 同步更简单 — RLix-mode 主流程不在
# train_step 内触发 ckpt save, 此 fail-fast 不影响功能. 若需 async_save, 在 sleep()
# 开头加 `if args.async_save: maybe_finalize_async_save(blocking=True); cuda.synchronize()`
# (follow-up).
assert not getattr(args, "async_save", False), (
    "first build does not support args.async_save — background ckpt flush races "
    "with actor.sleep() torch_memory_saver.pause() and segfaults. "
    "Implement maybe_finalize_async_save(blocking=True) + cuda.synchronize() "
    "in MegatronTrainRayActor.sleep() prologue as a follow-up."
)

# rollout_num_gpus divisibility (Critical Invariant): floor division 不整除会让 ROLL
# allocate_placement_group(world_size, ...) 与 MILES engine 数算出不同, 拓扑错位
assert args.rollout_num_gpus % args.rollout_num_gpus_per_engine == 0, (
    f"rollout_num_gpus ({args.rollout_num_gpus}) must divide evenly by "
    f"rollout_num_gpus_per_engine ({args.rollout_num_gpus_per_engine}); "
    f"otherwise ROLL allocate_placement_group world_size and MILES engine count diverge"
)

# M11.1 transport: cpu_serialize 是唯一 colocate transport (deployment-driven —
# vast.ai 受限容器无 --ipc=host / CAP_SYS_PTRACE). cuda_ipc colocate adapter 是
# M11.6 later milestone (production cluster, smoke-test capability check 不写 heuristics).
# Non-colocate receiver 仍走 dynamic NCCL broadcast (Case A 之外的另一条路径, M11.1
# load-bearing for partial overlap).
if DO_TIME_SHARING:
    assert args.model_update_transport == "cpu_serialize", (
        f"M11.1 first-build target = vast.ai restricted container, "
        f"only cpu_serialize colocate transport supported "
        f"(got model_update_transport={args.model_update_transport!r}). "
        f"cuda_ipc colocate adapter = M11.6 later milestone (production cluster)."
    )

    # Fix #14: M11.2 Gate 4 happy path requires fresh-base equivalence between
    # train cache_owner build_cpu_bucket_cache(-1) bytes AND SGLang
    # `args.hf_checkpoint` onload bytes. Resume / non-equivalent ref_load
    # would diverge them — and Fix #13's v=-1 no-transport short-circuit
    # would then serve stale weights on rollout side. CPU-only base sync
    # for resume is M11.2 follow-up; for now fail-fast.
    assert args.load is None or args.load == args.hf_checkpoint, (
        f"M11.2 Gate 4 happy path requires `args.load` to be either None or "
        f"equal to `args.hf_checkpoint` (got load={args.load!r}, "
        f"hf_checkpoint={args.hf_checkpoint!r}). Resume from Megatron "
        f"checkpoint is M11.2 follow-up — CPU-only base sync at v=-1 not "
        f"yet implemented; without it, Fix #13 v=-1 no-transport path would "
        f"serve stale rollout weights."
    )
    assert (
        getattr(args, "ref_load", None) is None
        or args.ref_load == args.hf_checkpoint
    ), (
        f"M11.2 Gate 4 happy path requires `args.ref_load` to be either None "
        f"or equal to `args.hf_checkpoint` (got ref_load={args.ref_load!r}, "
        f"hf_checkpoint={args.hf_checkpoint!r}). Non-equivalent ref_load is "
        f"M11.2 follow-up."
    )

# M11.1 forbids cross-node rollout engine, not multi-node DP (Cut 1').
# Dev gate runs on single-machine 4-GPU, but architecture must remain multi-node-compatible
# at placement / data-structure level. M11.1 OK: node0 train [0,1] + infer engines [0,1] [2,3];
# node1 infer engines [0,1] [2,3] (multi-node DP, each engine node-local). NOT M11.1:
# 单个 SGLang infer engine 占 4 GPU 跨 node0 gpu[0,1] + node1 gpu[0,1] (cross-node rollout engine,
# 不论 sharding 形式 TP/EP/etc., not supported in M11.1).
# Cross-node rollout engine gate captured by existing M3 assert below (rollout_num_gpus_per_engine
# <= num_gpus_per_node, 按 engine 总 GPU 数计) — no separate single-node assert needed.

# MoE / EP fail-fast (Out of Scope 显式 enforce, 不仅靠 divisibility 推断)
assert args.expert_model_parallel_size == 1, \
    "MoE / EP is out of scope for current milestone; F4 CPU bucket cache 仅覆盖 dense Megatron"
assert getattr(args, "moe_router_topk", 0) == 0, "MoE configs not allowed in RLix mode"

# Streaming generate 必须禁用 (F3 router metadata 注入要求 JSON body, SSE 不兼容)
assert not getattr(args, "rollout_force_stream", False), (
    "RLix mode requires non-streaming generate; metadata injection requires JSON body"
)

# S2: Bucket size GPU 上界 check 仅在 NCCL broadcast (non-colocate receiver) 存在时
# 触发. M11.1 cpu_serialize colocate path 不需要 sender GPU staging. NCCL broadcast
# path 需要 (NCCL 不支持 CPU tensor, sender 必须 H2D staging 到 cache_owner GPU 才能
# broadcast). M11.6 cuda_ipc 加回时也会触发此 check (cuda_ipc adapter 同样 per-bucket
# H2D staging 到 cache_owner GPU 才能 serialize IPC handle).
# Host RAM check (CPU bucket cache + N receivers torch.load 拷贝预算) 由 F4 §1
# total_cpu_cache_bytes 启动期 check 兜底.
bucket_size_bytes = args.miles_model_update_bucket_size_mb * 1024 * 1024
has_gpu_staging = _topology_has_non_colocate_engines(args)  # M11.6 + cuda_ipc 时 OR cuda_ipc transport
if has_gpu_staging:
    post_wake_free_vram_bytes = _estimate_post_wake_free_vram(args)  # SGLang weight + KV cache + cuda_graph 占满后剩余
    transport_scratch_bytes = 256 * 1024 * 1024  # NCCL scratch
    assert bucket_size_bytes + transport_scratch_bytes < post_wake_free_vram_bytes, (
        f"bucket_size_bytes ({bucket_size_bytes}) + transport_scratch ({transport_scratch_bytes}) "
        f"exceeds post-wake free VRAM ({post_wake_free_vram_bytes}); reduce miles_model_update_bucket_size_mb"
    )

# S3a-2: cpu_serialize 路径 wrapper 写 /dev/shm tmpfs; 启动期 check 容量
# (Docker 默认 /dev/shm = 64MB, bare metal 通常 8-16GB). 不够直接 fail-fast 提示
# 用户调小 bucket size 或 increase --shm-size. cpu_serialize-only 拓扑 (first
# build 默认) 才需要; cuda_ipc-only 拓扑无 tmpfs 操作.
if getattr(args, "model_update_transport", "cuda_ipc") == "cpu_serialize":
    import shutil
    assert os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK), (
        "cpu_serialize transport requires writable /dev/shm; container may need "
        "--shm-size (Docker) or tmpfs mount"
    )
    shm_free = shutil.disk_usage("/dev/shm").free
    required = bucket_size_bytes + 256 * 1024 * 1024  # bucket + 256MB safety margin
    assert shm_free >= required, (
        f"/dev/shm free space ({shm_free} bytes) < required ({required} bytes "
        f"= bucket_size + 256MB margin); increase --shm-size (Docker) or reduce "
        f"args.miles_model_update_bucket_size_mb"
    )

# First-build constraint: sorted contiguous inference mapping only.
# This guarantees RLix scheduler dp_rank == MILES rollout engine_index.
# 即使在此 first-build 限制下, F12 仍然必须构造 explicit dp_rank ↔ engine_index ↔ gpu_ids
# mapping table (供 MilesPlacementProvider / MilesCoordinator 使用), 否则解禁非连续 mapping
# 时会再踩同一个坑.
infer_device_mapping = list(range(args.rollout_num_gpus))  # 派生, MILES 没有显式 args
assert infer_device_mapping == sorted(infer_device_mapping), (
    "RLix MILES first build requires sorted infer_device_mapping; "
    "non-contiguous/custom ordering needs an explicit scheduler_dp_rank -> engine_index adapter (F12)"
)
for engine_index, start in enumerate(range(0, len(infer_device_mapping), args.rollout_num_gpus_per_engine)):
    group = infer_device_mapping[start : start + args.rollout_num_gpus_per_engine]
    expected = list(range(group[0], group[0] + args.rollout_num_gpus_per_engine))
    assert group == expected, (
        f"infer engine {engine_index} must occupy contiguous GPUs in first build; "
        f"got {group}, expected {expected}"
    )
assert "RadixTreeMiddleware" not in (args.miles_router_middleware_paths or []), (
    "RLix mode currently disables radix-tree middleware; partial_rollout + radix_tree is follow-up"
)

# Megatron 内部并行度 divisibility（资 NeMo F10）— 防 Megatron init 时才报错。
# MILES 的实际 args 名是 *_model_parallel_size 全名，不是缩写
# （参 [initialize.py:42-48](external/miles/miles/backends/megatron_utils/initialize.py#L42)
# 与 [bridge_lora_helpers.py:89-95](external/miles/miles/backends/megatron_utils/bridge_lora_helpers.py#L89)）
megatron_parallelism_product = (
    args.tensor_model_parallel_size
    * args.pipeline_model_parallel_size
    * args.context_parallel_size
    * args.expert_model_parallel_size  # = 1, 已在上面 assert
)
assert len(train_devices) % megatron_parallelism_product == 0, (
    f"train device_mapping ({len(train_devices)}) must divide evenly by "
    f"tp*pp*cp*ep ({megatron_parallelism_product})"
)
assert len(infer_devices) % args.rollout_num_gpus_per_engine == 0, (
    f"infer device_mapping must divide evenly by rollout_num_gpus_per_engine"
)
```

改动量：~50 行（含 derived qty 定义 + `single_updateable_model_and_server` helper + EP
fail-fast + stream/bucket-size 检查）

---

### Feature 11: Conditional RLix behavior flag

**作用：** MILES 代码在 standalone 与 RLix 模式下行为不同，flag 控制。

#### ROLL 怎么做的

- `roll/utils/constants.py` `DO_TIME_SHARING` 从 `RLIX_CONTROL_PLANE` 派生
- 用于：跳过 `ray.shutdown()`、pipeline-scoped naming、progress reporting、
  pipeline class 选择（`RollFullFinetunePipeline` vs `AgenticPipeline`）

#### MILES 现状

- 无此概念。`train.py` / `train_async.py` 总是 standalone

#### 移植方案

| 行为 | Standalone | RLix | 控制 |
|---|---|---|---|
| `ray.shutdown()` | 正常执行 | 跳过（用户 `ray stop` CLI 收尾, driver 不调） | Flag |
| Train step 后 | 直接进入 MILES 既有 weight update path: `RolloutManager.update_weights_from_distributed/tensor` ([rollout.py](external/miles/miles/ray/rollout.py)) 全量 broadcast/IPC | CPU bucket cache build + offload training GPU + destroy NCCL groups | Flag + Hook |
| Weight sync | `RolloutManager.update_weights_from_distributed` (broadcast) 或 `_send_to_colocated_engine` (cuda_ipc) 全量 | 跳过 — scheduler expand 触发 `MilesModelUpdateService.sync_selected_workers()` | Flag |
| Generation lifecycle | 训练 loop 内自启 | scheduler `request/release_cluster_gpus(actor_infer)` 驱动 | Flag |
| Progress reporting | 无 | fully_async path 上报 demand | Hook |
| Placement group | `create_placement_groups(args)` 自建 | 接受外部 PG + bundle indices | Flag + Hook |

实现：`RLIX_CONTROL_PLANE` env var → `DO_TIME_SHARING`（与 ROLL 一致）。

**入口结构（双 entry，不在 `train_async.py` 内部分支）：**

| Entry | 模式 | 文件 |
|---|---|---|
| Standalone | `DO_TIME_SHARING=False`（原 MILES 行为） | `train_async.py` **完全不动** — 仍是 `asyncio.run(train(args))`，标准 entry point |
| RLix | `DO_TIME_SHARING=True` | 新 driver 脚本 `examples/rlix/run_miles_rlix.py`：构造 orchestrator + MilesCoordinator + admit + `coordinator.create_pipeline_actor.remote(pipeline_config=miles_args)` (keyword-only) + `pipeline.initialize_pipeline()` + main loop |

`DO_TIME_SHARING` 的判定**只发生在 RLix entry 内部**（决定 build_cpu_bucket_cache /
hooks 等行为分支）；不在 `train_async.py` 内 if/else 包裹整 `train()` 函数 — 那样会把
两条路径耦合。

`train_async.py` 在 RLix mode 下**不被直接执行**；其内部逻辑（rollout_manager /
actor_model / loop）由 `MilesPipeline.run()` 重组（继承 / 调用 train_async 的 builder
helpers，但控制流由 pipeline actor 持有）。

**Fail-fast 守卫**（在 standalone entry 顶部 + RLix entry 顶部各一处）：

```python
# train_async.py 顶部
if os.environ.get("RLIX_CONTROL_PLANE") == "rlix":
    raise RuntimeError(
        "RLIX_CONTROL_PLANE=rlix detected; do not run train_async.py directly. "
        "Use examples/rlix/run_miles_rlix.py instead."
    )

# 检测 partial overlap 配置但 DO_TIME_SHARING=False
if not DO_TIME_SHARING and train_devices_subset_of_infer(args):
    raise RuntimeError(
        "Partial overlap topology detected (train ⊂ infer) but RLIX_CONTROL_PLANE not set; "
        "this would run standalone RolloutManager.update_weights_from_distributed full-broadcast "
        "with overlap GPU contention → silent OOM."
    )
```

```python
# examples/rlix/run_miles_rlix.py 内 RLix 模式实际行为
if DO_TIME_SHARING:  # always True at this entry
    build_cpu_bucket_cache(step)
    self._cache_ready_step = step
    offload_training_gpu()        # ReloadableProcessGroup destroy_process_groups
    # M10: sync HF base weights to active engines after offload, before hook fan-out.
    # coordinator handle 通过 lazy resolver `_get_coordinator_handle()` 拿到 (复用 ROLL
    # pattern). 不能在 hooks.after_training 之后调 — scheduler expand 触发的
    # `MilesModelUpdateService.sync_selected_workers` 需要 base weights 已就位 (active
    # set 包含的 engines 必须是 base-synced state).
    version = ray.get(coordinator.sync_base_weights_to_active.remote(step))
    self._current_weight_version = version
    hooks.after_training(step)    # notify scheduler → expand
# 没有 else 分支 — RLix entry 不应在 standalone path
```

**MilesPipeline actor 创建 (M8 — defensive concurrency)**:
`MilesCoordinator.create_pipeline_actor` 在内部用如下方式创建 `MilesPipeline`:
```python
MilesPipeline.options(
    namespace=self._ray_namespace,
    name=f"rlix:miles_pipeline:{self._pipeline_id}",
    max_concurrency=2,  # M8 defensive: actor 内主路径全 outbound RPC, 但 inbound
                        # stop/status/health-check RPC (RLix scheduler) 必须有窗口,
                        # 否则 main loop ray.get 阻塞时 inbound 死锁
).remote(...)
```
不展开成 design 章节; 仅一行 + 一行注释保持 defensive insurance.

`destroy_process_groups()` 已由 [miles/utils/reloadable_process_group.py](external/miles/miles/utils/reloadable_process_group.py)
+ [miles/backends/megatron_utils/actor.py:58](external/miles/miles/backends/megatron_utils/actor.py#L58)
提供，**不需要 NeMo F11 的 `destroy_megatron_nccl_groups()` helper 与 Gate 2.5
fallback**。这是 MILES 相对 NeMo 的最大省工。

**Resize safety story 自述：** MILES 在 RLix 模式下的 resize safety **不依赖**任何
admission control event，由 generation 层 routing-state 变更（`_active_engine_indices`
/ `_preempted_engines`）+ abort-drain-sleep 保证（Feature 2 + Feature 3）。
**routing-state 自身的并发安全见 Feature 2 第 5 条 `_routing_lock` compound operation
不变量**。等价 NeMo F11 中 "RLix resize safety 不依赖 `_refit_pause_cleared`" 的自述。

改动量：~40 行

---

### Feature 12: Shared PG cluster

**作用：** RLix 模式下不让 MILES 自建 PG；接受 `RollResourceManagerProxy` shared PG，
让 MILES 与其他 RLix pipeline（含 ROLL）共享同一组 GPU。

#### ROLL 怎么做的

- ROLL 创建一组 PG 覆盖所有 GPU；不同 role 通过 `device_mapping` 映射到 PG 中的不同
  GPU 子集
- PG 不变，shrink/expand 只改 worker 状态
- `RollResourceManagerProxy` 是 shared PG owner

#### MILES 现状

- [miles/ray/placement_group.py](external/miles/miles/ray/placement_group.py)
  `create_placement_groups(args)` 直接创建 Ray PG，内部切 actor / critic / rollout
  bundle 范围
- 无外部 PG 注入入口
- `RolloutManager` 本就接受 `pg=(placement_group, reordered_bundle_indices,
  reordered_gpu_ids)` 三元组（[miles/ray/rollout.py:72,338](external/miles/miles/ray/rollout.py#L72)），
  但 ROLL `RollResourceManagerProxy.allocate_placement_group()` 返回的是
  per-worker device dict list。两者 **不是天然同 shape**，必须有显式 adapter 把
  ROLL-style device allocation materialize 成 MILES 所需的 bundle index / gpu id
  三元组。

#### 移植方案

不让 MILES 在 RLix 模式下创建自己的 PG。增加 placement-group provider hook：

1. `create_placement_groups(args, *, external_provider=None)`：
   - `external_provider is None` → 原有路径（standalone）
   - `external_provider is not None` → 从 provider 获取 actor / rollout PG + bundle
     indices
2. 新增 `MilesPlacementProvider` adapter（`miles/ray/placement_provider.py`）：
   - **从 coordinator 注入 `RollResourceManagerProxy`, 不自建** — `PipelineCoordinator.__init__`
     已经构造了 singleton proxy ([coordinator.py:209](rlix/pipeline/coordinator.py#L209)
     `self._resource_manager_proxy = RollResourceManagerProxy(num_gpus_per_node=...)`),
     coordinator 在创建 `MilesPipeline` actor 时把 `proxy_handle` 传给 pipeline,
     pipeline 转手传给 `MilesPlacementProvider(proxy_handle, args, train_device_mapping=...,
     infer_device_mapping=...)`。**禁止 provider 自己 `RollResourceManagerProxy(...)`
     实例化** — 会 double-init 并 break shared PG 不变量. **`train/infer_device_mapping`
     必须与 F8 driver `register_pipeline` 时声明的 `cluster_device_mappings[role]` 同
     源** (driver register 时从 args 派生 → 注册给 scheduler → 同时缓存到 `MilesPipeline`
     传给 provider; provider 不重新派生)
   - 调用 `proxy.allocate_placement_group(world_size=..., device_mapping=declared_mapping)`
     获取 shared PG slot (per role, per pipeline)
   - 从 proxy 返回的 per-worker device dict list (运行时实际 placement) 构建
     `WorkerPlacement.placement_group / node_rank / gpu_ids`
3. `MilesPipeline.initialize_pipeline()` 在 RLix 模式下注入 provider；普通模式下
   provider 为 `None`，行为完全不变
4. 不修改 `miles/ray/placement_group.py` 的内部 bundle 切分逻辑 — 只在入口处替换 PG 来源

**真实 adapter (不能 `return (pg, range, infer_dm)` 蒙混过关)** —— 两边 PG shape
根本不同, 同时 MILES 需要扩 `start_rollout_servers` / `RayTrainGroup` 接受 ROLL
allocation list shape:

| | bundle 粒度 | PG 数量 | 返回类型 | MILES 既有消费 API |
|---|---|---|---|---|
| MILES `_create_placement_group(num_gpus)` ([placement_group.py:46](external/miles/miles/ray/placement_group.py#L46)) | per-GPU (`{"GPU": 1, "CPU": 1}` × N) | 1 PG, N bundles | `(PlacementGroup, list[int], list[int])` | `RolloutManager(args, pg=三元组)` |
| ROLL `RollResourceManagerProxy.allocate_placement_group(world_size, device_mapping)` ([resource_manager.py:122](external/ROLL/roll/distributed/scheduler/resource_manager.py#L122)) | per-node (`{ray_device_key: gpu_per_node}` × M) | M PGs | `List[List[Dict]]` (per-worker × per-GPU dict 含 `node_rank/gpu_rank/placement_group`) | n/a |

**(a) Provider 调 ROLL allocate, 切成 MILES 可消费的 per-worker view**:

```python
# miles/ray/placement_provider.py
from dataclasses import dataclass
from ray.util.placement_group import PlacementGroup

@dataclass(frozen=True)
class WorkerPlacement:
    """单个 worker (engine / train rank) 的 ROLL-derived placement.

    **Multi-node-compatible structural invariant (M11.1 Cut 1')**:
    Dev gate 是单机 4 卡, 但**实现不能把多机跑通堵死**. WorkerPlacement 必须 node-local —
    每个 worker 的所有 GPU 都在同节点 (`placement_group` + `node_rank` + 节点本地
    `gpu_ids` + `bundle_index`). M11.1 OK 拓扑示例: node0 train=[0,1] + infer engine0=[0,1] +
    infer engine1=[2,3]; node1 infer engine2=[0,1] + infer engine3=[2,3] (multi-node DP,
    每个 engine node-local). NOT M11.1: 单个 SGLang infer engine 占 4 GPU 跨 node0 gpu[0,1] + node1
    gpu[0,1] (cross-node rollout engine — 不论 sharding 形式 TP/EP/etc.). 跨节点 rollout engine 由
    [§F10 M3 assert](#) `rollout_num_gpus_per_engine <= num_gpus_per_node` 拒收 (规则按
    rollout engine 总 GPU 数计, 与 sharding 形式无关) — cross-node rollout engine not in scope
    (no later milestone re-enables it). Train 侧 (Megatron actors) 不受这条 assert 控制,
    走自己的 placement; M11.1 dev gate 单机 4 卡天然单节点, 跨节点 train placement 留 follow-up.

    架构层面**禁止假设 global GPU id == local id** — 即使 dev gate 单机 4 卡下两者偶然
    重合 (CUDA_VISIBLE_DEVICES remap 后), 多机部署时 `gpu_ids` 是节点本地索引, base_gpu_id
    必须是 0 (post-CVD local view, M9 invariant).
    """
    placement_group: PlacementGroup   # ROLL 节点级 PG 引用 (per-node, not cross-node)
    node_rank: int                    # 节点 rank (0..N-1); 用于 multi-node dispatch
    gpu_ids: list[int]                # 该 worker 占的 GPU id list (len = gpus_per_worker);
                                      # 节点本地索引 (per-node), 不是 global id
    bundle_index: int = 0             # PG bundle index (单节点 PG 通常只有 1 bundle)

class MilesPlacementProvider:
    """
    Critical Invariant: `device_mapping` arg passed to `proxy.allocate_placement_group(...)`
    必须与 driver 注册 pipeline 时声明的 `cluster_device_mappings[role]` 同源, 不能
    在 provider 内重新 `list(range(...))` 派生.

    RLix `cluster_device_mappings` 是 driver/pipeline 注册时声明的 GPU 拓扑
    ([orchestrator.py:register_pipeline]), scheduler 用它做拓扑校验 + 容量账, 不回传
    "新 mapping". 真正运行时 placement 来自 proxy.allocate_placement_group(...) 返回的
    per-worker device dict list, 这才是物理分配, 必须基于此构建 engine_index ↔ gpu_ids
    映射, 不能 range(...) 推导.

    Single pipeline first build 下 declared = list(range(N)) 偶然对上 proxy 返回, 蒙混
    过关; 多 pipeline (Gate 4) Pipeline B 注册 [4,5,6,7] 时, provider 若内部仍 `list(range(4))=[0,1,2,3]`
    与 Pipeline A 冲突. 因此 provider 必须接受外部传入的 declared mapping, 与 F8
    driver register_pipeline 时同一份.
    """

    def __init__(
        self,
        proxy: "RollResourceManagerProxy",
        args,
        *,
        train_device_mapping: list[int],   # F8 driver register 时声明 (来自 args 派生),
                                            # MilesPipeline 缓存后注入
        infer_device_mapping: list[int],
    ):
        self._proxy = proxy           # 由 MilesCoordinator __init__ 注入, 不自建
        self._args = args
        self._train_device_mapping = train_device_mapping
        self._infer_device_mapping = infer_device_mapping
        self._all_rollout_placements: list[WorkerPlacement] | None = None
        self._train_workers: list[WorkerPlacement] | None = None

    def get_all_rollout_engine_placements(self) -> list[WorkerPlacement]:
        """**Length == engine_count**, derived from declared `infer_device_mapping`.
        Independent of `allocated_actor_infer_gpus`.  Used by `RolloutManager.__init__`
        to populate every `EngineInfo.placement` at construction (active AND shell).

        Fix #8: replaces old `get_rollout_workers()` which conflated "declared
        full table" with "active subset".  Pipeline init Step 6.6a calls this;
        Pipeline init Step 7.2 separately calls `get_active_engine_indices` to
        derive which engines are active in the current allocation.
        """
        if self._all_rollout_placements is None:
            tp = self._args.rollout_num_gpus_per_engine
            assert len(self._infer_device_mapping) % tp == 0, "F10 already asserts"
            world_size = len(self._infer_device_mapping) // tp
            raw = self._proxy.allocate_placement_group(
                world_size=world_size,
                device_mapping=self._infer_device_mapping,  # declared, 不是 list(range)
            )
            self._all_rollout_placements = [
                WorkerPlacement(
                    placement_group=worker[0]["placement_group"],
                    node_rank=worker[0]["node_rank"],
                    gpu_ids=[d["gpu_rank"] for d in worker],
                )
                for worker in raw
            ]
        return self._all_rollout_placements

    def get_active_engine_indices(
        self,
        allocated_gpus: list[int],
        tp_size: int,
    ) -> frozenset[int]:
        """Map `allocated_gpus` to engine indices in declared `infer_device_mapping`.

        First-build contiguous mapping: each engine claims `tp_size` contiguous
        GPUs in the declared mapping; engine_idx = `infer_device_mapping.index(g) // tp_size`.

        Half-engine allocation (an engine with only some of its `tp_size` GPUs
        allocated) raises `RuntimeError` — RLix scheduler must allocate at
        engine boundary. Non-contiguous mapping support is M11.2 follow-up
        (would need `scheduler_dp_rank ↔ miles_engine_index` lookup table).
        """
        by_engine: dict[int, set[int]] = {}
        for g in allocated_gpus:
            idx = self._infer_device_mapping.index(g) // tp_size
            by_engine.setdefault(idx, set()).add(g)
        half_engines = [i for i, gs in by_engine.items() if len(gs) != tp_size]
        if half_engines:
            raise RuntimeError(
                f"half-engine allocation: engines {half_engines} have partial GPU sets; "
                f"M11.2 first build requires engine-boundary allocation"
            )
        return frozenset(by_engine.keys())

    def get_train_workers(self) -> list[WorkerPlacement]:
        if self._train_workers is None:
            world_size = len(self._train_device_mapping)  # 每 worker 1 GPU
            raw = self._proxy.allocate_placement_group(
                world_size=world_size,
                device_mapping=self._train_device_mapping,  # declared, 不是 list(range)
            )
            self._train_workers = [
                WorkerPlacement(
                    placement_group=worker[0]["placement_group"],
                    node_rank=worker[0]["node_rank"],
                    gpu_ids=[d["gpu_rank"] for d in worker],
                )
                for worker in raw
            ]
        return self._train_workers
```

**CUDA_VISIBLE_DEVICES invariant (M6)**:
provider 把每 worker `WorkerPlacement.gpu_ids` 转成 CSV (`",".join(str(g) for g in
wp.gpu_ids)`), 调用方 (`RayTrainGroup` / `start_rollout_servers_from_worker_placements`)
在创建 actor 时通过
`ray.remote(...).options(runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": csv,
"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}})` 注入. **禁止** actor 内部任何
`os.environ["CUDA_VISIBLE_DEVICES"] = ...` 在 import sglang / torch 之后再覆盖 ——
`cuInit()` 锁定时机问题: 若 `import sglang` 隐式触发 cuInit, 后再覆盖 env 已无效,
engine 永远只看到 Ray bootstrap 时分配的卡 (通常单卡), TP init 即崩.

既有 MILES 模式正确:
[external/miles/miles/ray/actor_group.py:88](external/miles/miles/ray/actor_group.py#L88)
`ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)`. SGLang 端
[external/miles/miles/backends/sglang_utils/sglang_engine.py:38](external/miles/miles/backends/sglang_utils/sglang_engine.py#L38)
`os.environ.get("CUDA_VISIBLE_DEVICES")` 是 read-only, 信任 Ray bootstrap 已设. Plan
保持此模式; 不需要 `WorkerPlacement` dataclass 加新字段 (env_vars 由 caller 直接构造
传给 `.options(runtime_env=...)`, first build 一行 doc 足够).

**`LOCAL_RANK` injection (Fix #9 — RLix path 必须显式传)**:

现有 [`TrainRayActor.get_local_gpu_id()`](external/miles/miles/ray/train_actor.py#L21)
做 `cvd.split(",").index(str(ray.get_gpu_ids()[0]))`. 在 standalone path
(`num_gpus_per_actor=0.4` Ray-managed CVD) 下能 work, 因为 Ray 自己写的 CVD
和 `ray.get_gpu_ids()` 一致.

但 RLix mode 用 `num_gpus_per_actor=0.01` (近零 fractional reservation,
Ray 仅占 placement bundle, 真实 GPU 隔离靠 manual CVD) + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`
+ 用户从 `WorkerPlacement.gpu_ids` 注入 manual CVD. 此时 `ray.get_gpu_ids()`
返回的是 Ray 内部 bookkeeping 的 GPU id (基于 fractional pool), **不一定**在
manual CVD 列表里 → `cvd.split(",").index(...)` 直接 `ValueError`, 或拿到错
LOCAL_RANK.

**Fix**: `TrainRayActor.__init__` 加可选 `local_rank: int | None = None`
kwarg. 为 None 时 fallback 到既有 `get_local_gpu_id()` (standalone parity).
RLix path (单 actor 单 GPU under fractional + manual CVD) 显式传 `local_rank=0`:

```python
class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port,
                 *, local_rank: int | None = None):
        ...
        os.environ["LOCAL_RANK"] = str(
            local_rank if local_rank is not None else get_local_gpu_id()
        )
```

`RayTrainGroup` 在 RLix path 创建每个 `TrainRayActor` 时传 `local_rank=0`
(每个 worker 单 GPU, local rank 永远 0). standalone path (`num_gpus_per_actor=0.4`,
Ray-managed CVD) 不传, 走旧 fallback.

**(b) MILES 侧 `start_rollout_servers` / `RayTrainGroup` 接受 worker list**:

现有 `RolloutManager.__init__(args, pg)` 三元组路径保留供 standalone 用. RLix mode
走新增路径 (Fix #8 — `worker_placements` 老 API 替换为 `all_engine_placements +
active_engine_indices` 双参数 — 前者是 declared 全表 length=engine_count, 后者是
active 子集 frozenset; partial allocation 下 active 是 strict subset, 余下 engine
进入 shell 状态):
```python
# miles/ray/rollout.py
class RolloutManager:
    def __init__(
        self,
        args,
        pg=None,
        *,
        all_engine_placements: list[WorkerPlacement] | None = None,
        active_engine_indices: frozenset[int] | None = None,
    ):
        ...
        if all_engine_placements is not None:
            # RLix mode: per-worker placement, 节点级 PG + capture_child_tasks.
            # `all_engine_placements` is declared full table (length engine_count);
            # `active_engine_indices` is the subset that gets a Ray/SGLang actor.
            # Engines NOT in active_engine_indices stay in `shell` state — slot
            # reserved (placement preserved on EngineInfo), but no Ray actor / no
            # SGLang server / no GPU.
            assert active_engine_indices is not None, (
                "RLix mode requires both all_engine_placements and active_engine_indices"
            )
            # Pre-allocate EngineInfo slot for every engine_idx; active subset
            # immediately spawn SGLang server, rest stay shell.
            self._engines = self._init_shell_slots(all_engine_placements)
            # Bind per-pipeline empty Miles router (Fix #1+#2 (g)): allocate router
            # port from provider's pool, set self._args.sglang_router_ip/port,
            # start router with empty worker list. activate_routing is the SOLE
            # add_worker entry point.
            self._start_empty_router()
            # Spawn SGLang servers ONLY for active engines.
            self._spawn_active_servers(args, active_engine_indices)
        elif pg is not None:
            # standalone mode: 单 PG + bundle_indices 三元组 (现状; 不变)
            self.servers = start_rollout_servers(args, pg)
        else:
            raise ValueError(
                "must pass either pg or (all_engine_placements + active_engine_indices)"
            )
```
`start_rollout_servers_from_worker_placements` (renamed conceptually as
`_spawn_active_servers`):
- 仅遍历 `active_engine_indices` 子集; 同 engine 的 GPU 共一个 SGLang server process
- **First build 强约束 (M3)**: `assert args.rollout_num_gpus_per_engine <= args.num_gpus_per_node`
  —— `WorkerPlacement.placement_group` 只指向单个节点 PG, 跨节点 rollout (SGLang) engine 当前
  不支持 (规则按 rollout engine 总 GPU 数计, 与 sharding 形式 TP/EP 无关). `args.num_gpus_per_node`
  是 MILES 既有顶层 rollout-side per-node GPU 数
  ([external/miles/miles/utils/arguments.py:70](external/miles/miles/utils/arguments.py#L70)
  `--num-gpus-per-node`, default=8), 同时被 SGLang engine 端
  ([external/miles/miles/backends/sglang_utils/sglang_engine.py:25-33](external/miles/miles/backends/sglang_utils/sglang_engine.py#L25-L33))
  消费. 不要用 `args.actor_num_gpus_per_node` (train side, 不同字段) 做 rollout
  single-node check —— 二者不同会导致 multi-node 判断错. 跨节点 rollout engine 留 follow-up
- **Ray actor resource: standalone 沿用 fractional, RLix 必须 tiny reservation (Critical Invariant)**:
  - **standalone**: `RolloutRayActor` 沿用 `num_gpus=0.2` ([rollout.py:118](external/miles/miles/ray/rollout.py#L118)),
    `RayTrainGroup` 沿用 `num_gpus_per_actor=0.4`
  - **RLix mode**: train actor offload 后通过 `_notify_release_cluster_gpus` 告诉
    scheduler GPU id 空闲, 但 Ray actor 还活着 (success path 保留, 后续 train_step
    反复用 offload→onload→offload). 如果 Ray-side 仍 reserve `0.4 * train_world_size`
    fractional GPU, scheduler 视角空闲 vs Ray placement 视角占用会冲突, 双 pipeline
    共节点时新 actor 调度卡死. 因此 **RLix worker_placements 路径**:
    `RayTrainGroup` 用 `num_gpus_per_actor=0.01` (近零, 只为占 placement bundle);
    `RolloutRayActor` (RLix mode) 同样 `num_gpus=0.01`
  - 实际 GPU 隔离靠 `PlacementGroupSchedulingStrategy + capture_child_tasks +
    CUDA_VISIBLE_DEVICES`
- `PlacementGroupSchedulingStrategy(placement_group=wp.placement_group,
  placement_group_bundle_index=0,  # 节点级 PG 只有 1 bundle
  placement_group_capture_child_tasks=True)`
- **手动设 `CUDA_VISIBLE_DEVICES = ",".join(str(g) for g in wp.gpu_ids)` + 必须同时
  设 `NOSET_VISIBLE_DEVICES_ENV_VARS_LIST`** 阻止 Ray 自动覆盖可见卡 (与 MILES 既有
  [rollout.py:130](external/miles/miles/ray/rollout.py#L130) /
  [actor_group.py:57](external/miles/miles/ray/actor_group.py#L57) 现状一致):
  ```python
  env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}
  env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in wp.gpu_ids)
  ```
- **`SGLangEngine` 必须显式传 `base_gpu_id=0` + `num_gpus_per_engine=len(wp.gpu_ids)`** (M9);
  **不能 fallback 到 `get_base_gpu_id(args, rank)`** ([sglang_engine.py:24-34](external/miles/miles/backends/sglang_utils/sglang_engine.py#L24)) —
  该公式依赖 `args.colocate / actor_num_gpus / use_critic` 等 standalone 拓扑假设,
  RLix-derived partial-overlap 下不可靠. **不传 `wp.gpu_ids[0]` (= physical GPU id)**:
  CVD 注入 (M6) 后, 进程内只能看到 `cuda:0..tp-1` (post-CVD local view), `base_gpu_id`
  按 SGLang 既有约定是 local index, 必须传 0; 传物理 id 会导致
  [sglang_engine.py:38-50](external/miles/miles/backends/sglang_utils/sglang_engine.py#L38)
  的 `physical_gpu_id ∈ CVD list` validation 错位
- **同样应用到 `RayTrainGroup`** (actor_group.py): 每 actor 用对应
  `WorkerPlacement.placement_group + bundle_index=0 + capture_child_tasks +
  CUDA_VISIBLE_DEVICES + NOSET env`. **`num_gpus_per_actor=0.01` (RLix mode)**
- **`get_engine_count()` method**: 新增, 返回派生量
  `sum(len(s.engines) for s in self.servers.values())` — 用 `srv.engines` (node-0
  router-facing engines, [rollout.py:225-232](external/miles/miles/ray/rollout.py#L225)),
  **不是 `srv.all_engines`**

**(c) Cleanup 失败语义 — `ray stop` CLI 收尾 (first build simplification)**:

First build **不引入 graceful cleanup APIs**, **不引入 driver 顶层 `ray.shutdown()`**.
失败语义完全靠 CLI side `ray stop` 收尾:

- `RayTrainGroup.__init__` / `RolloutManager.__init__` /
  `start_rollout_servers_from_worker_placements` 中途 raise → `MilesPipeline.initialize_pipeline()`
  抛异常 → driver process exit
- 用户在 shell 跑 `ray stop && ray start ...` 重启集群. **`ray stop` 杀 node 上所有
  Ray 进程** (raylet / GCS / actor / SGLang child process / placement group), 无视
  actor lifetime (detached 也杀), 无视 driver 是否正常退出 (ctrl-C / SIGTERM /
  uncaught exception 都 OK)
- 这意味着 plan **不约束 `MilesPipeline` / `MilesCoordinator` 的 `lifetime`**, 也
  不需要 driver 顶层 try/except + `ray.shutdown()`
- `ray stop` 是 per-node CLI; first build (Gate 1-3) 单节点一行命令; 多节点 (Gate 4
  之后) 需要 ansible / pdsh 批量 — 见 follow-up

driver 顶层模板 (`examples/rlix/run_miles_rlix.py`) 极简:
```python
def main():
    # ... orchestrator + coordinator + pipeline.initialize_pipeline() + main loop ...
    # 失败让 raise 自然 propagate; driver process exit; 用户 ray stop 收尾
```

Gate 4 (双 MILES pipeline 共节点) 仍能用 `ray stop` 兜底 — 但代价是杀掉**所有**
pipeline (包括别的 healthy pipeline). 那时需要 RLix orchestrator 接管 namespace-scoped
selective cleanup, 见 follow-up.

**not-cut 项** (production hardening, follow-up):
- graceful per-actor shutdown (`RolloutManager.shutdown` / `RayTrainGroup.shutdown` 含
  30s timeout + force-kill + dispose finally) + `__init__` self-cleanup (~150 行)
- multi-pipeline orchestrator-driven selective cleanup (namespace-scoped actor
  enumeration + selective `ray.kill`)
- receiver crash 容错

**(d) Round-trip structural validation (启动期 catch wiring bug, 仅 structural,
不做 identity self-check — first-build contiguous identity 永远 pass)**:
```python
all_placements = provider.get_all_rollout_engine_placements()
assert len(all_placements) == args.rollout_num_gpus // args.rollout_num_gpus_per_engine
for engine_index, wp in enumerate(all_placements):
    assert len(wp.gpu_ids) == args.rollout_num_gpus_per_engine
    assert wp.gpu_ids == sorted(wp.gpu_ids)  # first build identity 拓扑
# active subset 取决于 partial allocation; assert 不依赖 allocation 状态.
# get_active_engine_indices(allocated_gpus, tp_size) 在 init Step 7 派生.
```

NeMo 之所以要 `RayVirtualCluster`-shape adapter 是因为 `VllmGeneration` /
`RayWorkerGroup` 要求 cluster object 暴露 `world_size()` / `get_placement_groups()`
等方法. MILES 这边 `WorkerPlacement` 是 per-worker view, 不是 cluster wrapper, 但
仍是真实 adapter, 不能 simple `(pg, range, infer_dm)` wrapping.

**Gate 4 (双 MILES pipeline)** 通过此真实 adapter 实现: 两个 pipeline 共享 ROLL proxy
的同一组节点级 PG, scheduler 管 GPU id 分配, 不会 Ray 资源冲突.

改动量：~320 行 (`MilesPlacementProvider` adapter ~120 + `WorkerPlacement` dataclass +
identity validation ~20 + `start_rollout_servers_from_worker_placements` 新增 ~80 +
`RolloutManager.__init__` signature 扩展 + dispatch ~20 + `RayTrainGroup.__init__`
worker_placements 路径 ~80; **不含 cleanup API shutdown 方法**: 那些算 ~60 行单列
F12 cleanup 段)

**新增文件：** `miles/ray/placement_provider.py`

---

## 测试策略

### 验证环境

主 parity gates 按 **4 GPU** 设计，确保从首个端到端验证开始就覆盖
`rollout_num_gpus_per_engine=2` / SGLang TP>1 / partial overlap。唯一例外是 Gate 2
的负向 safety test：它故意使用 1 个 tp=2 engine 验证不能 shrink-to-zero。

- infer engines = 2
- `rollout_num_gpus_per_engine = 2`
- engine 0 → GPU 0,1；engine 1 → GPU 2,3
- training overlap engine 0 的 GPU 0,1
- engine 1 在 training 期间保持 active，覆盖 active in-flight refresh

### 测试工作负载

`examples/fully_async/fully_async_rollout.py` 作为 fully_async 外层；测试脚本需在
`run-qwen3-4b-fully_async.sh` baseline 基础上显式加入：

```bash
--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
--max-weight-staleness 2
```

因此测试目标不是默认 math single-turn example，而是 **fully_async + multi-turn custom
generate + staleness control**。默认脚本仅覆盖 fully_async single-turn，不覆盖 F3
turn-level redispatch。

### Parity 验证

以下 gates 不是降级阶段；每个 gate 都使用最终 tp>1 拓扑，只是验证侧重点不同。

### Gate 1: partial sleep/wake + routing 基础（infer_engines=2, tp=2）

```
配置：4 GPU, 2 SGLang engines (tp=2), fullasync rollout + multi_turn custom generate
测试：
1. 初始化 2 个 SGLang engine
2. 验证 shared-PG bundle mapping：engine 0 → GPU 0,1，engine 1 → GPU 2,3
3. fullasync round-robin 到 2 engines
4. sleep_partial([0]) — GPU 0,1 VRAM 释放，post-sleep assertion 通过
5. fullasync 自动跳过 sleeping engine 0，只用 engine 1
6. wake_up_partial([0]) — GPU 0,1 VRAM 恢复
7. fullasync round-robin 恢复

新增 invariant 验证：
(a) `_preempted_engines` 在 `sleep_partial(engine_indices)` 后含目标 engine_index；
    `wake_up_partial` 完成后清空
(b) post-sleep `assert_post_sleep_memory_below_threshold` 必须通过
(c) first-build 拓扑约束生效：非排序连续 `infer_device_mapping` fail fast；
    `RadixTreeMiddleware` 在 RLix mode fail fast 禁用

预期：全部通过，无 crash
```

### Gate 2: TP group sleep/wake safety（唯一 engine 禁止 shrink）

```
配置：2 GPU, 1 engine tp=2, fullasync + multi_turn custom generate
测试：dp=1 时 sleep_partial 不能 sleep 唯一 shard，TP NCCL 无错
```

### Gate 2.5: NCCL selective sync + Megatron NCCL destroy/re-init (engines=2, tp=2)

```
配置：4 GPU, 2 engines tp=2, fullasync + multi_turn custom generate + staleness control
测试：完整 training → offload → expand → sync 周期：
1. Megatron training step (tp=2 NCCL groups active)
2. build_cpu_bucket_cache — 所有 TP/PP/CP rank gather，cache owner 存
3. ReloadableProcessGroup.destroy_process_groups() — 销毁 TP NCCL，验证 VRAM 释放
4. SGLang wake_up overlap engine[0] (tp=2 collective_rpc 传播)
5. sync_selected_workers — 同时验证 active refresh engine[1] 与 expand sync engine[0]
   的 NCCL broadcast / receiver dual-mask 路径
6. 下一轮 training 前 reload_process_groups() — 重建 NCCL groups
7. 连续 3+ step，验证 destroy/reload 循环稳定

新增 invariant 验证：
(a) **M11.1 colocate sync = `cpu_serialize` 唯一路径** (M11.6 cuda_ipc Gate 补回):
    走新增 `update_weights_from_cpu_bucket` (Ray actor wrapper auto-deref'd `bytes` →
    tmpfs `/dev/shm/miles_cpu_bucket_{uuid}.pt` → HTTP `POST /update_weights_from_cpu_bucket`
    body 携带 file path, vendored fork patch `cpu_serialize_http_route.patch`)；
    receiver cpu_serialize-only ranks `destroy_collective_group` no-op
    (`is_group_exist` guard); Gate 验证: tmpfs file 写入 + 读取 + cleanup 全成功,
    sync 后 `ls /dev/shm/miles_cpu_bucket_*` 无 leak; per-bucket receiver 串行触发
    (sync_selected_workers 不并发); SGLang HTTP route 同步语义验证 (route 返回时
    所有 TP workers 已 ack)
(b) **M11.1 non-colocate sync = dynamic NCCL broadcast**: partial overlap 下 non-overlap
    engine 通过 NCCL broadcast 收 weight (cache_owner H2D staging → broadcast →
    receiver GPU). Receiver dual-mask `cpu_serialize_local_ranks +
    broadcast_local_ranks`. 验证: 同 bucket 同时有 cpu_serialize + broadcast receivers,
    receiver mask guard 各自 skip 无关 rank, no `KeyError` from `destroy_collective_group`
    on cpu_serialize-only ranks (`is_group_exist` guard).
(c) Warmup allreduce（F4-6）必须在每次 NCCL group USE 之前发出；故意制造 group 损坏验证
    raise 路径

预期：无 NCCL 错误，无 VRAM 泄漏，权重正确
关键：覆盖 M11.1 colocate (`cpu_serialize`) + non-colocate (NCCL broadcast) + receiver
      dual-mask (`cpu_serialize_local_ranks` + `broadcast_local_ranks`) + Megatron NCCL
      lifecycle。M11.6 加 cuda_ipc 路径 Gate (`ipc_local_ranks` mask 同步加回). 省下
      NeMo Gate 2.5 的 fallback rule — MILES 直接复用 ReloadableProcessGroup.
```

### Gate 3: 单 pipeline 端到端 fullasync GRPO (partial overlap)

```
配置：4 GPU, infer engines=2 (tp=2), train 占 GPU 0,1 (overlap engine[0]),
     fullasync + multi_turn custom generate + staleness control
测试：完整 fullasync loop —
     `actor_infer` 持有长期 GENERATION allocation，generation 持续后台，
     train 时 shrink overlap engine[0]，
     after_training: active refresh (in-flight sync engine[1]) → version publish → GPU release，
     scheduler expand engine[0] → selective sync + finalize + 原子激活，
     非重叠 engine 无全局 pause，
     `after_training(step)` 完成前 `before_training(step+1)` 不进入下一轮，
     version 一致性：两条路径 publish 同一 `_cache_ready_step`（无 double-bump），
     过渡窗口：in-flight active refresh 期间少量请求误标（tolerated）

新增 invariant 验证：
(a) Init bootstrap：`_cache_ready_step == -1` 在 `initialize_pipeline()` 完成后存在
(b) `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` (复用 RLix 既有 env var, 不再造 MILES_*)
    在 active refresh 人为 hang 时触发 crash (注入 receiver block, 验证 150s 内 raise)
(c) `begin_progress_batch` initial snapshot is `initial_completed=0` per
    Fix #1 contract; pre-queued groups in `worker.output_queue` flow through
    the first while-loop iter via `worker.get_completed_groups()` →
    `bump_completed(target_weight_version=current_weight_version)` and emit
    bucket-0 normally. Verify NO `worker.buffer / g.is_ready /
    g.target_weight_version` access in the implementation
(d) Multi-turn preempt: 注入 sleep_partial mid-turn，验证 turn-level redispatch 重做该
    turn 而不是从 turn 0 重做，已完成 turn 的 env state 保留
(e) Staleness control: 每个 turn 的 `weight_versions` 被记录；trajectory 使用
    `oldest_weight_version` 判断 staleness，超过 `--max-weight-staleness` 时 group
    recycle，未超过时进入 train_data / rollout metrics

此 gate 是 in-flight active refresh 在生产负载下的主要验证点
```

### Gate 4 (M11.2 milestone — NOT M11.1 MVP gate): 双 MILES pipeline 调度 (happy path only)

**M11.1 first build 不阻塞 Gate 4**. M11.2 multi-pipeline milestone 加回 happy path
only; production hardening (admission_epoch / cleanup race / graceful drain)
是 M11.2-tagged follow-up, **NOT a Gate 4 pass/fail criterion**. Crash cleanup
not validated by Gate 4 — 手动 `ray stop` 是 accepted recovery.

```
配置：4 GPU, infer engines=2 (tp=2), 两个 MILES fullasync pipeline
测试：两 pipeline 共享 GPU，通过 RLix scheduler 交替获得 training GPU，
     Perfetto trace 确认 GPU 时分复用
PG：两 pipeline 共享 RollResourceManagerProxy 的 shared PGs

新增 invariant 验证：
(a) 跨 pipeline `MilesCoordinator` / `MilesPipeline` / `MilesModelUpdateService`
    actor 具名隔离（F7 + F7-1），同一 namespace 不冲突
(b) 两 pipeline 各自的 selective sync 不互相干扰；master_port 在两个 pipeline 间
    通过 `SharedStorage` 原子 claim 不冲突（happy path 验证；receiver crash 容错
    超出本 milestone）
(c) Pipeline B init under contention (partial allocation main path):
    - Pipeline A holds 4 GPU as actor_infer (engines [0,1] active and serving).
    - Pipeline B `initialize_pipeline()` runs Step 1 → Step 6.6f
      (all-shell RolloutManager registered with empty Miles router on its own
      port; coordinator `_cache_ready_step=-1` published; bootstrap_active_engines
      called with frozenset() and `_active_engines_bootstrapped=True`).
    - Pipeline B Step 7 `_request_cluster_gpus(actor_infer, GENERATION)` blocks
      pending donor-shrink (see (e)).
    - Scheduler returns partial allocation (e.g., 2 of 4 GPU).
    - Verify ALL of:
        - Pipeline B `initialize_pipeline()` returns SUCCESS (no raise, no
          engine_count assertion fire).
        - `coordinator.get_active_engines()` returns the active subset
          (e.g., {0}); engine 1 stays in `shell` state on RolloutManager
          (handle is None, server not started, GPU not held).
        - Pipeline B Miles router's worker list contains exactly the active
          engine subset; shell engine URLs never appear; no generation
          traffic dispatched to shell engines.
        - M1 subset assert (`set(allocated) ⊆ set(declared)`) passed;
          `is_partial_allocation=True` was logged.
        - Pipeline B Step 7.3 consistency assert
          (`coordinator.get_active_engines() == active_engine_indices`) passed.

(d) Expand-before-first-after_training (base-version no-transport path):
    - Pipeline B is in (c) state (1 active + 1 shell), BEFORE its first
      after_training completes. So `coordinator._cache_ready_step == -1`
      from init Step 6.6e.
    - A subsequent `resize_infer(add=[shell_idx])` is triggered (e.g.,
      Pipeline A finishes a train_step and shrinks).
    - Verify ALL of:
        - `_expand_workers` does NOT raise on the
          `_cache_ready_step is not None` guard (init Step 6.6e took
          care of it).
        - `manager.expand_engines([shell_idx])` lazy-creates the SGLang
          Ray actor + onload from `args.hf_checkpoint` (state shell → loading).
        - `service.sync_selected_workers(sync_id, [shell_idx], -1)` enters
          the **`version == -1` short-circuit** (Fix #13 + #15): runs ONLY
          `manager.set_weight_version(-1, [shell_idx])`. Verify neither
          `cache_owner_actor.run_sync_session` nor
          `engine.finalize_weight_update` is called for the expand
          (instrumented call counters stay at pre-expand values).
        - `manager.activate_routing([shell_idx])` flips loading → active,
          adds worker to router; router worker list now contains both
          original active engine AND newly-activated former-shell engine.
        - Engine's reported version after sync = -1 (base label, equivalent
          to `args.hf_checkpoint` weights from onload; under Fix #14
          fail-fast `args.load`/`args.ref_load` ⊆ {None, hf_checkpoint}
          guarantees train and rollout base weights are equivalent).
    - Pipeline B 后续 train_step 完成 → `sync_base_weights_to_active(0)` 在
      `_resize_sync_lock` 内把 `coordinator._cache_ready_step` 从 -1 翻新到 0 +
      service.sync_selected_workers atomic unit (a)+(b)+(c) 推到 active engines
      → 进入正常闭环.

(e) Donor-shrink-before-recipient-expand ordering (control plane invariant):
    Required: T1 < T2 < T3 strict —
    - T1: A.coordinator.resize_infer(remove=[0]) returns (A engine[0]
      reaches `offloaded`, GPUs released from A's view).
    - T2: B.coordinator.resize_infer(add=[0]) starts (B's shell engine[0]
      → loading).
    - T3: B's pending Step 7 `_request_cluster_gpus` returns to driver.
    Failure prevented: B spawning Ray/SGLang actor on GPUs 0,1 while A
    still holds them. T2 < T3 specifically prevents Step 7.3 sanity from
    observing transient mismatch on `coordinator.get_active_engines()`.
    Test instrumentation is per-test concern (timestamp logger on
    coordinator + driver side); not part of production code path.

NOT a Gate 4 pass/fail criterion (M11.2-tagged follow-up, separate
deliverables; on crash, manual `ray stop` is accepted recovery):
admission_epoch race, orchestrator cleanup, graceful actor drain.
```

---

## 文件改动总清单

### MILES 侧

| 文件 | Feature | 改动 | 行数 |
|---|---|---|---|
| `miles/backends/sglang_utils/sglang_engine.py` | F1, F2, F5+6 | **M11.1**: post-sleep VRAM assertion (走 `/server_info` `memory_usage` GB 单位 + `args.miles_post_sleep_vram_threshold_gb`, 不读 actor `torch.cuda`) + `is_idle()` worker method (走 `/v1/loads` `slot["num_total_reqs"]`, **不是 `/server_info` 的 `num_running_reqs` (后者不存在)**, 带 timeout + raise_for_status) + `abort_all_requests()` (POST `/abort_request {"abort_all": true}`) + receiver-side 方法: **新增 `update_weights_from_cpu_bucket(payload_bytes, cpu_serialize_local_ranks, ...)`** (**`payload_bytes` 不是 `payload_ref` — Ray auto-deref top-level ObjectRef**; **wrapper tmpfs cleanup owner = `try/finally os.unlink(/dev/shm/miles_cpu_bucket_{uuid}.pt)`**, 见 §F4 §B tmpfs file lifecycle invariant) + setup/destroy_collective_group with `is_group_exist` no-op guard + broadcast_parameter + finalize_weight_update. **M11.6 加**: 既有 `update_weights_from_tensor` 扩展 `ipc_local_ranks` 参数 (tp>1 dual-mask, 对齐 ROLL `vllm_strategy.py:685`). **不引入 F1 coordinator ack publish flag** (actor.py:58 已无条件挂 monkey patch) | M11.1: +180; M11.6: +30 |
| `miles/ray/rollout.py` | F1, F2, F3, F9, F12 | `EngineInfo` dataclass (`state: Literal["shell", "active", "disabling", "offloaded", "loading"]`, **5 态** — 加 `shell` 表示 slot reserved/无 Ray actor/无 SGLang server/无 GPU; 字段 `handle` / `worker_url` / `server_group` 在 shell 状态下为 None, `bundle_index` / `gpu_ids` / `placement` 始终 populated; **取代 worker-local `is_engine_resident_on_gpu` flag**) + `RolloutManager.{offload,onload,onload_weights,onload_kv}(engine_indices=None)` 仅在 Manager 层 + `shrink_engines/expand_engines` 复合操作 (`expand_engines` 内部 dispatch on entry state: **`shell → loading`** lazy `_create_sglang_actor(engine_idx, placement)` + `onload()`, **`offloaded → loading`** existing `wake_up()`; 返回 engines in `loading` 状态, 不开 routing) + **`activate_routing(engine_indices)` method** (`loading → active` + open router; coordinator 在 service.sync_selected_workers 完成后调用) + abort-drain-sleep + `_routing_lock` compound 不变量 + worker registration with `engine_index` (F3 router metadata 数据源, **不注入 GenerateFnInput.preempt_state**, F3 改用 router-side classification) + progress callback hook (走 `RLixHooks` protocol, 不直接 import RLix 类型) + **`get_engine_count()` method** (返回 `len(self._engines)` == declared engine_count, 包含 shell; sanity check 用 `srv.engines` 不是 `srv.all_engines`) + **`set_weight_version(version, engine_indices=None)` method** (fan-out 到 per-engine `update_weight_version.remote`; 由 service.sync_selected_workers 内部调用, 不在 pipeline 直接调; engine_indices=None 退化到所有 active engines, shell engines 永远跳过) + **`get_engine_handles(engine_indices: Set[int]) -> dict[int, RayActorHandle]` method (Fix #5)** (read-only snapshot of per-engine SGLangEngineActor handles; raise if any engine_idx is in `shell` state; 由 `MilesModelUpdateService.sync_selected_workers` at sync entry 一次性 fetch — service 据此驱动 per-engine cpu_bucket / NCCL broadcast RPC, 不再每次 RPC 找 manager) + **`activate_routing(engine_indices)` method** (loading → active + add_worker to router; coordinator 在 service.sync_selected_workers 完成后调用, 是唯一的 add_worker 入口 — `shrink_engines` 是唯一的 remove/disable 入口; per-pipeline 独立 router port 在 `__init__` 内绑定 `args.sglang_router_port` from provider's port pool, 不 inherit peer pipeline 的 router port) + **`RolloutManager.__init__` 扩展接受 `all_engine_placements: list[WorkerPlacement]` (length engine_count, 全表) + `active_engine_indices: frozenset[int]` (active 子集; partial allocation 下是 strict subset of range(engine_count))** + 新增 `start_rollout_servers_from_worker_placements` (only iterates active engines per `active_engine_indices`; per-worker 节点级 PG + bundle_index=0 + capture_child_tasks + manual CUDA_VISIBLE_DEVICES + NOSET env + 显式 `base_gpu_id=0` (M9 — post-CVD local index, 不传 `wp.gpu_ids[0]` physical id) + 不 fallback `get_base_gpu_id`; shell engines 仅初始化 `EngineInfo` slot 持有 placement, 不 spawn Ray actor). **M4 self-cleanup**: `start_rollout_servers_from_worker_placements` ctor loop try/except — 中途失败 kill 已创建 SGLang engine actors. **M4 `RolloutManager.shutdown_hard()` MVP**: stop monitors (背景 task 取消) + `for h in self._engine_actors: ray.kill(h, no_restart=True)` (skip shell — `handle is None`); 不做 graceful drain / abort RPC / force-kill timeout (follow-up) | +320 |
| `miles/ray/actor_group.py` | F4, F12 | `RayTrainGroup.__init__` 扩展接受 `worker_placements: list[WorkerPlacement] \| None` (替代 pg=三元组), 完整新签名 `(args, num_nodes, num_gpus_per_node, *, pg=None, worker_placements=None, num_gpus_per_actor=0.4 standalone / 0.01 RLix mode, role, with_ref)`; 必须 pg xor worker_placements 二选一. 每 actor 用对应 `WorkerPlacement.placement_group + bundle_index=0 + capture_child_tasks + CUDA_VISIBLE_DEVICES + NOSET_VISIBLE_DEVICES_ENV_VARS_LIST`. **RLix path 显式传 `local_rank=0` 给 `TrainRayActor.__init__`** (Fix #9 — fractional GPU + manual CVD 下 `ray.get_gpu_ids()` 不在 manual CVD 列表里, 既有 `get_local_gpu_id()` 会 ValueError 或设错 LOCAL_RANK; 1 actor 1 GPU 下 local_rank 永远 0). standalone path 不传, 走旧 fallback. **M4 self-cleanup**: `__init__` 包 try/except, ctor loop 中途失败 (任何 `TrainRayActor.options(...).remote(...)` 或 `init.remote()` raise) 时 `for h in self._actor_handles: ray.kill(h, no_restart=True)` + `self._actor_handles = []` + raise 透传. **M2 sender API**: 加 `build_cpu_bucket_cache(step)` fan-out + `collect_cache_owner_roles() -> list[(rank, is_owner, actor_handle)]` (worker rank ↔ actor handle 配对). **不引入 graceful shutdown method** (follow-up) | +115 |
| `miles/ray/train_actor.py` (Fix #9) | F12 | `TrainRayActor.__init__` 加可选 `local_rank: int \| None = None` kwarg. None 时 fallback 到既有 `get_local_gpu_id()` (standalone parity); RLix path 由 RayTrainGroup 显式传 `local_rank=0` (1 actor 1 GPU under fractional + manual CVD, local rank 永远 0; 既有 `cvd.split(",").index(str(ray.get_gpu_ids()[0]))` 在 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 + manual CVD 路径下不成立). 设 `os.environ["LOCAL_RANK"] = str(local_rank if local_rank is not None else get_local_gpu_id())`. 不改 `get_local_gpu_id()` 自身 (兼容 standalone) | +5 |
| `miles/router/router.py` | F3 | `/disable_worker` / `/enable_worker` / `/remove_worker` endpoint + `/add_worker?engine_index=...` 扩展 + 4 个 router state (`worker_request_counts / worker_failure_counts / dead_workers / enabled_workers / worker_engine_index_map`) + 4 个 internal helper (`_add_worker_internal / _remove / _disable / _enable`) 完整 lifecycle 维护 (含 `setdefault` 避免 re-add 清零 + `add` 时 `dead_workers.discard` + `disable` reset failure_count) + `_use_url` 改用 `enabled_workers - dead_workers` (Critical Invariant: 不只 metadata) + `do_proxy` 内 mutate JSON body 注入 `meta_info["miles_engine_index", "miles_admission_disabled"]` (**仅 path == "generate"**, response 时刻读 `enabled_workers`; race window 边界见 F3 admission_disabled 语义边界段) + **header hardening**: `do_proxy` strip `Content-Encoding` (改 body 后旧 header 不能保留, 否则客户端解码崩); Content-Length 由 `build_proxy_response` 既有逻辑 strip + `JSONResponse` 重算 + `_health_check_loop` 只 probe `enabled_workers - dead_workers`. **不引入 admission_epoch race 防御** (turn retry 兜底; production 多 pipeline 高频 shrink/expand 触发再加, follow-up) | +135 |
| `miles/router/middleware_hub/radix_tree_middleware.py` | F3 | RLix mode 下不改成 pass-through；启动校验禁止加载 `RadixTreeMiddleware`，`partial_rollout + radix_tree` 留作 follow-up | +0 |
| `miles/rollout/generate_hub/multi_turn.py` | F3 | 删除 `assert not args.partial_rollout`（line 29）+ 强制 `payload["stream"] = False`（router metadata 注入要求 JSON body）+ snapshot-then-retry turn loop（`MAX_TURN_REDISPATCH_ATTEMPTS = args.rollout_num_gpus // args.rollout_num_gpus_per_engine`, retry 用尽 raise `EnginePreemptedError` fail fast）+ `_is_scheduler_preempt(output, rlix_mode=DO_TIME_SHARING)` 判定 (RLix mode 缺 metadata raise `RLixRouterMetadataError`, **不读 GenerateFnInput.preempt_state**) | +70 |
| `miles/rollout/base_types.py` | F3 | `class EnginePreemptedError(Exception)` + `class RLixRouterMetadataError(Exception)`. **不扩 GenerateFnInput** (避免改 sglang_rollout.py:266 / inference_rollout_common.py:82 两个构造点) | +5 |
| `miles/rollout/generate_utils/generate_endpoint_utils.py` | F3 | `_snapshot_turn_state` / `_restore_turn_state` 辅助 | +40 |
| `miles/backends/megatron_utils/update_weight/cpu_bucket_cache.py` (**新增**) | F4 | CPU bucket build + lookup + `_cache_ready_step` 单槽指针 | +180 |
| `miles/backends/megatron_utils/actor.py` | F4 (M2 sender API) | `MegatronTrainRayActor` 加 `build_cpu_bucket_cache(step)` (HF gather, cache_owner rank 真存; 其它 rank 参与 collective gather 但丢弃结果, 见 F4 §1 HF-format invariant; method 内 `with self._cache_lock` 持锁) + `report_cache_owner_role() -> tuple[int, bool]` (基于 `_is_distributed_src_rank`) + **`run_sync_session(plan: SyncSessionPlan) -> None`** (cache_owner 实现, 其它 rank raise; Fix #6 composite RPC — `with self._cache_lock` 单 method 单临界区: snapshot bucket list + `_cache_ready_step`, classify cpu_serialize vs broadcast targets via `plan.target_handles` + `plan.cpu_serialize_local_ranks` + `plan.broadcast_local_ranks`, per-bucket dispatch (cpu_serialize: `engine.update_weights_from_cpu_bucket.remote(payload_ref, ...)` — payload **不携带 weight_version**, Fix #3; broadcast: TCP-rendezvous group setup via `init_process_group(init_method='tcp://master_addr:master_port', ...)` + per-bucket `dist.broadcast(src=cache_owner_global_rank, group=tmp)` + `dist.destroy_process_group(tmp)`, Fix #7), 整段在锁内, 调用结束 = 所有 buckets 已被所有 targets load 完成). **不**: `get_bucket_count` / `serialize_bucket_to_objref` / `setup_collective_group` / `broadcast_bucket` / `destroy_collective_group` 不再是 top-level Ray methods, 全部降级为 `run_sync_session` 内部 helpers (Fix #6 — 解决跨 RPC 持锁矛盾). **M11.6 加**: `run_sync_session` 内部 cuda_ipc adapter (CPU cache → H2D staging → IPC handle, ~50-80 行) for cuda_ipc transport classification | +180 |
| `miles/ray/placement_provider.py` (**新增**) | F12 | `MilesPlacementProvider` 真实 adapter — 接收注入的 `RollResourceManagerProxy` (**不自建**, 从 coordinator 传入) + 接收注入的 `train_device_mapping` / `infer_device_mapping` (**与 F8 driver `register_pipeline` 时声明的 `cluster_device_mappings[role]` 同源**, 不在 provider 内 `list(range(...))` 重新派生 — 防止多 pipeline 场景下 Pipeline B 注册 `[4,5,6,7]` 时 provider 仍申请 `[0,1,2,3]` 与 Pipeline A 冲突) + `WorkerPlacement` dataclass (`placement_group / node_rank / gpu_ids / bundle_index`, per-worker view of ROLL `List[List[Dict]]`; **multi-node-compatible structural invariant** per Cut 1' — node-local gpu_ids, 不假设 global GPU id == local id, dev gate 单机 4 卡不堵死多机部署) + **`get_all_rollout_engine_placements() -> list[WorkerPlacement]`** (length `engine_count`, 从 declared `infer_device_mapping` 派生, **independent of `allocated_actor_infer_gpus`**; F4 partial GENERATION allocation 下 shell engine slot 仍持有声明的 placement) + **`get_active_engine_indices(allocated_gpus, tp_size) -> frozenset[int]`** (从 declared mapping `infer_device_mapping.index(g) // tp_size` 派生; 验证 each engine 的 tp_size GPU 全分配或全没分配, half-engine raise) + `get_train_workers()` 调 `proxy.allocate_placement_group(world_size, device_mapping=declared_mapping)` 切成 per-worker view + 启动期 structural assert (`len(all_engine_placements) == engine_count`, `len(wp.gpu_ids) == tp`, `wp.gpu_ids == sorted(...)` 验证 first build contiguous; **不做 identity scheduler_to_engine round-trip** — first build 拓扑下永远 pass, 是 dead assert; **不做 `scheduler.get_allocation` cross-check** — RLix 没此 public API; 二者都留 follow-up) | +130 |
| `miles/ray/placement_group.py` | F12 | `create_placement_groups(args, *, external_provider=None)` | +20 |
| `miles/utils/arguments.py` | F1, F4, F10 | 新增 RLix-mode tuning args (3 个 new): `args.miles_model_update_bucket_size_mb` (默认 512), `args.miles_post_sleep_vram_threshold_gb` (默认 1.0), `args.model_update_transport` (默认 `"cuda_ipc"`; **M11.1 RLix mode F10 强制 `cpu_serialize`** — vast.ai 受限容器无 IPC; M11.6 加 cuda_ipc 选项 + smoke-test capability check). **不新增 device_mapping args** (派生路径). `args.offload_train` 既有, F10 fail-fast 强制 True | +30 |
| `examples/fully_async/fully_async_rollout.py` | F2, F3, F9 | abort 触发口 + 接受 `rlix_hooks` 参数 (standalone 走 `NoOpRLixHooks`) + `begin_progress_batch(target_weight_version, step_target_groups, initial_completed, mode=None, adapter_id=None)` (M11 hook signature; **M11.5 forward-compat 字段 `mode/adapter_id` nullable 保留**) / `bump_completed(target_weight_version=...)` / `end_progress_batch()` 包裹 wait window (**只调 hook 抽象, 不 import RLix 类型, 不 `ray.get_actor` coordinator**) + **`initial_completed = 0` semantics**: `begin_progress_batch` 在 wait window 之前调用, 此时 `data = []` (caller-local), output_queue 内已 queued 的 group 在 while 循环首次 iter 通过 `worker.get_completed_groups()` → `bump_completed` 流转. **不读 worker 内部** — `AsyncRolloutWorker` 没有 `worker.buffer` / `g.is_ready` / `g.target_weight_version` 字段; 任何 worker-side scan 路径都是 fake. 不引入 `_completed_count_by_step` / `count_ready_groups_for_step` 兜底. `bump_completed` 用 caller-local `current_weight_version` 作 `target_weight_version` (group 单元自身没这个字段, Sample.weight_versions 才是 per-decision). + **`_FatalError` sentinel + `task_done_callback` catch `(EnginePreemptedError, RLixRouterMetadataError)` 走 queue sentinel + 主循环 dequeue 检测 `isinstance(result, _FatalError)` raise** (queue 单路径已足够) | +50 |
| `train_async.py` | F11 | RLix mode fail-fast guard（detect `RLIX_CONTROL_PLANE=rlix` → raise；不在此处分支 RLix 行为，仅 reject）+ partial overlap topology fail-fast (`train_devices_subset_of_infer(args)` helper 在 F10) | +20 |
| `examples/rlix/run_miles_rlix.py` (**新增**) | F8, F11 | RLix entry driver — orchestrator allocate/register/admit (`cluster_device_mappings` 派生自 `actor_num_nodes * actor_num_gpus_per_node` / `rollout_num_gpus`, 不依赖新 args) + `MilesCoordinator.options(...).remote(pipeline_id=, pipeline_config=)` (keyword-only) + `coordinator.create_pipeline_actor.remote(pipeline_config=)` (keyword-only) + `pipeline.initialize_pipeline()` + main loop. **不加 driver 顶层 try/except + ray.shutdown** (失败语义 = raise 自然 propagate → driver exit → 用户 `ray stop` 收尾, 详见 F12 (c)) | +110 |
| `miles/utils/rlix_hooks.py` (**新增**) | F9, F11 | `RLixHooks` protocol + `NoOpRLixHooks` 默认 + import seam | +30 |

### RLix 侧

| 文件 | Feature | 改动 | 行数 |
|---|---|---|---|
| `rlix/pipeline/miles_pipeline.py` (**新增**) | F5+6, F8, F10, F11, F12 | `MilesPipeline` actor — registration + validation (F10 含 single_updateable / EP fail-fast / stream / bucket-size / **`assert args.offload_train`** / **`rollout_num_gpus % rollout_num_gpus_per_engine == 0`** / **M11.1 RLix mode 强制 `model_update_transport == "cpu_serialize"`** (vast.ai 受限容器, cuda_ipc → M11.6). **M11.1 forbids cross-node rollout engine, not multi-node DP** — 既有 M3 assert `rollout_num_gpus_per_engine <= num_gpus_per_node` (按 rollout engine 总 GPU 数计, 与 sharding 形式 TP/EP 无关; train 侧 Megatron actor 不在此 assert 范围内) 已正确表达 "rollout engine fit within one node"; multi-node DP (rollout engines 跨节点, 每个 node-local) M11.1 OK; cross-node rollout engine (intra-engine sharding 跨节点拼) **not in scope (no later milestone re-enables it)**. 不引入冗余 `actor_num_nodes == 1` assert / **M7 `assert not args.async_save`**) + resize_infer + expand+selective sync + `_before_training(step)` hook (`_request_cluster_gpus(actor_train, ACTOR_TRAINING, global_step=step)` + `actor_train.onload()`; 每 step 必调一次, 包括 step 0 — init 末尾已 offload, ROLL strategy auto-wake 在 MILES `RayTrainGroup` 不存在; partial actor_train allocation INVALID, fail-fast `set(allocated)==set(declared)`) + `_after_training(step)` hook (调用 `version = ray.get(coordinator.sync_base_weights_to_active.remote(step))` **显式传 step**, coordinator 在 `_resize_sync_lock` 内同步更新 `_cache_ready_step` 与 service.sync 是同一原子操作, 避免 stale version label → `self._current_weight_version = version` → `notify_release_cluster_gpus(actor_train)`; **不再调 `publish_cache_ready_step`** (sync_base_weights_to_active 已在 lock 内更新 coordinator state); **不直接调用 `_finalize_weight_update` / `rollout_manager.set_weight_version`** — service.sync_selected_workers atomic unit 内含, 见 F5+6 ownership 表 + Sequence diagram) + `_init_lock: threading.Lock` 字段 + init bootstrap **sync def + `run(coro)`** (不 async, 避免半 async/sync 阻塞 event loop) **新顺序 (RLix scheduler 在 Phase 5 执行 `resize_infer(add)` RPC 早于 Phase 6 signal pending GEN waiter; 因此 RolloutManager + service resources + base version + bootstrap MUST 全部 6.6 完成于 Step 7 之前)** (Step 1 request train → Step 1b RayTrainGroup ctor with `worker_placements=` and `num_gpus_per_actor=0.01`, **RLix path 显式 `local_rank=0` per actor** → Step 2 `run(actor_train.init())` → Step 3 `run(actor_train.onload())` → Step 4 `run(actor_train.build_cpu_bucket_cache(step=-1))` (M2 fan-out, 不 driver-local) → Step 5 `run(actor_train.offload())` → **Phase 1 finally: `_notify_release_cluster_gpus(actor_train)`** (本地方法, NOT `.remote()`; 释放 scheduler-side allocation 让 Pipeline B Gate 4 拿得到 train) → Step 6.5 `run(actor_train.collect_cache_owner_roles())` 收 cache_owner_actor handle (M2) → **Step 6.6a `provider.get_all_rollout_engine_placements()`** (declared 全表, length engine_count, no scheduler interaction) → **Step 6.6b `RolloutManager.options(...).remote(args, all_engine_placements=..., active_engine_indices=frozenset())`** (all-shell ctor; **starts empty Miles router immediately**, binds per-pipeline `args.sglang_router_ip/port`; 不 spawn SGLang actors) → **Step 6.6c `get_engine_count()` sanity** (== len(declared), 包含 shell slots) → **Step 6.6d `coordinator.register_model_update_resources(cache_owner_actor, rollout_manager)`** (lazy service ctor args) → **Step 6.6e `coordinator.publish_cache_ready_step(-1)`** (init Step 4 已 build base cache; v=-1 short-circuit 路径不读 cache bytes 但需 version 标签; 必须在 Step 7 前发布) → **Step 6.6f `coordinator.bootstrap_active_engines(frozenset())`** (空集 bootstrap 合法; coordinator `_active_engines_bootstrapped: bool` flag 翻 True; `_active_engine_indices` 由 _expand_workers 在 Step 7 期间填充) → **Step 7 `_request_cluster_gpus(actor_infer, GENERATION)`** (期间 scheduler 调 `coordinator.resize_infer(dp_ranks_to_add=...)` 触发 `_expand_workers`: shell→loading lazy `_create_sglang_actor` + onload from `args.hf_checkpoint` → service.sync v=-1 short-circuit (no transport, no finalize, only `manager.set_weight_version(-1)`) → `manager.activate_routing` (loading→active + add_worker); 然后 Step 7 request 才返回) → **Step 7.1 M1 subset assert** `set(allocated) ⊆ set(declared)` + log `is_partial_allocation` (Gate 4 多 pipeline contention 必经路径) → **Step 7.2** `active_engine_indices = self._placement_provider.get_active_engine_indices(allocated_gpus, tp_size)` (frozenset, half-engine raise; provider 端 logic, 不在 pipeline 端复刻) → **Step 7.3 consistency assert** `coordinator.get_active_engines() == active_engine_indices` (验证 _expand_workers 填充与 pipeline 派生一致) → except (Phase 2 cleanup scope = Step 6.5 + Step 6.6a-f + Step 7 + Step 7.1-7.3): **M4 hard cleanup** order = `ray.kill` train actors (覆盖 Phase 1 finally 因 `train_init_succeeded=True` 而保留, 但 Phase 2 失败必须连带清) → `actor_infer.shutdown_hard.remote()` w/ 10s timeout + `ray.kill` actor_infer (manager 在 6.6b 后存在, 失败必须清) → conditional `_notify_release_cluster_gpus(infer)` (gated by `actor_infer_allocated` bool flag, 仅在 Step 7 successful return 后置 True) + raise) + `_get_coordinator_handle()` lazy resolver (复用 ROLL pattern) + bundle mapping consume `MilesPlacementProvider.get_train_workers` for actor_train + `get_all_rollout_engine_placements` (declared 全表) + `get_active_engine_indices(allocated_gpus, tp_size)` for actor_infer (active subset under partial allocation, frozenset) + **缓存 register_pipeline 时声明的 `train/infer_device_mapping`** (从 args 派生, 同时传给 scheduler `register_pipeline` + `MilesPlacementProvider.__init__`, 单一 source of truth, 不在 provider 内重新派生) + **F10 startup fail-fast (Fix #14)**: RLix mode `assert args.load is None or args.load == args.hf_checkpoint` AND `assert args.ref_load is None or args.ref_load == args.hf_checkpoint` (resume / 非等价 ref_load 会让 v=-1 no-transport 路径服 stale weights; 真正的 CPU-only base sync 是 follow-up) | +680 |
| `rlix/pipeline/miles_coordinator.py` (**新增**) | F3, F5+6, F7, F9 | `class MilesCoordinator(Coordinator)` — **不 subclass `PipelineCoordinator`, 不 call super().__init__** (避免触发 ROLL `_validate_config_schema` / `_validate_cpu_only_reward` / `_validate_vllm_sleep_level` / `_validate_offload_nccl` 4 个吃 ROLL config 字段 validator, MILES args 没这些字段). **手动 init**: `_pipeline_id`, `_ray_namespace`, `_pipeline_env_vars`, `_resource_manager_proxy` (singleton, 注入给 placement provider), `_resize_sync_lock`, `_progress_lock`, `_scheduler_reports`, `_coord_progress_last_bucket`, `_active_engine_indices: Set[int]` (init = `set()`), **`_active_engines_bootstrapped: bool` (init = False; flips True on first `bootstrap_active_engines`, used to detect double-bootstrap including empty-set bootstrap)**, `_cache_ready_step: Optional[int]` (init = None), `_rlix_scheduler` actor handle, `_model_update_service` lazy slot, `_model_update_resources` (cache_owner_actor + rollout_manager handles). **复制并适配** (不继承): `report_progress_from_scheduler` / `clear_progress_stream` / `_aggregate_and_emit` / `_inject_pipeline_env_vars` 4 个 backend-neutral 方法. **必须实现 Coordinator ABC abstract `sync_lora_weights`** (LoRA out of scope, raise unsupported stub, 否则 ABC 实例化 TypeError). **新增**: `bootstrap_active_engines(engine_indices: Set[int])` (允许空集; 由 `_active_engines_bootstrapped` flag detect double-call, 不靠 set 真值), **`get_active_engines() -> frozenset[int]`** (read-only snapshot for pipeline Step 7.3 consistency check), **`publish_cache_ready_step(step: int)` — 仅 init bootstrap 用, 发布 base 版本 -1 (Pipeline B 在 first after_training 之前可能就被 expand). active refresh path 不再调用此 RPC, sync_base_weights_to_active 已在 lock 内更新 coordinator state**, **M2 P0-1 `register_model_update_resources(*, cache_owner_actor, rollout_manager)` — 缓存 sender Megatron actor handle + receiver RolloutManager handle 进 `_model_update_resources`, lazy-init `MilesModelUpdateService` 时取出 (避免 init bootstrap 多一次 ray.get 创建 service)**, **`sync_base_weights_to_active(step: int) -> int`** (**signature 必须显式接收 step, 否则 coordinator 用 stale `_cache_ready_step` 给 service 推 version inversion**; 在 `_resize_sync_lock` 内同步操作: `self._cache_ready_step = step` → `service.sync_selected_workers(sync_id, _active_engine_indices, step)` (lazy 构造 service 若未创建); 返回 step. 退化拓扑 active set 为空时也更新 `_cache_ready_step` 后短路返回, 保证 coordinator state 始终最新), `resize_infer` / `_expand_workers` (**guard: `_cache_ready_step is not None`, raise if violated**; first build identity 拓扑, scheduler dp_rank == MILES engine_index, 调 `manager.expand_engines(...)` → `service.sync_selected_workers(sync_id, target, version)` → `manager.activate_routing(...)`; 非连续 mapping 解禁时引入 `MilesPlacementProvider.scheduler_to_engine` 双向 lookup, follow-up), `create_pipeline_actor(*, pipeline_config)` 创建 `MilesPipeline` actor. **`_active_engine_indices` 字段类型: `Set[int]`** | +280 |
| `rlix/pipeline/miles_model_update_service.py` (**新增**) | F4, F5+6 | 简化版 ModelUpdateService — **M11.1 实现一条 colocate transport `cpu_serialize`** (新 SGLang admin route `/update_weights_from_cpu_bucket` + tmpfs `/dev/shm` file path, 见 §F4 §B Case B) + **dynamic NCCL group 生命周期 (non-colocate broadcast, partial overlap 必需)** + 单槽 versioning + warmup allreduce + port claim 释放 (happy path 无条件回 pool; 不处理 receiver crash 容错) + **cache owner 由 init 阶段 worker `report_cache_owner_role.remote()` 上报, 不在 sync 路径运行时查询** + **M2: `__init__` 加 `cache_owner_actor` 参数 + `rollout_manager` 参数 (Megatron worker actor handle + RolloutManager handle), sender 走 single composite RPC `cache_owner_actor.run_sync_session(plan: SyncSessionPlan).remote()`** (Fix #6 — 替代之前的 5 个 split sender RPCs `serialize_bucket_to_objref / setup_collective_group / broadcast_bucket / destroy_collective_group / get_bucket_count`. 后者降级为 cache_owner 内部 helpers, 不再是 top-level Ray methods. `_cache_lock` 在 `run_sync_session` 单 method 内持有, 不跨 RPC.) + **`sync_selected_workers(sync_id, target_engines: Set[int], version: int) -> int` atomic unit**: 入口先 `handles = ray.get(rollout_manager.get_engine_handles.remote(target_engines))` (Fix #5 — service 调 manager 的 read-only entry, 拿 per-engine receiver handle); 然后 dispatch: **`if version == -1`** (Fix #13+#15): 跳过 (a)/(b)/run_sync_session, 只跑 (c) `manager.set_weight_version(-1, target_engines)` 后 return -1 (target engines 已通过 SGLang `onload()` 从 `args.hf_checkpoint` 加载 base, 无需 transport, 无需 finalize); **`else`**: build `SyncSessionPlan(sync_id, version, group_name, master_addr, master_port, timeout_s, target_handles=handles, cpu_serialize_local_ranks={…}, broadcast_local_ranks={…}, comm_ranks={…})` → 单次 `cache_owner_actor.run_sync_session(plan).remote()` 完成 (a) per-bucket transport (per-bucket payload **不携带 weight_version**, Fix #3) → 然后 service 跑 (b) `[h.finalize_weight_update.remote() for h in handles.values()]` fan-out → (c) `manager.set_weight_version.remote(version, engine_indices=target_engines)`; 整个 atomic unit wrap in 单次 `asyncio.wait_for(timeout=ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S)`, **不新增 MILES timeout env var**; pipeline / coordinator 不直接调 finalize 或 manager-level set_weight_version. **service-manager 关系**: service 调 manager 限于 `get_engine_handles` (read-only at sync entry) + `set_weight_version` (single write at sync exit); manager 从不调 service. + `master_port` 由 service `get_free_port()` + `SharedStorage MASTER_ADDR_PORT:*` claim 决定后传给 sender + 所有 receiver (禁 `master_port=0`) + `sync_id` 参数 + init bootstrap step=-1 path. **M11.6 加**: `cuda_ipc` colocate adapter (CPU cache → H2D staging → IPC handle), ~50-80 行 | +290 |
| `rlix/pipeline/miles_hooks.py` (**新增**) | F9 | `MilesRLixHooks` 实现 `RLixHooks` protocol — 把 `report_progress(collected, bucket, ...)` 等 plain kwargs 打包成 `ProgressReport(metrics={...})` fire-and-forget 给 `coordinator.report_progress_from_scheduler.remote(...)`; `clear_progress` 转 `coordinator.clear_progress_stream`. RLix-side import seam, MILES side 不依赖 | +50 |

### 测试

| 文件 | 改动 | 行数 |
|---|---|---|
| `tests/test_partial_sleep_wake.py` (**新增**) | Feature 1-3 单元测试 | +150 |
| `tests/test_miles_pipeline.py` (**新增**) | Feature 4-6 集成测试 | +200 |

**总计：~2700 行** (在 "~2790 production-grade" 基础上经多轮简化 + M-series patches):

第八轮简化 pass 砍掉 ~335 行防御性工程:
- F3 router admission_epoch race 防御 + epoch start/end 比对: -50 (follow-up)
- fully_async _fatal_error flag 双路径 (queue sentinel 单路径已足够): -20
- F12 round-trip identity self-check (first build contiguous identity 永远 pass): -15
- F1 coordinator-side NCCL teardown ack verification (actor.py:58 已无条件挂): -10
- F4 cuda_ipc Case A 详细描述折叠到 follow-up (RLix mode 不走): -20

MVP M-series 加回 ~245 行 (账本一致性 + sender API):
- M2 sender API on `MegatronTrainRayActor`: +160
- M2 RayTrainGroup `build_cpu_bucket_cache` fan-out + `collect_cache_owner_roles`: +50
- M4 `RayTrainGroup.__init__` self-cleanup + `RolloutManager.shutdown_hard` + ctor
  loop self-cleanup + `MilesPipeline` failure path hard-kill: ~+90 (graceful drain
  / abort RPC / force-kill timeout 仍 follow-up)
- M2 `MilesModelUpdateService.__init__(cache_owner_actor)` 注入 + sender path 走
  `cache_owner_actor.X.remote(...)`: +20
- M1 `allocated_actor_infer_gpus` capture + full-allocation assert: +5
- M5/M6/M7/M8/M9/M10/M11/NCCL-port docs: ~+30

保留 (核心 P0):
- F1-F12 happy path 正确性
- RLix protocol release_cluster_gpus (用 try/log/swallow 包裹避免异常掩盖)
- ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S (sync 不能 hang)
- payload_bytes (Ray auto-deref bug)
- F12 真实 adapter (WorkerPlacement + start_rollout_servers_from_worker_placements +
  RayTrainGroup worker_placements path)
- MilesCoordinator manual init + sync_lora_weights stub (Coordinator ABC 强制)

**失败语义 (M4 修正)**: first build 失败 = `MilesPipeline.initialize_pipeline()` 失败
路径 best-effort `ray.kill` (RayTrainGroup actors + RolloutManager + engine actors
via `shutdown_hard`) 然后 `_notify_release_cluster_gpus` 释放 scheduler allocation,
最后 raise. 这是账本一致性 (单 pipeline 失败不能让 scheduler 视 GPU 空闲但 actor 仍占),
不是 production hardening. graceful drain / abort RPC / force-kill timeout / multi-
pipeline cleanup race 留作 follow-up. 用户 `ray stop` CLI 仍是最后兜底, 但不能替代
单 pipeline 内部账本一致性.

NeMo RL port 是 ~1700 行 — 量级现在更接近. MILES 上调来自不能复用
PipelineCoordinator __init__, 真实 placement provider adapter (ROLL `List[List[Dict]]`
↔ MILES per-worker view).

（与 NeMo 工作量对比表已挪到 §关键风险与边界段以保持本节单一职责。）

---

## 实施顺序

以下只是工程排期，不代表降级 MVP。每个可合入 gate 都按最终 tp>1 parity 配置验证。

```
Week 1: Feature 1-3 — partial sleep/wake + routing skip + turn retry
  ├── Day 1: Feature 1 — post-sleep VRAM assertion + scheduler RPC
  ├── Day 2-3: Feature 2 — engine index map + subset API + abort-drain-sleep + routing lock
  ├── Day 4: Feature 3 — router admission + RLix mode 禁用 radix middleware
  ├── Day 5: Feature 3 — multi_turn.py turn-level redispatch + snapshot/restore + fail fast on exhaustion
  └── Day 6: Gate 1 + Gate 2 测试（含 multi-turn preempt → resume from aborted turn 验证）

Week 2: Feature 4-6 — CPU cache + selective sync + version
  ├── Day 1-2: Feature 4 — CPU bucket cache build + ModelUpdateService routing 层
  ├── Day 3-4: Feature 5+6 — 两路径 weight refresh + version accounting + receiver API
  ├── Day 5: Gate 2.5 测试（NCCL transport + ReloadableProcessGroup destroy/reload 循环）

Week 3: Feature 8-12 — RLix 适配
  ├── Day 1: Feature 12+8+10 — shared PG + registration + validation
  ├── Day 2-3: Feature 7+11 — namespace + DO_TIME_SHARING flag
  ├── Day 4: Feature 9 — progress reporting (begin/end batch)
  └── Day 5-6: Gate 3 (单 pipeline)

Week 4: 打磨 + post-MVP M11.2 happy path
  ├── Day 1-3: 文档、edge case、单 pipeline 稳定性回归
  └── Day 4-5: M11.2 happy path Gate 4 (c) + (d) + (e) (双 pipeline shell partial
              allocation; admission_epoch / cleanup safety / graceful drain 不在本
              scope, 各自 follow-up)
```

---

## 关键风险与边界

### 已决边界

- parity gate 绑定 `examples/fully_async` 外层 +
  `miles.rollout.generate_hub.multi_turn.generate` custom path +
  `--max-weight-staleness`；不再增加其他 example 作为当前 milestone gate。
- selective P2P weight transfer 不在当前 milestone 内；已放入 Implementation follow-up。
- 直接新建 `miles_coordinator.py`，不抽 `base_coordinator.py`；当第 3 个 backend 接入时
  再抽 backend-neutral base coordinator。
- F3 router metadata：Miles Router 在 JSON response 的 `meta_info` 注入
  `miles_engine_index` 与 `miles_worker_url`，作为 `_is_scheduler_preempt(output)` 的
  唯一数据源（不走 header-only 方案）。
- F8 RLix orchestrator API 调用一律用 keyword args（`pipeline_id=...,
  ray_namespace=...`），不走 positional。
- F9 progress ingress：MILES hook → `MilesCoordinator.report_progress_from_scheduler`
  入站用 `metrics["collected"]`；coordinator 聚合后向 RLix central scheduler 发
  `metrics["completed"]`。两层 wire 不混用。

### Optional diagnostics

- `train.py` 同步路径只允许作为 optional diagnostic smoke test，不能替代
  `examples/fully_async` parity gate。

### 非技术待定

- 是否创建 `rlops/miles` 社区 fork 作为 framework-side hook 落地位置。这是 repo
  ownership / 协作流程决策，不影响 implementation spec；默认先落在当前 MILES 工作树，
  最终归属由 repo owner 决定。

### 主要风险

- `sglang_data_parallel_size > 1` 在未来需要时引入新一层 DP 轴 — 当前 scope 强制 == 1
- `MilesModelUpdateService` 动态 NCCL group 生命周期是 NeMo 已经踩过的坑（ROLL
  原生有），实现路径要与 ROLL `model_update_service.py` 严格对齐而不是从头发明
- in-flight active refresh 误标的过渡窗口 — 与 NeMo `in_flight_weight_updates=True`
  同类，已在生产 tolerated；Gate 3 量化误标量级，无 drain-then-sync fallback 设计
- RLix mode 当前禁用 radix-tree middleware；`partial_rollout + radix_tree` 的 abort 透传、
  prefix-cache 污染防护与 sample 状态回滚留作 follow-up
- CPU cache `total_cpu_cache_bytes` 在 host RAM 紧张时 OOM — 首次 build 时 fail fast
  守住
- **SGLang TP NCCL `release_memory_occupation` 后保留**：与 vLLM 同设计，SGLang 自身
  release 路径就依赖此（用 `barrier(tp_cpu_group)`）。Gate 2 是回归确认而非真正风险
  点；NCCL buffer 残留 (~200 MB) 由 post-sleep VRAM 阈值兜住
- **F3 dispatch 与 scheduler sleep 的 in-flight abort race 由 turn redispatch 兜底**.
  不引入 admission_epoch 防御 (race window ms 级, 实际拓扑不发生; production 多
  pipeline 高频 shrink/expand 触发再加, follow-up). false negative race (response 时刻
  enabled_workers 已恢复) 在第 3 个数量级时间错开下不会发生; 详见 F3
  `miles_admission_disabled` 语义边界段
- **First build 失败语义 (M4 修正)**: `MilesPipeline.initialize_pipeline()` 失败路径
  best-effort hard-cleanup (`RayTrainGroup` ctor self-cleanup + `MilesPipeline` finally
  / except 内 `ray.kill` train actors + `RolloutManager.shutdown_hard` w/ 10s timeout +
  `ray.kill` actor_infer) → 然后 `_notify_release_cluster_gpus` 释放 scheduler
  allocation → 最后 raise 透传. 这是**单 pipeline 账本一致性**: scheduler 视 GPU
  空闲 + Ray actor/CUDA context 仍占的分裂状态会让下个 pipeline 调度同 GPU OOM. 不
  能只靠 `ray stop` CLI 兜底 (CLI 是用户级整集群杀, 单 pipeline 失败不能要求用户重启
  整集群). graceful drain / abort RPC / force-kill timeout / multi-pipeline cleanup
  race 仍 follow-up
- **NCCL `master_port` collision 由 `get_free_port()` + SharedStorage claim 兜底**;
  禁 `master_port=0` (multi-rank rendezvous 不成立). EADDRINUSE retry 新 port; cooldown
  queue 是 follow-up
- **Receiver-side bucket copy (cpu_serialize)** bounded by `args.miles_model_update_bucket_size_mb`
  上限 (~512MB-1GB), 非 whole-model. Plasma 真零拷贝 adapter (memoryview-backed reader,
  避免 `tobytes()` deep copy) 是 follow-up
- **CUDA_VISIBLE_DEVICES poisoning (M6)**: 必须 Ray actor `runtime_env={'env_vars': {...}}`
  在 process bootstrap 阶段静态注入, 禁 actor 内 `os.environ[...] = ...` 覆盖
  (`cuInit()` 锁定时机). 既有 MILES `actor_group.py:88` 模式正确
- **async_save not supported (M7)**: F10 fail-fast `assert not args.async_save`. 后台
  ckpt flush 与 actor.sleep() 的 torch_memory_saver.pause() 竞争 → segfault. 实现
  `maybe_finalize_async_save(blocking=True)` + `cuda.synchronize()` 是 follow-up
- **跨节点 rollout (SGLang) engine 当前不支持** (F12 first build 强约束
  `rollout_num_gpus_per_engine <= args.num_gpus_per_node`, M3, 按 rollout engine 总 GPU 数计,
  与 sharding 形式 TP/EP 无关; train 侧 Megatron actor 不在此 assert 范围内, 单节点是 dev
  gate 的 incidental). 跨节点 rollout engine follow-up:
  扩 `WorkerPlacement` 成 per-node placement list, 按 node-rank 分别建 SGLang sub-actor
- **cuda_ipc selective sync 路径 = M11.6 later milestone** — M11.1 first build target =
  vast.ai 受限容器 (无 `--ipc=host` / `CAP_SYS_PTRACE`), F10 强制
  `args.model_update_transport == "cpu_serialize"`. M11.6 加新 cuda_ipc adapter
  (CPU cache → per-bucket H2D staging → `MultiprocessingSerializer.serialize` IPC
  handle, ~50-80 行) + 启动期 smoke test capability check (不写脆弱环境探测器). 既有
  `_send_to_colocated_engine` 依赖 live `dist.gather_object` 与 F4 destroy NCCL 顺序
  不兼容, RLix mode 不复用 (standalone mode 仍可走既有路径)

### MILES vs NeMo 工作量对比

| 维度 | MILES vs NeMo |
|---|---|
| **MILES 省下的** | `ReloadableProcessGroup` 已提供 NCCL teardown（NeMo 需 `nccl_offload.py` ~90 行 + Gate 2.5 fallback rule）；MILES 不需要 NeMo 那种完整 `RayVirtualCluster` object adapter |
| **MILES 多花的** | Miles Router 是独立进程，需要 `/disable_worker` / `/enable_worker` / `/remove_worker` endpoint（~80 行；NeMo 无独立 router，直接通过 vLLM `collective_rpc`）；turn snapshot/restore 辅助（~40 行；NeMo 的 turn retry 由于 message_log 累加在 `_async_generate_base` 之上，不需要逐字段 rollback）；ROLL-style device mapping 到 MILES PG 三元组的 `MilesPlacementProvider` 显式 adapter；radix-tree middleware 兼容当前不做，减少 first-build 风险 |
| **总差额** | ≈ 持平。MILES ~2195 行 vs NeMo ~1700 行，差额来自 partial overlap + 5+6 selective sync 全套基础设施都是新的；ROLL/NeMo 已经踩过的坑（动态 NCCL group / warmup allreduce / cpu_serialize-broadcast dual-mask）都直接对齐 ROLL `model_update_service.py`，不重新发明。Receiver crash 容错（conditional port leak、周期性 GC）超出本 milestone，留作 follow-up |

### 推荐的立即决策

- 批准以 **partial overlap + subset sleep/wake + CPU bucket cache + selective sync** 
  作为本 port 的统一目标（与 ROLL/NeMo F1-F12 同形态）
- **不复刻** request-level deterministic migration — handoff 主路径绑定 multi_turn
  turn-level redispatch；single_turn 路径走 group recycle；其他 abort 来源（tool
  error 等）由 fully_async 既有 group recycle 处理（与本 port 正交）
- **不重新实现** Megatron NCCL teardown — 直接复用 `ReloadableProcessGroup`
- first build 强制排序连续 `infer_device_mapping`；非连续 / 自定义顺序 mapping 直接 fail fast，
  不实现通用 DP-rank adapter
- RLix mode 禁用 `RadixTreeMiddleware`；`partial_rollout + radix_tree` 兼容不进入当前
  milestone
- 第一批 scheduler 集成验证目标绑定 `examples/fully_async` 外层，但必须显式启用
  `multi_turn.generate` custom path + `--max-weight-staleness`
- M11.1 MVP gate: Gate 1, 2, **2.5 (NCCL selective sync)**, 3 (single pipeline).
  Gate 4 (双 MILES pipeline) 推迟到 M11.2. ROLL + MILES 混合调度不在本
  plan 验证范围内。

---

## Implementation follow-up（不在本 plan 范围，仅留挂钩）

| 目标 | 触发条件 |
|---|---|
| 抽 `rlix/pipeline/coordinator.py` 的 backend-neutral hook（base coordinator） | 当第 3 个 backend（NeMo / 新 framework）port 到位、共享逻辑足够清晰时 |
| MoE / EP support — `get_expert_tensor_parallel_group()` 等 | 当业务方启用 MoE 模型且需要 RLix 调度时；预计是中等改动（~200-400 行 + 专门 MoE parity gate），不是当前 plan 的小补丁 |
| Selective P2P weight transfer | 当 broadcast/tensor subset sync 成为吞吐瓶颈时 |
| **M11.5** Receiver crash 容错（fault-tolerant port 管理 + conditional leak + 周期性 GC） | 当生产观察到 receiver crash 导致 master_port collision，或 selective sync 失败率非 zero 时 |
| 非连续 / 自定义顺序 `infer_device_mapping` adapter | 当需要跨机/非连续 GPU 或自定义 engine ordering 时；实现 `scheduler_dp_rank -> miles_engine_index -> gpu_ids` 显式映射 |
| `partial_rollout + radix_tree` compatibility | 当 RLix 主路径稳定后；需要 scheduler-preempt abort 透传、prefix-cache 污染防护与 sample 状态回滚 |
| **M11.5** Cleanup API graceful path (`RolloutManager.shutdown` + `RayTrainGroup.shutdown` + `__init__` self-cleanup + 30s timeout + force-kill fallback + dispose finally) | 当 multi-pipeline 共节点 cleanup race 实际触发, 或单 pipeline 失败 cleanup 时长 unacceptable; ~150 行 |
| **M11.2** 多 pipeline orchestrator-driven cleanup (Gate 4 dual-pipeline 同 milestone) | orchestrator 接管 pipeline-level 隔离, 单 pipeline crash 不影响别的; 需要 namespace-scoped actor enumeration + selective `ray.kill` (避免 `ray stop` 杀掉别的 healthy pipeline). M11.1 Gate 1-3 单 pipeline 用户 shell `ray stop` 即可 |
| **M11.6 cuda_ipc colocate adapter** (CPU cache → per-bucket H2D staging → IPC handle serialize, ~50-80 行) + smoke-test capability check (不写脆弱 heuristics) | M11.1 → M11.6 (production colocate transport, after M11.5 hardening) |
| **M11.2** router `admission_epoch` race 防御 (start/end epoch 比对消除 disable→abort→enable race) | production 多 pipeline 高频 shrink/expand 触发 false negative; M11.1 turn retry 兜底, race window ms 级 single-pipeline 拓扑不发生 |
| F12 identity round-trip self-check + dp_rank ↔ engine_index 双向 lookup table | 当解禁非连续 / 自定义顺序 `infer_device_mapping` 时; first build contiguous 拓扑下 identity 永远 pass, 是 dead assert |
| Plasma true zero-copy adapter (cpu_serialize) | Receiver-side bucket copy 成 RAM 瓶颈 (chunk_size_mb 不足以兜住, 多 pipeline 多 receiver 累计); memoryview-backed file-like reader + `np.ndarray[uint8]` / `torch.ByteTensor` (避免 `.tobytes()` deep copy) + 实测验证 |
| **M11.5** NCCL port cooldown queue / port pool TIME_WAIT mitigation | EADDRINUSE retry 仍频繁失败 (cluster 高频 resize); port pool + TIME_WAIT cooldown |
| Async save support (`args.async_save` flush) | RLix-mode 需要 train-step 内异步 ckpt flush; sleep() 开头 `if args.async_save: maybe_finalize_async_save(blocking=True); cuda.synchronize()` |
| Train-side post-offload VRAM assert | torch_memory_saver leak 实测 (silent VRAM retention); `RayTrainGroup.offload()` 末尾 per-actor `torch.cuda.memory_allocated() < threshold_mb` |
| Engine-level overlap assert (set intersection) | colocate/non-colocate 边界 bug 触发 (allocation 错位); F10 加 `non_overlap_engines >= 1` |
| Router `enable_worker discard dead` + reset failure counter | dead worker 误标 enable 后无法恢复 (production 多 pipeline 长跑触发); `_enable_worker_internal` 加 `dead_workers.discard(url)` + `worker_failure_counts[url] = 0` |
| Router `do_proxy except` 扩 (KeyError/AttributeError/TypeError) | router metadata parse 异常 (data 不是 dict 时 setdefault 抛 AttributeError); except clause 扩成 `(JSONDecodeError, KeyError, AttributeError, TypeError)` |
| **M11.5** 5xx → preempt synthesis | engine crash mid-abort 实测 (router fail-fast 上抛 RLixRouterMetadataError 不够; scheduler 重排); 5xx 路径 synthesize preempt sentinel |
| **M11.5** SGLang ingress 503 middleware | TCP/FastAPI race in abort-drain-sleep 实测复现; SGLang 端 ingress middleware 在 sleeping 状态返回 503, 替代 `assert is_fully_idle()` fail-fast 兜底 |
| **M11.2** `MilesPipeline` graceful actor drain (替代 `ray.kill`) | multi-pipeline cleanup race (Gate 4 dual-pipeline 同 milestone); `actor.shutdown()` RPC + force-kill timeout 替代 hard `ray.kill` |
| Non-contiguous / 自定义 `infer_device_mapping` 下的 partial allocation 反推 | M11.2 first build assumes contiguous mapping (`infer_device_mapping.index(g) // tp_size` 派生 active engine_idx); 解禁非连续映射时需 `MilesPlacementProvider.scheduler_to_engine` 双向 lookup, 与 follow-up "非连续 / 自定义顺序 `infer_device_mapping` adapter" 同里程碑 |
| **M11.2 follow-up**: CPU-only base sync at `version=-1` for resume / non-equivalent ref_load | M11.2 Gate 4 happy path 用 Fix #14 fail-fast 拒绝 resume 配置 (`args.load`/`args.ref_load` ⊆ {None, args.hf_checkpoint}). 解禁需 service `sync_selected_workers(..., -1)` 实现 CPU-only push from `build_cpu_bucket_cache(step=-1)` bytes through receiver `update_weights_from_cpu_bucket` (NOT NCCL broadcast — broadcast 仍需 sender GPU 而 init Step 6 已 release train); 所有 targets 强制走 cpu_serialize 不论 transport 配置. ~30 行 service spec 改动. 解锁后可 drop F10 fail-fast |
