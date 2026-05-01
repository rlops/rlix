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
  args.rollout_num_gpus_per_engine`，`args.sglang_data_parallel_size == 1`
- **Sleep/wake 粒度：** subset (per-engine indices)
- **Routing 粒度：** subset (跳过 sleeping engines)
- **Sync 粒度：** subset (sync-on-expand + active in-flight refresh)
- **权重路径：** CPU bucket cache + selective sync（不再用 global
  `actor_model.update_weights()` 作为 RLix 模式 baseline）
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

### Final parity only

本 plan 的目标是 **一步到位实现 ROLL / NeMo port feature parity**。Feature 1-12 都属于
同一批最终交付范围；不设置 tp=1 MVP，不用简化 workload 作为第一阶段替代品：

- F1-F3：必须实现 subset sleep/wake、router admission、scheduler-preempt retry。
  parity workload 绑定 `multi_turn.generate + --max-weight-staleness`，因此 turn-level
  redispatch 是主路径；single-turn generation-call retry / group recycle 只能作为兼容
  路径，不能替代 multi-turn retry。
- F4-F6：最终必须覆盖 CPU bucket cache、active in-flight refresh、expand selective
  sync、version accounting。首个端到端验证必须覆盖 `tp>1` mixed receiver mask：
  colocated Ray ObjectRef `cpu_serialize` + non-colocated dynamic NCCL broadcast +
  receiver-side dual-mask / `is_group_exist` hardening，不后置为第二阶段。
- F7-F8-F11：implementation 可合并到一个 `MilesRLixAdapter` / `rlix_hooks.py` 以减少
  分散 glue，但三个 feature spec 必须独立保留；namespace isolation、registration
  lifecycle、conditional standalone/RLix behavior 都必须保留。
- F9：progress reporting 必须保留 ROLL/NeMo 的 batch-begin snapshot 语义；不能退化成
  只报累计 completed。
- F10：topology validation 是 parity 必需项，不是 nice-to-have。必须 fail fast 覆盖
  `train_devices ⊂ infer_devices`、`infer_engine_count >= 2`、fullasync enabled、
  `sglang_data_parallel_size == 1`、Megatron 并行度 divisibility 与单 updateable
  model/server 约束。
- F12：shared PG 不是可选 glue。MILES 必须接受 RLix/ROLL shared PG，并显式完成
  ROLL-style device allocation 到 MILES `(pg, bundle_indices, gpu_ids)` 的 adapter。

**验证 caveat**：一步到位不等于线性串行落地。F1-F3 可以先用 unit / mock integration
覆盖 router metadata、preempted-engine 判定、snapshot/restore 与 retry exhaustion；但
最终 e2e gate 只在 F4-F6 weight refresh、F7-F11 control plane、F12 placement 全部到位后，
按 `tp>1` final parity 配置判定通过。

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
- 缺：post-sleep VRAM assertion（`memory_allocated()` 校验）
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
3. **Post-sleep VRAM assertion**：每个 target engine 在 `offload()` 完成后必须验证
   `torch.cuda.memory_allocated() < post_sleep_vram_threshold_bytes`（默认 1 GiB 量级，
   做成显式配置）。MILES 当前路径无此 check，不补会出现"逻辑上 slept、实际显存未释放"，
   导致 scheduler 误以为 GPU 可分给另一 pipeline 而 OOM
4. **Idempotency guard**：worker 本地维护 `is_engine_resident_on_gpu`，重复 sleep /
   wake 视为 no-op（对齐 ROLL）
5. **Coordinator 层 fail-fast 验证 NCCL teardown wired up**（等价 ROLL `coordinator.py:136`
   `_validate_offload_nccl`）：`MilesCoordinator.__init__` 通过 RPC 校验每个 actor 已
   apply `monkey_patch_torch_dist`（参 [actor.py:58](external/miles/miles/backends/megatron_utils/actor.py#L58)）。
   实现：让每个 Megatron actor 在 init 完成时 publish 一个状态 flag（如
   `dist.old_new_group is not None` 或显式 `nccl_teardown_ready`）；coordinator init
   收 ack 失败即 raise

**SGLang TP NCCL communicator 状态说明：** SGLang `release_memory_occupation()` 仅释放
weights + KV cache 的 CuMem，TP NCCL communicator 保留有效（与 vLLM `LLM.sleep()` 同形态；
SGLang 内部 release 路径用 `barrier(self.tp_cpu_group)`，本身就依赖 TP group 保留 —
[scheduler_update_weights_mixin.py:159](external/sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py#L159)）。
Gate 2 是回归确认，不是真正风险点 — 强制 destroy SGLang TP NCCL 会 break SGLang 自己
的 release 路径，**不应作为 fallback 设计**。NCCL communicator buffer 残留（~50-200 MB）
由 `post_sleep_vram_threshold_bytes` 兜底，partial overlap 可接受。

改动量：~30 行（含 +5 行 coordinator-side validate hook）

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
       handle: SGLangEngineActor       # Ray actor handle
       worker_url: str                  # http://host:port for Miles Router admission
       server_group: ServerGroup
       bundle_index: int                # placement bundle
       gpu_ids: list[int]
       state: Literal["active", "disabled", "offloaded"]
   ```
   `RolloutManager._engines: Dict[int, EngineInfo]`（单一 dict，`engine_index` 为 key）。
   文档可保留字段说明，但实现层不要拆 5 个并行 map（容易状态漂移）。

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
       # 每个 engine 内部 abort all running requests
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

   - `EngineInfo.state` 反映 engine 当前生命周期阶段（active / disabling / offloaded /
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

4. **Drain 机制：worker-side `is_idle()` API**（等价 NeMo `vllm:num_requests_running`）：
   - 新增 `SGLangEngine.is_idle() -> bool` 方法（[sglang_engine.py](external/miles/miles/backends/sglang_utils/sglang_engine.py)），
     内部调用 SGLang `/server_info` HTTP endpoint（已有 `get_server_info()` 包装在
     [sglang_engine.py:326](external/miles/miles/backends/sglang_utils/sglang_engine.py#L326)
     使用 `/server_info`；不要复用 line 193 的 deprecated `/get_server_info`）。
     `/server_info` response 含 `internal_states: List[Dict]`，每个 state 含
     `num_running_reqs`（资 [sglang/observability/metrics_collector.py:80](external/sglang/python/sglang/srt/observability/metrics_collector.py#L80)
     与 [sglang/entrypoints/http_server.py:631](external/sglang/python/sglang/srt/entrypoints/http_server.py#L631)）。
     `is_idle()` 实现：`all(s["num_running_reqs"] == 0 for s in resp["internal_states"])`
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
   - engine state 用 `active / disabling / offloaded / loading`，不要只靠
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

6. **复合操作命名对齐 ROLL：`shrink_engines` / `expand_engines`**（资 ROLL
   `generate_scheduler.py:1885,1973`）：
   - 在 `RolloutManager` 上加 `shrink_engines(engine_indices)` / `expand_engines(engine_indices)`
     高层方法，封装 `admission_close + abort_partial + drain + offload`（shrink）和
     `wake + load + admission_open`（expand）
   - Coordinator 只调这两个高层方法，不直接拼装 abort/drain/offload 序列（避免漏 step
     或顺序错位）

改动量：~150 行（engine index map + subset API + abort-drain-sleep + shrink/expand_engines 复合）

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

**2. Partial-rollout 解禁（两处 assert 都要删）**

MILES 当前在两个独立位置硬 reject `args.partial_rollout=True`，**两处都必须删**才能让
turn-level redispatch / partial trajectory 走通：

- [miles/router/middleware_hub/radix_tree_middleware.py:180](external/miles/miles/router/middleware_hub/radix_tree_middleware.py#L180)
  `assert not args.partial_rollout, "Currently partial rollout is not supported when using miles router"`
  — 删除该 assert / reject 改为 pass-through
- [miles/rollout/generate_hub/multi_turn.py:29](external/miles/miles/rollout/generate_hub/multi_turn.py#L29)
  `assert not args.partial_rollout, "Partial rollout is not supported"` — 删除该 assert

**路由决策由 Router admission API 单点负责，radix tree 不重复路由过滤** — 两层都做会
导致 dispatch 重复决策路径与 routing 状态分裂。multi_turn.py 在删除 assert 之后，
partial trajectory 通过 turn-level redispatch（本 §3）+ snapshot/restore（[generate_endpoint_utils.py](external/miles/miles/rollout/generate_utils/generate_endpoint_utils.py)）
正确处理。

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

**Router response metadata + `EnginePreemptedError` 闭环：**

(a) 新增 `class EnginePreemptedError(Exception)`，**位置**：
[miles/rollout/base_types.py](external/miles/miles/rollout/base_types.py)（与
`GenerateFnInput / GenerateFnOutput` 同模块；multi_turn.py 已经 import 这个模块，
零额外 import seam）。

(b) **Router 必须注入 engine metadata**：Miles Router 在选中 worker 并收到 SGLang
response 后，把 routing 结果写入 JSON response 的 `meta_info`：
`meta_info["miles_engine_index"] = engine_index` 与
`meta_info["miles_worker_url"] = worker_url`。这是 `_is_scheduler_preempt(output)` 的
唯一数据源；不采用 header-only 方案，因为 `multi_turn.py` 当前只消费 JSON body。
若上游 response 没有 `meta_info`，router 必须创建空 dict 后再注入。

(c) **抛出位置**：[miles/rollout/generate_hub/multi_turn.py](external/miles/miles/rollout/generate_hub/multi_turn.py)
中 `output = await post(url, payload)` 返回后立刻检查（在 `update_sample_from_response`
调用之前），若 `output["meta_info"]["finish_reason"]["type"] == "abort"` 且通过
`output["meta_info"]["miles_engine_index"] ∈ _preempted_engines`，立即 raise
`EnginePreemptedError`。缺少 `miles_engine_index` 时 fail fast，而不是把 scheduler
preempt 当普通 abort 处理。

(d) **捕获位置**：multi_turn.py snapshot/restore retry loop 内（包住 `await post()` 那
段），catch 后调 `_restore_turn_state(sample, snapshot)` 然后 `continue` 重 dispatch。

(e) `_is_scheduler_preempt(output)` 等价于
`output.meta_info.miles_engine_index ∈ _preempted_engines` 检查，逻辑仅一处实现（避免
散落判定）。

(f) `_preempted_engines` set/clear 时机：`sleep_partial` 第 1 步 set（Feature 2 已
列），`wake_up_partial` 完成后 clear（与 NeMo `ShardPreemptedError` 同形态）。

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
for _turn in range(max_turns):
    payload, halt_status = compute_request_payload(args, sample.tokens, ...)
    if payload is None:
        sample.status = halt_status; break

    # ─── TURN-LEVEL REDISPATCH（主路径）───
    snapshot = _snapshot_turn_state(sample)   # tokens / weight_versions / loss_mask / spec_info / prefix_cache_info 长度
    output = None
    for attempt in range(MAX_TURN_REDISPATCH_ATTEMPTS):
        output = await post(url, payload)
        finish = output["meta_info"]["finish_reason"]["type"]
        if finish == "abort" and _is_scheduler_preempt(output):
            _restore_turn_state(sample, snapshot)             # 丢弃 preempted partial 增量
            if attempt == MAX_TURN_REDISPATCH_ATTEMPTS - 1:
                sample.status = Sample.Status.ABORTED         # 落到兜底
                return GenerateFnOutput(samples=...)
            continue                                          # router 跳过 sleeping engine 自动选新 engine
        break                                                  # 成功 / length / 非 preempt abort

    await update_sample_from_response(args, sample, payload=payload, output=output, ...)

    if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
        break

    # ... tool_calls, env step（commit point — turn generation 成功后才执行）...
```

**关键不变量：**

- **Commit point 在 generation 成功之后**：tool_call / env step 只在 turn generation
  非 abort 时执行（[multi_turn.py:70-75](external/miles/miles/rollout/generate_hub/multi_turn.py#L70-L75)
  当前已是这个顺序）。Aborted turn retry 不会重复 commit env side effect。这是
  ROLL agentic env-manager 与 NeMo F3 retry safety invariant 的同形态。
- **完成 turn 全保留**：snapshot 只覆盖当前 turn 增量；turn N-1 及之前的 `tokens` /
  tool_responses 不动。
- **结构性不会失败**：`MAX_TURN_REDISPATCH_ATTEMPTS = active_engine_count`，配合
  Feature 10 的拓扑验证（非重叠 engine 至少 1 个常驻 active）+ Feature 3 router
  admission（新 dispatch 跳过 sleeping engines），turn retry 在拓扑合法时不会用尽
- **真用尽就 fail fast**：如果 turn retry 仍用尽，那是拓扑违规或 scheduler bug，**直接
  raise**，不静默落到 group recycle — 让上层暴露问题，而不是静默吞掉一条 trajectory

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

改动量：~130 行（router endpoints ~40，middleware 改造 ~20，routing lock + abort 触发口
~20，turn snapshot/restore + redispatch loop ~50）

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
  - **Colocate engines**（同 GPU receiver）→ 由 `args.model_update_transport` 选择
    cuda_ipc（默认，复用 MILES 既有 IPC handle 路径）或 cpu_serialize（fallback，新增
    Ray ObjectRef bytes 路径）；wire format 详见本 feature 段下方 "Bucket payload 格式"
    Case A / Case B 两节
  - **Non-colocate engines**（跨 GPU receiver）→ 逐 bucket H2D staging + 临时 NCCL
    group broadcast + bucket-level barrier + group destroy
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
- **不存在** CPU bucket cache 概念 — `actor_model.update_weights()` 直接从 GPU 张量
  发出
- 流程固定：pause generation → flush cache → transfer weights → continue generation

#### 移植方案

**问题：refit 时训练权重在哪里？**

partial overlap 中，训练 GPU = 重叠 GPU = inference 需要 wake_up 的 GPU。要求权重在
GPU 上的 refit 路径（MILES 现有 distributed/tensor 路径）会与 inference wake 抢同一份
VRAM → OOM。**这是必须引入 CPU bucket cache 的核心原因，与 NeMo RL F4 同因。**

**方案：参照 ROLL `ModelUpdateService` 的简化版（与 NeMo RL port plan F4 同形态）**

**Step 0（Init bootstrap base model cache）：**

⚠️ **时间窗硬约束**：必须落在 `create_training_models()` 返回**之后**（model 已 load
到 GPU）且
`self._notify_release_cluster_gpus(cluster_id="actor_train", global_step=...)` **之前**
（GPU 仍占用，gather 才能进行）。**放错位置失败模式**：

- (a) 放到 model load 之前 → 拿到未初始化权重 → 首次 expand 把垃圾 weights 推给 inference
- (b) 放到 GPU release 之后 → gather 死锁（rank 已离开 NCCL group）或读到空 cache → 首次 expand 拿不到任何 weights

**精确落点（与 [full_finetune_pipeline.py:114](rlix/pipeline/full_finetune_pipeline.py#L114)
`initialize_pipeline()` 同形态，inline 单 method 不抽 hook）**：

`MilesPipeline.initialize_pipeline(self) -> ActionResponse:` 单一 method，用
`self._init_lock` 守卫（避免重入）。method 内部按以下严格顺序，**inline 不抽子方法**：

```python
def initialize_pipeline(self) -> ActionResponse:
    with self._init_lock:
        if self._initialized:
            return ActionResponse(success=True)

        # Step 1: 申请 actor_train GPU
        init_global_step = -1
        self._request_cluster_gpus(
            cluster_id=self._actor_train_cluster_id,
            priority=Priority.INITIALIZATION,
            global_step=init_global_step,
        )
        try:
            # Step 2: actor_train initialize（Megatron worker setup + model load）
            self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)

            # Step 3: build base-model cache（_cache_ready_step = -1）
            #   ⚠ 时间窗：必须在 Step 2 之后（model 已 load）+ Step 4 之前
            #   （GPU 仍占用，gather 才能 cross-rank 推进）
            self._build_cpu_bucket_cache(step=-1)
            self._cache_ready_step = -1

            # Step 4: 申请 actor_infer + 启动 inference engines
            self._request_cluster_gpus(
                cluster_id=self._actor_infer_cluster_id,
                priority=Priority.GENERATION,
                global_step=init_global_step,
            )
            self.actor_infer.initialize(...)

        finally:
            # Step 5: release actor_train（init bootstrap 完成）
            self._notify_release_cluster_gpus(
                cluster_id=self._actor_train_cluster_id,
                global_step=init_global_step,
            )

        self._initialized = True
        return ActionResponse(success=True)
```

**放错位置失败模式**：

- (a) 放到 Step 2（model load）之前 → 拿到未初始化权重 → 首次 expand 把垃圾 weights 推给 inference
- (b) 放到 Step 5（GPU release）之后 → gather 死锁（rank 已离开 NCCL group）或读到空 cache → 首次 expand 拿不到任何 weights

第一次 expand（来自 runtime 后续 `_request_cluster_gpus(cluster_id="actor_train",
priority=Priority.ACTOR_TRAINING, global_step=N)` 重新分配）直接读这个 cache。对齐 ROLL
`full_finetune_pipeline.py:289-301` 的 `setup()` 阶段时机。

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
   - Bucket layout：`List[BucketRecord]`，每条至少含 `param_names`, `shapes`,
     `dtypes`, `used_bytes`, `cpu_uint8_bucket`（contiguous CPU tensor / bytes）
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
  单 sender（cache owner）推到 woken engines。**ROLL H2D staging 模式**：CPU bucket
  先拷到 sender GPU staging buffer；同 GPU receiver 走 Ray ObjectRef `cpu_serialize`，
  跨 GPU receiver 从 GPU staging buffer 发 dynamic NCCL broadcast。每 bucket 传完立即
  free staging。**必须严格逐 bucket 滚动，不能整模型一次性回 GPU**。锁语义为“整段持
  锁”：`acquire _cache_lock → 读取 cache pointer + 版本 → for bucket in buckets: 分配 GPU
  staging buffer → CPU→GPU 拷贝 → Ray ObjectRef cpu_serialize / NCCL broadcast 发出 →
  等该 bucket 所有 receivers ack → free staging → 下一 bucket → (全部完成) destroy 动态
  NCCL group → release _cache_lock`。

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
   - `bucket_size_bytes` 显式配置；启动时用"wake_up 后剩余 VRAM"做上界 check，
     确保 `bucket_size_bytes + transport scratch < overlap GPU 可用余量`
   - **同 GPU receiver**（overlap GPU）→ `cpu_serialize` Ray ObjectRef bytes（portable
     path，不依赖 CUDA IPC / `--ipc=host` / CAP_SYS_PTRACE）
   - **跨 GPU receiver**（target 有 TP 跨 GPU 的 rank）→ H2D staging → 动态 NCCL group
     broadcast
   - 同一 bucket 可能 **同时**有 cpu_serialize receivers 与 broadcast receivers
     （receiver-side dual-mask），但 staging buffer 仍只 H2D staging 一次

**Receiver-side dual-mask + `is_group_exist` no-op guard（tp>1 必须正确）：**

`comm_plan` 携带 per-engine **`cpu_serialize_local_ranks`** 与
**`broadcast_local_ranks`**：

- `update_parameter_in_bucket(payload_ref, cpu_serialize_local_ranks, ...)` 必须检查
  `self.rank in cpu_serialize_local_ranks`，不在则 skip（该 rank 通过 broadcast 接收）
- `destroy_collective_group(group_name)` 必须用 `is_group_exist(group_name)` no-op
  guard — cpu_serialize-only ranks 从未 join group，没有 guard 会 KeyError
- 对齐 ROLL `worker.py:640,757`。所有 parity gate 都按 tp>1 运行，必须覆盖
  cpu_serialize-only ranks、broadcast ranks 与混合 receiver mask；不能依赖 tp=1
  绕过该 bug。

**Cache 安全性 4 个不变量：**

1. **单 writer**：training hook 是唯一调用 `build_cpu_bucket_cache(step)` 的入口；
   写完即原子更新 `_cache_ready_step`
2. **单 reader 路径**：active refresh + expand sync 都通过 `MilesModelUpdateService.
   sync_selected_workers(...)` 读同一份 `_bucket_cache`
3. **顺序契约**：`before_training(step+1)` 阻塞到前一个 `after_training(step)` 触发的
   active refresh + expand 完成后才返回（由 RLix `request_cluster_gpus` 的 blocking
   `ray.get` 保证）。这是覆盖式 cache 安全的**主要**保证 — 因为有这条契约，build
   永不会发生在 broadcast 进行时
4. **Cache owner `_cache_lock`（强制硬约束，整段临界区）**：必须覆盖完整 "读 cache
   pointer / `_cache_ready_step` / bucket 列表 → 逐 bucket transport（含 cpu_serialize +
   broadcast）→ 最后一个 bucket receiver barrier → 动态 NCCL group destroy" 整段。
   `build_cpu_bucket_cache()` 写入新 buckets + publish `_cache_ready_step` 时也必须持
   同一把锁。**禁止只锁 cache lookup 或只锁 pointer swap 的半截实现**（对齐 ROLL
   `megatron_strategy.py:2095-2099`）。正常路径不变量 3 已经够；`_cache_lock` 是抗超时
   与异常恢复的最后一道

**Bucket payload 格式：colocate 双 transport + non-colocate broadcast**

**Colocate（同 GPU receiver）有两条 transport，由 `args.model_update_transport`
（命名与 ROLL 一致）选择**：

| 配置 | 路径 | 容器要求 | 性能 | 何时用 |
|---|---|---|---|---|
| `"cuda_ipc"`（默认） | **沿用 MILES 现有 IPC handle 路径** | CAP_SYS_PTRACE 或 `--ipc=host` | 基线（zero-copy） | 生产 / 物理机 / 完整权限容器 |
| `"cpu_serialize"`（fallback） | 新增 Ray ObjectRef bytes 路径 | 无特殊要求 | 比 IPC 慢（D2H + 序列化 + ray.put） | 受限容器（vast.ai 等） |

未识别值 fail-fast。不引入 env var；env var 仅在临时调试时通过 launcher 注入到 args
（避免两套 source-of-truth）。

##### Case A：CUDA IPC 默认路径（**沿用 MILES 既有实现，不重写**）

完全复用现有 [`_send_to_colocated_engine`](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L280)
+ [`SGLangEngine.update_weights_from_tensor`](external/miles/miles/backends/sglang_utils/sglang_engine.py#L282)
路径，**不新增 sender / receiver 代码**：

1. `FlattenedTensorBucket(named_tensors=named_tensors)` 打平（[update_weight_from_tensor.py:311](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L311)）
2. `MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)` 生成
   IPC handle 字符串（[update_weight_from_tensor.py:317](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L317)）
3. `dist.gather_object` 收 string 到 `ipc_gather_src`
4. src 端 per-target 调
   `ipc_engine.update_weights_from_tensor.remote(serialized_named_tensors=...)`
   （[update_weight_from_tensor.py:357](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L357)）
5. Receiver 走 SGLang 现有 `update_weights_from_tensor` HTTP endpoint
   （[sglang_engine.py:282](external/miles/miles/backends/sglang_utils/sglang_engine.py#L282)），
   内部 `MultiprocessingSerializer` 反序列化拿到 IPC handle，`rebuild_cuda_tensor_from_ipc()`
   后 load_weights

**MILES F4 改造对 cuda_ipc 路径的影响**：

- F4 §5 per-bucket barrier / `_cache_lock` 由 MilesModelUpdateService 在外层 wrap
  现有 send 路径，**不改 IPC wire format**
- F4 §1 cache owner（`_is_distributed_src_rank`）已经是 IPC 路径的 `ipc_gather_src`
  rank，复用即可
- Receiver-side mask（cuda_ipc 路径）使用对称的 `ipc_local_ranks`；与 cpu_serialize
  路径的 `cpu_serialize_local_ranks` 互斥（一个 receiver engine 不会同时被两条 transport
  寻址）
- LoRA `load_lora_adapter_from_tensors`（[update_weight_from_tensor.py:341](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L341)）
  cuda_ipc 路径**不动**

##### Case B：cpu_serialize fallback（仅在受限容器启用）

仅在 `args.model_update_transport == "cpu_serialize"` 时启用。所有以下新增代码都
**仅服务这条分支**，cuda_ipc 默认路径不受影响。

Wire payload（与 ROLL `roll/utils/send_recv_utils.py` 同 schema，**不复用 SGLang
`FlattenedTensorBucket` 嵌套结构**）：

  ```python
  {"bucket": pinned_cpu_uint8_tensor, "tensors_meta": list[dict]}
  ```

    Sender 路径（cache owner 端，per bucket）：

    0. **与 cuda_ipc 保持一致：每个 bucket 必须先 H2D staging 到 GPU**。
      即使 cache 在 CPU，也先把该 bucket staging 到 GPU，再决定走 cpu_serialize
      或 cuda_ipc。cpu_serialize 只是传输通道不同，不改变 staging 顺序。
    1. bucket 在 GPU 上构造（避免 N 次 D2H）：
      `bucket_gpu = _bucket_named_tensors(hf_named_tensors)`
    2. DMA 到 pinned CPU：
     ```
     pinned = torch.empty_like(bucket_gpu, device="cpu").pin_memory()
     pinned.copy_(bucket_gpu, non_blocking=True)
     torch.cuda.current_stream().synchronize()
     ```
  3. 序列化：
     `payload_bytes = torch.save({"bucket": pinned, "tensors_meta": meta}, BytesIO).getvalue()`
  4. Ray put：`payload_ref = ray.put(payload_bytes)`
  5. Per-target receiver invocation：
     `SGLangEngine.update_weights_from_cpu_bucket.remote(payload_ref, load_format=..., flush_cache=..., weight_version=...)`

  **`payload_bytes` 不进入 HTTP request body，不落 `/dev/shm` 文件，不走 ZMQ。**

  Receiver 路径（SGLang Ray actor method，新增）：

  ```python
  # miles/backends/sglang_utils/sglang_engine.py
  def update_weights_from_cpu_bucket(self, payload_ref, load_format=..., flush_cache=False, weight_version=None):
      raw = ray.get(payload_ref)
      payload = torch.load(BytesIO(raw), weights_only=True)
      bucket = payload["bucket"].contiguous().pin_memory().to(self.device, non_blocking=True)
      torch.cuda.current_stream().synchronize()
      named_params = named_tensors_from_bucket(bucket, payload["tensors_meta"])
      # vendored SGLang in-process load helper（见下方 SGLang fork patch 范围）
      self._in_process_load_cpu_bucket(named_params, load_format=load_format, ...)
  ```

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

  bump SGLang 版本时，patches 与上游 source 不兼容则**重新 rebase**单个 patch 文件，
  不需要新建版本目录。

  **明确禁止**：

  - 不新增 `/update_weights_from_cpu_bucket` HTTP route
  - 不扩展 `/update_weights_from_tensor` 的 `transport` 字段
  - 不保留 HTTP fallback / debug path
  - 不在 actor method 内向 SGLang HTTP server 发 POST
  - cpu_serialize 路径不复用 `load_format="flattened_bucket"`（那是 SGLang
    `FlattenedTensorBucket` 内部协议，与 ROLL wire format 不兼容）

  **背景（仅适用 Case B）**：MILES 是首个把 cpu_serialize 引入 SGLang 链路的实现。
  ROLL 的 cpu_serialize 仅覆盖 vLLM（`infer_strategy == "sglang"` 分支恒走
  `MultiprocessingSerializer` + CUDA IPC）；SGLang 上游 `weight_sync/utils.py` 也只
  支持 `MultiprocessingSerializer`。CUDA IPC 等价路径在受限容器（vast.ai 等无
  CAP_SYS_PTRACE / `--ipc=host`）下不可用，故 cpu_serialize 是受限场景的必需 fallback；
  在能用 cuda_ipc 的环境下走 Case A，**不强制 cpu_serialize**。

  **LoRA 平行处理（仅 Case B）**：cuda_ipc 路径下 `load_lora_adapter_from_tensors`
  ([update_weight_from_tensor.py:341](external/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L341))
  不动；cpu_serialize 路径平行新增 `update_lora_adapter_from_cpu_bucket(payload_ref,
  lora_name, ipc_local_ranks, ...)` Ray actor method（命名与 `update_weights_from_cpu_bucket`
  对称）+ SGLang vendored patch 对应 dispatch。Wire payload schema 与 weight bucket
  相同（`{"bucket": pinned, "tensors_meta": meta}`），sender 端把 LoRA tensors 走相同
  打包路径。

##### 两条 colocate 路径共享的属性

无论选 cuda_ipc 还是 cpu_serialize：

- F4 §5 per-bucket barrier / `_cache_lock` invariant 同样适用（MilesModelUpdateService
  在外层 wrap，wire format 透明）
- Cache owner 由 `_is_distributed_src_rank` 决定（F4 §1）
- F5+6 receiver-side `finalize_weight_update` 由 pipeline 一次性调用，与 transport
  无关
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

**Comm plan 与动态 NCCL group（用 PyTorch 原生 helper，不依赖 ROLL utilities）**：

- 参照 ROLL `_build_comm_plan_for_sender` 分类逻辑（cpu_serialize vs broadcast，
  per-engine `cpu_serialize_local_ranks` / `broadcast_local_ranks` mask）
- `init_collective_group` / `destroy_collective_group` 用 PyTorch 原生 `dist.new_group`
  / `dist.destroy_process_group` 实现，**不**依赖 ROLL `roll/utils/collective/collective.py`
  （避免 import 依赖）
- `group_name = f"miles_model_update_{sync_id}"`，sync_id 含 pipeline_id + step + uuid，
  避免跨 pipeline / 跨调用冲突
- 临时 NCCL group 每次 sync 后 destroy，4 步生命周期 CLASSIFY/CREATE/USE/DESTROY：
  - **CREATE 步骤补 warmup allreduce**：group rendezvous 完成后立即发
    `collective.allreduce(torch.zeros(1, device='cuda'), group)` 验证 group 健康；失败
    立即 destroy + raise，不进入 USE 阶段
  - **DESTROY 步骤释放 port claim**：sync 完成（含成功 / sync 内部 fail-fast crash）后
    `shared_storage.release_port_claim(master_port)`，port 回 pool 供下次 sync 复用。
    **本 milestone 不处理 receiver crash 容错**——sync 路径上的 fail-fast 由 F5+6
    `MILES_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` 触发整 pipeline crash，pipeline restart
    时 SharedStorage 整体清理 port pool；不需要 conditional leak 设计。如未来生产观察到
    receiver crash 导致的 port collision，再单独立 plan 加 fault-tolerant port 管理

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

| 路径 | Engine 状态 | Owner | 机制 |
|---|---|---|---|
| **Active refresh** | 非重叠 active engines | training loop（`after_training` hook） | `coordinator.sync_base_weights_to_active()` → in-flight 推到非重叠 engines |
| **Expand sync** | 重叠 slept/woken engines | scheduler（`resize_infer(add=...)`） | `_expand_workers()` → wake → sync → activate |

**不变量：** 所有已 active 的 engine 由 training loop 刷新；所有后续被激活的 engine 由
expand 刷新。两条路径共享同一份 CPU bucket cache（单 cache owner）。

#### 为什么不复用 MILES 现有 `actor_model.update_weights()`

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

误标数量受 in-flight batch size 与单次 decode step 延迟约束（个位数请求量级），不与
fully_async 的 staleness window 混淆。

refresh 完成后的语义必须是 exact-at-engine/trajectory-granularity：

- refreshed engine 必须完成 receiver-side barrier / finalize，并发布
  `_current_weight_version == _cache_ready_step`
- 后续 trajectory 的 `weight_versions` 必须能被 `--max-weight-staleness` 可靠消费
- staleness cutoff 允许丢弃旧 trajectory，但不能替代 refresh 完成后的 version publish

Drain-then-sync **不在本移植方案范围内**。

#### Hardening：timeout + fail-fast

- **单一 `MILES_SELECTIVE_MODEL_UPDATE_TIMEOUT_S = 150` 秒**（默认值，对齐 RLix
  `rlix/pipeline/model_update_service.py:55` 中的 `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S`
  env var，default 150；该 const 实际归属 RLix，不在 ROLL）。在 `sync_selected_workers()`
  整体外加 `asyncio.wait_for` / `ray.get(timeout=)`。超时 → raise → coordinator →
  pipeline crash（**fail fast，no retry**）。Sender-side `_cache_lock` 持有期间也受
  timeout 约束。
- **不预先拆 active refresh / expand 两个常量**：active refresh 面向 serving workers
  （GPU 争用），expand 面向 idle workers（无争用），理论上 active 可能需要更宽 timeout，
  但**没有实测证据**支持预先拆分。先用单 const 跑 Gate 3，如真观察到 active refresh
  逼近 150s 上限再拆，YAGNI。

#### Control-plane 不变量

**Pipeline 在 `sync_base_weights_to_active()` 完成且 version 发布之前，不得调用
`self._notify_release_cluster_gpus(cluster_id="actor_train", ...)`
（[full_finetune_pipeline.py:497](rlix/pipeline/full_finetune_pipeline.py#L497) 已有
方法）。** GPU 释放信号表示"我的 active engines 权重一致"，不是"训练完成"。由
`after_training` hook 序列强制。

#### Actor call graph

避免 pipeline actor re-entrant self-call（参照 NeMo `sync_lora_weights` 模式）：

```
Pipeline actor (after_training):
  ├── ray.get(coordinator.sync_base_weights_to_active.remote())
  │     └── Coordinator actor:
  │           acquire _resize_sync_lock
  │           active_engines = _active_engine_indices
  │           sync_id = make_sync_id(pipeline_id, step)
  │           ray.get(model_update_service.sync_selected_workers.remote(sync_id, active_engines))
  │           release _resize_sync_lock
  │           return                         ← 不回调 pipeline
  ├── _finalize_weight_update(active_non_overlap_engines)  ← 一次性 post-load hook
  ├── self._current_weight_version = self._cache_ready_step   ← 本地，无 remote call
  ├── ray.get(rollout_manager.set_weight_version.remote(version))
  └── notify_release_cluster_gpus(actor_train)
```

#### Training step 序列（finalize 由 pipeline 驱动，per-bucket 之后**只调一次**）

```
1. train_step()
2. build_cpu_bucket_cache(step)             ← 所有 training rank gather；cache owner 存
3. _cache_ready_step = step
4. offload training GPU / destroy NCCL groups（ReloadableProcessGroup）
5. coordinator.sync_base_weights_to_active()  ← 内部: per-bucket apply N 次
                                                 (update_parameter_in_bucket / broadcast_parameter)
5b. pipeline → finalize_weight_update.remote(active_engines)
                                              ← 一次性 RPC 到每个 target engine，
                                                 在 sync_selected_workers 返回后执行；
                                                 不在 receiver 自驱、不在 sender 内嵌；
                                                 对齐 ROLL sync_selected_workers →
                                                 process_weights_after_loading →
                                                 load_states_partial 顺序
6. pipeline 本地更新 version
7. notify_release_cluster_gpus(actor_train)
8. (later) scheduler resize_infer(add=...)  ← 同一 cache 推 woken overlap engines；
                                                 同样 per-bucket apply N 次后由 pipeline
                                                 调一次 finalize_weight_update
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

         [training on overlap GPUs 0,1]                       😴  😴  ●   ●  (dp2,3 serve v2)

                                    after_training(step=3):
                                      build_cpu_cache(step=3)
                                      _cache_ready_step = 3
                                      offload training GPU
                                      destroy NCCL groups

                     <─ sync_base_weights ─── ray.get(coordinator.sync_base_weights_to_active())
                     lock
                     _active={2,3}
                     ── model_update_service.sync(sync_id, [2,3]) ──> 😴  😴 ●→v3 ●→v3 (in-flight)
                     unlock
                     ── return ─────>
                                    finalize_weight_update.remote(2,3)  ← pipeline 驱动一次性
                                    self._current_weight_version = 3
                                    set_weight_version(3) on rollout_manager
                                    self._notify_release_cluster_gpus(cluster_id="actor_train", ...)

resize_infer(dp_ranks_to_remove=[], dp_ranks_to_add=[0,1])
─────────────────>   lock
                     ───────────>   _expand_workers([0,1])
                                    ── wake([0,1])           ⏳  ⏳  ●v3 ●v3
                                    ── sync(sync_id, [0,1])  ✓v3 ✓v3 ●v3 ●v3
                                    ── finalize_weight_update.remote(0,1)
                                    ── publish version 3 (no bump)
                                    ── activate_routing([0,1])  ●v3 ●v3 ●v3 ●v3
                     _active={0,1,2,3}
                     unlock
```

#### Edge cases

1. **全部 engines 重叠（退化 = ROLL 拓扑）**：shrink 后 `_active_engine_indices` 为空。
   `sync_base_weights_to_active()` 立即短路返回（active set 为空）。Expand sync 所有
   engines on wake。**正确**。
2. **无重叠（所有 engines 非重叠）**：不发生 shrink/expand。`sync_base_weights_to_active()`
   in-flight sync 所有 engines。**正确**。
3. **Init 后首步**：CPU cache 含 base weights（`_cache_ready_step = -1`，由 step 0 init
   bootstrap 提供）。Active refresh / expand 都推送 base weights，全部 version = -1。
   **正确**。

#### Version accounting（修复 double-bump）

Version = `_cache_ready_step`（绑定到产生 cache 的 training step，不是 sync 操作）：

```python
# training step 3 之后：e
self._cache_ready_step = 3

# sync_base_weights:
self._current_weight_version = self._cache_ready_step  # = 3
ray.get(self._rollout_manager.set_weight_version.remote(self._current_weight_version))

# _expand_workers:
# 已经是 3，确保 collector 看到，不 bump
ray.get(self._rollout_manager.set_weight_version.remote(self._current_weight_version))
```

MILES 已在 [miles/utils/types.py](external/miles/miles/utils/types.py) `Sample` 上有
`weight_versions`，[miles/rollout/generate_utils/generate_endpoint_utils.py](external/miles/miles/rollout/generate_utils/generate_endpoint_utils.py)
在 train data 上传播 version。直接复用，无需新发明 version tracking。

#### Receiver-side：target engine API surface

SGLang engine 必须实现（参照 NeMo `vllm_backend.py`）。首个 parity gate 就是 `tp>1`，
因此 receiver API 必须覆盖 mixed receiver mask：同 GPU receiver 按
`args.model_update_transport` 走 cuda_ipc（既有 `update_weights_from_tensor`）或
cpu_serialize（新增 `update_weights_from_cpu_bucket`）；non-colocate receiver 走
dynamic NCCL broadcast；不能先降级成 tp=1-only 路径。

| 方法 | 调用者 | 路径 | 时机 |
|---|---|---|---|
| `setup_collective_group(name, comm_plan, mode, timeout_s)` | ModelUpdateService | NCCL | 跨 GPU TP peers |
| `update_weights_from_tensor(serialized_named_tensors, ...)` | ModelUpdateService | CUDA IPC | colocate, **既有** ([sglang_engine.py:282](external/miles/miles/backends/sglang_utils/sglang_engine.py#L282))，cuda_ipc 路径复用 |
| `update_weights_from_cpu_bucket(payload_ref, ipc_local_ranks, ...)` | ModelUpdateService | Ray ObjectRef bytes | colocate **新增**，仅 cpu_serialize 路径调用 |
| `broadcast_parameter(group_name, names, dtypes, shapes, broadcast_local_ranks)` | ModelUpdateService | NCCL | 动态 group |
| `destroy_collective_group(group_name)` | ModelUpdateService | NCCL | sync 完后；colocate-only ranks（不论 cuda_ipc 还是 cpu_serialize）必须有 `is_group_exist` no-op guard |
| `finalize_weight_update()` | Pipeline | — | 所有 bucket 完成后；worker 内执行 SGLang 端 post-load hook |

**`verify_model` 不在本 milestone receiver API surface**：传输健壮性由 per-bucket
barrier + warmup allreduce 保证；权重错误更可能从 trajectory reward 异常 / 模型行为
退化更早暴露而不是 hash 比对。如未来需要 debug-time validation，再单独加。

`finalize_weight_update()` 必须在 **engine 进程内**执行（涉及 SGLang model_runner 本地
对象，不可序列化）。

receiver-side hardening、version publish、timeout crash、sync barrier 都是 parity
必需项。

改动量：~230 行（两条路径 + version accounting + receiver API surface；不含 verify_model）

**修改/新增文件：**
- `rlix/pipeline/miles_pipeline.py` — `_expand_workers()`、`_after_training` hook、`_finalize_weight_update()`
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

cluster_device_mappings = {
    "actor_train": list(train_device_mapping),
    "actor_infer": list(infer_device_mapping),
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

coordinator = MilesCoordinator.options(
    name=f"rlix:coordinator:{pipeline_id}", namespace=ray_namespace,
).remote(pipeline_id, miles_args)
ray.get(coordinator.create_pipeline_actor.remote())
```

`cluster_device_mappings` 必须来自 **配置中声明的实际 train/infer device_mapping**，
不是 `list(range(n))` 这种连续 toy 示例。

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

**本地最小 counter 集（删除 dead state）**：本地只维护与 RLix wire 对齐的最小集：

- `step_target_trajectories`（来自 config，按 MILES 现有 args 派生：
  `args.rollout_batch_size * args.n_samples_per_prompt // (args.num_steps_per_rollout or 1)`，
  与 [arguments.py:2031](external/miles/miles/utils/arguments.py#L2031) `global_batch_size`
  公式一致；MILES 没有 NeMo / ROLL 那种 `num_groups_per_step` / `num_prompts_per_step`
  独立 arg）
- `_local_completed`（本地 increment counter，仅累加 `target_weight_version ==
  _progress_target_step` 的 sample）
- `_last_progress_bucket`（2% 阈值跟踪，0..50）
- `_progress_target_step`（当前 wait 的 weight version）

**显式不维护**（RLix scheduler 不消费，纯 dead state）：`queued / inflight / consumed
/ oldest_unfinished_creation_ts / active_weight_version / oldest_generation_weight_version`。
sample 自身已带的 `weight_versions` 沿用，但**不**作为 progress 上报字段。

**ProgressReport wire 字段表（MILES hook → MilesCoordinator aggregation ingress）**：

| 字段 | 值 | 来源 |
|---|---|---|
| `pipeline_id` | 注册时分配 | Feature 8 |
| `step_target_trajectories` | config | `args.rollout_batch_size * args.n_samples_per_prompt // (args.num_steps_per_rollout or 1)` |
| `metrics["collected"]` | local counter | `_local_completed`（reporter 端，**raw 不 clamp**；coordinator 端聚合 + clamp 才得到 scheduler-side `completed`） |
| `bucket` | 0..50 | `math.floor(_local_completed / step_target_trajectories * 50)`（reporter 端计算；只在 `bucket != _progress_last_bucket` 时才发 RPC，**2% gate 在 reporter 层**） |
| `current_train_step` | 当前 step | `_progress_target_step` |
| `mode` | `"train"` | full-finetune 单 stream |
| `new_batch` | True/False | begin_progress_batch 触发 True，后续 bucket 更新 False |

**两层职责分工（对齐 ROLL [rollout_scheduler.py:601-635](external/ROLL/roll/distributed/scheduler/rollout_scheduler.py#L601)）**：

| 层 | 字段 | 职责 |
|---|---|---|
| **Reporter（MILES rollout hook）** | `_progress_last_bucket: int`、`_local_completed: int`、`step_target_trajectories: int` | 持有 target context；每 increment 算 `bucket = floor(collected/target * 50)`；**只在 `bucket != _progress_last_bucket` / `remaining == 0` / `_progress_new_batch` 时**才发 RPC 给 coordinator（**2% bucket gate 在这一层**）|
| **Coordinator（`MilesCoordinator(PipelineCoordinator)`）** | streams keyed by `(mode, adapter_id)` | 聚合跨 stream 的 `collected`；`_aggregate_and_emit` 算 `completed = min(total_collected, total_required)`；fire-and-forget 给 RLix central scheduler |

1. **Reporter 实现**（`miles/utils/rlix_hooks.py` 或 `examples/fully_async/fully_async_rollout.py` 内嵌）：
   - 持有 `_local_completed`（仅累加 `target_weight_version == _progress_target_step` 的 group）
   - 持有 `_progress_last_bucket`（per-stream，初始 -1）
   - `bump_completed(group, target_weight_version)`：
     ```python
     if target_weight_version != self._progress_target_step:
         return  # 跨 step / age window 不一致的 group 不计
     self._local_completed += 1
     bucket = math.floor(self._local_completed / self._step_target_trajectories * 50)
     remaining = max(self._step_target_trajectories - self._local_completed, 0)
     should_emit = (bucket != self._progress_last_bucket
                    or remaining == 0
                    or self._progress_new_batch)
     if not should_emit:
         return  # ← 2% gate 在 reporter 这一层
     self._progress_last_bucket = bucket
     self._progress_new_batch = False
     # 只有 should_emit 时才 RPC
     coordinator.report_progress_from_scheduler.remote(
         ProgressReport(
             pipeline_id=...,
             step_target_trajectories=self._step_target_trajectories,
             metrics={"collected": self._local_completed, "bucket": bucket,
                      "mode": "train", "new_batch": ..., ...}
         )
     )
     ```
2. **Coordinator ingress**：`MilesCoordinator.report_progress_from_scheduler(report)`
   （[coordinator.py:299](rlix/pipeline/coordinator.py#L299) 既有方法，inherit 不改）。
   入站 report 必须带 `metrics["collected"]`，**不能直接带 scheduler-level `completed`**
3. **Coordinator → central scheduler**：`_aggregate_and_emit`
   ([coordinator.py:359](rlix/pipeline/coordinator.py#L359) 既有，inherit 不改) 聚合
   跨 stream `collected`，算 `completed = min(total_collected, total_required)`，
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

- `begin_progress_batch(target_weight_version, step_target_trajectories)`（**硬不变量**）：
  进入"等待当前 step 凑齐 trajectory"之前调用。reporter 端**必须**：
  - 持有 `step_target_trajectories`（从外部传入；reporter 没有 RLix wire 上下文时无法
    自己算 target，必须由 caller 传）
  - 从 fully_async output queue 一次性读取**当前已就绪且 `target_weight_version == 当前
    step` 的 group count** 作为初始 `_local_completed`，**不能直接 reset 为 0**。
    否则与 ROLL `get_batch()` batch-open snapshot 语义不一致 → scheduler 误判 demand →
    过早 shrink
  - reset `_progress_last_bucket = -1`（强制首个 emit 触发，因为初始 bucket 必然 ≠ -1）
  - 立即发送 `new_batch=True` 首个 RPC（这时 `should_emit` 一定 True，bypass 2% gate）
- `end_progress_batch()`：放在 wait window 的 `finally` 中，无论成功或异常都清除
  progress stream（`coordinator.clear_progress_stream(...)`）
- 后续 group/sample completion hook（`bump_completed`）仅在 `_progress_active and
  target_weight_version == _progress_target_step` 时累加 `_local_completed` + 算 bucket，
  本地 2% gate 决定是否 RPC，避免 hot path 上每次 push 都 `ray.get`；RPC 的报文打
  `new_batch=False`，对齐 ROLL `RolloutScheduler` wire-level 行为

**MILES 的 wait window 精确锚点**（不是 `train_async.py:37`）：

`train_async.py:37` `rollout_data_curr_ref = await rollout_data_next_future` 是 await
Ray ObjectRef，从 trainer 视角看不到 generation 内部进度事件，**不能在这里 hook**。

实际 demand window 在 [examples/fully_async/fully_async_rollout.py:218-220](external/miles/examples/fully_async/fully_async_rollout.py#L218)
`generate_rollout_async()` 内部的 `while len(data) < target_data_size:` 主循环。这是
真正轮询完成 group 的地方，是 begin/end_progress_batch 的目标位置：

```python
# examples/fully_async/fully_async_rollout.py:generate_rollout_async()
async def generate_rollout_async(args, rollout_id, data_buffer):
    ...
    target_data_size = ...

    if DO_TIME_SHARING:
        rlix_hooks.begin_progress_batch(target_weight_version=current_weight_version)
    try:
        while len(data) < target_data_size:                    # ← 实际 wait window
            completed = worker.get_completed_groups()
            for group_id, group in completed:
                # ... existing logic ...
                if DO_TIME_SHARING:
                    rlix_hooks.bump_completed(group, target_weight_version)  # 2% bucket trigger
            ...
    finally:
        if DO_TIME_SHARING:
            rlix_hooks.end_progress_batch()                    # ← 必须 finally，异常也清

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

```python
assert args.sglang_data_parallel_size == 1, "RLix mode requires sglang dp == 1"
# PD disaggregation 走 SglangConfig property，不是 args field
# （[sglang_config.py:95,189](external/miles/miles/backends/sglang_utils/sglang_config.py#L95)）
assert not sglang_config.has_pd_disaggregation, "PD disaggregation out of scope"
assert train_devices.issubset(infer_devices), \
    "partial overlap requires train ⊂ infer"
assert infer_engine_count >= 2, "partial overlap requires >= 2 engines"
assert async_generation_enabled, "partial overlap requires fullasync"
assert len(infer_devices - train_devices) >= args.rollout_num_gpus_per_engine, \
    "at least 1 full inference engine must stay active after shrink"
assert single_updateable_model_and_server(args)

# Megatron 内部并行度 divisibility（资 NeMo F10）— 防 Megatron init 时才报错。
# MILES 的实际 args 名是 *_model_parallel_size 全名，不是缩写
# （参 [initialize.py:42-48](external/miles/miles/backends/megatron_utils/initialize.py#L42)
# 与 [bridge_lora_helpers.py:89-95](external/miles/miles/backends/megatron_utils/bridge_lora_helpers.py#L89)）
megatron_parallelism_product = (
    args.tensor_model_parallel_size
    * args.pipeline_model_parallel_size
    * args.context_parallel_size
    * args.expert_model_parallel_size
)
assert len(train_devices) % megatron_parallelism_product == 0, (
    f"train device_mapping ({len(train_devices)}) must divide evenly by "
    f"tp*pp*cp*ep ({megatron_parallelism_product})"
)
assert len(infer_devices) % args.rollout_num_gpus_per_engine == 0, (
    f"infer device_mapping must divide evenly by rollout_num_gpus_per_engine"
)
```

改动量：~30 行

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
| `ray.shutdown()` | 正常执行 | 跳过（RLix 管 Ray 生命周期） | Flag |
| Train step 后 | 直接进入 `actor_model.update_weights()` | CPU bucket cache build + offload training GPU + destroy NCCL groups | Flag + Hook |
| Weight sync | `update_weights_from_distributed/tensor` 全量 | 跳过 — scheduler expand 触发 `MilesModelUpdateService.sync_selected_workers()` | Flag |
| Generation lifecycle | 训练 loop 内自启 | scheduler `request/release_cluster_gpus(actor_infer)` 驱动 | Flag |
| Progress reporting | 无 | fully_async path 上报 demand | Hook |
| Placement group | `create_placement_groups(args)` 自建 | 接受外部 PG + bundle indices | Flag + Hook |

实现：`RLIX_CONTROL_PLANE` env var → `DO_TIME_SHARING`（与 ROLL 一致）。

**入口结构（双 entry，不在 `train_async.py` 内部分支）：**

| Entry | 模式 | 文件 |
|---|---|---|
| Standalone | `DO_TIME_SHARING=False`（原 MILES 行为） | `train_async.py` **完全不动** — 仍是 `asyncio.run(train(args))`，标准 entry point |
| RLix | `DO_TIME_SHARING=True` | 新 driver 脚本 `examples/rlix/run_miles_rlix.py`：构造 orchestrator + MilesCoordinator + admit + `coordinator.create_pipeline_actor()` + `pipeline.initialize_pipeline()` + main loop |

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
        "this would run standalone actor_model.update_weights() with overlap GPU contention → silent OOM."
    )
```

```python
# examples/rlix/run_miles_rlix.py 内 RLix 模式实际行为
if DO_TIME_SHARING:  # always True at this entry
    build_cpu_bucket_cache(step)
    self._cache_ready_step = step
    offload_training_gpu()        # ReloadableProcessGroup destroy_process_groups
    hooks.after_training(step)    # notify scheduler → expand
# 没有 else 分支 — RLix entry 不应在 standalone path
```

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
   - 调用 `RollResourceManagerProxy.allocate_placement_group(...)` 获取 shared PG
   - 从 RLix/ROLL 的 per-worker device mapping 显式推导
     `actor_pg / rollout_pg / actor_bundle_indices / rollout_bundle_indices /
     actor_gpu_ids / rollout_gpu_ids`
3. `MilesPipeline.initialize_pipeline()` 在 RLix 模式下注入 provider；普通模式下
   provider 为 `None`，行为完全不变
4. 不修改 `miles/ray/placement_group.py` 的内部 bundle 切分逻辑 — 只在入口处替换 PG 来源

**Bundle adapter 是真实工作项，但比 NeMo `RLixVirtualClusterAdapter` 薄：**

provider 必须把 RLix shared PG + device mapping 映射成 `RolloutManager` 期望的三元组：

- `placement_group` ← `RollResourceManagerProxy.allocate_placement_group(...)` 返回的
  Ray PG handle
- `reordered_bundle_indices` ← RLix `cluster_device_mappings[role]` 在 PG 内对应的
  bundle index list
- `reordered_gpu_ids` ← 与 bundle 对应的 GPU id

NeMo 之所以要 `RayVirtualCluster`-shape adapter 是因为 `VllmGeneration` /
`RayWorkerGroup` 要求 cluster object 暴露 `world_size()` / `get_placement_groups()`
等方法；MILES 直接消费 PG handle + bundle index，因此 adapter 更薄，但不能省略。

改动量：~120 行（hook + adapter）

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
(c) F3-4：`partial_rollout=True` 的 sample 通过 radix_tree_middleware 不被 reject
    （routing skip 才有效，否则 partial trajectory 被 radix tree 挡掉）

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
(a) **`args.model_update_transport="cuda_ipc"` 默认路径**：colocate sync 走既有
    `update_weights_from_tensor`（IPC handle string）；不调用 `update_weights_from_cpu_bucket`；
    receiver IPC-only ranks `destroy_collective_group` no-op
(b) **`args.model_update_transport="cpu_serialize"` fallback**：colocate sync 走新增
    `update_weights_from_cpu_bucket`（Ray ObjectRef）；不发 HTTP POST；receiver
    cpu_serialize-only ranks `destroy_collective_group` no-op（`is_group_exist` guard）
(c) Warmup allreduce（F4-6）必须在每次 NCCL group USE 之前发出；故意制造 group 损坏验证
    raise 路径
(d) 两种 colocate transport 下，non-colocate broadcast receiver ranks 行为一致

预期：无 NCCL 错误，无 VRAM 泄漏，权重正确
关键：覆盖两条 colocate transport（cuda_ipc / cpu_serialize）+ NCCL broadcast +
      receiver dual-mask + Megatron NCCL lifecycle。省下 NeMo Gate 2.5 的 fallback
      rule — MILES 直接复用 ReloadableProcessGroup。
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
(b) `MILES_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` 在 active refresh 人为 hang 时触发 crash
    （注入 receiver block，验证 150s 内 raise）
(c) `begin_progress_batch` 启动时读到 buffer 内已就绪 group count 非 0（不 reset 为 0）
(d) Multi-turn preempt: 注入 sleep_partial mid-turn，验证 turn-level redispatch 重做该
    turn 而不是从 turn 0 重做，已完成 turn 的 env state 保留
(e) Staleness control: 每个 turn 的 `weight_versions` 被记录；trajectory 使用
    `oldest_weight_version` 判断 staleness，超过 `--max-weight-staleness` 时 group
    recycle，未超过时进入 train_data / rollout metrics

此 gate 是 in-flight active refresh 在生产负载下的主要验证点
```

### Gate 4: 双 MILES pipeline 调度

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
```

---

## 文件改动总清单

### MILES 侧

| 文件 | Feature | 改动 | 行数 |
|---|---|---|---|
| `miles/backends/sglang_utils/sglang_engine.py` | F1, F2, F5+6 | post-sleep VRAM assertion + idempotency guard + `is_idle()` worker method + receiver-side 方法：cuda_ipc 路径复用既有 `update_weights_from_tensor`，cpu_serialize fallback 新增 `update_weights_from_cpu_bucket(payload_ref, ipc_local_ranks, ...)` Ray actor method；setup/destroy_collective_group with `is_group_exist` no-op guard, broadcast_parameter, finalize_weight_update | +180 |
| `miles/ray/rollout.py` | F1, F2, F9 | `EngineInfo` dataclass（替代 5 个并行 map）+ `RolloutManager.{offload,onload,onload_weights,onload_kv}(engine_indices=None)` 仅在 Manager 层（不在 Server/Group 层）+ `shrink_engines/expand_engines` 复合操作 + abort-drain-sleep + `_routing_lock` compound 不变量 + progress callback hook | +170 |
| `miles/router/router.py` | F3 | `/disable_worker` / `/enable_worker` / `/remove_worker` endpoint + `/list_workers` 扩展 metadata | +80 |
| `miles/router/middleware_hub/radix_tree_middleware.py` | F3 | 删除 partial_rollout reject（pass-through，路由决策让给 Router admission API 单点负责） | +30 |
| `miles/rollout/generate_hub/multi_turn.py` | F3 | 删除 `assert not args.partial_rollout`（line 29）+ snapshot-then-retry turn loop（`_snapshot_turn_state` + `_restore_turn_state` + `MAX_TURN_REDISPATCH_ATTEMPTS` redispatch 循环）+ `_is_scheduler_preempt` 判定 | +50 |
| `miles/rollout/generate_utils/generate_endpoint_utils.py` | F3 | `_snapshot_turn_state` / `_restore_turn_state` 辅助（`tokens` / `weight_versions` / `loss_mask` / `prefix_cache_info` / `spec_info` 5 字段可逆增量） | +40 |
| `miles/backends/megatron_utils/update_weight/cpu_bucket_cache.py` (**新增**) | F4 | CPU bucket build + lookup + `_cache_ready_step` 单槽指针 | +180 |
| `miles/ray/placement_provider.py` (**新增**) | F12 | `MilesPlacementProvider` adapter | +100 |
| `miles/ray/placement_group.py` | F12 | `create_placement_groups(args, *, external_provider=None)` | +20 |
| `examples/fully_async/fully_async_rollout.py` | F2, F9 | abort 触发口 + begin/end_progress_batch 包裹 wait window（group recycle 路径不动 — 处理 MILES 既有非 RLix abort，与本 port 正交） | +30 |
| `train_async.py` | F11 | RLix mode fail-fast guard（detect `RLIX_CONTROL_PLANE=rlix` → raise；不在此处分支 RLix 行为，仅 reject）+ partial overlap topology fail-fast | +20 |
| `examples/rlix/run_miles_rlix.py` (**新增**) | F8, F11 | RLix entry driver — orchestrator allocate/register/admit + MilesCoordinator 创建 + `pipeline.initialize_pipeline()` + main loop（不调 `train_async.train()`，独立控制流） | +120 |
| `miles/utils/rlix_hooks.py` (**新增**) | F9, F11 | `RLixHooks` protocol + `NoOpRLixHooks` 默认 + import seam | +30 |

### RLix 侧

| 文件 | Feature | 改动 | 行数 |
|---|---|---|---|
| `rlix/pipeline/miles_pipeline.py` (**新增**) | F5+6, F8, F10, F12 | `MilesPipeline` actor — registration + validation + resize_infer + expand+selective sync + `_after_training` hook + bundle mapping helper | +450 |
| `rlix/pipeline/miles_coordinator.py` (**新增**) | F3, F5+6, F7, F9 | `MilesCoordinator(PipelineCoordinator)` — **subclass 既有 `PipelineCoordinator`** ([coordinator.py:183](rlix/pipeline/coordinator.py#L183))。**Inherit 不变**：`report_progress_from_scheduler` (299)、`clear_progress_stream` (326)、`_aggregate_and_emit` (359)、`_inject_pipeline_env_vars` (412)、`_resize_sync_lock`、`_validate_offload_nccl` (136)、`resize_infer` (502)、`sync_lora_weights` (440，LoRA out-of-scope 不调用)。**Override**：`__init__` 注入 MILES-specific deps（`MilesPlacementProvider` / `MilesModelUpdateService` / `MilesPipeline` 类）+ `create_pipeline_actor` (242) 创建 `MilesPipeline` 而非 `FullFinetunePipeline`。**新增**：`sync_base_weights_to_active(self) -> None` | +180 |
| `rlix/pipeline/miles_model_update_service.py` (**新增**) | F4, F5+6 | 简化版 ModelUpdateService（selective sync 双 colocate transport：cuda_ipc 默认走 MILES 既有 `_send_to_colocated_engine` 不重写；cpu_serialize fallback 走 Ray ObjectRef + SGLang vendored fork actor method）+ 动态 NCCL group 生命周期 + 单槽 versioning + warmup allreduce + port claim 释放（happy path 无条件回 pool；不处理 receiver crash 容错）+ 复用 MILES `_is_distributed_src_rank` 选 cache owner + `sync_id` 参数 + init bootstrap step=-1 path | +290 |
| 新增 `EnginePreemptedError`（位于 `miles/utils/types.py` 或 `miles/rollout/`） | F3 | 异常定义 + import seam | +5 |

### 测试

| 文件 | 改动 | 行数 |
|---|---|---|
| `tests/test_partial_sleep_wake.py` (**新增**) | Feature 1-3 单元测试 | +150 |
| `tests/test_miles_pipeline.py` (**新增**) | Feature 4-6 集成测试 | +200 |

**总计：~2225 行**（不含可选 `base_coordinator.py` — 已删除）。NeMo RL port 是 ~1700
行 — 量级相当，**不是 MILES 多出"genuinely new"的东西**：partial sleep/wake、turn-level
redispatch、CPU bucket cache、selective sync、engine indexing、registration 等 NeMo 同样
从零搭建。

（与 NeMo 工作量对比表已挪到 §关键风险与边界段以保持本节单一职责。）

---

## 实施顺序

以下只是工程排期，不代表降级 MVP。每个可合入 gate 都按最终 tp>1 parity 配置验证。

```
Week 1: Feature 1-3 — partial sleep/wake + routing skip + turn retry
  ├── Day 1: Feature 1 — post-sleep VRAM assertion + scheduler RPC
  ├── Day 2-3: Feature 2 — engine index map + subset API + abort-drain-sleep + routing lock
  ├── Day 4: Feature 3 — router admission + radix tree partial 解禁
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
  └── Day 5-6: Gate 3 (单 pipeline) + Gate 4 (双 MILES pipeline)

Week 4: 打磨
  └── Day 1-5: 文档、edge case、双 MILES pipeline 稳定性回归
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
- partial rollout 在 MILES radix tree middleware 当前显式 reject —
  解禁后要保证 cache hit 行为正确（partial trajectory 不污染 prefix cache）
- CPU cache `total_cpu_cache_bytes` 在 host RAM 紧张时 OOM — 首次 build 时 fail fast
  守住
- **SGLang TP NCCL `release_memory_occupation` 后保留**：与 vLLM 同设计，SGLang 自身
  release 路径就依赖此（用 `barrier(tp_cpu_group)`）。Gate 2 是回归确认而非真正风险
  点；NCCL buffer 残留 (~200 MB) 由 post-sleep VRAM 阈值兜住

### MILES vs NeMo 工作量对比

| 维度 | MILES vs NeMo |
|---|---|
| **MILES 省下的** | `ReloadableProcessGroup` 已提供 NCCL teardown（NeMo 需 `nccl_offload.py` ~90 行 + Gate 2.5 fallback rule）；MILES 不需要 NeMo 那种完整 `RayVirtualCluster` object adapter |
| **MILES 多花的** | Miles Router 是独立进程，需要 `/disable_worker` / `/enable_worker` / `/remove_worker` endpoint（~80 行；NeMo 无独立 router，直接通过 vLLM `collective_rpc`）；radix-tree middleware partial-rollout 解禁（~30 行；NeMo 无 prefix-cache routing 限制）；turn snapshot/restore 辅助（~40 行；NeMo 的 turn retry 由于 message_log 累加在 `_async_generate_base` 之上，不需要逐字段 rollback）；ROLL-style device mapping 到 MILES PG 三元组的 `MilesPlacementProvider` 显式 adapter |
| **总差额** | ≈ 持平。MILES ~2200 行 vs NeMo ~1700 行，差额来自 partial overlap + 5+6 selective sync 全套基础设施都是新的；ROLL/NeMo 已经踩过的坑（动态 NCCL group / warmup allreduce / cpu_serialize-broadcast dual-mask）都直接对齐 ROLL `model_update_service.py`，不重新发明。Receiver crash 容错（conditional port leak、周期性 GC）超出本 milestone，留作 follow-up |

### 推荐的立即决策

- 批准以 **partial overlap + subset sleep/wake + CPU bucket cache + selective sync** 
  作为本 port 的统一目标（与 ROLL/NeMo F1-F12 同形态）
- **不复刻** request-level deterministic migration — handoff 主路径绑定 multi_turn
  turn-level redispatch；single_turn 路径走 group recycle；其他 abort 来源（tool
  error 等）由 fully_async 既有 group recycle 处理（与本 port 正交）
- **不重新实现** Megatron NCCL teardown — 直接复用 `ReloadableProcessGroup`
- 第一批 scheduler 集成验证目标绑定 `examples/fully_async` 外层，但必须显式启用
  `multi_turn.generate` custom path + `--max-weight-staleness`
- 最终 gate 到 Gate 4 为止：双 MILES pipeline 调度。ROLL + MILES 混合调度不在本
  plan 验证范围内。

---

## Implementation follow-up（不在本 plan 范围，仅留挂钩）

| 目标 | 触发条件 |
|---|---|
| 抽 `rlix/pipeline/coordinator.py` 的 backend-neutral hook（base coordinator） | 当第 3 个 backend（NeMo / 新 framework）port 到位、共享逻辑足够清晰时 |
| MoE / EP support — `get_expert_tensor_parallel_group()` 等 | 当业务方启用 MoE 模型且需要 RLix 调度时；预计是中等改动（~200-400 行 + 专门 MoE parity gate），不是当前 plan 的小补丁 |
| Selective P2P weight transfer | 当 broadcast/tensor subset sync 成为吞吐瓶颈时 |
| Receiver crash 容错（fault-tolerant port 管理 + conditional leak + 周期性 GC） | 当生产观察到 receiver crash 导致 master_port collision，或 selective sync 失败率非 zero 时 |
