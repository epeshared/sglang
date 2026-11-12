# Fast Loop 热点专家统计与卸载：按层（per-layer）改造

## 背景 / 动机

在改造之前，fast loop（`FastLoopScheduler`）对每个 rank 只维护**一个全局的**热点专家统计
（`CudaGraphOffloadContext._expert_count_acc`，一个 `[512]` 的计数 tensor）和**一个全局的**
CPU 卸载槽位（`CpuShadowRankState.cpu_slot_expert_id`，一个标量）。

问题：不同 MoE 层的专家路由分布通常并不相同，但旧实现会把所有层的流量混合累加到同一个计数器里，
决策出的"最热专家"再统一应用到该 rank 负责的**每一层**上——即使某一层的真实热点专家与全局热点完全不同，
也只能卸载全局那一个，无法做到"每层各自卸载最合适的热点专家"。

## 改造目标

把热点专家统计和 CPU 卸载槽位分配，从 **per-rank 全局** 改为 **per-(rank, layer) 独立**：
每一层都有自己独立的流量计数器和独立的 CPU 卸载槽位决策，互不影响。

## 核心 API 变更

### `shadow_state.py`

- `CpuShadowRankState.cpu_slot_expert_id: int` → `cpu_slot_by_layer: dict[int, int]`
  （`{layer_id: expert_id}`，`-1` 表示该层当前不卸载）。
- 新增方法：
  - `get_slot(layer_id) -> int`
  - `assign_slot(layer_id, expert_id) -> None`
  - `clear_slot(layer_id=None) -> None`（不传 `layer_id` 则清空该 rank 所有层）
- `CpuShadowRegistry.set_slots(dict)` → `set_layer_slots({layer_id: {rank_id: expert_id}})`。
- `apply_initial_slots()` 的所有引导策略（`SGLANG_CPU_OFFLOAD_INIT_SLOT(S)`、
  `CPU_OFFLOAD_USE_LAST_GPU_SLOT` 等）现在都按 `range(num_layers)` 逐层写入初始槽位，
  其中 "last GPU slot" 策略会正确使用每层各自的 `physical_to_logical_map` 行
  （而不是像以前一样只用 layer 0 的行）。

### `cuda_graph_offload.py`

- `_cpu_slot_buf` / `_expert_count_acc`（单个 GPU tensor）→
  `_cpu_slot_bufs` / `_expert_count_accs`（`dict[layer_id, tensor]`），在
  `register_layer_from_daemon(layer_id, ...)` 时按需惰性分配（幂等）。
- 新增：
  - `registered_layer_ids() -> list[int]`
  - `get_layer_expert_counts(layer_id) -> Optional[Tensor]`
- `update_slot(expert_id)` → `update_slot(layer_id, expert_id)`。
- `begin_offload()` 内部的 `scatter_add_`、槽位比较（`topk_ids == cpu_slot`）等全部改为
  按 `layer_id` 取对应的 buffer，不再共用一份。

### `fast_loop.py`（决策逻辑核心重写）

`_run_once()` 改为：

1. **Phase 1（统计）**：遍历 `gctx.registered_layer_ids()`，每层单独从
   `gctx.get_layer_expert_counts(layer_id)` 读取累计计数，与上次 tick 的快照
   （`self._expert_count_snap[layer_id]`）做差得到本轮增量 `delta`，按层存入
   `layer_expert_counts: dict[layer_id, Tensor]`。
   （若 CUDA graph 路径无数据，回退到旧的 Python 快照 `_load_snapshot_fn()`，
   该快照本身就是按 `layer_id` 分桶的。）
2. **Phase 2（决策）**：对 `layer_expert_counts` 中的**每一层**独立执行原有的
   "按热度排序 + 排除本地 GPU 专家 + top-M" 逻辑，得到该层各 rank 的卸载专家，
   汇总进 `layer_assignments: dict[layer_id, dict[rank_id, expert_id]]`。
3. **Phase 3（发布）**：`self._registry.set_layer_slots(layer_assignments)` 写回
   shadow state；本 rank 每层新槽位通过 `gctx.update_slot(layer_id, new_slot)`
   推送到对应层的 GPU 槽位 buffer（CUDA graph 捕获的是 tensor 读取而非常量，
   所以可以在两次 replay 之间安全更新，无需重新 capture）。
4. **打印**：每个 tick 打印一行紧凑日志（避免层数多时刷屏）：

   ```
   [fast_loop tick#<N>] src=cuda_graph_acc layers_active_this_tick=<K> \
       slot_changes=<M> local_rank=<R> local_slots_by_layer={<layer_id>: <expert_id>, ...}
   ```

   其中 `local_slots_by_layer` 直接体现了"不同层选出不同热点专家"的效果。

### 调用方同步改造（`local_cpu_slot()` 现在必须传 `layer_id`）

- `runtime.py`：`local_cpu_slot()` → `local_cpu_slot(layer_id)`。
- `dispatch_hook.py`：CUDA graph 捕获分支与 eager 分支的两处调用都传入当前 `layer_id`。
- `layers/moe/ep_moe/layer.py`：两处调用改为传 `self.layer_id`。
- `model_executor/model_runner.py`：`init_cuda_graph_offload()` 中"capture 前预分配初始槽位"
  的逻辑从"只做一次、只用 layer 0 的 EPLB 映射"改为**移入按层注册的循环内部**，
  每层使用自己在 `physical_to_logical_map` 中对应行来推导初始槽位。

## 测试

- 测试目录：`python/sglang/srt/eplb/cpu_offload/tests/`
  （`pytest` 在当前环境未安装，需用 `python -m unittest` 运行）：

  ```bash
  cd sglang_heto
  python -m unittest \
    python.sglang.srt.eplb.cpu_offload.tests.test_basic \
    python.sglang.srt.eplb.cpu_offload.tests.test_dispatch_hook \
    python.sglang.srt.eplb.cpu_offload.tests.test_daemon_backend
  ```

- `test_basic.py` / `test_dispatch_hook.py` 中所有引用旧标量 API
  （`cpu_slot_expert_id`、`set_slots(...)`、`local_cpu_slot()` 无参数形式）的用例
  已同步更新为新的按层 API。
- 与改造前的基线（`git stash` 对比）逐条比对失败测试名，确认改造**未引入新的回归**——
  剩余失败均为当前环境缺少 `moe_cpu_engine` 原生模块 / CUDA 导致的既有失败，与本次改动无关。

## 涉及文件一览

| 文件 | 改动类型 |
| --- | --- |
| `shadow_state.py` | 核心：槽位数据结构改为按层字典 |
| `cuda_graph_offload.py` | 核心：GPU 侧计数器/槽位 buffer 改为按层字典 |
| `fast_loop.py` | 核心：决策循环按层独立执行 |
| `runtime.py` | 接口：`local_cpu_slot()` 增加 `layer_id` 参数 |
| `dispatch_hook.py` | 调用点适配 |
| `layers/moe/ep_moe/layer.py` | 调用点适配 |
| `model_executor/model_runner.py` | CUDA graph capture 前的初始槽位引导逻辑按层适配 |
| `tests/test_basic.py`、`tests/test_dispatch_hook.py` | 测试用例适配新 API |
