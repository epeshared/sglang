# 接入与使用边界

> **摘要**：新版基线已有 Hy4 原生模型实现。本次迁入动态 CPU 专家逻辑；现有 CPU 后端接收分块 FP8 权重，Hy4 BF16 的 CPU 路径仍需适配。验证边界见[迁移记录](validation.md)。

## 版本来源

- 上游：`sgl-project/sglang` 的 `main`，固定在 [`725b98b114f3af937636e4e709f1898b9fcb0ad0`](https://github.com/sgl-project/sglang/commit/725b98b114f3af937636e4e709f1898b9fcb0ad0)。
- Hy4 的[模型支持改动](https://github.com/sgl-project/sglang/pull/36805)已合入；首个包含该支持的正式版本是 [v0.5.20](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)。本 fork 沿用创建时的上游 `main` 历史。
- 本地逻辑：工作区 `repos/sglang_heto-fused-prepare`，分支 `experiment/fused-prepare-20261110`，提交 `9bcfaac58dfea473e150bd530fdc546c82242b0f`；包含动态槽位、逐步追踪、pinned 通信选择和融合准备路径。
- 配套 CPU 后端：[`intel-sandbox/moe-cpu-engine-xtang`](https://github.com/intel-sandbox/moe-cpu-engine-xtang)，提交 `3491bc78b670309464960fe2b3fd739114acb294`。该后端是独立仓库，需要对应权限和原生编译；本 fork 没有内嵌它。

## 新版接线

[参数声明](../python/sglang/srt/arg_groups/fields/cpu_experts.py)进入新版 `exec.moe` 命名空间，默认关闭。`ModelRunner` 通过 [runner_hooks.py](../python/sglang/srt/eplb/cpu_offload/runner_hooks.py) 初始化运行时、登记加载完成的专家权重、准备图捕获及驱动快速调度。

新版 `FusedMoE.forward_impl` 通过 [layer_hooks.py](../python/sglang/srt/eplb/cpu_offload/layer_hooks.py) 复用原分流过程：先启动 CPU 计算，把对应 GPU 专家编号改成 `-1`，GPU 完成后加回 CPU 结果，再进行跨卡求和。原始路由张量不在原地修改。

逐步追踪通过 [graph_hooks.py](../python/sglang/srt/eplb/cpu_offload/graph_hooks.py) 在暖机后开始捕获，重放记录实际 token 数；请求身份从新版调度批次读取。pinned 通信保留首次导入 torch 之前的线程限制，以及所有 rank 在初始化末尾的线程核验。

## 可接入范围

当前动态 CPU 路径要求：

- 输入权重的量化配置是 `quant_method=fp8`、`weight_block_size=[128,128]`。
- 流水线并行数为 1，专家并行数与张量并行数相同；设置 `--moe-a2a-backend none`，关闭投机解码。
- 使用能够提供显式专家编号的 Triton MoE 路径，建议显式设置 `--moe-runner-backend triton`。
- 设置 `--cuda-graph-backend-prefill disabled`；decode 图捕获支持 `full` 或 `disabled`，缓冲区容量按新版已解析的捕获批量分配。
- 模型权重完整加载后才能登记 CPU 专家；启动时的权重加载重叠与此路径不兼容。

沿用 `--enable-cpu-offload-fast-loop`、`--cpu-active-ranks` 等开关及 CPU 后端原来的环境配置。这里列的是必要条件，不能代替真机验证，也不能作为直接启动 Hy4 BF16 CPU offload 的命令。

关闭动态 CPU 开关时，Hy4 的[原生模型](../python/sglang/srt/models/hunyuan_v4.py)和[配置](../python/sglang/srt/configs/hy_v4.py)保持上游实现。开启开关加载 Hy4 BF16 会明确报错：CPU 后端还缺原始 BF16 权重加载和计算适配。`MOE_DTYPE=bf16` 仍表示将后端已有的分块 FP8 输入转成 BF16，不能替代这项适配。Hy4 官方 MXFP8 权重也不等于这里的分块 FP8 格式。

当前逻辑仍保留 GPU 上的专家权重，尚未实现靠冷热专家划分来减少完整模型的 GPU 权重占用。

## 旧入口去向

旧版静态 Heto、定制基准和配套脚本保存在 [legacy 目录](../python/sglang/srt/eplb/cpu_offload/legacy/README.md)，便于逐项对照。新版动态路径没有导入旧静态类；单独启用 `--enable-ep-moe-heto` 会报错。

当前基准使用上游入口 `python -m sglang.benchmark.one_batch`；`python -m sglang.bench_one_batch` 只转发到它。旧版自定义扫描参数仍留在封存原件中，尚未接入新版基准入口。
