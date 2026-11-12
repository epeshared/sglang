# 旧版入口原件

> **摘要**：本目录保存迁移来源的原始文件，供行为对照；新版运行路径不导入旧静态实现，也不能直接运行这些旧实验脚本。

来源提交和当前可用范围见[接入说明](../../../../../../wiki/moe-offload.md)。

- `ep_moe_v055.py`：旧版 `layers/moe/ep_moe/layer.py` 原件，包含静态 Heto 类。
- `bench_one_batch_v055.py`：旧版定制基准原件，保留扫描及采样逻辑；现有纯函数单测从此处提取函数进行对照。
- `hetero-test/`：原实验脚本，原路径和环境依赖按来源版本保留。

这些文件与来源逐字节核对，并从自动格式化中排除。当前基准入口是 `python -m sglang.benchmark.one_batch`；旧命令名 `sglang.bench_one_batch` 已转发到该上游入口。
