# 迁移验证记录

> **摘要**：记录 2026-11-12 本地迁移检查。单测不能代替完整 GPU 推理；此处不宣称 Hy4 BF16 CPU offload 已跑通。版本来源见[接入说明](moe-offload.md#版本来源)。

## 环境和证据

检查在工作区独立虚拟环境 `tmp/hy4-migration-20261112/venv` 中运行，沿用本机 CPU 版 PyTorch，没有安装配套 GPU 内核和完整 CPU 原生扩展。按当前上游依赖安装了 `transformers==5.19.0`、`tokenizers==0.23.2`、`apache-tvm-ffi==0.1.11`；继承的其他研究包存在依赖版本警告，未做全环境兼容验收。

原始日志保存在工作区 `tmp/hy4-migration-20261112/`，不随 fork 提交。下列命令从 `python/` 运行，`python` 指该独立环境。

```bash
python -m unittest sglang.srt.eplb.cpu_offload.tests.test_upstream_integration
python -m unittest discover -s sglang/srt/eplb/cpu_offload/tests -p 'test_*.py'
python -m unittest sglang.test.test_bench_one_batch_sweep sglang.test.test_bench_profile_window sglang.test.test_bench_real_text
```

## 检查结果

| 检查 | 实际结果 | 能说明什么 |
| --- | --- | --- |
| 来源分支 CPU offload 单测 | 153 项，6 项失败、7 项错误 | 本地对照基线，日志 `source-tests-own-env.log` |
| 迁移后同套单测及新增接入用例 | 172 项，6 项失败、7 项错误；未通过的用例名单与基线完全一致 | 新增 19 项全部通过，日志 `target-tests-final.log` |
| 基准辅助逻辑 | 归档前后均为 50 项通过 | 旧扫描、真实文本与定点窗口纯函数保持原测试结果，最终日志 `bench-tests-final.log` |

新增用例执行从真实新版 `FusedMoE.forward_impl` 提取的方法，检查 GPU 分流与 CPU 结果合并顺序；其余覆盖图捕获暖机边界、实际 token 数、请求身份、参数开关、量化格式拒绝、完整权重要求、已解析的图批量和 pinned 启动核验接线。测试使用 CPU 张量与替代依赖，没有执行 GPU 内核。

原有未通过项包括对象身份断言、daemon 替代依赖的断言／异常，以及本机缺 CUDA 和运行时依赖时的调用或导入错误。没有把这些问题全部归为环境问题，也没有删除用例、放宽断言或将结果记为全部通过。

已运行上游 `pre-commit run --all-files`，格式、导入、私钥检查、CI 登记检查和 Rust 检查均通过，日志 `pre-commit-verified.log`。按用户要求直接提交 `main`，只跳过禁止向该分支提交的 `no-commit-to-branch` 规则。旧入口原件从自动改写中排除，其余迁移代码参加检查。

旧静态类、旧基准及三个旧实验脚本与来源逐字节一致；其中保留了原有行尾空格。因此 `git diff --check` 对原件仍会提示这些历史空格，排除 `legacy/` 后检查通过。新增文档共检查 17 个相对链接，没有断链。Hy4 原生模型文件和配置文件相对上游基线无差异。

使用本 fork 的实际 `HYV4Config` 读取官方 BF16 仓库的 `config.json`，确认架构为 `HYV4ForCausalLM`、模型类型为 `hy_v4`，层数 78、隐藏维度 6144、专家数 256。这只验证配置能解析，没有加载完整权重。

## 未验证事项

尚未在 aiserver 构建这套新版 GPU 内核和配套 CPU 后端，也未执行完整 Hy4 推理、GPU 与 CPU 数值比对、图捕获重放或性能测量。此前旧分支上的实验结论不能直接当作新版基线的验收结果。
