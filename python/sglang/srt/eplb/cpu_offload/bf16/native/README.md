# 原始 BF16 CPU 内核补丁

> 摘要｜此补丁为真实权重数值验证增加原始 BF16 张量入口和 Hy4 SwiGLU 限幅。正式服务的 BF16 offload 仍未接通。

基线是 `intel-sandbox/moe-cpu-engine-xtang` 的 `3491bc78b670309464960fe2b3fd739114acb294`。在独立本地工作树应用 [01-raw-bf16-swiglu.patch](01-raw-bf16-swiglu.patch)，本地 commit 后同步到远端编译。补丁同时包含原生回归测试。

```bash
git apply --check /绝对路径/01-raw-bf16-swiglu.patch
git apply /绝对路径/01-raw-bf16-swiglu.patch
# 完成本地 commit 和同步后，在对应 GPU 环境执行：
cd ext
python setup.py build_ext --inplace
PYTHONPATH="$PWD" python ../tests/test_bf16_experts.py
```

新增 `fused_experts_bf16_cpu` 接收原始 CPU BF16 张量，复用 BF16 小矩阵和 AMX 批量计算路径，不将权重转换为 FP8 或 INT8。gate 分支截上界，up 分支截到正负限值。无限限值保留旧入口计算行为；原有 `fused_experts_cpu` 参数不变。

从本 SGLang 副本的 `python/` 目录运行真实权重对照。将配套扩展的 `ext/` 目录加入 `PYTHONPATH`，并确认导入的是新编译模块：

```bash
python -m sglang.srt.eplb.cpu_offload.bf16.validate \
  --model-path /data/Hy4-preview --layers 1 2 \
  --output /运行结果目录/hy4-bf16.json
```

对照使用两个 MoE 层中的真实专家权重、合成隐状态和固定路由；参考端是逐专家 PyTorch GPU 计算。每个使用的完整分片都会对照 `.download/verified/` 下载回执重新计算 SHA256。报告必须保留失败结果。

默认判据预先固定为：CPU 和混合输出相对二范数误差均不超过 3%，余弦相似度不低于 0.999，输出有限，且每层至少一组输入实际触发限幅。这是本阶段数值筛查标准，不等于完整模型质量标准；不包含注意力、共享专家、真实模型生成或 CPU 守护进程通信的验收。
