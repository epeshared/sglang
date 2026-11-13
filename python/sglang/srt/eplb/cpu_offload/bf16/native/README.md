# 原始 BF16 CPU 内核与守护进程补丁

> 摘要｜此补丁增加原始 BF16 张量入口、Hy4 SwiGLU 限幅和正式共享内存通道的少量层验证。完整模型服务的 BF16 offload 仍未接通。

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

默认判据预先固定为：CPU 和混合输出相对二范数误差均不超过 3%，余弦相似度不低于 0.999，输出有限，且每层至少一组输入实际触发限幅。这是本阶段数值筛查标准，不等于完整模型质量标准；默认进程内模式不包含 CPU 守护进程通信，通信验证使用下节的独立模式；两种模式都不包含注意力、共享专家或真实模型生成的验收。

## 正式通信通道验证

补丁还包含 CPU 仓库 `9311c92` 的守护进程适配。客户端和服务端均须重编译并使用这套扩展；Python 模块随 `setup.py` 安装，源码运行时把对应 `ext/` 加入 `PYTHONPATH`。权重上传使用独立操作码 4、权重文件版本 2；旧服务会拒绝，不会把 BF16 当成 FP8。版本、形状、大小和限幅不一致时拒绝，原始上传副本在服务端完成打包后释放。

在独立终端启动一个测试守护进程，使用未被其它实验占用的前缀：

```bash
ENABLE_NUMA=0 MOE_KERNEL_BACKEND=torch MOE_DTYPE=bf16 \
CPU_OFFLOAD_PREFETCH_NEXT_LAYER=0 OMP_NUM_THREADS=16 \
python -m cpu_expertd_main --socket 0 --num-workers 1 \
  --cores-per-numa 16 --reserved-cores 0 \
  --shm-prefix /dev/shm/hy4-bf16-test
```

然后从 SGLang 的 `python/` 目录执行上面的数值对照命令，额外传入 `--daemon-prefix /dev/shm/hy4-bf16-test`。该模式通过正式 CUDA 提交、等待和返回通道调用 CPU 守护进程；每次调用检查错误码、请求完成序号及真实计算计数，报告保存前后计数。完成后停止自己启动的守护进程。

协议测试在 CPU 仓库运行 `python tests/test_bf16_weight_protocol.py`；连接测试使用 `HY4_TEST_SHM_PREFIX=/dev/shm/hy4-bf16-test python tests/test_bf16_daemon.py`。连接测试使用层号 101—103，不能对正在运行模型的服务执行。

当前只验单客户端、单 worker、CPU 完整局部专家张量（TP=1），使用合成隐状态和固定路由，不捕获 CUDA Graph。重复加载已占用的层号会拒绝，需要新进程或新层号。此模式补齐少量层的通信验证，不放开正式 runner 的 BF16 保护，也不实现完整冷热权重加载、动态专家迁移或性能验收。
