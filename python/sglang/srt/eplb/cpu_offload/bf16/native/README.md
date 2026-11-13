# Raw BF16 CPU kernel and daemon patch

> Summary: raw BF16 tensors, Hy4 SwiGLU clipping, and small-layer validation over the production shared-memory transport. Full-model BF16 offload is not yet integrated.

Apply [01-raw-bf16-swiglu.patch](01-raw-bf16-swiglu.patch) to `intel-sandbox/moe-cpu-engine-xtang` at `3491bc78b670309464960fe2b3fd739114acb294` in an isolated local worktree. Commit locally before synchronizing to the build machine. The patch includes native regression tests.

```bash
git apply --check /absolute/path/01-raw-bf16-swiglu.patch
git apply /absolute/path/01-raw-bf16-swiglu.patch
# After committing and synchronizing, build in the target GPU environment:
cd ext
python setup.py build_ext --inplace
PYTHONPATH="$PWD" python ../tests/test_bf16_experts.py
```

`fused_experts_bf16_cpu` accepts raw CPU BF16 tensors and reuses the BF16 small-matrix and AMX batch paths without converting weights to FP8 or INT8. The gate branch is clipped only above the limit; the up branch is clipped symmetrically. An infinite limit preserves existing computation. The `fused_experts_cpu` signature is unchanged.

Run real-weight validation from this SGLang checkout's `python/` directory. Add the matching extension's `ext/` directory to `PYTHONPATH` and verify that the newly built module is imported:

```bash
python -m sglang.srt.eplb.cpu_offload.bf16.validate \
  --model-path /data/Hy4-preview --layers 1 2 \
  --output /results/hy4-bf16.json
```

The comparison uses real expert weights from two MoE layers, synthetic hidden states, and fixed routing. The reference is an independent per-expert PyTorch GPU implementation. Each complete shard is rehashed against its `.download/verified/` receipt. Preserve failed results in the report.

The predefined criteria require relative L2 error no greater than 3%, cosine similarity at least 0.999, finite CPU and hybrid outputs, and at least one input exercising clipping in each layer. These are numerical screening criteria, not full-model quality criteria. The default in-process mode does not exercise daemon communication; use the mode below for that. Neither mode validates attention, shared experts, or full-model generation.

## Production transport validation

The patch also includes the daemon integration introduced through CPU commit `9311c92`, with English comments and docstrings. Rebuild both client and daemon with this extension. Python modules are installed by `setup.py`; for source execution, add the matching `ext/` directory to `PYTHONPATH`. Weight uploads use opcode 4 and weight-file version 2. Older daemons reject the request rather than interpreting BF16 as FP8. Version, shape, size, or clipping-limit mismatches are rejected. The raw upload copy is released after daemon-side packing completes.

Start a dedicated test daemon in a separate terminal with a prefix that is not used by other experiments:

```bash
ENABLE_NUMA=0 MOE_KERNEL_BACKEND=torch MOE_DTYPE=bf16 \
CPU_OFFLOAD_PREFETCH_NEXT_LAYER=0 OMP_NUM_THREADS=16 \
python -m cpu_expertd_main --socket 0 --num-workers 1 \
  --cores-per-numa 16 --reserved-cores 0 \
  --shm-prefix /dev/shm/hy4-bf16-test
```

Then run the validation command above from SGLang's `python/` directory with `--daemon-prefix /dev/shm/hy4-bf16-test`. This uses the production CUDA submit, wait, and return path. Each call checks the error code, completion sequence, and actual computation count; the report saves counters before and after validation. Stop the daemon you started when finished.

Run `python tests/test_bf16_weight_protocol.py` in the CPU repository for protocol tests. Run `HY4_TEST_SHM_PREFIX=/dev/shm/hy4-bf16-test python tests/test_bf16_daemon.py` for connected tests. The latter use layer IDs 101–103 and must not target a daemon serving a model.

The tested configuration has one client, one worker, complete local CPU expert tensors (TP=1), synthetic hidden states, fixed routing, and no CUDA Graph capture. Reloading an occupied layer ID is rejected; use a new process or a different layer ID. This mode validates small-layer communication. It does not remove the runner's BF16 guard or implement full-model hot/cold loading, dynamic expert migration, or performance acceptance.
