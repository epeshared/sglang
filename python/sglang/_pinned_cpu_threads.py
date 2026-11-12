"""pinned 的进程启动约束；不向 shm 或逐请求路径添加工作。"""

import os
import sys
import threading

_bootstrap = None


def enabled():
    return os.environ.get("CPU_OFFLOAD_TRANSPORT") in ("pinned", "anonymous")


def bootstrap():
    """必须在 SGLang 的其他导入之前执行，不能靠事后缩池补救。"""
    global _bootstrap
    if not enabled():
        return
    if _bootstrap is not None:
        require_bootstrap()
        return
    if "torch" in sys.modules:
        raise RuntimeError("pinned 必须在首次导入 torch 前选择；请使用全新进程")
    previous = os.environ.get("OMP_NUM_THREADS")
    # libgomp 的等待策略是进程级的；先限制默认团队，再由专家线程显式设为 16。
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    _bootstrap = {
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
        "stage": "sglang_package_before_public_imports",
        "previous_omp_num_threads": previous,
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "main_threads": torch.get_num_threads(),
    }


def require_bootstrap():
    """拒绝晚开开关、fork 继承回执或后续扩大主线程团队。"""
    import torch

    if (
        _bootstrap is None
        or _bootstrap["pid"] != os.getpid()
        or _bootstrap["tid"] != threading.get_native_id()
        or torch.get_num_threads() != 1
        or os.environ.get("OMP_NUM_THREADS") != "1"
    ):
        raise RuntimeError("pinned 缺少本 rank 的早期单线程初始化，或线程设置已改变")
    return dict(_bootstrap)
