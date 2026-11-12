"""定点诊断窗口：只改变 bench 的记录范围，不改变模型计算。"""

import json
import sys


def focus_spec(values):
    """五个整数为 batch、输入长、输出长、中点前步数、中点后步数。"""
    if not values:
        return None
    if len(values) != 5 or any(type(v) is not int for v in values):
        raise ValueError("诊断窗口须给五个整数")
    bs, il, ol, before, after = values
    if min(bs, il, ol) <= 0 or min(before, after) < 0:
        raise ValueError("batch 和长度须为正数，窗口跨度须非负")
    start, stop = ol // 2 - before, ol // 2 + after
    if start < 0 or stop > ol - 2:
        raise ValueError("诊断窗口超出实际解码循环")
    return {
        "args": list(values),
        "segment": f"bs{bs}-in{il}-out{ol}",
        "capture_steps": [start + 1, stop + 1],
        "capture_count": stop - start + 1,
        "sample_every": 1,
    }


def validate_focus(
    values, batches, pairs, profile, activities, stage, correctness=False
):
    spec = focus_spec(values)
    if spec is None:
        return
    if (
        correctness
        or not profile
        or tuple(activities) != ("CUDA_PROFILER",)
        or stage != "decode"
    ):
        raise ValueError("定点窗口只用于扫描负载的 CUDA_PROFILER 解码诊断")
    bs, il, ol = values[:3]
    if list(batches).count(bs) != 1 or list(pairs).count(f"{il}:{ol}") != 1:
        raise ValueError("目标组必须在扫描网格中出现且只出现一次")


def focus_matches(values, bs, il, ol):
    return not values or tuple(values[:3]) == (bs, il, ol)


def profile_steps(output_len, values=()):
    spec = focus_spec(values)
    if spec is None:
        return output_len // 2, output_len // 2
    if output_len != values[2]:
        raise ValueError("窗口与解码长度不一致")
    return tuple(step - 1 for step in spec["capture_steps"])


if __name__ == "__main__":
    # 采集脚本复用同一份校验和元数据，避免 shell 再实现一套窗口定义。
    values = tuple(int(v) for v in sys.argv[1].split(":"))
    validate_focus(
        values,
        [int(v) for v in sys.argv[2].split()],
        sys.argv[3].split(),
        True,
        ("CUDA_PROFILER",),
        "decode",
    )
    print(json.dumps(focus_spec(values), ensure_ascii=False))
