"""用真实 Hy4 权重比较 CPU、GPU 和混合专家输出；不代表完整模型验收。"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import moe_cpu_engine
import torch

from .layer import Bf16Experts
from .reference import error_metrics, forward
from .weights import load_layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--experts", type=int, nargs="+", default=[0, 1, 7, 63, 64, 127, 128, 255]
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("此对照需要真实 CUDA GPU")
    if min(args.tokens) < 1 or args.threads < 1:
        raise ValueError("token 数和 CPU 线程数必须为正数")
    config = json.loads((Path(args.model_path) / "config.json").read_text())
    topk = int(config["num_experts_per_tok"])
    if len(args.experts) < topk:
        raise ValueError("抽取专家数不能小于模型的 top-k")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(20261113)
    report = {
        "scope": "真实权重、合成隐状态、固定路由的 MoE 子层对照；不是整层注意力或完整模型生成验收",
        "host": platform.node(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "threads": args.threads,
        "seed": 20261113,
        "native_module": moe_cpu_engine.__file__,
        "native_module_sha256": hashlib.sha256(
            Path(moe_cpu_engine.__file__).read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            (Path(args.model_path) / "config.json").read_bytes()
        ).hexdigest(),
        "swiglu_limit": config["swiglu_limit"],
        "routed_scaling_factor": config["routed_scaling_factor"],
        "model_path": args.model_path,
        "layers": args.layers,
        "experts": args.experts,
        "criteria": {
            "relative_l2_max": 0.03,
            "cosine_min": 0.999,
            "finite_required": True,
            "clamp_must_be_exercised": True,
        },
        "verified_shards": {},
        "cases": [],
        "passed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        for layer_id in args.layers:
            print(f"校验并读取层 {layer_id} 的原始 BF16 专家", flush=True)
            weights = load_layer(
                args.model_path,
                layer_id,
                args.experts,
                verify=True,
                verified=report["verified_shards"],
            )
            cpu = Bf16Experts(weights)
            gpu_gate_up, gpu_down = weights.gate_up.cuda(), weights.down.cuda()
            for tokens in args.tokens:
                for gain in (1.0, 8.0):
                    hidden = (torch.randn(tokens, weights.gate_up.shape[-1]) * gain).to(
                        torch.bfloat16
                    )
                    ids = torch.stack(
                        [
                            torch.randperm(len(args.experts))[:topk]
                            for _ in range(tokens)
                        ]
                    ).to(torch.int32)
                    routing = (
                        torch.softmax(torch.randn(tokens, topk), -1)
                        * weights.routed_scaling_factor
                    )
                    output = cpu.forward(hidden, ids, routing)
                    gh, gi, gw = hidden.cuda(), ids.cuda(), routing.cuda()
                    reference, clipped = forward(
                        gh, gpu_gate_up, gpu_down, gi, gw, weights.swiglu_limit
                    )
                    # 固定把局部专家 0 分给 CPU，其余留在 GPU，覆盖排除和加回的次序。
                    cpu_ids = ids.masked_fill(ids != 0, -1)
                    gpu_ids = gi.masked_fill(gi == 0, -1)
                    cpu_part = cpu.forward(hidden, cpu_ids, routing)
                    gpu_part, _ = forward(
                        gh, gpu_gate_up, gpu_down, gpu_ids, gw, weights.swiglu_limit
                    )
                    hybrid = (gpu_part.float() + cpu_part.cuda().float()).to(
                        torch.bfloat16
                    )
                    full_error = error_metrics(output, reference)
                    hybrid_error = error_metrics(hybrid, reference)
                    case = {
                        "layer": layer_id,
                        "tokens": tokens,
                        "input_gain": gain,
                        "clipped_elements": clipped,
                        "cpu": full_error,
                        "hybrid": hybrid_error,
                    }
                    case["passed"] = all(
                        e["finite"]
                        and e["relative_l2"] <= 0.03
                        and e["cosine"] >= 0.999
                        for e in (full_error, hybrid_error)
                    )
                    report["cases"].append(case)
                    print(json.dumps(case, ensure_ascii=False), flush=True)
            del cpu, weights, gpu_gate_up, gpu_down
            torch.cuda.empty_cache()
        report["passed"] = all(c["passed"] for c in report["cases"]) and all(
            any(
                c["clipped_elements"] > 0
                for c in report["cases"]
                if c["layer"] == layer_id
            )
            for layer_id in args.layers
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("BF16 数值对照未通过，见报告中的原始读数")


if __name__ == "__main__":
    main()
