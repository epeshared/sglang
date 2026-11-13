"""Read packed Hy4 experts through the safetensors index while preserving raw BF16 weights."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ExpertWeights:
    layer_id: int
    expert_ids: tuple[int, ...]
    gate_up: torch.Tensor
    down: torch.Tensor
    swiglu_limit: float
    routed_scaling_factor: float


def verify_shard(path: Path) -> dict:
    """Rehash the complete shard against its download receipt and reject incomplete downloads."""
    receipt_path = path.parent / ".download" / "verified" / (path.name + ".json")
    receipt = json.loads(receipt_path.read_text())
    if path.stat().st_size != receipt["字节数"]:
        raise ValueError(f"分片大小与校验回执不一致：{path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != receipt["SHA256"]:
        raise ValueError(f"分片哈希与校验回执不一致：{path.name}")
    return {"file": path.name, "sha256": actual, "bytes": path.stat().st_size}


def load_layer(model_path, layer_id, expert_ids, *, verify=False, verified=None):
    """Copy only selected experts; their returned order defines local CPU weight indices."""
    from safetensors import safe_open

    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    if "HYV4ForCausalLM" not in config.get("architectures", []):
        raise ValueError("此入口只验证 Hy4 的打包 BF16 专家布局")
    if config.get("quantization_config"):
        raise ValueError("此入口要求原始 BF16 权重")
    ids = tuple(int(value) for value in expert_ids)
    count = int(config["n_routed_experts"])
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= count for i in ids):
        raise ValueError("专家列表必须非空、不重复，且落在模型专家范围内")
    hidden, intermediate = (
        int(config["hidden_size"]),
        int(config["moe_intermediate_size"]),
    )
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = []
    verified = {} if verified is None else verified
    for name, shape in (
        ("gate_up_proj", (count, 2 * intermediate, hidden)),
        ("down_proj", (count, hidden, intermediate)),
    ):
        key = f"model.layers.{int(layer_id)}.mlp.experts.{name}"
        path = root / weight_map[key]
        if path.parent.resolve() != root.resolve():
            raise ValueError("分片路径必须位于模型目录中")
        if verify and path.name not in verified:
            verified[path.name] = verify_shard(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            view = handle.get_slice(key)
            if tuple(view.get_shape()) != shape or view.get_dtype() != "BF16":
                raise ValueError(f"专家形状或精度不符：{key}")
            tensors.append(torch.stack([view[i] for i in ids]).contiguous())
    return ExpertWeights(
        int(layer_id),
        ids,
        *tensors,
        float(config["swiglu_limit"]),
        float(config["routed_scaling_factor"]),
    )
