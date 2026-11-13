"""验证 Hy4 原始打包权重的索引、专家顺序及下载完整性。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.srt.eplb.cpu_offload.bf16.weights import load_layer


class Bf16WeightsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = {
            "architectures": ["HYV4ForCausalLM"],
            "hidden_size": 4,
            "moe_intermediate_size": 2,
            "n_routed_experts": 3,
            "swiglu_limit": 10.0,
            "routed_scaling_factor": 2.827,
        }
        (self.root / "config.json").write_text(json.dumps(config))
        self.gate = torch.arange(48).reshape(3, 4, 4).to(torch.bfloat16)
        self.down = torch.arange(24).reshape(3, 4, 2).to(torch.bfloat16)
        self.path = self.root / "model-00001.safetensors"
        self.tensors = {
            "model.layers.1.mlp.experts.gate_up_proj": self.gate,
            "model.layers.1.mlp.experts.down_proj": self.down,
        }
        self.write_shard()
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: self.path.name for key in self.tensors}})
        )

    def write_shard(self):
        save_file(self.tensors, self.path)
        verified = self.root / ".download" / "verified"
        verified.mkdir(parents=True, exist_ok=True)
        (verified / (self.path.name + ".json")).write_text(
            json.dumps(
                {
                    "字节数": self.path.stat().st_size,
                    "SHA256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                }
            )
        )

    def test_raw_bf16_and_requested_expert_order_are_preserved(self):
        verified = {}
        weights = load_layer(self.root, 1, [2, 0], verify=True, verified=verified)
        self.assertEqual(weights.expert_ids, (2, 0))
        self.assertEqual(weights.gate_up.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(weights.gate_up, self.gate[[2, 0]]))
        self.assertTrue(torch.equal(weights.down, self.down[[2, 0]]))
        self.assertEqual(weights.swiglu_limit, 10.0)
        self.assertEqual(weights.routed_scaling_factor, 2.827)
        self.assertEqual(len(verified), 1)

    def test_corrupted_shard_is_rejected_before_loading(self):
        data = bytearray(self.path.read_bytes())
        data[-1] ^= 1
        self.path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "哈希"):
            load_layer(self.root, 1, [0], verify=True)

    def test_missing_download_receipt_is_rejected(self):
        (self.root / ".download" / "verified" / (self.path.name + ".json")).unlink()
        with self.assertRaises(FileNotFoundError):
            load_layer(self.root, 1, [0], verify=True)

    def test_fp32_weights_are_not_silently_converted(self):
        self.tensors = {key: value.float() for key, value in self.tensors.items()}
        self.write_shard()
        with self.assertRaisesRegex(ValueError, "精度"):
            load_layer(self.root, 1, [0])

    def test_invalid_expert_lists_are_rejected(self):
        for ids in ([], [-1], [3], [1, 1]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "专家列表"):
                load_layer(self.root, 1, ids)


if __name__ == "__main__":
    unittest.main()
