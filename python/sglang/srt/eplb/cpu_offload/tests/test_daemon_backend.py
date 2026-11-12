"""Unit tests for ``DaemonBackend`` using mocks.

We can't bind real CUDA streams or talk to a real ``cpu_expertd`` here,
so these tests stub out ``moe_cpu_engine`` and ``cpu_offload_remote_moe``
to verify control-flow + slot allocation + delegation through
``CpuOffloadEngine``.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics


def _install_fake_engine_modules(attach_pid: int = 12345, slot_handle: int = 0xABCD):
    """Inject fake ``moe_cpu_engine`` + ``cpu_offload_remote_moe`` into sys.modules.

    Returns the (engine_mod, remote_moe_cls, RemoteMOE_calls) for assertions.
    """
    engine_mod = types.ModuleType("moe_cpu_engine")
    engine_mod.cpu_client_attach = MagicMock(return_value=attach_pid)
    engine_mod.cpu_client_open_slot = MagicMock(return_value=slot_handle)
    sys.modules["moe_cpu_engine"] = engine_mod

    rm_mod = types.ModuleType("cpu_offload_remote_moe")

    instances: list = []

    class FakeRemoteMOE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.set_weights_calls: list = []
            self.kernel_enabled = False
            self.forward_calls: list = []
            instances.append(self)

        def set_weights(self, *args):
            self.set_weights_calls.append(args)

        def enable_kernel_path(self):
            self.kernel_enabled = True

        def forward_experts_kernel(self, stream, in_ptr, ids_ptr, w_ptr, out_ptr, n):
            self.forward_calls.append((stream, in_ptr, ids_ptr, w_ptr, out_ptr, n))

    rm_mod.RemoteMOE = FakeRemoteMOE
    sys.modules["cpu_offload_remote_moe"] = rm_mod

    return engine_mod, FakeRemoteMOE, instances


def _uninstall_fake_engine_modules():
    sys.modules.pop("moe_cpu_engine", None)
    sys.modules.pop("cpu_offload_remote_moe", None)
    # Also drop the daemon backend module so it re-imports the fakes next time.
    sys.modules.pop("sglang.srt.eplb.cpu_offload.engine_daemon", None)


def _make_cfg(**overrides) -> CpuOffloadConfig:
    base = dict(
        enable_cpu_offload_fast_loop=True,
        cpu_active_ranks=[0],
        cpu_offload_backend="daemon",
        cpu_offload_daemon_launcher="",  # attach-only; no subprocess
        cpu_offload_shm_prefix="/dev/shm/test_daemon_backend",
    )
    base.update(overrides)
    return CpuOffloadConfig(**base)


class _FakeTensor:
    """Minimal stand-in for a torch tensor; only data_ptr/device/is_contiguous needed."""

    def __init__(self, ptr: int = 0xDEAD, on_cpu: bool = True):
        self._ptr = ptr
        self._on_cpu = on_cpu

    def data_ptr(self):
        return self._ptr

    @property
    def device(self):
        return types.SimpleNamespace(type="cpu" if self._on_cpu else "cuda")

    def is_contiguous(self):
        return True


class TestDaemonBackendInit(unittest.TestCase):
    def setUp(self):
        self._engine_mod, self._RM, self._instances = _install_fake_engine_modules()

    def tearDown(self):
        _uninstall_fake_engine_modules()

    def test_init_attaches_and_opens_slot(self):
        from sglang.srt.eplb.cpu_offload.engine_daemon import DaemonBackend

        cfg = _make_cfg()
        be = DaemonBackend(cfg, host_rank=0, world_size=1, num_numa_nodes=1)
        be.init_runtime()
        self.assertTrue(be.initialized)
        self.assertEqual(be.socket_id, 0)
        self.assertEqual(be.slot_id, 0)
        self.assertEqual(be.slot_handle, 0xABCD)
        self._engine_mod.cpu_client_attach.assert_called_once()
        self._engine_mod.cpu_client_open_slot.assert_called_once()

    def test_init_raises_when_rank_not_active_on_numa(self):
        from sglang.srt.eplb.cpu_offload.engine_daemon import DaemonBackend

        cfg = _make_cfg(cpu_active_ranks=[3])  # rank 0 is not active
        be = DaemonBackend(cfg, host_rank=0, world_size=4, num_numa_nodes=1)
        with self.assertRaises(RuntimeError):
            be.init_runtime()


class TestDaemonBackendRegisterAndLaunch(unittest.TestCase):
    def setUp(self):
        self._engine_mod, self._RM, self._instances = _install_fake_engine_modules()

    def tearDown(self):
        _uninstall_fake_engine_modules()

    def _make_spec_and_weights(self, layer_id: int = 7):
        from sglang.srt.eplb.cpu_offload.engine import LayerSpec, LayerWeights

        spec = LayerSpec(
            layer_id=layer_id,
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=64,
            intermediate_size=32,
        )
        w = LayerWeights(
            gate_up_weights=_FakeTensor(ptr=0x1000),
            down_weights=_FakeTensor(ptr=0x2000),
            gate_up_scales=_FakeTensor(ptr=0x3000),
            down_scales=_FakeTensor(ptr=0x4000),
            e_score_correction_bias=None,
        )
        return spec, w

    def test_register_and_launch_round_trip(self):
        from sglang.srt.eplb.cpu_offload.engine_daemon import DaemonBackend

        cfg = _make_cfg()
        be = DaemonBackend(
            cfg,
            host_rank=0,
            world_size=1,
            num_numa_nodes=1,
            metrics=CpuOffloadMetrics(),
        )
        be.init_runtime()

        spec, weights = self._make_spec_and_weights(layer_id=7)
        be.register_layer(spec, weights)
        self.assertTrue(be.has_layer(7))
        self.assertEqual(be.registered_layers(), [7])

        rm = self._instances[-1]
        # set_weights was called with the right pointers + (tp_size=1, tp_rank=0)
        (args,) = rm.set_weights_calls
        self.assertEqual(args[0], 0x1000)  # gate_up
        self.assertEqual(args[1], 0x2000)  # down
        self.assertEqual(args[5], 1)  # tp_size
        self.assertEqual(args[6], 0)  # tp_rank
        self.assertTrue(rm.kernel_enabled)

        # launch_layer forwards to forward_experts_kernel.
        be.launch_layer(
            7,
            dev_input=_FakeTensor(ptr=0xAA, on_cpu=False),
            dev_topk_ids=_FakeTensor(ptr=0xBB, on_cpu=False),
            dev_topk_weights=_FakeTensor(ptr=0xCC, on_cpu=False),
            dev_output=_FakeTensor(ptr=0xDD, on_cpu=False),
            n_tokens=4,
            cuda_stream=0xCAFE,
        )
        self.assertEqual(len(rm.forward_calls), 1)
        stream, ip, ids, w, op, n = rm.forward_calls[0]
        self.assertEqual(stream, 0xCAFE)
        self.assertEqual((ip, ids, w, op, n), (0xAA, 0xBB, 0xCC, 0xDD, 4))

    def test_register_requires_cpu_contiguous_weights(self):
        from sglang.srt.eplb.cpu_offload.engine_daemon import DaemonBackend

        cfg = _make_cfg()
        be = DaemonBackend(cfg, host_rank=0, world_size=1, num_numa_nodes=1)
        be.init_runtime()
        from sglang.srt.eplb.cpu_offload.engine import LayerSpec, LayerWeights

        spec = LayerSpec(
            layer_id=0,
            num_experts=4,
            num_experts_per_tok=1,
            hidden_size=16,
            intermediate_size=8,
        )
        bad = LayerWeights(
            gate_up_weights=_FakeTensor(ptr=1, on_cpu=False),  # GPU tensor -> reject
            down_weights=_FakeTensor(ptr=2),
            gate_up_scales=_FakeTensor(ptr=3),
            down_scales=_FakeTensor(ptr=4),
        )
        with self.assertRaises(ValueError):
            be.register_layer(spec, bad)

    def test_launch_zero_tokens_is_noop(self):
        from sglang.srt.eplb.cpu_offload.engine_daemon import DaemonBackend

        cfg = _make_cfg()
        be = DaemonBackend(cfg, host_rank=0, world_size=1, num_numa_nodes=1)
        be.init_runtime()
        spec, weights = self._make_spec_and_weights(layer_id=0)
        be.register_layer(spec, weights)
        rm = self._instances[-1]
        be.launch_layer(
            0,
            dev_input=_FakeTensor(),
            dev_topk_ids=_FakeTensor(),
            dev_topk_weights=_FakeTensor(),
            dev_output=_FakeTensor(),
            n_tokens=0,
            cuda_stream=1,
        )
        self.assertEqual(rm.forward_calls, [])

    def test_commit_layer_is_noop_for_daemon(self):
        from sglang.srt.eplb.cpu_offload.engine_daemon import DaemonBackend

        cfg = _make_cfg()
        be = DaemonBackend(cfg, host_rank=0, world_size=1, num_numa_nodes=1)
        be.init_runtime()
        # Should not raise even though nothing is registered.
        be.commit_layer(42, dev_output=_FakeTensor(), n_tokens=10, cuda_stream=1)


class TestEngineFacadeUsesDaemon(unittest.TestCase):
    def setUp(self):
        self._engine_mod, self._RM, self._instances = _install_fake_engine_modules()

    def tearDown(self):
        _uninstall_fake_engine_modules()

    def test_engine_with_daemon_config_routes_to_backend(self):
        from sglang.srt.eplb.cpu_offload.engine import (
            CpuOffloadEngine,
            LayerSpec,
            LayerWeights,
        )

        cfg = _make_cfg()
        eng = CpuOffloadEngine(
            cfg, host_rank=0, num_hosts=1, ep_size=1, num_numa_nodes=1
        )
        eng.init_runtime()
        self.assertIsNotNone(eng._daemon)

        spec = LayerSpec(
            layer_id=3,
            num_experts=4,
            num_experts_per_tok=1,
            hidden_size=16,
            intermediate_size=8,
        )
        w = LayerWeights(
            gate_up_weights=_FakeTensor(ptr=0x100),
            down_weights=_FakeTensor(ptr=0x200),
            gate_up_scales=_FakeTensor(ptr=0x300),
            down_scales=_FakeTensor(ptr=0x400),
        )
        eng.register_layer(spec, w)
        self.assertTrue(eng.has_layer(3))
        self.assertEqual(eng.registered_layers(), [3])

        eng.launch_layer(
            3,
            dev_input=_FakeTensor(ptr=0x11),
            dev_topk_ids=_FakeTensor(ptr=0x22),
            dev_topk_weights=_FakeTensor(ptr=0x33),
            dev_output=_FakeTensor(ptr=0x44),
            n_tokens=2,
            cuda_stream=0xBEEF,
        )
        # commit_layer is a no-op on daemon; should not raise.
        eng.commit_layer(
            3, dev_output=_FakeTensor(ptr=0x55), n_tokens=2, cuda_stream=0xBEEF
        )
        rm = self._instances[-1]
        self.assertEqual(len(rm.forward_calls), 1)

    def test_engine_with_daemon_config_rejects_launch_without_dev_output(self):
        from sglang.srt.eplb.cpu_offload.engine import CpuOffloadEngine

        cfg = _make_cfg()
        eng = CpuOffloadEngine(
            cfg, host_rank=0, num_hosts=1, ep_size=1, num_numa_nodes=1
        )
        eng.init_runtime()
        with self.assertRaises(ValueError):
            eng.launch_layer(
                0,
                dev_input=_FakeTensor(),
                dev_topk_ids=_FakeTensor(),
                dev_topk_weights=_FakeTensor(),
                n_tokens=1,
                cuda_stream=1,
            )


if __name__ == "__main__":
    unittest.main()
