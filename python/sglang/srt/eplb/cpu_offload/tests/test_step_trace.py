"""逐步 trace 的纯 Python 合同测试，不需要 GPU 或 daemon。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.eplb.cpu_offload import engine_daemon, step_trace


class _FakeEvent:
    def __init__(self, timestamp_ms: float):
        self.timestamp_ms = float(timestamp_ms)
        self.record_count = 0

    def record(self, *_args, **_kwargs):
        self.record_count += 1

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return other.timestamp_ms - self.timestamp_ms


class _EventFactory:
    def __init__(self):
        self.next_timestamp_ms = 0.0

    def __call__(self):
        event = _FakeEvent(self.next_timestamp_ms)
        self.next_timestamp_ms += 1.0
        return event


class _FakeScalar:
    def __init__(self, value: int):
        self.value = int(value)

    def item(self):
        return self.value

    def clone(self):
        return _FakeScalar(self.value)


class _FakeActiveTokens:
    def __init__(self, value: int):
        self.value = int(value)

    def fill_(self, value: int):
        self.value = int(value)
        return self


def _read_records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class StepTraceTests(unittest.TestCase):
    def test_reused_slot_disables_only_step_trace(self):
        class FakeEngineModule:
            def __init__(self):
                self.legacy_calls = []
                self.step_calls = []

            def cpu_client_slot_set_trace(self, handle, enabled):
                self.legacy_calls.append((handle, enabled))

            def cpu_client_slot_set_step_trace(self, handle, enabled):
                self.step_calls.append((handle, enabled))

        fake_engine = FakeEngineModule()
        backend = engine_daemon.DaemonBackend.__new__(engine_daemon.DaemonBackend)
        backend._engine_mod = fake_engine
        backend._slot_handle = 17
        backend._socket_id = 0
        backend._slot_id = 3
        old_moe_trace = engine_daemon._MOE_TRACE
        old_step_trace = engine_daemon._STEP_TRACE_ENABLED
        try:
            engine_daemon._MOE_TRACE = True
            engine_daemon._STEP_TRACE_ENABLED = True
            backend._maybe_arm_trace()
            engine_daemon._STEP_TRACE_ENABLED = False
            backend._maybe_arm_trace()
        finally:
            engine_daemon._MOE_TRACE = old_moe_trace
            engine_daemon._STEP_TRACE_ENABLED = old_step_trace

        self.assertEqual(fake_engine.step_calls, [(17, True), (17, False)])
        self.assertEqual(fake_engine.legacy_calls, [(17, True), (17, True)])

    def test_disabled_switch_does_not_touch_recorder(self):
        old_enabled = step_trace.ENABLED
        old_recorder = step_trace._recorder
        sentinel = object()
        try:
            step_trace.ENABLED = False
            step_trace._recorder = sentinel
            self.assertIsNone(step_trace.get_step_trace())
            self.assertIsNone(step_trace.begin_step(SimpleNamespace()))
        finally:
            step_trace.ENABLED = old_enabled
            step_trace._recorder = old_recorder

    def test_request_layer_and_cpu_task_keys_are_linked(self):
        snapshots = [
            {
                "valid": True,
                "supported": True,
                "enabled": True,
                "write_task_id": 0,
                "lost_records": 0,
                "inconsistent_records": 0,
                "records": [],
            },
            {
                "valid": True,
                "supported": True,
                "enabled": True,
                "write_task_id": 9,
                "lost_records": 0,
                "inconsistent_records": 0,
                "records": [
                    {
                        "task_id": 9,
                        "req_seq": 21,
                        "layer_id": 3,
                        "worker_id": 2,
                        "expert_id": 7,
                        "offloaded_branches": 4,
                        "active_tokens": 4,
                        "cpu_pairs": 4,
                        "error_code": 0,
                        "did_compute": False,
                        "status": "no_compute",
                        "cpu_compute_ns": None,
                        "daemon_service_ns": 20_000,
                        "dispatch_ns": None,
                        "flush_ns": 2_000,
                    }
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = step_trace.StepTraceRecorder(
                rank_id=1,
                hetero_mode="dynamic_dual_loop",
                cpu_trace_applicable=True,
                output_dir=temp_dir,
                run_id="keys",
                event_factory=_EventFactory(),
                snapshot_fn=lambda _cursor: snapshots.pop(0),
                write_run_record=False,
            )
            recorder.note_plan_update(3, 7)
            step = recorder.begin_step(
                SimpleNamespace(
                    request_ids=["req-a", "req-b"],
                    request_decode_steps=[5, 8],
                    request_id_source="scheduler",
                    batch_size=2,
                )
            )
            layer = step_trace.LayerEvents(
                layer_id=3,
                start=_FakeEvent(0),
                dispatch_end=_FakeEvent(1),
                local_moe_end=_FakeEvent(3),
                end=_FakeEvent(5),
                branches_received=_FakeScalar(12),
            )
            step.layers.append(layer)
            cpu_events = step_trace.CpuGpuEvents(
                layer_id=3,
                d2h_start=_FakeEvent(0),
                d2h_end=_FakeEvent(1),
                wait_start=_FakeEvent(1),
                wait_end=_FakeEvent(4),
                h2d_start=_FakeEvent(4),
                h2d_end=_FakeEvent(5),
                visible_wait_start=_FakeEvent(4.5),
                visible_wait_end=_FakeEvent(5.2),
                wait_status=_FakeScalar(0),
                business_offloaded_branches=_FakeScalar(4),
            )
            step.cpu_events.append(cpu_events)
            recorder.end_step(step)
            recorder.close()

            records = _read_records(Path(temp_dir) / "step_trace_rank_1.jsonl")
            by_kind = {record["kind"]: record for record in records}
            self.assertEqual(by_kind["step"]["request_ids"], ["req-a", "req-b"])
            self.assertEqual(by_kind["step"]["request_decode_steps"], [5, 8])
            self.assertEqual(by_kind["layer_rank"]["batch_step_id"], 1)
            self.assertEqual(by_kind["layer_rank"]["layer_id"], 3)
            self.assertEqual(by_kind["cpu_task"]["task_id"], 9)
            self.assertEqual(by_kind["cpu_task"]["plan_epoch"], 1)
            self.assertEqual(by_kind["cpu_task"]["offloaded_branches"], 4)
            self.assertEqual(by_kind["cpu_task"]["executed_offloaded_branches"], 4)
            self.assertEqual(by_kind["cpu_task"]["padding_offloaded_branches"], 0)
            self.assertEqual(by_kind["cpu_task"]["visible_wait_us"], 500.0)
            self.assertEqual(by_kind["cpu_task"]["status"], "no_compute")
            self.assertIsNone(by_kind["cpu_task"]["cpu_compute_us"])
            self.assertIsNone(by_kind["cpu_task"]["daemon_queue_us"])
            self.assertIsNone(by_kind["cpu_task"]["daemon_protocol_us"])

    def test_graph_replay_uses_captured_layer_events(self):
        empty_snapshot = {
            "valid": True,
            "supported": True,
            "enabled": False,
            "write_task_id": 0,
            "lost_records": 0,
            "inconsistent_records": 0,
            "records": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = step_trace.StepTraceRecorder(
                rank_id=0,
                hetero_mode="gpu_only",
                cpu_trace_applicable=False,
                output_dir=temp_dir,
                run_id="graph",
                event_factory=_EventFactory(),
                snapshot_fn=lambda _cursor: dict(empty_snapshot),
                write_run_record=False,
            )
            graph = step_trace.GraphTrace(
                key=(77, 4),
                graph_batch_size=4,
                graph_num_tokens=4,
                active_tokens=_FakeActiveTokens(4),
                row_indices=None,
                layers=[
                    step_trace.LayerEvents(
                        layer_id=11,
                        start=_FakeEvent(10),
                        dispatch_end=_FakeEvent(11),
                        local_moe_end=_FakeEvent(13),
                        end=_FakeEvent(15),
                        branches_received=_FakeScalar(18),
                    )
                ],
            )
            recorder._graphs[graph.key] = graph
            step = recorder.begin_step(
                SimpleNamespace(
                    request_ids=["r0", "r1", "r2"],
                    request_decode_steps=[1, 1, 1],
                    request_id_source="scheduler",
                    batch_size=3,
                )
            )
            recorder.mark_graph_replay(77, 4, 3)
            recorder.end_step(step)
            recorder.close()

            records = _read_records(Path(temp_dir) / "step_trace_rank_0.jsonl")
            step_record = next(x for x in records if x["kind"] == "step")
            layer_record = next(x for x in records if x["kind"] == "layer_rank")
            self.assertEqual(step_record["execution_mode"], "cuda_graph_replay")
            self.assertEqual(step_record["active_batch_size"], 3)
            self.assertEqual(step_record["graph_batch_size"], 4)
            self.assertEqual(graph.active_tokens.value, 3)
            self.assertEqual(layer_record["branches_received"], 18)

    def test_graph_padding_is_removed_from_count_without_changing_execution_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = step_trace.StepTraceRecorder(
                rank_id=0,
                hetero_mode="dynamic_dual_loop",
                cpu_trace_applicable=True,
                output_dir=temp_dir,
                run_id="padding",
                event_factory=_EventFactory(),
                snapshot_fn=lambda _cursor: {},
                write_run_record=False,
            )
            graph = step_trace.GraphTrace(
                key=(1, 4),
                graph_batch_size=4,
                graph_num_tokens=4,
                active_tokens=torch.tensor([3], dtype=torch.int32),
                row_indices=torch.arange(4, dtype=torch.int32),
            )
            recorder.enter_graph_capture(graph)
            events = recorder.begin_cpu_gpu_task(7)
            execution_mask = torch.tensor([True, False, True, True])
            original = execution_mask.clone()
            recorder.record_cpu_business_branches(
                events, execution_mask, max_branches=4
            )

            self.assertTrue(torch.equal(execution_mask, original))
            self.assertEqual(events.business_offloaded_branches.item(), 2)
            recorder.close()

    def test_cursor_reset_rebases_task_ids_and_nulls_step_totals(self):
        snapshots = [
            {
                "valid": True,
                "supported": True,
                "enabled": True,
                "write_task_id": 8,
                "cursor_reset": False,
                "lost_records": 0,
                "inconsistent_records": 0,
                "records": [],
            },
            {
                "valid": True,
                "supported": True,
                "enabled": True,
                "write_task_id": 1,
                "cursor_reset": True,
                "lost_records": 0,
                "inconsistent_records": 0,
                "records": [
                    {
                        "task_id": 1,
                        "req_seq": 1,
                        "layer_id": 0,
                        "worker_id": 0,
                        "expert_id": 3,
                        "offloaded_branches": 2,
                        "active_tokens": 2,
                        "cpu_pairs": 2,
                        "error_code": 0,
                        "did_compute": False,
                        "status": "no_compute",
                        "cpu_compute_ns": None,
                        "daemon_service_ns": 10_000,
                        "dispatch_ns": None,
                        "flush_ns": 1_000,
                        "queue_ns": None,
                        "protocol_ns": None,
                    }
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = step_trace.StepTraceRecorder(
                rank_id=0,
                hetero_mode="dynamic_dual_loop",
                cpu_trace_applicable=True,
                output_dir=temp_dir,
                run_id="reset",
                event_factory=_EventFactory(),
                snapshot_fn=lambda _cursor: snapshots.pop(0),
                write_run_record=False,
            )
            recorder.note_plan_update(0, 3)
            step = recorder.begin_step(
                SimpleNamespace(
                    request_ids=["r0"],
                    request_decode_steps=[2],
                    request_id_source="scheduler",
                    batch_size=1,
                )
            )
            step.layers.append(
                step_trace.LayerEvents(
                    layer_id=0,
                    start=_FakeEvent(0),
                    dispatch_end=_FakeEvent(1),
                    local_moe_end=_FakeEvent(2),
                    end=_FakeEvent(3),
                    branches_received=_FakeScalar(2),
                )
            )
            recorder.end_step(step)
            recorder.close()

            records = _read_records(Path(temp_dir) / "step_trace_rank_0.jsonl")
            step_record = next(x for x in records if x["kind"] == "step")
            task_records = [x for x in records if x["kind"] == "cpu_task"]
            self.assertIsNone(step_record["total_offloaded_branches"])
            self.assertIsNone(step_record["gpu_wait_cpu_total_us"])
            self.assertTrue(step_record["cpu_trace_cursor_reset"])
            self.assertIn("cpu_trace_cursor_reset", step_record["unsupported_reason"])
            self.assertEqual(step_record["cpu_trace_unassigned_task_records"], 1)
            self.assertEqual(recorder._cpu_cursor, 1)
            self.assertEqual(task_records, [])

    def test_missing_native_data_is_explicitly_unsupported(self):
        unsupported = {
            "valid": False,
            "supported": False,
            "enabled": False,
            "write_task_id": 0,
            "records": [],
            "reason": "task_trace_snapshot_api_unavailable",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = step_trace.StepTraceRecorder(
                rank_id=0,
                hetero_mode="dynamic_dual_loop",
                cpu_trace_applicable=True,
                output_dir=temp_dir,
                run_id="missing",
                event_factory=_EventFactory(),
                snapshot_fn=lambda _cursor: dict(unsupported),
                write_run_record=False,
            )
            step = recorder.begin_step(
                SimpleNamespace(
                    request_ids=[],
                    request_decode_steps=[],
                    request_id_source="unavailable",
                    batch_size=1,
                )
            )
            recorder.end_step(step)
            recorder.close()

            record = _read_records(Path(temp_dir) / "step_trace_rank_0.jsonl")[0]
            self.assertEqual(record["kind"], "step")
            self.assertIn(record["completeness"], {"partial", "unsupported"})
            self.assertIn(
                "task_trace_snapshot_api_unavailable",
                record["unsupported_reason"],
            )
            self.assertIn("no_layer_events", record["unsupported_reason"])


def _empty_snapshot(_cursor):
    return {
        "valid": True,
        "supported": True,
        "enabled": True,
        "write_task_id": 0,
        "lost_records": 0,
        "inconsistent_records": 0,
        "records": [],
    }


def _decode_batch():
    return SimpleNamespace(
        request_ids=["synthetic:test:0"],
        request_decode_steps=[1],
        request_id_source="synthetic",
        batch_size=1,
    )


class StepTraceSamplingTests(unittest.TestCase):
    def _recorder(self, temp_dir, sync_calls, **kwargs):
        options = {
            "rank_id": 0,
            "hetero_mode": "dynamic_dual_loop",
            "cpu_trace_applicable": True,
            "output_dir": temp_dir,
            "run_id": "base",
            "event_factory": _EventFactory(),
            "snapshot_fn": _empty_snapshot,
            "sync_fn": lambda: sync_calls.append(1),
        }
        options.update(kwargs)
        return step_trace.StepTraceRecorder(**options)

    @staticmethod
    def _decode(recorder):
        step = recorder.begin_step(_decode_batch())
        recorder.end_step(step)
        return step

    @staticmethod
    def _steps(path):
        return [r for r in _read_records(path) if r["kind"] == "step"]

    def test_sampling_keeps_real_decode_numbering(self):
        sync_calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = self._recorder(temp_dir, sync_calls, sample_every=3)
            traced = [self._decode(recorder) is not None for _ in range(7)]
            recorder.close()

            self.assertEqual(traced, [True, False, False, True, False, False, True])
            records = _read_records(Path(temp_dir) / "step_trace_rank_0.jsonl")
            run = records[0]
            self.assertEqual(run["kind"], "run")
            self.assertEqual(run["sample_every"], 3)
            self.assertEqual(run["sampling"], "every_3_decode_forwards")
            self.assertEqual(run["diagnostic_sync"], "once_per_sampled_decode_step")
            self.assertEqual(
                [r["batch_step_id"] for r in records if r["kind"] == "step"],
                [1, 4, 7],
            )
            # 第 1 步前面没有漏记的步，不用同步；第 4、7 步前各同步一次。
            self.assertEqual(len(sync_calls), 2)

    def test_every_step_mode_matches_old_records(self):
        sync_calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = self._recorder(temp_dir, sync_calls)
            for _ in range(3):
                self._decode(recorder)
            recorder.close()

            records = _read_records(Path(temp_dir) / "step_trace_rank_0.jsonl")
            self.assertEqual(records[0]["sampling"], "every_decode_forward")
            self.assertEqual(records[0]["diagnostic_sync"], "once_per_decode_step")
            self.assertEqual(
                [r["batch_step_id"] for r in records if r["kind"] == "step"],
                [1, 2, 3],
            )
            self.assertEqual(sync_calls, [])

    def test_pause_does_not_count_and_forces_sync(self):
        sync_calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = self._recorder(temp_dir, sync_calls, write_run_record=False)
            self._decode(recorder)
            recorder.pause()
            self.assertIsNone(self._decode(recorder))
            self.assertIsNone(self._decode(recorder))
            recorder.resume()
            self._decode(recorder)
            recorder.close()

            steps = self._steps(Path(temp_dir) / "step_trace_rank_0.jsonl")
            self.assertEqual([r["batch_step_id"] for r in steps], [1, 2])
            self.assertEqual(len(sync_calls), 1)

    def test_segments_write_own_files_and_restart_numbering(self):
        sync_calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = self._recorder(temp_dir, sync_calls)
            self._decode(recorder)
            segment_dir = recorder.start_segment(
                "bs1-in8-out4", {"batch_size": 1}, sample_every=2
            )
            for _ in range(3):
                self._decode(recorder)
            recorder.end_segment()
            self._decode(recorder)
            recorder.close()

            self.assertEqual(segment_dir, Path(temp_dir) / "bs1-in8-out4")
            top = _read_records(Path(temp_dir) / "step_trace_rank_0.jsonl")
            self.assertEqual([r["kind"] for r in top], ["run", "step", "step"])
            self.assertEqual([r["batch_step_id"] for r in top[1:]], [1, 2])
            self.assertTrue(all(r["run_id"] == "base" for r in top))

            segment = _read_records(segment_dir / "step_trace_rank_0.jsonl")
            run = segment[0]
            self.assertEqual(run["kind"], "run")
            self.assertEqual(run["segment"], "bs1-in8-out4")
            self.assertEqual(run["run_id"], "base/bs1-in8-out4")
            self.assertEqual(run["base_run_id"], "base")
            self.assertEqual(run["workload"], {"batch_size": 1})
            self.assertEqual(run["sample_every"], 2)
            steps = [r for r in segment if r["kind"] == "step"]
            self.assertEqual([r["batch_step_id"] for r in steps], [1, 3])
            self.assertTrue(all(r["run_id"] == "base/bs1-in8-out4" for r in steps))

    def test_top_level_file_is_created_only_when_needed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = self._recorder(temp_dir, [])
            recorder.start_segment("only-segment")
            self._decode(recorder)
            recorder.close()
            self.assertFalse((Path(temp_dir) / "step_trace_rank_0.jsonl").exists())
            self.assertTrue(
                (Path(temp_dir) / "only-segment" / "step_trace_rank_0.jsonl").exists()
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            # 一步没记也没分段时，照旧留下 run 记录。
            self._recorder(temp_dir, []).close()
            records = _read_records(Path(temp_dir) / "step_trace_rank_0.jsonl")
            self.assertEqual([r["kind"] for r in records], ["run"])

        with tempfile.TemporaryDirectory() as temp_dir:
            self._recorder(temp_dir, [], write_run_record=False).close()
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_end_segment_drops_unfinished_step(self):
        sync_calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = self._recorder(temp_dir, sync_calls, write_run_record=False)
            recorder.start_segment("first")
            step = recorder.begin_step(_decode_batch())
            self.assertIsNotNone(step)
            recorder.end_segment()
            recorder.end_step(step)
            recorder.start_segment("second")
            self._decode(recorder)
            recorder.close()

            first = Path(temp_dir) / "first" / "step_trace_rank_0.jsonl"
            self.assertEqual(first.read_text(), "")
            second = self._steps(Path(temp_dir) / "second" / "step_trace_rank_0.jsonl")
            self.assertEqual([r["batch_step_id"] for r in second], [1])
            self.assertEqual(len(sync_calls), 1)

    def test_segment_name_must_be_a_plain_directory_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = self._recorder(temp_dir, [], write_run_record=False)
            for name in ["", ".", "..", "a/b", "a\\b", "bs 1", "../x"]:
                with self.assertRaises(ValueError, msg=name):
                    recorder.start_segment(name)
            self.assertIsNotNone(recorder.start_segment("bs1-in1024-out1024"))
            recorder.close()

    def test_module_wrappers_are_noops_when_disabled(self):
        old_enabled = step_trace.ENABLED
        old_recorder = step_trace._recorder
        try:
            step_trace.ENABLED = False
            step_trace._recorder = object()
            self.assertIsNone(step_trace.start_segment("x"))
            step_trace.end_segment()
            step_trace.pause()
            step_trace.resume()
        finally:
            step_trace.ENABLED = old_enabled
            step_trace._recorder = old_recorder


if __name__ == "__main__":
    unittest.main()
