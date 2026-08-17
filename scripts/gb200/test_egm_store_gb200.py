#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.gb200 import egm_store_bench as bench
from scripts.gb200 import egm_store_consumer as consumer
from scripts.gb200 import egm_store_orchestrator as orchestrator
from scripts.gb200 import egm_store_provider as provider
from scripts.gb200 import egm_validation_common as validation_common


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


class FakeCuda:
    def __init__(self):
        self.memory: dict[int, bytes] = {}

    def copy_from_host(self, pointer: int, payload: bytes) -> None:
        self.memory[pointer] = payload

    def memset(self, pointer: int, value: int, size: int) -> None:
        self.memory[pointer] = bytes([value]) * size

    def synchronize(self) -> None:
        pass

    def copy_to_host(self, pointer: int, size: int) -> bytes:
        return self.memory[pointer][:size]


class FakeStore:
    def __init__(self, cuda: FakeCuda):
        self.cuda = cuda
        self.objects: dict[str, bytes] = {}

    def put_from(self, key: str, pointer: int, size: int) -> int:
        self.objects[key] = self.cuda.memory[pointer][:size]
        return 0

    def get_into(self, key: str, pointer: int, size: int) -> int:
        payload = self.objects[key]
        self.cuda.memory[pointer] = payload
        return len(payload)

    def remove(self, key: str, force: bool) -> int:
        self.objects.pop(key)
        return 0 if force else -1


class EgmStoreGb200Test(unittest.TestCase):
    def test_wait_helpers_initialize_timeouts_under_nounset(self):
        source = (SCRIPT_DIR / "egm_store_gb200.sh").read_text()

        def function_source(name: str) -> str:
            start = source.index(f"{name}() {{")
            end = source.index("\n}", start) + 2
            return source[start:end]

        cases = [
            (
                "wait_for_tcp",
                "tcp_reachable() { return 0; }\nwait_for_tcp 127.0.0.1 1 1",
            ),
            (
                "wait_for_pid_command",
                "pid_exists() { return 0; }\n"
                "pid_live() { return 0; }\n"
                "wait_for_pid_command /tmp/test-pid expected 1",
            ),
        ]
        for name, invocation in cases:
            script = f"set -u\n{function_source(name)}\n{invocation}\n"
            subprocess.run(["bash", "-c", script], check=True)

    def test_consumer_actions_enable_nvlink_auto_discovery(self):
        source = (SCRIPT_DIR / "egm_store_gb200.sh").read_text()
        self.assertIn('"MC_MS_AUTO_DISC=0"', source)
        self.assertEqual(source.count('run_env "$NODE_B_IP" env MC_MS_AUTO_DISC=1'), 2)

    def test_bandwidth_conversion(self):
        self.assertEqual(consumer.bandwidth_gib_s(1024**3, 1_000_000_000), 1.0)

    def test_nvidia_route_evidence_and_rdma_latency_are_parsed(self):
        nvlink = validation_common.parse_nvlink_data(
            "GPU 0: NVIDIA GB200 (UUID: GPU-test)\n"
            "         Link 0: Data Tx: 1,234 KiB\n"
            "         Link 0: Data Rx: 5,678 KiB\n"
        )
        self.assertEqual(nvlink["gpu0/link0/tx_bytes"], 1234 * 1024)
        self.assertEqual(nvlink["gpu0/link0/rx_bytes"], 5678 * 1024)
        self.assertEqual(
            validation_common.parse_c2c_status(
                "GPU 0: NVIDIA GB200 (UUID: GPU-test)\n"
                "         C2C Link 0: 44.712 GB/s\n"
            )["gpu0/link0/capacity_gb_s"],
            44.712,
        )
        self.assertEqual(
            validation_common.parse_c2c_errors(
                "GPU 0: NVIDIA GB200 (UUID: GPU-test)\n"
                " C2C Link 0: Error Back-to-Back Replay Count: 7\n"
            )["gpu0/link0/back_to_back_replay_errors"],
            7,
        )
        fabric = validation_common.parse_fabric(
            "    Fabric\n"
            "        State : Completed\n"
            "        Status : Success\n"
            "        CliqueId : 32766\n"
            "        ClusterUUID : test\n"
            "        Health\n"
            "            Summary : Healthy\n"
            "            Bandwidth : Full\n"
            "            Route Recovery in progress : False\n"
        )
        self.assertEqual(fabric["gpu0"]["bandwidth"], "Full")
        self.assertEqual(
            orchestrator.parse_te_latency_samples(
                "I0000 Latency sample: duration 91 ns\n"
                "I0000 Latency sample: duration 103 ns\n"
            ),
            [91.0, 103.0],
        )

    def test_inferred_c2c_route_requires_nvlink_traffic_and_no_new_errors(self):
        nvlink = {
            f"gpu0/link{link}/{direction}_bytes": 1000
            for link in range(18)
            for direction in ("tx", "rx")
        }
        capacity = {f"gpu0/link{link}/capacity_gb_s": 44.712 for link in range(5)}
        errors = {
            f"gpu0/link{link}/{name}_errors": 0
            for link in range(5)
            for name in ("interrupt", "replay", "back_to_back_replay")
        }
        fabric = {
            "gpu0": {
                "state": "Completed",
                "status": "Success",
                "health": "Healthy",
                "bandwidth": "Full",
                "route_recovery": "False",
            }
        }
        before = {
            "backend": "nvidia-smi",
            "command_status": {
                "nvlink_data": "PASS",
                "c2c_status": "PASS",
                "c2c_errors": "PASS",
                "fabric": "PASS",
            },
            "nvlink_bytes": nvlink,
            "c2c_capacity_gb_s": capacity,
            "c2c_errors": errors,
            "fabric": fabric,
        }
        after = {
            **before,
            "nvlink_bytes": {name: value + 4096 for name, value in nvlink.items()},
            "c2c_errors": dict(errors),
        }
        evidence = validation_common.build_route_evidence(before, after, "test")
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["route_verification"], "C2C_ROUTE_INFERRED")

        after["c2c_errors"]["gpu0/link0/replay_errors"] = 1
        evidence = validation_common.build_route_evidence(before, after, "test")
        self.assertEqual(evidence["status"], "FAIL")

    def test_raw_ce_ceiling_requires_stable_bounded_matrix(self):
        records: list[dict[str, object]] = []
        payloads = [2 * 1024**3, 4 * 1024**3]
        streams = [4, 8]
        for path in ("EGM_H2D", "EGM_D2H"):
            for payload in payloads:
                for stream_count in streams:
                    for sample in range(10):
                        records.append(
                            {
                                "event": "raw_bandwidth_sample",
                                "group": "1GPU",
                                "path": path,
                                "engine": "CE",
                                "device": 0,
                                "bytes": payload,
                                "streams": stream_count,
                                "sample": sample,
                                "duration_seconds": 1.1,
                                "bandwidth_gb_s": 100.0,
                            }
                        )
                        records.append(
                            {
                                "event": "raw_aggregate_sample",
                                "group": "1GPU",
                                "path": path,
                                "engine": "CE",
                                "bytes_per_device": payload,
                                "streams": stream_count,
                                "sample": sample,
                                "duration_seconds": 1.1,
                                "aggregate_bandwidth_gb_s": 100.0,
                            }
                        )
        records.append({"event": "raw_benchmark_gate", "status": "PASS"})
        self.assertEqual(orchestrator.raw_summaries(records)["status"], "PASS")
        for record in records:
            if (
                record.get("event") == "raw_aggregate_sample"
                and record.get("bytes_per_device") == 4 * 1024**3
                and record.get("streams") == 8
            ):
                record["aggregate_bandwidth_gb_s"] = 110.0
        summary = orchestrator.raw_summaries(records)
        self.assertEqual(summary["status"], "FAIL")
        self.assertIn(
            "UNBOUNDED_BY_MATRIX",
            {ceiling["status"] for ceiling in summary["ceilings"]},
        )

    def test_fake_hbm_egm_round_trip_emits_timed_results(self):
        cuda = FakeCuda()
        store = FakeStore(cuda)
        args = argparse.Namespace(
            iterations=2,
            warmups=0,
            payload_size=4096,
            device=1,
            key_prefix="test",
            run_id="run",
            source_sha="a" * 40,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            records = consumer.run_transfers(store, cuda, args, 100, 200)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["sequence_phase"], "lazy_init_probe")
        self.assertEqual(records[1]["sequence_phase"], "steady")
        self.assertGreater(records[0]["put_duration_ns"], 0)
        self.assertGreater(records[0]["get_bandwidth_gib_s"], 0)
        self.assertEqual(store.objects, {})
        self.assertEqual(len(output.getvalue().splitlines()), 2)

    def test_master_segment_detail_parser(self):
        detail = "node-a:12345\nUsed(bytes): 0\nCapacity(bytes): 629145600\n"
        self.assertEqual(
            provider.parse_segment_detail(detail, "node-a:12345"),
            (0, 629145600),
        )
        with self.assertRaises(ValueError):
            provider.parse_segment_detail(detail, "wrong:12345")

    def test_failed_setup_still_closes_provider_and_consumer(self):
        class FailedStore:
            def __init__(self):
                self.closed = 0

            def setup(self, _config):
                return -1

            def close(self):
                self.closed += 1
                return 0

        class Module:
            __file__ = "fake-store.so"

            def __init__(self, instance):
                self.instance = instance

            def MooncakeDistributedStore(self):
                return self.instance

        provider_store = FailedStore()
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "egm_store_provider.py",
                "--local-hostname",
                "node-a:12345",
                "--metadata-server",
                "http://node-a:8079/metadata",
                "--master-server",
                "node-a:50051",
                "--master-admin-url",
                "http://node-a:9003",
                "--run-id",
                "run",
                "--source-sha",
                "a" * 40,
                "--ready-file",
                str(pathlib.Path(directory) / "ready.json"),
            ]
            with (
                mock.patch.object(
                    provider, "import_store_module", return_value=Module(provider_store)
                ),
                mock.patch("sys.argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(provider.main(), 2)
        self.assertEqual(provider_store.closed, 1)

        consumer_store = FailedStore()
        fake_runtime = mock.Mock()
        argv = [
            "egm_store_consumer.py",
            "--local-hostname",
            "node-b:12400",
            "--metadata-server",
            "http://node-a:8079/metadata",
            "--master-server",
            "node-a:50051",
            "--device",
            "0",
            "--payload-size",
            "4096",
            "--run-id",
            "run",
            "--source-sha",
            "a" * 40,
        ]
        with (
            mock.patch.object(consumer, "CudaRuntime", return_value=fake_runtime),
            mock.patch.object(
                consumer, "import_store_module", return_value=Module(consumer_store)
            ),
            mock.patch("sys.argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(consumer.main(), 1)
        self.assertEqual(consumer_store.closed, 1)

    def make_child_records(self, cleanup_status: str = "PASS"):
        run_id = "run"
        sha = "b" * 40
        records: list[dict[str, object]] = []
        for iteration in range(2):
            start = 1_000_000 + iteration * 100_000
            records.append(
                {
                    "event": "transfer_result",
                    "status": "PASS",
                    "run_id": run_id,
                    "source_sha": sha,
                    "device": 0,
                    "iteration": iteration,
                    "sequence_phase": (
                        "lazy_init_probe" if iteration == 0 else "steady"
                    ),
                    "bytes": 4096,
                    "sha256": "c" * 64,
                    "put_path": "consumer_hbm_to_provider_egm",
                    "put_started_ns": start,
                    "put_ended_ns": start + 1000,
                    "put_duration_ns": 1000,
                    "put_bandwidth_gib_s": 3.0,
                    "get_path": "provider_egm_to_consumer_hbm",
                    "get_started_ns": start + 2000,
                    "get_ended_ns": start + 4000,
                    "get_duration_ns": 2000,
                    "get_bandwidth_gib_s": 2.0,
                }
            )
        records.extend(
            [
                {"event": "consumer_gate", "status": "PASS"},
                {"event": "consumer_cleanup", "status": cleanup_status},
            ]
        )
        return records

    def test_benchmark_validation_requires_cleanup(self):
        records = self.make_child_records()
        results = bench.validate_consumer_records(
            records, 0, 4096, 2, 0, "run", "b" * 40
        )
        aggregate = bench.aggregate_iteration([results[0]], "put")
        self.assertEqual(aggregate["window_duration_ns"], 1000)
        self.assertGreater(aggregate["aggregate_window_gib_s"], 0)
        with self.assertRaisesRegex(RuntimeError, "clean up"):
            bench.validate_consumer_records(
                self.make_child_records("FAIL"), 0, 4096, 2, 0, "run", "b" * 40
            )

    def test_shell_scripts_parse_and_wrapper_config_is_deterministic(self):
        scripts = [
            SCRIPT_DIR / "egm_store_build.sh",
            SCRIPT_DIR / "egm_store_preflight.sh",
            SCRIPT_DIR / "egm_store_gb200.sh",
        ]
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)
        wrapper = SCRIPT_DIR / "egm_store_gb200.sh"
        help_result = subprocess.run(
            ["bash", str(wrapper), "--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        self.assertIn("provider-start", help_result.stdout)
        for action, required_flag in (
            ("full-provider", "--listen-ip"),
            ("full-consumer", "--provider"),
        ):
            result = subprocess.run(
                ["bash", str(wrapper), action, "--help"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertIn(required_flag, result.stdout)

        source = (SCRIPT_DIR / "egm_store_gb200.conf.example").read_text()
        source = source.replace('NODE_A_IP="CHANGE_ME"', 'NODE_A_IP="192.0.2.10"')
        source = source.replace('NODE_B_IP="CHANGE_ME"', 'NODE_B_IP="192.0.2.11"')
        source = source.replace(
            'RUN_ID="egm-gb200-CHANGE_ME"', 'RUN_ID="egm-gb200-script-test"'
        )
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "config"
            result_root = pathlib.Path(directory) / "results"
            config.write_text(
                source.replace(
                    'RESULT_ROOT="/tmp/mooncake-egm-gb200"',
                    f'RESULT_ROOT="{result_root}"',
                )
            )
            result = subprocess.run(
                ["bash", str(wrapper), "--config", str(config), "print-config"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                cwd=REPO_ROOT,
                env={"PATH": str(pathlib.Path("/usr/bin")) + ":/bin"},
            )
            stop_result = subprocess.run(
                ["bash", str(wrapper), "--config", str(config), "provider-stop"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                cwd=REPO_ROOT,
            )
        self.assertIn("NODE_A_IP=192.0.2.10", result.stdout)
        self.assertIn("NODE_B_IP=192.0.2.11", result.stdout)
        self.assertIn("SOURCE_SHA=", result.stdout)
        self.assertIn("EGM_NUMA_NODES=auto", result.stdout)
        self.assertIn("EGM_POOL_SIZE=20 GB", result.stdout)
        self.assertIn("BUILD_UNIT_TESTS=0", result.stdout)
        self.assertIn("MC_IMEX_DAEMON_EXTERNAL=1", result.stdout)
        self.assertIn("Provider cleanup skipped", stop_result.stdout)

    def test_report_requires_matching_teardown_and_renders_performance(self):
        run_id = "report-run"
        sha = "d" * 40
        provider_ready = {
            "event": "provider_ready",
            "status": "PASS",
            "run_id": run_id,
            "source_sha": sha,
            "requested_capacity_bytes": 600 * 1024**2,
            "effective_capacity_bytes": 600 * 1024**2,
            "master_chunk_count": 4,
        }
        provider_records = [
            {
                "event": "provider_cleanup",
                "status": "PASS",
                "run_id": run_id,
                "source_sha": sha,
                "duration_ns": 2_000_000,
            },
            {
                "event": "provider_unpublished",
                "status": "PASS",
                "run_id": run_id,
                "source_sha": sha,
            },
        ]
        benchmark_records = [
            {
                "event": "performance_summary",
                "run_id": run_id,
                "source_sha": sha,
                "operation": "put",
                "path": "consumer_hbm_to_provider_egm",
                "sequence_phase": "steady",
                "bytes": 128 * 1024**2,
                "samples": 4,
                "duration_p50_us": 1000.0,
                "bandwidth_p50_gib_s": 125.0,
            },
            {
                "event": "aggregate_iteration",
                "run_id": run_id,
                "source_sha": sha,
                "operation": "put",
                "sequence_phase": "steady",
                "bytes_per_device": 128 * 1024**2,
                "aggregate_window_gib_s": 400.0,
            },
            {
                "event": "benchmark_gate",
                "status": "PASS",
                "run_id": run_id,
                "source_sha": sha,
                "devices": [0, 1, 2, 3],
                "transfer_samples": 4,
                "threshold_payload_size": 128 * 1024**2,
                "min_put_gib_s": 0,
                "min_get_gib_s": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            ready_path = root / "ready.json"
            provider_log = root / "provider.log"
            benchmark_log = root / "bench.jsonl"
            ready_path.write_text(json.dumps(provider_ready) + "\n")
            provider_log.write_text(
                "native log line\n"
                + "\n".join(json.dumps(record) for record in provider_records)
                + "\n"
            )
            benchmark_log.write_text(
                "\n".join(json.dumps(record) for record in benchmark_records) + "\n"
            )
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT_DIR / "egm_store_report.py"),
                    "--provider-ready",
                    str(ready_path),
                    "--provider-log",
                    str(provider_log),
                    "--benchmark-log",
                    str(benchmark_log),
                    "--branch",
                    "codex/test",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
        self.assertIn("Result: **PASS**", result.stdout)
        self.assertIn("Provider teardown: `PASS`", result.stdout)
        self.assertIn("125.0000", result.stdout)


if __name__ == "__main__":
    unittest.main()
