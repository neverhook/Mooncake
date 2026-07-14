#!/usr/bin/env python3

import argparse
import http.server
import io
import json
import pathlib
import threading
import types
import unittest
import urllib.parse
from contextlib import redirect_stdout
from unittest import mock

import nvlink_host_numa_consumer as consumer
import nvlink_host_numa_provider as provider
import nvlink_host_numa_bench as bench
from nvlink_host_numa_metrics import (
    classify_cache_phase,
    consumer_delta,
    consumer_metrics,
    provider_capacity,
    validate_miss_then_hit,
)


PROVIDER_METRICS = """
# TYPE mooncake_nvlink_host_numa_requested_capacity_bytes gauge
mooncake_nvlink_host_numa_requested_capacity_bytes 600
mooncake_nvlink_host_numa_effective_capacity_bytes 576
mooncake_nvlink_host_numa_node_effective_bytes{numa_node="0"} 288
mooncake_nvlink_host_numa_node_effective_bytes{numa_node="1"} 288
mooncake_nvlink_host_numa_node_chunks{numa_node="0"} 2
mooncake_nvlink_host_numa_node_chunks{numa_node="1"} 2
"""


def consumer_metrics_text(hits: int, misses: int, imports: int, duration: int) -> str:
    return f"""
mooncake_nvlink_consumer_mapping_cache_total{{result="hit"}} {hits}
mooncake_nvlink_consumer_mapping_cache_total{{result="miss"}} {misses}
mooncake_nvlink_consumer_lazy_import_observations_total {imports}
mooncake_nvlink_consumer_lazy_import_duration_us_total {duration}
"""


class _AdminHandler(http.server.BaseHTTPRequestHandler):
    segment_name = "10.0.0.1:12345"

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/get_all_segments":
            body = ((self.segment_name + "\n") * 4 + "other:9999\n").encode()
            self.send_response(200)
        elif parsed.path == "/query_segment":
            query = urllib.parse.parse_qs(parsed.query)
            if query.get("segment") != [self.segment_name]:
                self.send_response(400)
                body = b"bad segment"
            else:
                self.send_response(200)
                body = (
                    f"{self.segment_name}\nUsed(bytes): 0\nCapacity(bytes) : 576\n"
                ).encode()
        else:
            self.send_response(404)
            body = b"not found"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


class _FakeCudaRuntime:
    def __init__(self):
        self.next_pointer = 0x100000
        self.memory: dict[int, bytearray] = {}

    def malloc(self, size: int) -> int:
        pointer = self.next_pointer
        self.next_pointer += size + 4096
        self.memory[pointer] = bytearray(size)
        return pointer

    def free(self, pointer: int) -> None:
        del self.memory[pointer]

    def copy_from_host(self, pointer: int, payload: bytes) -> None:
        self.memory[pointer][:] = payload

    def copy_to_host(self, pointer: int, size: int) -> bytes:
        return bytes(self.memory[pointer][:size])

    def memset(self, pointer: int, value: int, size: int) -> None:
        self.memory[pointer][:size] = bytes([value]) * size

    def synchronize(self) -> None:
        pass


class _FakeConsumerStore:
    def __init__(self, cuda: _FakeCudaRuntime):
        self.cuda = cuda
        self.objects: dict[str, bytes] = {}
        self.registered: set[int] = set()
        self.hits = 0
        self.misses = 0
        self.imports = 0
        self.unregister_result = 0

    def register_buffer(self, pointer: int, _size: int) -> int:
        self.registered.add(pointer)
        return 0

    def unregister_buffer(self, pointer: int) -> int:
        if self.unregister_result != 0:
            return self.unregister_result
        self.registered.remove(pointer)
        return 0

    def put_from(self, key: str, pointer: int, size: int) -> int:
        if self.imports == 0:
            self.misses += 1
            self.imports += 1
        else:
            self.hits += 1
        self.objects[key] = bytes(self.cuda.memory[pointer][:size])
        return 0

    def get_into(self, key: str, pointer: int, size: int) -> int:
        self.hits += 1
        self.cuda.memory[pointer][:size] = self.objects[key]
        return size

    def remove(self, key: str, _force: bool) -> int:
        del self.objects[key]
        return 0

    def serialize_metrics(self) -> str:
        return consumer_metrics_text(
            self.hits, self.misses, self.imports, self.imports * 11
        )


class HarnessTest(unittest.TestCase):
    def test_payload_size_is_capped_at_128_mb(self):
        maximum = 128 * 1024 * 1024
        self.assertEqual(consumer.payload_size(str(maximum)), maximum)
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "128 MB"):
            consumer.payload_size(str(maximum + 1))
        self.assertEqual(bench.MAX_PAYLOAD_SIZE, maximum)

    def test_harness_prefers_source_build_store_module_with_packaged_fallback(self):
        for harness_module in (consumer, provider):
            with self.subTest(module=harness_module.__name__, path="source-build"):
                source_build = types.SimpleNamespace(__file__="/build/store.so")
                with mock.patch.object(
                    harness_module.importlib,
                    "import_module",
                    return_value=source_build,
                ) as import_module:
                    self.assertIs(harness_module.import_store_module(), source_build)
                    import_module.assert_called_once_with("store")

            with self.subTest(module=harness_module.__name__, path="installed"):
                missing = ModuleNotFoundError("No module named 'store'", name="store")
                installed = types.SimpleNamespace(__file__="/site/mooncake/store.so")
                with mock.patch.object(
                    harness_module.importlib,
                    "import_module",
                    side_effect=(missing, installed),
                ) as import_module:
                    self.assertIs(harness_module.import_store_module(), installed)
                    self.assertEqual(
                        import_module.call_args_list,
                        [mock.call("store"), mock.call("mooncake.store")],
                    )

    def test_consumer_runs_hbm_round_trip_without_torch(self):
        source = pathlib.Path(consumer.__file__).read_text()
        self.assertNotIn("import torch", source)

        cuda = _FakeCudaRuntime()
        store = _FakeConsumerStore(cuda)
        args = types.SimpleNamespace(
            iterations=2,
            payload_size=4096,
            device=3,
            key_prefix="nvlink-host-numa",
            run_id="no-torch-test",
        )
        output = io.StringIO()
        quarantine = consumer.HbmAllocationQuarantine()
        with redirect_stdout(output):
            consumer.run_iterations(store, cuda, args, quarantine)

        records = [
            json.loads(line) for line in output.getvalue().splitlines() if line.strip()
        ]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["event"] == "result" for record in records))
        self.assertEqual(records[0]["cache_phase"], "mixed")
        self.assertEqual(records[1]["cache_phase"], "warm")
        self.assertEqual(cuda.memory, {})
        self.assertEqual(len(quarantine), 0)
        self.assertEqual(store.registered, set())
        self.assertEqual(store.objects, {})

    def test_consumer_quarantines_hbm_until_failed_registration_is_closed(self):
        cuda = _FakeCudaRuntime()
        store = _FakeConsumerStore(cuda)
        store.unregister_result = 7
        quarantine = consumer.HbmAllocationQuarantine()

        cleanup_output = io.StringIO()
        with redirect_stdout(cleanup_output):
            with self.assertRaisesRegex(RuntimeError, "unregister returned 7"):
                with consumer.allocated_hbm_buffers(
                    cuda, (4096,), quarantine
                ) as pointers:
                    pointer = pointers[0]
                    with consumer.registered_buffers(
                        store, ((pointer, 4096, "source"),), quarantine
                    ):
                        pass

        self.assertIn(pointer, cuda.memory)
        self.assertEqual(len(quarantine), 1)
        self.assertIn(pointer, store.registered)

        # A successful Store close removes the registration before the
        # quarantined cudaMalloc allocation is released.
        store.registered.clear()
        self.assertEqual(quarantine.release_after_store_close(cuda), [])
        self.assertEqual(len(quarantine), 0)
        self.assertEqual(cuda.memory, {})

    def test_bench_uses_counter_derived_phase(self):
        record = {
            "event": "result",
            "run_id": "run-1",
            "device": 0,
            "iteration": 0,
            "bytes": 4096,
            "cache_phase": "mixed",
            "put_cache_delta": {
                "hit": 0,
                "miss": 1,
                "lazy_imports": 1,
                "lazy_import_duration_us": 11,
            },
            "get_cache_delta": {
                "hit": 1,
                "miss": 0,
                "lazy_imports": 0,
                "lazy_import_duration_us": 0,
            },
            "put_latency_ns": 100,
            "get_latency_ns": 100,
            "put_gib_s": 1.0,
            "get_gib_s": 1.0,
        }
        parsed = bench.result_records(json.dumps(record), 0, 4096, 1, "run-1")
        self.assertEqual(parsed, [record])

        record["cache_phase"] = "cold"
        with self.assertRaisesRegex(RuntimeError, "invalid cache phase"):
            bench.result_records(json.dumps(record), 0, 4096, 1, "run-1")

    def test_provider_capacity_requires_exact_per_node_sum(self):
        capacity = provider_capacity(PROVIDER_METRICS)
        self.assertEqual(capacity.requested_bytes, 600)
        self.assertEqual(capacity.effective_bytes, 576)
        self.assertEqual(capacity.chunk_count, 4)

        broken = PROVIDER_METRICS.replace(
            'node_effective_bytes{numa_node="1"} 288',
            'node_effective_bytes{numa_node="1"} 287',
        )
        with self.assertRaisesRegex(ValueError, "does not sum"):
            provider_capacity(broken)

    def test_master_admin_mock_and_exact_publication_validation(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _AdminHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            all_segments, detail = provider.fetch_master_publication(
                base, _AdminHandler.segment_name
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        status = provider.validate_publication(
            PROVIDER_METRICS,
            600,
            _AdminHandler.segment_name,
            all_segments,
            detail,
        )
        self.assertEqual(status["master_chunk_count"], 4)
        self.assertEqual(status["master_capacity_bytes"], 576)

        with self.assertRaisesRegex(ValueError, "mounted 3 chunks"):
            provider.validate_publication(
                PROVIDER_METRICS,
                600,
                _AdminHandler.segment_name,
                (_AdminHandler.segment_name + "\n") * 3,
                detail,
            )
        with self.assertRaisesRegex(ValueError, "expected 576"):
            provider.validate_publication(
                PROVIDER_METRICS,
                600,
                _AdminHandler.segment_name,
                all_segments,
                detail.replace("576", "575"),
            )

    def test_consumer_counter_delta_proves_miss_then_hit(self):
        before = consumer_metrics(consumer_metrics_text(7, 3, 3, 40))
        after_first = consumer_metrics(consumer_metrics_text(7, 4, 4, 51))
        after_second = consumer_metrics(consumer_metrics_text(8, 4, 4, 51))
        first = consumer_delta(before, after_first)
        second = consumer_delta(after_first, after_second)
        validate_miss_then_hit(first, second)
        self.assertEqual(classify_cache_phase(first), "cold")
        self.assertEqual(classify_cache_phase(second), "warm")
        self.assertEqual(classify_cache_phase(first, second), "mixed")

        with self.assertRaisesRegex(ValueError, "unexpectedly imported"):
            validate_miss_then_hit(
                first,
                consumer_delta(
                    after_first, consumer_metrics(consumer_metrics_text(8, 5, 5, 52))
                ),
            )


if __name__ == "__main__":
    unittest.main()
