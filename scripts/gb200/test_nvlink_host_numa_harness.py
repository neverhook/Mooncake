#!/usr/bin/env python3

import http.server
import json
import threading
import unittest
import urllib.parse

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


class HarnessTest(unittest.TestCase):
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
