#!/usr/bin/env python3

import argparse
import json
import math
import os
import pathlib
import re
import signal
import time
import urllib.parse
import urllib.request

from nvlink_host_numa_metrics import provider_capacity


def import_store_module():
    try:
        from mooncake import store as store_module  # type: ignore
    except ImportError:
        import store as store_module  # type: ignore
    return store_module


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([KMGT]?B?|B)?\s*", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}")
    number = int(match.group(1))
    unit = (match.group(2) or "B").upper()
    factors = {
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }
    size = number * factors[unit]
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be greater than zero")
    return size


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to zero")
    return number


def redact(config: dict[str, str]) -> dict[str, str]:
    hidden = ("password", "secret", "token", "credential", "handle")
    return {
        key: ("<redacted>" if any(part in key.lower() for part in hidden) else value)
        for key, value in config.items()
    }


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_master_publication(
    master_admin_url: str, segment_name: str
) -> tuple[str, str]:
    base = master_admin_url.rstrip("/")
    segments = fetch_text(f"{base}/get_all_segments")
    query = urllib.parse.urlencode({"segment": segment_name})
    detail = fetch_text(f"{base}/query_segment?{query}")
    return segments, detail


def parse_query_segment(detail: str, expected_name: str) -> tuple[int, int]:
    lines = detail.splitlines()
    if not lines or lines[0] != expected_name:
        raise ValueError(
            f"Master query returned segment {lines[0] if lines else None!r}, "
            f"expected {expected_name!r}"
        )
    used_match = re.search(r"^Used\(bytes\)\s*:\s*(\d+)\s*$", detail, re.M)
    capacity_match = re.search(r"^Capacity\(bytes\)\s*:\s*(\d+)\s*$", detail, re.M)
    if used_match is None or capacity_match is None:
        raise ValueError("Master /query_segment response is missing used/capacity")
    used = int(used_match.group(1))
    capacity = int(capacity_match.group(1))
    if used > capacity:
        raise ValueError("Master reports segment used bytes above capacity")
    return used, capacity


def validate_publication(
    metrics: str,
    requested_bytes: int,
    segment_name: str,
    all_segments: str,
    segment_detail: str,
) -> dict[str, int]:
    expected = provider_capacity(metrics)
    if expected.requested_bytes != requested_bytes:
        raise ValueError(
            "Provider requested-capacity metric mismatch: "
            f"{expected.requested_bytes} != {requested_bytes}"
        )
    names = [line for line in all_segments.splitlines() if line]
    mounted_chunks = sum(name == segment_name for name in names)
    if mounted_chunks != expected.chunk_count:
        raise ValueError(
            f"Master mounted {mounted_chunks} chunks for {segment_name!r}, "
            f"expected {expected.chunk_count}"
        )
    used, capacity = parse_query_segment(segment_detail, segment_name)
    if capacity != expected.effective_bytes:
        raise ValueError(
            f"Master capacity for {segment_name!r} is {capacity}, "
            f"expected {expected.effective_bytes}"
        )
    return {
        "effective_capacity_bytes": expected.effective_bytes,
        "expected_chunk_count": expected.chunk_count,
        "master_chunk_count": mounted_chunks,
        "master_used_bytes": used,
        "master_capacity_bytes": capacity,
    }


def write_ready_file(path: pathlib.Path, status: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(status, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_for_readiness(
    provider, args: argparse.Namespace, requested_bytes: int
) -> dict[str, int]:
    deadline = time.monotonic() + args.readiness_timeout_sec
    last_error: Exception | None = None
    while True:
        try:
            metrics = provider.serialize_metrics()
            all_segments, detail = fetch_master_publication(
                args.master_admin_url, args.local_hostname
            )
            return validate_publication(
                metrics,
                requested_bytes,
                args.local_hostname,
                all_segments,
                detail,
            )
        except Exception as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Provider readiness validation timed out: {last_error}"
                ) from last_error
            print(
                json.dumps(
                    {"event": "readiness_pending", "error": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(args.readiness_retry_sec)


def serve(args: argparse.Namespace, provider, requested_bytes: int) -> int:
    publication = wait_for_readiness(provider, args, requested_bytes)
    status: dict[str, object] = {
        "event": "ready",
        "requested_capacity_bytes": requested_bytes,
        "local_hostname": args.local_hostname,
        **publication,
    }
    print(json.dumps(status, sort_keys=True), flush=True)
    if args.ready_file:
        write_ready_file(pathlib.Path(args.ready_file), status)

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    deadline = time.monotonic() + args.run_seconds if args.run_seconds > 0 else None
    next_metrics = time.monotonic()
    while not stopping and (deadline is None or time.monotonic() < deadline):
        if time.monotonic() >= next_metrics:
            try:
                metrics = (
                    fetch_text(args.metrics_url)
                    if args.metrics_url
                    else provider.serialize_metrics()
                )
                capacity = provider_capacity(metrics)
                print(
                    json.dumps(
                        {
                            "event": "capacity_metrics",
                            "requested_capacity_bytes": capacity.requested_bytes,
                            "effective_capacity_bytes": capacity.effective_bytes,
                            "chunk_count": capacity.chunk_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(
                    json.dumps({"event": "metrics_error", "error": str(exc)}),
                    flush=True,
                )
            next_metrics = time.monotonic() + args.metrics_interval_sec
        time.sleep(0.25)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="GB200 HOST_NUMA Provider")
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--master-server", required=True)
    parser.add_argument(
        "--master-admin-url",
        required=True,
        help="Master admin HTTP base URL (metrics_port, default 9003)",
    )
    parser.add_argument("--global-segment-size", default="600 GB")
    parser.add_argument("--local-buffer-size", default="0")
    parser.add_argument("--nodes", default="auto")
    parser.add_argument("--metrics-url", default="", help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--metrics-interval-sec", type=positive_float, default=10.0)
    parser.add_argument("--readiness-timeout-sec", type=positive_float, default=30.0)
    parser.add_argument("--readiness-retry-sec", type=positive_float, default=0.5)
    parser.add_argument("--run-seconds", type=nonnegative_float, default=0.0)
    args = parser.parse_args()

    try:
        requested_bytes = parse_size(args.global_segment_size)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.ready_file:
        pathlib.Path(args.ready_file).unlink(missing_ok=True)
    config = {
        "local_hostname": args.local_hostname,
        "metadata_server": args.metadata_server,
        "master_server_addr": args.master_server,
        "global_segment_size": args.global_segment_size,
        "local_buffer_size": args.local_buffer_size,
        "protocol": "nvlink",
        "enable_nvlink_host_numa": "true",
        "nvlink_host_numa_nodes": args.nodes,
    }
    print(
        json.dumps({"event": "config", "config": redact(config)}, sort_keys=True),
        flush=True,
    )

    store_module = import_store_module()
    provider = store_module.MooncakeDistributedStore()
    exit_code = 0
    active_error = False
    try:
        setup_result = provider.setup(config)
        if setup_result != 0:
            print(
                json.dumps({"event": "setup_failed", "result": setup_result}),
                flush=True,
            )
            exit_code = int(setup_result) if int(setup_result) > 0 else 2
        else:
            exit_code = serve(args, provider, requested_bytes)
    except BaseException:
        active_error = True
        raise
    finally:
        if args.ready_file:
            pathlib.Path(args.ready_file).unlink(missing_ok=True)
        try:
            close_result = provider.close() if hasattr(provider, "close") else 0
        except Exception as exc:
            close_result = f"exception: {exc}"
        if close_result not in (None, 0):
            print(
                json.dumps({"event": "cleanup_failed", "close_result": close_result}),
                flush=True,
            )
            if not active_error and exit_code == 0:
                exit_code = 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
