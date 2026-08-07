#!/usr/bin/env python3

"""Run the Store EGM Provider and publish machine-readable readiness evidence."""

import argparse
import importlib
import json
import math
import os
import pathlib
import re
import signal
import time
import urllib.parse
import urllib.request


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


class ProviderSetupFailed(Exception):
    pass


def import_store_module():
    try:
        return importlib.import_module("store")
    except ModuleNotFoundError as exc:
        if exc.name != "store":
            raise
    return importlib.import_module("mooncake.store")


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([KMGT]?B?|B)?\s*", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}")
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
    size = int(match.group(1)) * factors[(match.group(2) or "B").upper()]
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
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError(
            "value may contain only letters, digits, dot, underscore, and dash"
        )
    return value


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_segment_detail(detail: str, expected_name: str) -> tuple[int, int]:
    lines = detail.splitlines()
    if not lines or lines[0] != expected_name:
        actual = lines[0] if lines else None
        raise ValueError(
            f"Master returned segment {actual!r}, expected {expected_name!r}"
        )
    used_match = re.search(r"^Used\(bytes\)\s*:\s*(\d+)\s*$", detail, re.M)
    capacity_match = re.search(r"^Capacity\(bytes\)\s*:\s*(\d+)\s*$", detail, re.M)
    if used_match is None or capacity_match is None:
        raise ValueError("Master segment response is missing used/capacity")
    used = int(used_match.group(1))
    capacity = int(capacity_match.group(1))
    if capacity <= 0 or used > capacity:
        raise ValueError(
            f"invalid Master segment usage: used={used} capacity={capacity}"
        )
    return used, capacity


def query_publication(admin_url: str, segment_name: str) -> dict[str, int]:
    base = admin_url.rstrip("/")
    all_segments = fetch_text(f"{base}/get_all_segments")
    # MasterAdmin currently matches the literal host:port segment name.
    query = urllib.parse.urlencode({"segment": segment_name}, safe=":")
    detail = fetch_text(f"{base}/query_segment?{query}")
    chunk_count = sum(
        line.strip() == segment_name for line in all_segments.splitlines()
    )
    if chunk_count <= 0:
        raise ValueError(f"Master has no chunks for {segment_name!r}")
    used, capacity = parse_segment_detail(detail, segment_name)
    return {
        "master_chunk_count": chunk_count,
        "master_used_bytes": used,
        "effective_capacity_bytes": capacity,
    }


def wait_for_publication(
    admin_url: str,
    segment_name: str,
    requested_bytes: int,
    timeout_sec: float,
    retry_sec: float,
    run_id: str,
) -> dict[str, int]:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            publication = query_publication(admin_url, segment_name)
            effective = publication["effective_capacity_bytes"]
            if effective > requested_bytes:
                raise ValueError(
                    f"effective capacity {effective} exceeds request {requested_bytes}"
                )
            return publication
        except Exception as exc:  # readiness is intentionally retryable
            last_error = exc
            emit("provider_readiness_pending", run_id=run_id, error=str(exc))
            time.sleep(retry_sec)
    raise RuntimeError(f"Provider publication timed out: {last_error}")


def write_json_atomic(path: pathlib.Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="GB200 Store EGM Provider")
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--master-server", required=True)
    parser.add_argument("--master-admin-url", required=True)
    parser.add_argument("--pool-size", default="600 MB")
    parser.add_argument("--numa-nodes", default="auto")
    parser.add_argument("--run-id", type=safe_identifier, required=True)
    parser.add_argument("--source-sha", type=safe_identifier, required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--readiness-timeout-sec", type=positive_float, default=120)
    parser.add_argument("--readiness-retry-sec", type=positive_float, default=1)
    parser.add_argument("--run-seconds", type=nonnegative_float, default=0)
    args = parser.parse_args()

    requested_bytes = parse_size(args.pool_size)
    ready_path = pathlib.Path(args.ready_file)
    ready_path.unlink(missing_ok=True)
    config = {
        "local_hostname": args.local_hostname,
        "metadata_server": args.metadata_server,
        "master_server_addr": args.master_server,
        "global_segment_size": args.pool_size,
        "local_buffer_size": "0",
        "protocol": "nvlink",
        "enable_egm_store_pool": "true",
        "egm_numa_nodes": args.numa_nodes,
    }
    emit(
        "provider_config",
        run_id=args.run_id,
        source_sha=args.source_sha,
        pid=os.getpid(),
        requested_capacity_bytes=requested_bytes,
        config=config,
    )

    module = import_store_module()
    emit(
        "provider_module",
        run_id=args.run_id,
        source_sha=args.source_sha,
        path=getattr(module, "__file__", getattr(module, "__name__", "unknown")),
    )
    provider = module.MooncakeDistributedStore()
    exit_code = 0
    setup_attempted = False
    try:
        setup_started = time.perf_counter_ns()
        setup_attempted = True
        setup_result = provider.setup(config)
        setup_ns = time.perf_counter_ns() - setup_started
        if setup_result != 0:
            emit(
                "provider_setup",
                status="FAIL",
                run_id=args.run_id,
                source_sha=args.source_sha,
                result=setup_result,
                duration_ns=setup_ns,
            )
            raise ProviderSetupFailed
        publication = wait_for_publication(
            args.master_admin_url,
            args.local_hostname,
            requested_bytes,
            args.readiness_timeout_sec,
            args.readiness_retry_sec,
            args.run_id,
        )
        ready: dict[str, object] = {
            "event": "provider_ready",
            "status": "PASS",
            "run_id": args.run_id,
            "source_sha": args.source_sha,
            "pid": os.getpid(),
            "local_hostname": args.local_hostname,
            "requested_capacity_bytes": requested_bytes,
            "setup_duration_ns": setup_ns,
            **publication,
        }
        write_json_atomic(ready_path, ready)
        print(json.dumps(ready, sort_keys=True), flush=True)

        stopping = False

        def request_stop(_signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        deadline = time.monotonic() + args.run_seconds if args.run_seconds > 0 else None
        while not stopping and (deadline is None or time.monotonic() < deadline):
            time.sleep(0.25)
    except ProviderSetupFailed:
        exit_code = 2
    except Exception as exc:
        emit(
            "provider_error",
            status="FAIL",
            run_id=args.run_id,
            source_sha=args.source_sha,
            error=str(exc),
        )
        exit_code = 1
    finally:
        if setup_attempted:
            cleanup_started = time.perf_counter_ns()
            try:
                close_result = provider.close()
            except Exception as exc:
                close_result = f"exception: {exc}"
            cleanup_ns = time.perf_counter_ns() - cleanup_started
            cleanup_ok = close_result in (None, 0)
            emit(
                "provider_cleanup",
                status="PASS" if cleanup_ok else "FAIL",
                run_id=args.run_id,
                source_sha=args.source_sha,
                close_result=close_result,
                duration_ns=cleanup_ns,
            )
            if not cleanup_ok and exit_code == 0:
                exit_code = 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
