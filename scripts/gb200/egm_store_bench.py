#!/usr/bin/env python3

"""Run concurrent GB200 Consumers and emit PR-ready JSONL summaries."""

import argparse
import concurrent.futures
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
from collections.abc import Iterable


MAX_PAYLOAD_SIZE = 2 * 1024**3


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError(
            "value may contain only letters, digits, dot, underscore, and dash"
        )
    return value


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def parse_integer_list(
    parser: argparse.ArgumentParser, value: str, name: str
) -> list[int]:
    try:
        numbers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        parser.error(f"{name} must be a comma-separated integer list: {exc}")
    if not numbers:
        parser.error(f"{name} must not be empty")
    return numbers


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def run_consumer(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def parse_child_records(stdout: str, device: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            emit("consumer_stdout_text", device=device, text=line)
            continue
        if not isinstance(record, dict):
            raise RuntimeError(f"consumer GPU {device} emitted non-object JSON")
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return records


def require_positive_number(record: dict[str, object], field: str) -> float:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid {field} in result: {record!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"nonpositive {field} in result: {record!r}")
    return value


def validate_consumer_records(
    records: list[dict[str, object]],
    device: int,
    payload_size: int,
    iterations: int,
    warmups: int,
    run_id: str,
    source_sha: str,
) -> list[dict[str, object]]:
    results = [record for record in records if record.get("event") == "transfer_result"]
    expected_iterations = set(range(iterations))
    actual_iterations = {record.get("iteration") for record in results}
    if len(results) != iterations or actual_iterations != expected_iterations:
        raise RuntimeError(
            f"consumer GPU {device} emitted {len(results)} results with iterations "
            f"{actual_iterations!r}, expected {expected_iterations!r}"
        )
    gates = [record for record in records if record.get("event") == "consumer_gate"]
    cleanup = [
        record for record in records if record.get("event") == "consumer_cleanup"
    ]
    if len(gates) != 1 or gates[0].get("status") != "PASS":
        raise RuntimeError(f"consumer GPU {device} did not emit a PASS gate")
    if len(cleanup) != 1 or cleanup[0].get("status") != "PASS":
        raise RuntimeError(f"consumer GPU {device} did not clean up successfully")

    for record in results:
        if (
            record.get("status") != "PASS"
            or record.get("device") != device
            or record.get("bytes") != payload_size
            or record.get("run_id") != run_id
            or record.get("source_sha") != source_sha
        ):
            raise RuntimeError(
                f"consumer GPU {device} emitted mismatched result: {record!r}"
            )
        expected_phase = (
            "lazy_init_probe"
            if record["iteration"] == 0
            else "warmup"
            if int(record["iteration"]) <= warmups
            else "steady"
        )
        if record.get("sequence_phase") != expected_phase:
            raise RuntimeError(f"invalid sequence phase: {record!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
            raise RuntimeError(f"invalid payload digest: {record!r}")
        for operation in ("put", "get"):
            started = int(require_positive_number(record, f"{operation}_started_ns"))
            ended = int(require_positive_number(record, f"{operation}_ended_ns"))
            duration = int(require_positive_number(record, f"{operation}_duration_ns"))
            require_positive_number(record, f"{operation}_bandwidth_gib_s")
            if ended - started != duration:
                raise RuntimeError(f"inconsistent {operation} timing: {record!r}")
    return results


def sample_summary(
    records: list[dict[str, object]], operation: str, phase: str
) -> dict[str, object]:
    durations_us = [
        float(record[f"{operation}_duration_ns"]) / 1000 for record in records
    ]
    bandwidths = [float(record[f"{operation}_bandwidth_gib_s"]) for record in records]
    return {
        "event": "performance_summary",
        "operation": operation,
        "path": records[0][f"{operation}_path"],
        "sequence_phase": phase,
        "bytes": records[0]["bytes"],
        "samples": len(records),
        "duration_p50_us": percentile(durations_us, 0.50),
        "duration_p95_us": percentile(durations_us, 0.95),
        "duration_p99_us": percentile(durations_us, 0.99),
        "bandwidth_min_gib_s": min(bandwidths),
        "bandwidth_p50_gib_s": percentile(bandwidths, 0.50),
        "bandwidth_mean_gib_s": statistics.fmean(bandwidths),
        "bandwidth_max_gib_s": max(bandwidths),
    }


def aggregate_iteration(
    records: list[dict[str, object]], operation: str
) -> dict[str, object]:
    started = min(int(record[f"{operation}_started_ns"]) for record in records)
    ended = max(int(record[f"{operation}_ended_ns"]) for record in records)
    window_ns = ended - started
    total_bytes = sum(int(record["bytes"]) for record in records)
    if window_ns <= 0:
        raise RuntimeError("aggregate operation window is nonpositive")
    return {
        "event": "aggregate_iteration",
        "operation": operation,
        "path": records[0][f"{operation}_path"],
        "sequence_phase": records[0]["sequence_phase"],
        "bytes_per_device": records[0]["bytes"],
        "iteration": records[0]["iteration"],
        "devices": sorted(int(record["device"]) for record in records),
        "total_bytes": total_bytes,
        "window_started_ns": started,
        "window_ended_ns": ended,
        "window_duration_ns": window_ns,
        "window_duration_us": window_ns / 1000.0,
        "aggregate_window_gib_s": total_bytes * 1e9 / window_ns / 1024**3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concurrent GB200 Store EGM correctness/performance benchmark"
    )
    parser.add_argument("--node-b-ip", required=True)
    parser.add_argument("--consumer-port-base", type=int, default=12400)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--master-server", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument(
        "--payload-sizes",
        default="134217728,536870912,1073741824,2147483648",
    )
    parser.add_argument("--iterations", type=int, default=13)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--key-prefix", type=safe_identifier, default="egm-gb200")
    parser.add_argument("--run-id", type=safe_identifier, required=True)
    parser.add_argument("--source-sha", type=safe_identifier, required=True)
    parser.add_argument("--threshold-payload-size", type=int, default=MAX_PAYLOAD_SIZE)
    parser.add_argument("--min-put-gib-s", type=nonnegative_float, default=0)
    parser.add_argument("--min-get-gib-s", type=nonnegative_float, default=0)
    args = parser.parse_args()

    devices = parse_integer_list(parser, args.devices, "--devices")
    payload_sizes = parse_integer_list(parser, args.payload_sizes, "--payload-sizes")
    if any(device < 0 for device in devices) or len(set(devices)) != len(devices):
        parser.error("--devices must contain unique nonnegative IDs")
    if any(size <= 0 or size > MAX_PAYLOAD_SIZE for size in payload_sizes):
        parser.error("payload sizes must be in 1..2 GiB")
    if args.warmups < 0 or args.iterations <= args.warmups + 1:
        parser.error("iterations must include one probe, warmups, and steady samples")
    if not 1 <= args.consumer_port_base <= 65535:
        parser.error("--consumer-port-base is outside 1..65535")
    if any(args.consumer_port_base + device > 65535 for device in devices):
        parser.error("generated consumer port is outside 1..65535")
    if (args.min_put_gib_s > 0 or args.min_get_gib_s > 0) and (
        args.threshold_payload_size not in payload_sizes
    ):
        parser.error("threshold payload size must be present in --payload-sizes")

    consumer_script = pathlib.Path(__file__).with_name("egm_store_consumer.py")
    all_results: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    aggregate_records: list[dict[str, object]] = []
    try:
        for size in payload_sizes:
            commands: list[tuple[int, list[str]]] = []
            for device in devices:
                command = [
                    sys.executable,
                    str(consumer_script),
                    "--local-hostname",
                    f"{args.node_b_ip}:{args.consumer_port_base + device}",
                    "--metadata-server",
                    args.metadata_server,
                    "--master-server",
                    args.master_server,
                    "--device",
                    str(device),
                    "--payload-size",
                    str(size),
                    "--iterations",
                    str(args.iterations),
                    "--warmups",
                    str(args.warmups),
                    "--key-prefix",
                    args.key_prefix,
                    "--run-id",
                    args.run_id,
                    "--source-sha",
                    args.source_sha,
                ]
                commands.append((device, command))

            failures: list[str] = []
            size_results: list[dict[str, object]] = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(commands)
            ) as executor:
                futures = [
                    (device, executor.submit(run_consumer, command))
                    for device, command in commands
                ]
                for device, future in futures:
                    try:
                        completed = future.result()
                    except Exception as exc:
                        failures.append(f"GPU {device} launcher failed: {exc}")
                        continue
                    if completed.stderr:
                        print(
                            f"===== consumer GPU {device} stderr, bytes={size} =====",
                            file=sys.stderr,
                        )
                        print(completed.stderr, end="", file=sys.stderr)
                    records = parse_child_records(completed.stdout, device)
                    if completed.returncode != 0:
                        failures.append(
                            f"GPU {device} consumer exited rc={completed.returncode}"
                        )
                        continue
                    try:
                        size_results.extend(
                            validate_consumer_records(
                                records,
                                device,
                                size,
                                args.iterations,
                                args.warmups,
                                args.run_id,
                                args.source_sha,
                            )
                        )
                    except RuntimeError as exc:
                        failures.append(str(exc))
            if failures:
                raise RuntimeError("; ".join(failures))
            all_results.extend(size_results)

            for operation in ("put", "get"):
                for phase in ("lazy_init_probe", "warmup", "steady"):
                    phase_records = [
                        record
                        for record in size_results
                        if record["sequence_phase"] == phase
                    ]
                    summary = sample_summary(phase_records, operation, phase)
                    summary.update(run_id=args.run_id, source_sha=args.source_sha)
                    summaries.append(summary)
                    print(json.dumps(summary, sort_keys=True), flush=True)
                for iteration in range(args.iterations):
                    iteration_records = [
                        record
                        for record in size_results
                        if record["iteration"] == iteration
                    ]
                    aggregate = aggregate_iteration(iteration_records, operation)
                    aggregate.update(run_id=args.run_id, source_sha=args.source_sha)
                    aggregate_records.append(aggregate)
                    print(json.dumps(aggregate, sort_keys=True), flush=True)

        threshold_failures: list[str] = []
        for operation, minimum in (
            ("put", args.min_put_gib_s),
            ("get", args.min_get_gib_s),
        ):
            if minimum <= 0:
                continue
            matches = [
                summary
                for summary in summaries
                if summary["operation"] == operation
                and summary["sequence_phase"] == "steady"
                and summary["bytes"] == args.threshold_payload_size
            ]
            actual = float(matches[0]["bandwidth_p50_gib_s"])
            if actual < minimum:
                threshold_failures.append(
                    f"{operation} steady p50 {actual:.4f} GiB/s < {minimum:.4f} GiB/s"
                )
        if threshold_failures:
            raise RuntimeError("; ".join(threshold_failures))

        emit(
            "benchmark_gate",
            status="PASS",
            run_id=args.run_id,
            source_sha=args.source_sha,
            devices=devices,
            payload_sizes=payload_sizes,
            iterations=args.iterations,
            warmups=args.warmups,
            transfer_samples=len(all_results),
            threshold_payload_size=args.threshold_payload_size,
            min_put_gib_s=args.min_put_gib_s,
            min_get_gib_s=args.min_get_gib_s,
        )
        return 0
    except Exception as exc:
        emit(
            "benchmark_gate",
            status="FAIL",
            run_id=args.run_id,
            source_sha=args.source_sha,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
