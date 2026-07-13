#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import math
import pathlib
import statistics
import subprocess
import sys


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


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


def validate_endpoint(parser: argparse.ArgumentParser, endpoint: str) -> None:
    host, separator, port_text = endpoint.rpartition(":")
    if not separator or not host or not port_text.isdigit():
        parser.error(f"generated local hostname is not host:port: {endpoint!r}")
    port = int(port_text)
    if not 1 <= port <= 65535:
        parser.error(f"generated local hostname port is out of range: {endpoint!r}")


def run_consumer(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def result_records(
    stdout: str, device: int, payload_size: int, iterations: int
) -> list[dict[str, object]]:
    records = []
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("event") == "result":
            records.append(record)
    expected_iterations = set(range(iterations))
    actual_iterations = {record.get("iteration") for record in records}
    if len(records) != iterations or actual_iterations != expected_iterations:
        raise RuntimeError(
            f"consumer GPU {device} emitted {len(records)} result records; "
            f"iterations={sorted(actual_iterations, key=str)!r}, "
            f"expected={sorted(expected_iterations)!r}"
        )
    for record in records:
        if record.get("device") != device or record.get("bytes") != payload_size:
            raise RuntimeError(
                f"consumer GPU {device} emitted mismatched result record: {record!r}"
            )
        iteration = record["iteration"]
        expected_phase = "cold" if iteration == 0 else "warm"
        if record.get("cache_phase") != expected_phase:
            raise RuntimeError(
                f"consumer GPU {device} emitted invalid cache phase: {record!r}"
            )
        for field in (
            "put_latency_ns",
            "get_latency_ns",
            "put_gib_s",
            "get_gib_s",
        ):
            try:
                value = float(record[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"consumer GPU {device} emitted invalid {field}: {record!r}"
                ) from exc
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(
                    f"consumer GPU {device} emitted nonpositive {field}: {record!r}"
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent GB200 HOST_NUMA benchmark")
    parser.add_argument("--local-hostname-prefix", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--master-server", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--payload-sizes", default="4096,1048576,16777216")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--key-prefix", default="nvlink-host-numa-bench")
    args = parser.parse_args()

    consumer_script = pathlib.Path(__file__).with_name("nvlink_host_numa_consumer.py")
    devices = parse_integer_list(parser, args.devices, "--devices")
    sizes = parse_integer_list(parser, args.payload_sizes, "--payload-sizes")
    if any(device < 0 for device in devices):
        parser.error("--devices values must be nonnegative")
    if len(set(devices)) != len(devices):
        parser.error("--devices values must be unique")
    if any(size <= 0 for size in sizes):
        parser.error("--payload-sizes values must be greater than zero")
    if args.iterations < 2:
        parser.error("--iterations must be at least two (cold and warm)")
    endpoints = {device: f"{args.local_hostname_prefix}{device}" for device in devices}
    for endpoint in endpoints.values():
        validate_endpoint(parser, endpoint)
    if len(set(endpoints.values())) != len(endpoints):
        parser.error("--local-hostname-prefix generated duplicate endpoints")
    all_results: list[dict[str, object]] = []

    for payload_size in sizes:
        commands: list[tuple[int, list[str]]] = []
        for device in devices:
            command = [
                sys.executable,
                str(consumer_script),
                "--local-hostname",
                endpoints[device],
                "--metadata-server",
                args.metadata_server,
                "--master-server",
                args.master_server,
                "--device",
                str(device),
                "--payload-size",
                str(payload_size),
                "--iterations",
                str(args.iterations),
                "--key-prefix",
                args.key_prefix,
            ]
            commands.append((device, command))

        failures = []
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
                    failures.append(f"consumer GPU {device} could not run: {exc}")
                    continue
                print(completed.stdout, end="")
                if completed.returncode != 0:
                    failures.append(
                        f"consumer GPU {device} failed with rc={completed.returncode}"
                    )
                    continue
                try:
                    all_results.extend(
                        result_records(
                            completed.stdout, device, payload_size, args.iterations
                        )
                    )
                except RuntimeError as exc:
                    failures.append(str(exc))
        if failures:
            raise RuntimeError("; ".join(failures))

    for payload_size in sizes:
        records = [record for record in all_results if record["bytes"] == payload_size]
        expected_samples = len(devices) * args.iterations
        if len(records) != expected_samples:
            raise RuntimeError(
                f"payload {payload_size} has {len(records)} samples, "
                f"expected {expected_samples}"
            )
        for operation in ("put", "get"):
            latency_us = [
                float(record[f"{operation}_latency_ns"]) / 1000 for record in records
            ]
            throughput = [float(record[f"{operation}_gib_s"]) for record in records]
            summary = {
                "event": "summary",
                "operation": operation,
                "bytes": payload_size,
                "samples": len(records),
                "latency_p50_us": percentile(latency_us, 0.50),
                "latency_p95_us": percentile(latency_us, 0.95),
                "latency_p99_us": percentile(latency_us, 0.99),
                "throughput_mean_gib_s": statistics.fmean(throughput)
                if throughput
                else 0.0,
            }
            print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
