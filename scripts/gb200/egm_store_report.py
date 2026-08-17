#!/usr/bin/env python3

"""Render Provider readiness and benchmark JSONL as a PR-ready Markdown report."""

import argparse
import json
import pathlib


def read_object(path: pathlib.Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSON at {path}:{number}")
        records.append(value)
    return records


def read_mixed_log(path: pathlib.Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def format_bytes(value: int) -> str:
    for unit, factor in (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if value >= factor and value % factor == 0:
            return f"{value // factor} {unit}"
    return f"{value} B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render GB200 Store EGM PR evidence")
    parser.add_argument("--provider-ready", type=pathlib.Path, required=True)
    parser.add_argument("--provider-log", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-log", type=pathlib.Path, required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()

    provider = read_object(args.provider_ready)
    provider_records = read_mixed_log(args.provider_log)
    records = read_jsonl(args.benchmark_log)
    gates = [record for record in records if record.get("event") == "benchmark_gate"]
    if provider.get("event") != "provider_ready" or provider.get("status") != "PASS":
        raise SystemExit("Provider readiness evidence is not PASS")
    if len(gates) != 1 or gates[0].get("status") != "PASS":
        raise SystemExit("Benchmark must contain exactly one PASS gate")
    gate = gates[0]
    if provider.get("run_id") != gate.get("run_id") or provider.get(
        "source_sha"
    ) != gate.get("source_sha"):
        raise SystemExit("Provider and Consumer provenance do not match")
    cleanup = [
        record
        for record in provider_records
        if record.get("event") == "provider_cleanup"
        and record.get("run_id") == gate.get("run_id")
        and record.get("source_sha") == gate.get("source_sha")
    ]
    unpublished = [
        record
        for record in provider_records
        if record.get("event") == "provider_unpublished"
        and record.get("run_id") == gate.get("run_id")
        and record.get("source_sha") == gate.get("source_sha")
    ]
    if len(cleanup) != 1 or cleanup[0].get("status") != "PASS":
        raise SystemExit("Provider cleanup evidence is not exactly one PASS record")
    if len(unpublished) != 1 or unpublished[0].get("status") != "PASS":
        raise SystemExit(
            "Provider unpublication evidence is not exactly one PASS record"
        )

    summaries = [
        record for record in records if record.get("event") == "performance_summary"
    ]
    aggregates = [
        record for record in records if record.get("event") == "aggregate_iteration"
    ]
    if not summaries or not aggregates:
        raise SystemExit("Benchmark performance summaries are incomplete")

    print("## GB200/NVL72 Store EGM validation")
    print()
    print("- Result: **PASS**")
    print(f"- Branch: `{args.branch}`")
    print(f"- Source SHA: `{gate['source_sha']}`")
    print(f"- Run ID: `{gate['run_id']}`")
    print(
        "- Provider capacity: "
        f"requested `{provider['requested_capacity_bytes']}` bytes, "
        f"effective `{provider['effective_capacity_bytes']}` bytes, "
        f"chunks `{provider['master_chunk_count']}`"
    )
    print(f"- Consumer GPUs: `{gate['devices']}`")
    print(f"- Correct transfer samples: `{gate['transfer_samples']}`")
    print(
        "- Provider teardown: `PASS`, cleanup "
        f"`{float(cleanup[0]['duration_ns']) / 1e6:.3f}` ms, "
        "all Provider chunks removed from Master"
    )
    if (
        float(gate.get("min_put_gib_s", 0)) > 0
        or float(gate.get("min_get_gib_s", 0)) > 0
    ):
        print(
            "- Enforced steady-state thresholds at "
            f"`{format_bytes(int(gate['threshold_payload_size']))}`: "
            f"Put `{float(gate['min_put_gib_s']):.4f}` GiB/s, "
            f"Get `{float(gate['min_get_gib_s']):.4f}` GiB/s"
        )
    else:
        print(
            "- Bandwidth thresholds: not configured; performance is recorded as evidence"
        )
    print()
    print(
        "| Payload | Path | Sequence | Samples | Transfer p50 (us) | "
        "Per-stream p50 (GiB/s) | Concurrent-window p50 (GiB/s) |"
    )
    print("|---:|---|---|---:|---:|---:|---:|")
    for summary in sorted(
        summaries,
        key=lambda item: (
            int(item["bytes"]),
            str(item["operation"]),
            str(item["sequence_phase"]),
        ),
    ):
        matching = [
            float(record["aggregate_window_gib_s"])
            for record in aggregates
            if record["bytes_per_device"] == summary["bytes"]
            and record["operation"] == summary["operation"]
            and record["sequence_phase"] == summary["sequence_phase"]
        ]
        print(
            f"| {format_bytes(int(summary['bytes']))} "
            f"| `{summary['path']}` "
            f"| {summary['sequence_phase']} "
            f"| {summary['samples']} "
            f"| {float(summary['duration_p50_us']):.3f} "
            f"| {float(summary['bandwidth_p50_gib_s']):.4f} "
            f"| {percentile(matching, 0.50):.4f} |"
        )
    print()
    print(
        "All samples performed SHA-256 and byte-for-byte verification. Put measures "
        "Consumer HBM to Provider EGM; Get measures Provider EGM to Consumer HBM "
        "including destination CUDA synchronization. `lazy_init_probe`, `warmup`, "
        "and `steady` explicitly separate transport initialization from reported "
        "performance samples. Concurrent-window "
        "bandwidth is total bytes divided by the earliest-start/latest-end operation "
        "window across Consumer processes on the same node."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
