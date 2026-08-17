#!/usr/bin/env python3

"""Shared configuration, evidence, and CUDA helpers for GB200 validation."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pathlib
import socket
import statistics
import subprocess
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping


EVIDENCE_SCHEMA = "MOONCAKE_GB200_EVIDENCE_V1"
DCGM_NVLINK_ERROR_FIELDS = list(range(1204, 1220))
DCGM_NVLINK_COUNT_FIELDS = [1201, 1203, *DCGM_NVLINK_ERROR_FIELDS]
DCGM_C2C_PROFILE_FIELDS = [1077, 1079]


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": 0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "cv": 0.0,
        }
    mean = statistics.fmean(samples)
    deviation = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return {
        "samples": len(samples),
        "min": min(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "max": max(samples),
        "mean": mean,
        "cv": deviation / mean if mean > 0 else 0.0,
    }


def config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def route_address(peer: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((peer, 9))
        return str(sock.getsockname()[0])


def free_port(address: str, start: int, used: set[int] | None = None) -> int:
    reserved = used if used is not None else set()
    for port in range(start, min(start + 1000, 65536)):
        if port in reserved:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((address, port))
            except OSError:
                continue
        reserved.add(port)
        return port
    raise RuntimeError(f"no free TCP port found from {start}")


def visible_devices() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    devices = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not devices:
        raise RuntimeError("nvidia-smi reported no visible GPUs")
    return devices


def metadata_url(base: str, key: str) -> str:
    return f"{base.rstrip('/')}?{urllib.parse.urlencode({'key': key})}"


def metadata_put(base: str, key: str, value: Mapping[str, object]) -> None:
    request = urllib.request.Request(
        metadata_url(base, key),
        data=json.dumps(value, sort_keys=True).encode(),
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def metadata_get(base: str, key: str) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(metadata_url(base, key), timeout=10) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if not isinstance(value, dict):
        raise RuntimeError(f"metadata key {key!r} did not contain an object")
    return value


def metadata_delete(base: str, key: str) -> None:
    request = urllib.request.Request(metadata_url(base, key), method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise


def read_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(root: pathlib.Path) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    if not root.exists():
        return artifacts
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        artifacts.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def command_output(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "status": "UNAVAILABLE", "output": str(exc)}
    return {
        "command": command,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "output": result.stdout[-32768:],
    }


def collect_rdma_counters() -> dict[str, int]:
    counters: dict[str, int] = {}
    root = pathlib.Path("/sys/class/infiniband")
    if not root.exists():
        return counters
    names = {
        "port_xmit_data",
        "port_rcv_data",
        "port_xmit_packets",
        "port_rcv_packets",
        "port_xmit_discards",
        "port_rcv_errors",
        "symbol_error",
        "link_downed",
        "link_error_recovery",
    }
    for device in sorted(root.iterdir()):
        ports = device / "ports"
        if not ports.exists():
            continue
        for port in sorted(ports.iterdir()):
            for directory in (port / "counters", port / "hw_counters"):
                if not directory.exists():
                    continue
                for path in directory.iterdir():
                    if path.name not in names:
                        continue
                    try:
                        counters[f"{device.name}/port{port.name}/{path.name}"] = int(
                            path.read_text().strip()
                        )
                    except (OSError, ValueError):
                        continue
    return counters


def counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    return {
        name: int(value) - int(before.get(name, value)) for name, value in after.items()
    }


def parse_dcgm_dmon(output: str, field_ids: Iterable[int]) -> dict[str, list[float]]:
    """Parse the stable `dcgmi dmon` GPU row format without relying on headers."""
    fields = list(field_ids)
    values: dict[str, list[float]] = {}
    for line in output.splitlines():
        match = re.match(r"^\s*GPU\s+(\d+)\s+(.+?)\s*$", line, re.IGNORECASE)
        if match is None:
            continue
        tokens = match.group(2).split()
        if len(tokens) < len(fields):
            continue
        gpu = int(match.group(1))
        for field, token in zip(fields, tokens, strict=False):
            try:
                number = float(token.replace(",", ""))
            except ValueError:
                continue
            values.setdefault(f"gpu{gpu}/field{field}", []).append(number)
    return values


def collect_dcgm_count_snapshot() -> dict[str, float]:
    command = [
        "dcgmi",
        "dmon",
        "-e",
        ",".join(str(field) for field in DCGM_NVLINK_COUNT_FIELDS),
        "-c",
        "1",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    parsed = parse_dcgm_dmon(result.stdout, DCGM_NVLINK_COUNT_FIELDS)
    return {name: samples[-1] for name, samples in parsed.items() if samples}


def dcgm_count_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float]:
    return {
        name: float(value) - float(before[name])
        for name, value in after.items()
        if name in before
    }


def dcgm_fields_present(
    snapshot: Mapping[str, float], field_ids: Iterable[int]
) -> bool:
    names = tuple(snapshot)
    return all(
        any(name.endswith(f"field{field_id}") for name in names)
        for field_id in field_ids
    )


def collect_system() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,pci.bus_id,clocks.current.sm,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "fabric": command_output(["nvidia-smi", "-q", "-d", "FABRIC"]),
        "nvlink": command_output(["nvidia-smi", "nvlink", "--status"]),
        "dcgm": command_output(
            [
                "dcgmi",
                "dmon",
                "-e",
                ",".join(str(field) for field in DCGM_NVLINK_COUNT_FIELDS),
                "-c",
                "1",
            ]
        ),
        "rdma_counters": collect_rdma_counters(),
    }


def emit_evidence(evidence: Mapping[str, object]) -> None:
    role = str(evidence["role"])
    print(f"BEGIN_{EVIDENCE_SCHEMA} NODE={role}")
    print(json.dumps(dict(evidence), sort_keys=True, separators=(",", ":")))
    print(f"END_{EVIDENCE_SCHEMA}")


class ValidationCuda:
    """ctypes wrapper around libegm_validation_cuda.so."""

    def __init__(self, library_path: str):
        self.library_path = library_path
        self._library = ctypes.CDLL(library_path)
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        uint64 = ctypes.c_uint64
        integer = ctypes.c_int
        self._library.egmValidationFill.argtypes = [pointer, size, uint64, integer]
        self._library.egmValidationFill.restype = integer
        self._library.egmValidationVerify.argtypes = [
            pointer,
            size,
            uint64,
            integer,
            ctypes.POINTER(uint64),
        ]
        self._library.egmValidationVerify.restype = integer

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {result}")

    def fill(self, address: int, length: int, seed: int, device: int) -> None:
        self._check(
            self._library.egmValidationFill(address, length, seed, device),
            "egmValidationFill",
        )

    def verify(self, address: int, length: int, seed: int, device: int) -> int:
        mismatches = ctypes.c_uint64()
        self._check(
            self._library.egmValidationVerify(
                address, length, seed, device, ctypes.byref(mismatches)
            ),
            "egmValidationVerify",
        )
        return int(mismatches.value)
