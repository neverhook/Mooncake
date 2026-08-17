#!/usr/bin/env python3

"""Shared configuration, evidence, and CUDA helpers for GB200 validation."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pathlib
import re
import socket
import statistics
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping


EVIDENCE_SCHEMA = "MOONCAKE_GB200_EVIDENCE_V1"
EXPECTED_GB200_NVLINKS = 18
EXPECTED_GB200_C2C_LINKS = 5


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


def parse_nvlink_data(output: str) -> dict[str, int]:
    values: dict[str, int] = {}
    gpu: int | None = None
    for line in output.splitlines():
        gpu_match = re.match(r"^GPU\s+(\d+):", line)
        if gpu_match is not None:
            gpu = int(gpu_match.group(1))
            continue
        data_match = re.match(
            r"^\s*Link\s+(\d+):\s+Data\s+(Tx|Rx):\s+([0-9,]+)\s+KiB\s*$",
            line,
            re.IGNORECASE,
        )
        if gpu is None or data_match is None:
            continue
        link = int(data_match.group(1))
        direction = data_match.group(2).lower()
        values[f"gpu{gpu}/link{link}/{direction}_bytes"] = (
            int(data_match.group(3).replace(",", "")) * 1024
        )
    return values


def parse_c2c_status(output: str) -> dict[str, float]:
    values: dict[str, float] = {}
    gpu: int | None = None
    for line in output.splitlines():
        gpu_match = re.match(r"^GPU\s+(\d+):", line)
        if gpu_match is not None:
            gpu = int(gpu_match.group(1))
            continue
        link_match = re.match(r"^\s*C2C Link\s+(\d+):\s+([0-9.]+)\s+GB/s\s*$", line)
        if gpu is not None and link_match is not None:
            values[f"gpu{gpu}/link{int(link_match.group(1))}/capacity_gb_s"] = float(
                link_match.group(2)
            )
    return values


def parse_c2c_errors(output: str) -> dict[str, int]:
    values: dict[str, int] = {}
    gpu: int | None = None
    for line in output.splitlines():
        gpu_match = re.match(r"^GPU\s+(\d+):", line)
        if gpu_match is not None:
            gpu = int(gpu_match.group(1))
            continue
        error_match = re.match(
            r"^\s*C2C Link\s+(\d+):\s+Error\s+(.+?)\s+Count:\s+([0-9,]+)\s*$",
            line,
        )
        if gpu is None or error_match is None:
            continue
        error_name = re.sub(r"[^a-z0-9]+", "_", error_match.group(2).lower()).strip("_")
        values[f"gpu{gpu}/link{int(error_match.group(1))}/{error_name}_errors"] = int(
            error_match.group(3).replace(",", "")
        )
    return values


def parse_fabric(output: str) -> dict[str, dict[str, str]]:
    pattern = re.compile(
        r"^\s+Fabric\s*$\n"
        r"\s+State\s+:\s+([^\n]+)\n"
        r"\s+Status\s+:\s+([^\n]+)\n"
        r"(?:\s+CliqueId\s+:\s+[^\n]+\n)?"
        r"(?:\s+ClusterUUID\s+:\s+[^\n]+\n)?"
        r"\s+Health\s*$\n"
        r"\s+Summary\s+:\s+([^\n]+)\n"
        r"\s+Bandwidth\s+:\s+([^\n]+)\n"
        r"\s+Route Recovery in progress\s+:\s+([^\n]+)",
        re.MULTILINE,
    )
    return {
        f"gpu{gpu}": {
            "state": match.group(1).strip(),
            "status": match.group(2).strip(),
            "health": match.group(3).strip(),
            "bandwidth": match.group(4).strip(),
            "route_recovery": match.group(5).strip(),
        }
        for gpu, match in enumerate(pattern.finditer(output))
    }


def _capture(command: list[str]) -> tuple[str, str]:
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
        return "UNAVAILABLE", str(exc)
    return ("PASS" if result.returncode == 0 else "FAIL"), result.stdout


def collect_route_snapshot() -> dict[str, object]:
    commands = {
        "nvlink_data": ["nvidia-smi", "nvlink", "-gt", "d"],
        "c2c_status": ["nvidia-smi", "c2c", "-s"],
        "c2c_errors": ["nvidia-smi", "c2c", "-e"],
        "fabric": ["nvidia-smi", "-q"],
    }
    outputs: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for name, command in commands.items():
        statuses[name], outputs[name] = _capture(command)
    return {
        "backend": "nvidia-smi",
        "command_status": statuses,
        "nvlink_bytes": parse_nvlink_data(outputs["nvlink_data"]),
        "c2c_capacity_gb_s": parse_c2c_status(outputs["c2c_status"]),
        "c2c_errors": parse_c2c_errors(outputs["c2c_errors"]),
        "fabric": parse_fabric(outputs["fabric"]),
    }


def route_snapshot_errors(snapshot: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    command_status = snapshot.get("command_status", {})
    if not isinstance(command_status, Mapping) or any(
        value != "PASS" for value in command_status.values()
    ):
        errors.append(f"nvidia-smi command failure: {dict(command_status)}")

    nvlink = snapshot.get("nvlink_bytes", {})
    c2c_capacity = snapshot.get("c2c_capacity_gb_s", {})
    c2c_errors = snapshot.get("c2c_errors", {})
    fabric = snapshot.get("fabric", {})
    if not all(
        isinstance(value, Mapping)
        for value in (nvlink, c2c_capacity, c2c_errors, fabric)
    ):
        return [*errors, "route snapshot has invalid field types"]

    gpu_names = sorted({name.split("/", 1)[0] for name in nvlink})
    if not gpu_names:
        errors.append("no NVLink byte counters were parsed")
    for gpu in gpu_names:
        links = {
            name.split("/")[1]
            for name in nvlink
            if name.startswith(f"{gpu}/") and name.endswith("_bytes")
        }
        tx = [
            name
            for name in nvlink
            if name.startswith(f"{gpu}/") and name.endswith("/tx_bytes")
        ]
        rx = [
            name
            for name in nvlink
            if name.startswith(f"{gpu}/") and name.endswith("/rx_bytes")
        ]
        if (
            len(links) != EXPECTED_GB200_NVLINKS
            or len(tx) != len(links)
            or len(rx) != len(links)
        ):
            errors.append(f"{gpu} NVLink counters are incomplete")
        c2c_links = {
            name.split("/")[1] for name in c2c_capacity if name.startswith(f"{gpu}/")
        }
        if len(c2c_links) != EXPECTED_GB200_C2C_LINKS or any(
            float(value) <= 0
            for name, value in c2c_capacity.items()
            if name.startswith(f"{gpu}/")
        ):
            errors.append(f"{gpu} C2C capability is incomplete")
        expected_error_fields = EXPECTED_GB200_C2C_LINKS * 3
        actual_error_fields = sum(name.startswith(f"{gpu}/") for name in c2c_errors)
        if actual_error_fields != expected_error_fields:
            errors.append(f"{gpu} C2C error counters are incomplete")
        fabric_state = fabric.get(gpu)
        if not isinstance(fabric_state, Mapping) or fabric_state != {
            "state": "Completed",
            "status": "Success",
            "health": "Healthy",
            "bandwidth": "Full",
            "route_recovery": "False",
        }:
            errors.append(f"{gpu} Fabric is not completed, healthy, and full-bandwidth")
    if set(fabric) != set(gpu_names):
        errors.append("Fabric GPU set does not match NVLink counters")
    return errors


def build_route_evidence(
    before: Mapping[str, object],
    after: Mapping[str, object],
    semantic: str,
) -> dict[str, object]:
    diagnostics = [
        *(f"before: {error}" for error in route_snapshot_errors(before)),
        *(f"after: {error}" for error in route_snapshot_errors(after)),
    ]
    before_nvlink = before.get("nvlink_bytes", {})
    after_nvlink = after.get("nvlink_bytes", {})
    before_errors = before.get("c2c_errors", {})
    after_errors = after.get("c2c_errors", {})
    nvlink_delta: dict[str, int] = {}
    c2c_error_delta: dict[str, int] = {}
    if isinstance(before_nvlink, Mapping) and isinstance(after_nvlink, Mapping):
        nvlink_delta = {
            str(name): int(value) - int(before_nvlink[name])
            for name, value in after_nvlink.items()
            if name in before_nvlink
        }
    if isinstance(before_errors, Mapping) and isinstance(after_errors, Mapping):
        c2c_error_delta = {
            str(name): int(value) - int(before_errors[name])
            for name, value in after_errors.items()
            if name in before_errors
        }

    gpu_names = sorted({name.split("/", 1)[0] for name in nvlink_delta})
    byte_delta_by_gpu = {
        gpu: {
            direction: sum(
                value
                for name, value in nvlink_delta.items()
                if name.startswith(f"{gpu}/") and name.endswith(f"/{direction}_bytes")
            )
            for direction in ("tx", "rx")
        }
        for gpu in gpu_names
    }
    byte_delta_by_direction = {
        direction: sum(
            value
            for name, value in nvlink_delta.items()
            if name.endswith(f"/{direction}_bytes")
        )
        for direction in ("tx", "rx")
    }
    error_delta_by_gpu = {
        gpu: sum(
            value
            for name, value in c2c_error_delta.items()
            if name.startswith(f"{gpu}/")
        )
        for gpu in gpu_names
    }
    if any(value < 0 for value in nvlink_delta.values()):
        diagnostics.append("an NVLink byte counter decreased")
    if not all(value > 0 for value in byte_delta_by_direction.values()):
        diagnostics.append("both NVLink TX and RX byte deltas must be positive")
    if any(value != 0 for value in c2c_error_delta.values()):
        diagnostics.append("a C2C error counter changed")

    c2c_capacity = after.get("c2c_capacity_gb_s", {})
    c2c_capacity_by_gpu: dict[str, float] = {}
    if isinstance(c2c_capacity, Mapping):
        c2c_capacity_by_gpu = {
            gpu: sum(
                float(value)
                for name, value in c2c_capacity.items()
                if name.startswith(f"{gpu}/")
            )
            for gpu in gpu_names
        }
    return {
        "status": "PASS" if not diagnostics else "FAIL",
        "route_verification": "C2C_ROUTE_INFERRED",
        "counter_backend": "nvidia-smi",
        "direct_c2c_byte_counter": "UNAVAILABLE",
        "nvlink_byte_delta_by_direction": byte_delta_by_direction,
        "nvlink_byte_delta_by_gpu": byte_delta_by_gpu,
        "c2c_reported_link_capacity_sum_gb_s_by_gpu": c2c_capacity_by_gpu,
        "c2c_error_delta_by_gpu": error_delta_by_gpu,
        "fabric": after.get("fabric", {}),
        "diagnostics": diagnostics,
        "semantic": semantic,
    }


def collect_system() -> dict[str, object]:
    route = collect_route_snapshot()
    capacity = route["c2c_capacity_gb_s"]
    c2c_errors = route["c2c_errors"]
    gpu_names = sorted({name.split("/", 1)[0] for name in capacity})
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
        "nvlink_status": command_output(["nvidia-smi", "nvlink", "--status"]),
        "route_observation": {
            "backend": route["backend"],
            "command_status": route["command_status"],
            "diagnostics": route_snapshot_errors(route),
            "nvlink_byte_counter_count": len(route["nvlink_bytes"]),
            "c2c_reported_link_capacity_sum_gb_s_by_gpu": {
                gpu: sum(
                    float(value)
                    for name, value in capacity.items()
                    if name.startswith(f"{gpu}/")
                )
                for gpu in gpu_names
            },
            "c2c_error_count_by_gpu": {
                gpu: sum(
                    int(value)
                    for name, value in c2c_errors.items()
                    if name.startswith(f"{gpu}/")
                )
                for gpu in gpu_names
            },
            "fabric": route["fabric"],
        },
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
