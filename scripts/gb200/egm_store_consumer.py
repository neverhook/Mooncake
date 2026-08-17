#!/usr/bin/env python3

"""Validate HBM -> remote EGM -> HBM and record transfer performance."""

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib
import json
import os
import pathlib
import re
import time

try:
    from scripts.gb200.egm_validation_common import ValidationCuda
except ModuleNotFoundError:
    from egm_validation_common import ValidationCuda


MAX_PAYLOAD_SIZE = 2 * 1024**3
DEFAULT_OBJECT_SIZE = 128 * 1024**2


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def import_store_module():
    try:
        return importlib.import_module("store")
    except ModuleNotFoundError as exc:
        if exc.name != "store":
            raise
    return importlib.import_module("mooncake.store")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def payload_size(value: str) -> int:
    number = positive_int(value)
    if number > MAX_PAYLOAD_SIZE:
        raise argparse.ArgumentTypeError("payload must not exceed 2 GiB")
    return number


def safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError(
            "value may contain only letters, digits, dot, underscore, and dash"
        )
    return value


def bandwidth_gib_s(byte_count: int, duration_ns: int) -> float:
    if byte_count <= 0 or duration_ns <= 0:
        raise ValueError("byte count and duration must be positive")
    return byte_count * 1e9 / duration_ns / 1024**3


def partition_buffer(
    pointer: int, size: int, object_size: int
) -> tuple[list[int], list[int]]:
    pointers: list[int] = []
    sizes: list[int] = []
    offset = 0
    while offset < size:
        part_size = min(object_size, size - offset)
        pointers.append(pointer + offset)
        sizes.append(part_size)
        offset += part_size
    return pointers, sizes


def deterministic_payload(size: int, device: int, iteration: int) -> bytes:
    seed = f"egm-store-gb200:{device}:{iteration}:{size}".encode("ascii")
    return hashlib.shake_256(seed).digest(size)


class CudaRuntime:
    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, explicit_library: str | None = None):
        candidates: list[str] = []
        if explicit_library:
            candidates.append(explicit_library)
        configured = os.environ.get("MC_CUDART_LIBRARY")
        if configured:
            candidates.append(configured)
        discovered = ctypes.util.find_library("cudart")
        if discovered:
            candidates.append(discovered)
        candidates.extend(
            [
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so",
                "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so",
            ]
        )
        for pattern in (
            "/usr/local/cuda-*/lib64/libcudart.so",
            "/usr/local/cuda-*/targets/*/lib/libcudart.so",
        ):
            candidates.extend(str(path) for path in pathlib.Path("/").glob(pattern[1:]))

        errors: list[str] = []
        self._library = None
        self.library_path = ""
        for candidate in dict.fromkeys(candidates):
            try:
                self._library = ctypes.CDLL(candidate)
                self.library_path = candidate
                break
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        if self._library is None:
            detail = "; ".join(errors) if errors else "no candidates found"
            raise RuntimeError(
                "libcudart could not be loaded; set MC_CUDART_LIBRARY: " + detail
            )

        library = self._library
        library.cudaGetErrorString.argtypes = [ctypes.c_int]
        library.cudaGetErrorString.restype = ctypes.c_char_p
        library.cudaSetDevice.argtypes = [ctypes.c_int]
        library.cudaSetDevice.restype = ctypes.c_int
        library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        library.cudaMalloc.restype = ctypes.c_int
        library.cudaFree.argtypes = [ctypes.c_void_p]
        library.cudaFree.restype = ctypes.c_int
        library.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        library.cudaMemcpy.restype = ctypes.c_int
        library.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        library.cudaMemset.restype = ctypes.c_int
        library.cudaDeviceSynchronize.argtypes = []
        library.cudaDeviceSynchronize.restype = ctypes.c_int
        library.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.cudaRuntimeGetVersion.restype = ctypes.c_int
        library.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.cudaDriverGetVersion.restype = ctypes.c_int

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        raw = self._library.cudaGetErrorString(result)
        message = raw.decode("utf-8", errors="replace") if raw else "unknown"
        raise RuntimeError(f"{operation} failed: CUDA {result}: {message}")

    def set_device(self, device: int) -> None:
        self._check(self._library.cudaSetDevice(device), "cudaSetDevice")

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self._check(self._library.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        if pointer.value is None:
            raise RuntimeError("cudaMalloc returned null")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        self._check(self._library.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def copy_from_host(self, pointer: int, payload: bytes) -> None:
        host = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        self._check(
            self._library.cudaMemcpy(
                ctypes.c_void_p(pointer),
                ctypes.cast(host, ctypes.c_void_p),
                len(payload),
                self.CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy(H2D)",
        )

    def copy_to_host(self, pointer: int, size: int) -> bytes:
        host = (ctypes.c_ubyte * size)()
        self._check(
            self._library.cudaMemcpy(
                ctypes.cast(host, ctypes.c_void_p),
                ctypes.c_void_p(pointer),
                size,
                self.CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy(D2H)",
        )
        return bytes(host)

    def memset(self, pointer: int, value: int, size: int) -> None:
        self._check(
            self._library.cudaMemset(ctypes.c_void_p(pointer), value, size),
            "cudaMemset",
        )

    def synchronize(self) -> None:
        self._check(self._library.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def _version(self, function_name: str) -> str:
        encoded = ctypes.c_int()
        function = getattr(self._library, function_name)
        self._check(function(ctypes.byref(encoded)), function_name)
        return f"{encoded.value // 1000}.{(encoded.value % 1000) // 10}"

    def runtime_version(self) -> str:
        return self._version("cudaRuntimeGetVersion")

    def driver_version(self) -> str:
        return self._version("cudaDriverGetVersion")


def process_context(cuda: CudaRuntime, device: int, module) -> dict[str, object]:
    status: dict[str, str] = {}
    try:
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"Cpus_allowed_list", "Mems_allowed_list"}:
                status[name] = value.strip()
    except OSError:
        pass
    return {
        "pid": os.getpid(),
        "device": device,
        "cuda_runtime_library": cuda.library_path,
        "cuda_runtime_version": cuda.runtime_version(),
        "cuda_driver_version": cuda.driver_version(),
        "store_module": getattr(
            module, "__file__", getattr(module, "__name__", "unknown")
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "cpus_allowed_list": status.get("Cpus_allowed_list", "unknown"),
        "mems_allowed_list": status.get("Mems_allowed_list", "unknown"),
    }


def run_transfers(
    store,
    cuda: CudaRuntime,
    args: argparse.Namespace,
    source_ptr: int,
    destination_ptr: int,
    validation_cuda: ValidationCuda | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for iteration in range(args.iterations):
        source_parts, part_sizes = partition_buffer(
            source_ptr, args.payload_size, args.object_size
        )
        destination_parts, _ = partition_buffer(
            destination_ptr, args.payload_size, args.object_size
        )
        seed = (
            (args.device + 1) << 56 ^ args.payload_size ^ iteration ^ 0x45474D53544F5245
        )
        if validation_cuda is None:
            expected = deterministic_payload(args.payload_size, args.device, iteration)
            expected_hash = hashlib.sha256(expected).hexdigest()
            cuda.copy_from_host(source_ptr, expected)
            verification = "host_sha256_and_byte_compare"
        else:
            expected = None
            expected_hash = hashlib.sha256(
                f"egm-validation-pattern-v1:{seed}:{args.payload_size}".encode()
            ).hexdigest()
            validation_cuda.fill(source_ptr, args.payload_size, seed, args.device)
            verification = "device_full_pattern_compare"
        cuda.memset(destination_ptr, 0, args.payload_size)
        cuda.synchronize()
        key_prefix = (
            f"{args.key_prefix}-{args.run_id}-gpu{args.device}-"
            f"{args.payload_size}-{iteration}"
        )
        keys = [f"{key_prefix}-part{index}" for index in range(len(part_sizes))]

        put_started_ns = time.perf_counter_ns()
        put_results = store.batch_put_from(keys, source_parts, part_sizes)
        put_ended_ns = time.perf_counter_ns()
        if put_results != [0] * len(keys):
            raise RuntimeError(f"batch_put_from failed for {key_prefix}: {put_results}")

        get_started_ns = time.perf_counter_ns()
        get_results = store.batch_get_into(keys, destination_parts, part_sizes)
        cuda.synchronize()
        get_ended_ns = time.perf_counter_ns()
        if get_results != part_sizes:
            raise RuntimeError(
                f"batch_get_into returned {get_results}, expected {part_sizes}"
            )

        if validation_cuda is None:
            actual = cuda.copy_to_host(destination_ptr, args.payload_size)
            actual_hash = hashlib.sha256(actual).hexdigest()
            if actual_hash != expected_hash or actual != expected:
                raise RuntimeError(f"HBM payload mismatch for {key_prefix}")
            mismatches = 0
        else:
            mismatches = validation_cuda.verify(
                destination_ptr, args.payload_size, seed, args.device
            )
            if mismatches != 0:
                raise RuntimeError(
                    f"HBM payload mismatch for {key_prefix}: {mismatches} words"
                )
            actual_hash = expected_hash
        remove_results = [store.remove(key, True) for key in keys]
        if remove_results != [0] * len(keys):
            raise RuntimeError(f"remove failed for {key_prefix}: {remove_results}")

        put_duration_ns = put_ended_ns - put_started_ns
        get_duration_ns = get_ended_ns - get_started_ns
        record: dict[str, object] = {
            "event": "transfer_result",
            "status": "PASS",
            "run_id": args.run_id,
            "source_sha": args.source_sha,
            "device": args.device,
            "iteration": iteration,
            "sequence_phase": (
                "lazy_init_probe"
                if iteration == 0
                else "warmup"
                if iteration <= args.warmups
                else "steady"
            ),
            "bytes": args.payload_size,
            "payload_semantics": "aggregate_store_object_batch",
            "object_count": len(keys),
            "object_size": args.object_size,
            "sha256": actual_hash,
            "verification": verification,
            "mismatches": mismatches,
            "put_path": "consumer_hbm_to_provider_egm",
            "put_started_ns": put_started_ns,
            "put_ended_ns": put_ended_ns,
            "put_duration_ns": put_duration_ns,
            "put_duration_us": put_duration_ns / 1000.0,
            "put_bandwidth_gib_s": bandwidth_gib_s(args.payload_size, put_duration_ns),
            "get_path": "provider_egm_to_consumer_hbm",
            "get_started_ns": get_started_ns,
            "get_ended_ns": get_ended_ns,
            "get_duration_ns": get_duration_ns,
            "get_duration_us": get_duration_ns / 1000.0,
            "get_bandwidth_gib_s": bandwidth_gib_s(args.payload_size, get_duration_ns),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GB200 HBM <-> remote EGM correctness/performance consumer"
    )
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--master-server", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--payload-size", type=payload_size, default=MAX_PAYLOAD_SIZE)
    parser.add_argument("--object-size", type=positive_int, default=DEFAULT_OBJECT_SIZE)
    parser.add_argument("--iterations", type=positive_int, default=13)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--key-prefix", type=safe_identifier, default="egm-gb200")
    parser.add_argument("--run-id", type=safe_identifier, required=True)
    parser.add_argument("--source-sha", type=safe_identifier, required=True)
    parser.add_argument("--cuda-runtime-library")
    parser.add_argument(
        "--validation-cuda-library",
        default=os.environ.get("MC_EGM_VALIDATION_CUDA_LIBRARY", ""),
    )
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.warmups < 0 or args.iterations <= args.warmups + 1:
        parser.error("--iterations must include one probe, warmups, and steady samples")
    if args.object_size > DEFAULT_OBJECT_SIZE:
        parser.error("--object-size must not exceed 128 MiB")

    cuda = CudaRuntime(args.cuda_runtime_library)
    cuda.set_device(args.device)
    module = import_store_module()
    validation_cuda = (
        ValidationCuda(args.validation_cuda_library)
        if args.validation_cuda_library
        else None
    )
    if args.payload_size > 128 * 1024**2 and validation_cuda is None:
        parser.error("payloads above 128 MiB require --validation-cuda-library")
    store = module.MooncakeDistributedStore()
    config = {
        "local_hostname": args.local_hostname,
        "metadata_server": args.metadata_server,
        "master_server_addr": args.master_server,
        "global_segment_size": "0",
        "local_buffer_size": "0",
        "protocol": "nvlink",
    }
    emit(
        "consumer_config",
        run_id=args.run_id,
        source_sha=args.source_sha,
        device=args.device,
        payload_size=args.payload_size,
        payload_semantics="aggregate_store_object_batch",
        object_size=args.object_size,
        iterations=args.iterations,
        hbm_allocation="cudaMalloc local endpoint (not published)",
        config=config,
    )

    pointers: list[int] = []
    setup_attempted = False
    exit_code = 0
    try:
        setup_attempted = True
        setup_result = store.setup(config)
        if setup_result != 0:
            raise RuntimeError(f"consumer setup failed: {setup_result}")
        emit(
            "consumer_context",
            run_id=args.run_id,
            source_sha=args.source_sha,
            **process_context(cuda, args.device, module),
        )
        pointers = [cuda.malloc(args.payload_size), cuda.malloc(args.payload_size)]
        records = run_transfers(
            store, cuda, args, pointers[0], pointers[1], validation_cuda
        )
        emit(
            "consumer_gate",
            status="PASS",
            run_id=args.run_id,
            source_sha=args.source_sha,
            device=args.device,
            payload_size=args.payload_size,
            samples=len(records),
        )
    except Exception as exc:
        emit(
            "consumer_gate",
            status="FAIL",
            run_id=args.run_id,
            source_sha=args.source_sha,
            device=args.device,
            payload_size=args.payload_size,
            error=str(exc),
        )
        exit_code = 1
    finally:
        close_result: object = 0
        if setup_attempted:
            try:
                close_result = store.close()
            except Exception as exc:
                close_result = f"exception: {exc}"
        close_ok = close_result in (None, 0)
        free_errors: list[str] = []
        for pointer in reversed(pointers):
            try:
                cuda.free(pointer)
            except Exception as exc:
                free_errors.append(f"cudaFree({pointer:#x}) failed: {exc}")
        cleanup_ok = close_ok and not free_errors
        emit(
            "consumer_cleanup",
            status="PASS" if cleanup_ok else "FAIL",
            run_id=args.run_id,
            source_sha=args.source_sha,
            device=args.device,
            close_result=close_result,
            errors=free_errors,
        )
        if not cleanup_ok and exit_code == 0:
            exit_code = 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
