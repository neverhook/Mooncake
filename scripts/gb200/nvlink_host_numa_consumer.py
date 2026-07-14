#!/usr/bin/env python3

import argparse
import contextlib
import ctypes
import ctypes.util
import hashlib
import importlib
import json
import os
import pathlib
import re
import time
import uuid

from nvlink_host_numa_metrics import (
    classify_cache_phase,
    consumer_delta,
    consumer_delta_dict,
    consumer_metrics,
    validate_miss_then_hit,
)


MAX_PAYLOAD_SIZE = 128 * 1024 * 1024


def import_store_module():
    try:
        return importlib.import_module("store")
    except ModuleNotFoundError as exc:
        if exc.name != "store":
            raise
    return importlib.import_module("mooncake.store")


class CudaRuntime:
    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, library: str | None = None):
        candidates: list[str] = []
        if library:
            candidates.append(library)
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
                "CUDA runtime library libcudart could not be loaded; "
                "set MC_CUDART_LIBRARY to its path: " + detail
            )

        self._library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self._library.cudaGetErrorString.restype = ctypes.c_char_p
        self._library.cudaSetDevice.argtypes = [ctypes.c_int]
        self._library.cudaSetDevice.restype = ctypes.c_int
        self._library.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self._library.cudaMalloc.restype = ctypes.c_int
        self._library.cudaFree.argtypes = [ctypes.c_void_p]
        self._library.cudaFree.restype = ctypes.c_int
        self._library.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._library.cudaMemcpy.restype = ctypes.c_int
        self._library.cudaMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self._library.cudaMemset.restype = ctypes.c_int
        self._library.cudaDeviceSynchronize.argtypes = []
        self._library.cudaDeviceSynchronize.restype = ctypes.c_int
        self._library.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._library.cudaRuntimeGetVersion.restype = ctypes.c_int
        self._library.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._library.cudaDriverGetVersion.restype = ctypes.c_int

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        error = self._library.cudaGetErrorString(result)
        message = error.decode("utf-8", errors="replace") if error else "unknown"
        raise RuntimeError(f"{operation} failed: CUDA {result}: {message}")

    def set_device(self, device: int) -> None:
        self._check(self._library.cudaSetDevice(device), "cudaSetDevice")

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self._check(self._library.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        if pointer.value is None:
            raise RuntimeError("cudaMalloc returned a null pointer")
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


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def payload_size(value: str) -> int:
    number = positive_int(value)
    if number > MAX_PAYLOAD_SIZE:
        raise argparse.ArgumentTypeError("payload size must not exceed 128 MB")
    return number


def at_least_two(value: str) -> int:
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("value must be at least two")
    return number


def safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError(
            "value may contain only letters, digits, dot, underscore, and dash"
        )
    return value


def process_context(cuda, args: argparse.Namespace, store_module) -> dict[str, object]:
    status: dict[str, str] = {}
    try:
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"Cpus_allowed_list", "Mems_allowed_list"}:
                status[name] = value.strip()
    except OSError:
        pass
    return {
        "event": "consumer_context",
        "run_id": args.run_id,
        "pid": os.getpid(),
        "device": args.device,
        "cuda_runtime_library": cuda.library_path,
        "cuda_runtime_version": cuda.runtime_version(),
        "cuda_driver_version": cuda.driver_version(),
        "store_module": getattr(store_module, "__file__", store_module.__name__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "cpus_allowed_list": status.get("Cpus_allowed_list", "unknown"),
        "mems_allowed_list": status.get("Mems_allowed_list", "unknown"),
    }


def deterministic_payload(size: int, device: int, iteration: int) -> bytes:
    seed = f"nvlink-host-numa:{device}:{iteration}".encode("ascii")
    return hashlib.shake_256(seed).digest(size)


class HbmAllocationQuarantine:
    def __init__(self):
        self._pointers: set[int] = set()

    def retain(self, pointer: int) -> None:
        self._pointers.add(pointer)

    def contains(self, pointer: int) -> bool:
        return pointer in self._pointers

    def release_after_store_close(self, cuda) -> list[str]:
        errors = []
        for pointer in list(self._pointers):
            try:
                cuda.free(pointer)
                self._pointers.remove(pointer)
            except Exception as exc:
                errors.append(f"quarantined cudaFree raised {exc}")
        return errors

    def __len__(self) -> int:
        return len(self._pointers)


@contextlib.contextmanager
def allocated_hbm_buffers(cuda, sizes, quarantine: HbmAllocationQuarantine):
    allocated: list[int] = []
    body_failed = False
    try:
        for size in sizes:
            allocated.append(cuda.malloc(size))
        yield allocated
    except BaseException:
        body_failed = True
        raise
    finally:
        cleanup_errors = []
        for pointer in reversed(allocated):
            if quarantine.contains(pointer):
                continue
            try:
                cuda.free(pointer)
            except Exception as exc:
                quarantine.retain(pointer)
                cleanup_errors.append(f"cudaFree raised {exc}")
        if cleanup_errors:
            print(
                json.dumps({"event": "cleanup_failed", "errors": cleanup_errors}),
                flush=True,
            )
            if not body_failed:
                raise RuntimeError("; ".join(cleanup_errors))


@contextlib.contextmanager
def registered_buffers(store, buffers, quarantine: HbmAllocationQuarantine):
    registered: list[tuple[int, str]] = []
    body_failed = False
    try:
        for pointer, size, name in buffers:
            result = store.register_buffer(pointer, size)
            if result != 0:
                raise RuntimeError(f"{name} HBM registration failed: {result}")
            registered.append((pointer, name))
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        cleanup_errors = []
        for pointer, name in reversed(registered):
            try:
                result = store.unregister_buffer(pointer)
                if result != 0:
                    quarantine.retain(pointer)
                    cleanup_errors.append(f"{name} unregister returned {result}")
            except Exception as exc:
                quarantine.retain(pointer)
                cleanup_errors.append(f"{name} unregister raised {exc}")
        if cleanup_errors:
            print(
                json.dumps({"event": "cleanup_failed", "errors": cleanup_errors}),
                flush=True,
            )
            if not body_failed:
                raise RuntimeError("; ".join(cleanup_errors))


def run_iterations(
    consumer,
    cuda,
    args: argparse.Namespace,
    quarantine: HbmAllocationQuarantine,
) -> None:
    for iteration in range(args.iterations):
        expected = deterministic_payload(args.payload_size, args.device, iteration)
        expected_hash = hashlib.sha256(expected).hexdigest()
        with allocated_hbm_buffers(
            cuda, (args.payload_size, args.payload_size), quarantine
        ) as pointers:
            source_ptr, destination_ptr = pointers
            cuda.copy_from_host(source_ptr, expected)
            cuda.memset(destination_ptr, 0, args.payload_size)
            cuda.synchronize()
            buffers = (
                (source_ptr, args.payload_size, "source"),
                (destination_ptr, args.payload_size, "destination"),
            )
            with registered_buffers(consumer, buffers, quarantine):
                key = f"{args.key_prefix}-{args.run_id}-gpu{args.device}-{iteration}"
                before_put = consumer_metrics(consumer.serialize_metrics())
                put_started = time.perf_counter_ns()
                put_result = consumer.put_from(key, source_ptr, args.payload_size)
                put_ns = time.perf_counter_ns() - put_started
                if put_result != 0:
                    raise RuntimeError(f"put_from failed: {put_result}")
                after_put = consumer_metrics(consumer.serialize_metrics())
                put_cache_delta = consumer_delta(before_put, after_put)

                get_started = time.perf_counter_ns()
                get_result = consumer.get_into(key, destination_ptr, args.payload_size)
                cuda.synchronize()
                get_ns = time.perf_counter_ns() - get_started
                if get_result != args.payload_size:
                    raise RuntimeError(
                        f"get_into returned {get_result}, expected {args.payload_size}"
                    )
                after_get = consumer_metrics(consumer.serialize_metrics())
                get_cache_delta = consumer_delta(after_put, after_get)
                if iteration == 0:
                    validate_miss_then_hit(put_cache_delta, get_cache_delta)
                actual = cuda.copy_to_host(destination_ptr, args.payload_size)
                actual_hash = hashlib.sha256(actual).hexdigest()
                if actual_hash != expected_hash or actual != expected:
                    raise RuntimeError("HBM payload mismatch")
                remove_result = consumer.remove(key, True)
                if remove_result != 0:
                    raise RuntimeError(f"force remove failed: {remove_result}")
                print(
                    json.dumps(
                        {
                            "event": "result",
                            "run_id": args.run_id,
                            "device": args.device,
                            "iteration": iteration,
                            "cache_phase": classify_cache_phase(
                                put_cache_delta, get_cache_delta
                            ),
                            "put_cache_delta": consumer_delta_dict(put_cache_delta),
                            "get_cache_delta": consumer_delta_dict(get_cache_delta),
                            "bytes": args.payload_size,
                            "sha256": actual_hash,
                            "put_latency_ns": put_ns,
                            "get_latency_ns": get_ns,
                            "put_gib_s": (args.payload_size / put_ns * 1e9 / 1024**3),
                            "get_gib_s": (args.payload_size / get_ns * 1e9 / 1024**3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GB200 HBM Consumer correctness test (CUDA runtime, no Torch)"
    )
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--master-server", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--payload-size", type=payload_size, default=MAX_PAYLOAD_SIZE)
    parser.add_argument("--iterations", type=at_least_two, default=2)
    parser.add_argument("--key-prefix", default="nvlink-host-numa")
    parser.add_argument("--run-id", type=safe_identifier, default=uuid.uuid4().hex)
    parser.add_argument(
        "--cuda-runtime-library",
        help="explicit libcudart.so path (or set MC_CUDART_LIBRARY)",
    )
    args = parser.parse_args()

    cuda = CudaRuntime(args.cuda_runtime_library)
    cuda.set_device(args.device)
    store_module = import_store_module()
    consumer = store_module.MooncakeDistributedStore()
    config = {
        "local_hostname": args.local_hostname,
        "metadata_server": args.metadata_server,
        "master_server_addr": args.master_server,
        "global_segment_size": "0",
        "local_buffer_size": "0",
        "protocol": "nvlink",
    }
    exit_code = 0
    active_error = False
    quarantine = HbmAllocationQuarantine()
    try:
        setup_result = consumer.setup(config)
        if setup_result != 0:
            print(
                json.dumps({"event": "setup_failed", "result": setup_result}),
                flush=True,
            )
            exit_code = int(setup_result) if int(setup_result) > 0 else 2
        else:
            print(
                json.dumps(process_context(cuda, args, store_module), sort_keys=True),
                flush=True,
            )
            run_iterations(consumer, cuda, args, quarantine)
    except BaseException:
        active_error = True
        raise
    finally:
        try:
            close_result = consumer.close() if hasattr(consumer, "close") else 0
        except Exception as exc:
            close_result = f"exception: {exc}"
        cleanup_errors = []
        if close_result in (None, 0):
            cleanup_errors.extend(quarantine.release_after_store_close(cuda))
        elif len(quarantine) > 0:
            cleanup_errors.append(
                f"preserving {len(quarantine)} HBM allocation(s) until process "
                "exit because Store close did not succeed"
            )
        if close_result not in (None, 0) or cleanup_errors:
            print(
                json.dumps(
                    {
                        "event": "cleanup_failed",
                        "close_result": close_result,
                        "errors": cleanup_errors,
                    }
                ),
                flush=True,
            )
            if not active_error and exit_code == 0:
                exit_code = 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
