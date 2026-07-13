#!/usr/bin/env python3

import argparse
import contextlib
import hashlib
import json
import time
import uuid


def import_store_module():
    try:
        from mooncake import store as store_module  # type: ignore
    except ImportError:
        import store as store_module  # type: ignore
    return store_module


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def at_least_two(value: str) -> int:
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("value must be at least two (cold and warm)")
    return number


@contextlib.contextmanager
def registered_buffers(store, buffers):
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
                    cleanup_errors.append(f"{name} unregister returned {result}")
            except Exception as exc:
                cleanup_errors.append(f"{name} unregister raised {exc}")
        if cleanup_errors:
            print(
                json.dumps({"event": "cleanup_failed", "errors": cleanup_errors}),
                flush=True,
            )
            if not body_failed:
                raise RuntimeError("; ".join(cleanup_errors))


def run_iterations(consumer, torch, args: argparse.Namespace) -> None:
    run_id = uuid.uuid4().hex
    for iteration in range(args.iterations):
        generator = torch.Generator(device=f"cuda:{args.device}")
        generator.manual_seed((args.device + 1) * 1_000_003 + iteration)
        source = torch.randint(
            0,
            256,
            (args.payload_size,),
            dtype=torch.uint8,
            device=f"cuda:{args.device}",
            generator=generator,
        )
        destination = torch.zeros_like(source)
        torch.cuda.synchronize(args.device)
        expected_hash = hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest()
        source_ptr = source.data_ptr()
        destination_ptr = destination.data_ptr()
        buffers = (
            (source_ptr, args.payload_size, "source"),
            (destination_ptr, args.payload_size, "destination"),
        )
        with registered_buffers(consumer, buffers):
            key = f"{args.key_prefix}-{run_id}-gpu{args.device}-{iteration}"
            put_started = time.perf_counter_ns()
            put_result = consumer.put_from(key, source_ptr, args.payload_size)
            put_ns = time.perf_counter_ns() - put_started
            if put_result != 0:
                raise RuntimeError(f"put_from failed: {put_result}")

            get_started = time.perf_counter_ns()
            get_result = consumer.get_into(key, destination_ptr, args.payload_size)
            torch.cuda.synchronize(args.device)
            get_ns = time.perf_counter_ns() - get_started
            if get_result != args.payload_size:
                raise RuntimeError(
                    f"get_into returned {get_result}, expected {args.payload_size}"
                )
            actual_hash = hashlib.sha256(
                destination.cpu().numpy().tobytes()
            ).hexdigest()
            if actual_hash != expected_hash or not torch.equal(source, destination):
                raise RuntimeError("HBM payload mismatch")
            remove_result = consumer.remove(key, True)
            if remove_result != 0:
                raise RuntimeError(f"force remove failed: {remove_result}")
            print(
                json.dumps(
                    {
                        "event": "result",
                        "device": args.device,
                        "iteration": iteration,
                        "cache_phase": "cold" if iteration == 0 else "warm",
                        "bytes": args.payload_size,
                        "sha256": actual_hash,
                        "put_latency_ns": put_ns,
                        "get_latency_ns": get_ns,
                        "put_gib_s": args.payload_size / put_ns * 1e9 / 1024**3,
                        "get_gib_s": args.payload_size / get_ns * 1e9 / 1024**3,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="GB200 HBM Consumer correctness test")
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--master-server", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--payload-size", type=positive_int, default=16 * 1024 * 1024)
    parser.add_argument("--iterations", type=at_least_two, default=2)
    parser.add_argument("--key-prefix", default="nvlink-host-numa")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(args.device)
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
    try:
        setup_result = consumer.setup(config)
        if setup_result != 0:
            print(
                json.dumps({"event": "setup_failed", "result": setup_result}),
                flush=True,
            )
            exit_code = int(setup_result) if int(setup_result) > 0 else 2
        else:
            run_iterations(consumer, torch, args)
    except BaseException:
        active_error = True
        raise
    finally:
        try:
            close_result = consumer.close() if hasattr(consumer, "close") else 0
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
