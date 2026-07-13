# GB200 NVLink HOST_NUMA validation

These scripts validate the default-off NVL72 HOST_NUMA V1 path. They are not a
fallback or an H2H benchmark: the Provider owns Fabric-exportable HOST_NUMA
DRAM and Consumers transfer from/to registered HBM through protocol `nvlink`.

## Preconditions

- Build with `USE_CUDA=ON` and `USE_MNNVL=ON`.
- Use a CTest version that supports `--output-junit`.
- Leave `MC_USE_NVLINK_IPC` unset.
- Set `MC_MS_AUTO_DISC=0` (recommended), or set `MC_FORCE_MNNVL=1` when
  auto-discovery is required; the Provider readiness gate rejects an actual
  RDMA, intra-node, or mixed transport installation.
- Give the Provider and Consumer users access to the same IMEX channel.
- Start the Mooncake Master and metadata service before the Provider.
- Make the Master's admin HTTP port reachable from the Provider (default
  `--metrics_port=9003`).

CUDA requires `nvidia-caps-imex-channels` in `/proc/devices` and an accessible
`/dev/nvidia-caps-imex-channels/channel*` for Fabric handles. See the official
[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
and [NVIDIA IMEX channel guide](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/imexchannels.html).

Strict build, preflight, and hardware tests:

```bash
export BUILD_DIR="${BUILD_DIR:-$PWD/build-nvlink-host-numa}"
RUN_HARDWARE_TESTS=1 scripts/gb200/nvlink_host_numa_build.sh
export PYTHONPATH="${BUILD_DIR}/mooncake-integration${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${BUILD_DIR}/mooncake-common${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python3 -c 'import store; print("Store binding:", store.__file__)'
```

Run this build/provenance check on both nodes. The printed binding must resolve
under this `BUILD_DIR`, not to an older site-packages installation. Provider
and Consumer JSON also record the selected module path. The build-tree binding
links `libasio.so` from `BUILD_DIR/mooncake-common`; keep the exported
`LD_LIBRARY_PATH` in the Provider and Consumer shells.

In hardware mode the build script first builds every target, sets strict
Fabric/RDMA requirements, defaults `MC_MNNVL_FABRIC_PROBE` to the newly built
`nvlink_host_numa_fabric_test`, and only then runs preflight and the required
CTest labels. An explicit executable `MC_MNNVL_FABRIC_PROBE` still overrides
that default. Strict standalone preflight requires this variable and rejects a
missing or non-executable probe:

```bash
MC_REQUIRE_MNNVL_FABRIC=1 \
MC_MNNVL_FABRIC_PROBE="$PWD/build-nvlink-host-numa/mooncake-transfer-engine/tests/nvlink_host_numa_fabric_test" \
scripts/gb200/nvlink_host_numa_preflight.sh
```

`SKIP_PREFLIGHT=1` remains available only for a preflight result already
captured for the same build and host; hardware CTest strict/zero-skip gates
still run.

Torch is not used anywhere in this validation path. The HBM Consumer loads
`libcudart.so` directly with Python `ctypes`, allocates source/destination HBM
with `cudaMalloc`, and performs host-to-device/device-to-host verification with
`cudaMemcpy`. A normal CUDA-enabled Mooncake build container is sufficient. If
the runtime library is outside the loader's search path, set
`MC_CUDART_LIBRARY=/absolute/path/to/libcudart.so` or pass
`--cuda-runtime-library`.
The build wrapper applies its fixed `WITH_EP=OFF`, CUDA, and MNNVL settings
after `CMAKE_EXTRA_ARGS`, so optional CMake arguments cannot re-enable the
Torch-backed EP build in this validation path.

The Fabric probe tests every online NUMA node discovered from the visible GPUs.
Set `MC_NVLINK_HOST_NUMA_TEST_NODES=0,1` to override that discovery explicitly;
the single-node `MC_NVLINK_HOST_NUMA_TEST_NODE` override remains available for
focused diagnosis.

Without `RUN_HARDWARE_TESTS=1`, the build script runs only the required unit
label plus the no-Torch Python fake-runtime suite and a `libcudart` load/version
check. JUnit files are written under `BUILD_DIR`; required hardware/RDMA suites
must contain testcases and must report zero skipped tests.

When the IMEX daemon is deliberately managed outside a container, set
`MC_IMEX_DAEMON_EXTERNAL=1`; do not use that override to hide a missing host
daemon.

## Provider

The ConfigDict size parser accepts `GB`, not `GiB`:

```bash
RUN_ID="gb200-$(date -u +%Y%m%dT%H%M%SZ)"
MC_MAX_MR_SIZE=$((150*1024*1024*1024)) \
python3 scripts/gb200/nvlink_host_numa_provider.py \
  --local-hostname "${NODE_A_IP}:12345" \
  --metadata-server "http://${NODE_A_IP}:8080/metadata" \
  --master-server "${NODE_A_IP}:50051" \
  --master-admin-url "http://${NODE_A_IP}:9003" \
  --global-segment-size "600 GB" \
  --nodes auto \
  --run-id "${RUN_ID}" \
  --ready-file /tmp/mooncake-host-numa.ready
```

Readiness is printed and optionally written atomically only after
`setup(config)` succeeds and two independent views agree exactly:

- the Provider's read-only `serialize_metrics()` snapshot supplies requested
  capacity, effective capacity, and the sum of per-NUMA chunk counts;
- Master `/get_all_segments` must contain exactly that many entries for the
  Provider hostname, and `/query_segment` must report exactly the same
  effective capacity.

A mismatch or an unreachable admin endpoint times out without creating the
ready file. The Provider removes any stale ready file before setup, removes its
ready file on every exit path, and validates the Store close result.

## Consumers and concurrency

Run one Consumer per visible GPU on Node B. Each process verifies SHA256 plus a
byte-for-byte HBM round trip. It snapshots `serialize_metrics()` around the
first `put_from` and the following `get_into`; acceptance requires a real
mapping miss/lazy import followed by a cache hit with no second import. The
JSON records contain measured `put_cache_delta` / `get_cache_delta` counters;
`cache_phase` is derived from those counters as `cold`, `warm`, `mixed`, or
`none`, never from the iteration number:

```bash
python3 scripts/gb200/nvlink_host_numa_consumer.py \
  --local-hostname "${NODE_B_IP}:12400" \
  --metadata-server "http://${NODE_A_IP}:8080/metadata" \
  --master-server "${NODE_A_IP}:50051" \
  --device 0 --iterations 2 --payload-size 16777216 \
  --run-id "${RUN_ID}"
```

The benchmark launches GPU Consumers concurrently, rejects duplicate GPU IDs,
and requires at least two iterations. The hostname prefix must include the
colon and a distinct port prefix, for example `10.0.0.2:1240` produces ports
`12400` through `12403`:

```bash
python3 scripts/gb200/nvlink_host_numa_bench.py \
  --local-hostname-prefix "${NODE_B_IP}:1240" \
  --metadata-server "http://${NODE_A_IP}:8080/metadata" \
  --master-server "${NODE_A_IP}:50051" \
  --run-id "${RUN_ID}" \
  --devices 0,1,2,3 --iterations 4
```

Use the same explicit `RUN_ID` on both nodes. Consumer context records include
the PID, loaded CUDA runtime path, CUDA runtime/driver versions,
`CUDA_VISIBLE_DEVICES`, and Linux CPU/NUMA affinity; summaries are split by
counter-derived cache phase and include mapping hit ratio and cold import
latency.

Record topology, driver/toolkit/IMEX versions, CTest XML, Provider metrics, and
the JSON result stream. Hardware results are `PASS`, `FAIL`, or `NOT RUN`; a
skip is not acceptance evidence.

The harness parsers, Master admin mock, no-Torch Consumer fake-runtime round
trip, and metric-delta assertions can be tested in a build container without a
GPU:

```bash
python3 -m unittest discover -s scripts/gb200 \
  -p 'test_nvlink_host_numa_harness.py' -v
python3 -m py_compile scripts/gb200/*.py
bash scripts/gb200/test_nvlink_host_numa_scripts.sh
```
