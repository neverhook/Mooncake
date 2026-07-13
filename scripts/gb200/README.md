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

Strict preflight and build:

```bash
MC_REQUIRE_MNNVL_FABRIC=1 \
MC_MNNVL_FABRIC_PROBE=/path/to/nvlink_host_numa_fabric_test \
scripts/gb200/nvlink_host_numa_preflight.sh

BUILD_DIR=$PWD/build-nvlink-host-numa \
SKIP_PREFLIGHT=1 \
RUN_HARDWARE_TESTS=1 \
scripts/gb200/nvlink_host_numa_build.sh
```

Torch is not a build prerequisite. The Provider and all parser/mock gates run
without it. Only the HBM Consumer needs a CUDA-enabled Torch wheel; install an
internally approved, driver-compatible wheel into the workspace venv before
the data-plane step if the base image must remain unchanged.

The Fabric probe tests every online NUMA node discovered from the visible GPUs.
Set `MC_NVLINK_HOST_NUMA_TEST_NODES=0,1` to override that discovery explicitly;
the single-node `MC_NVLINK_HOST_NUMA_TEST_NODE` override remains available for
focused diagnosis.

Without `RUN_HARDWARE_TESTS=1`, the build script runs only the required unit
label. JUnit files are written under `BUILD_DIR`; required hardware/RDMA suites
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
byte-for-byte tensor comparison. It snapshots `serialize_metrics()` around the
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
  --device 0 --iterations 2 --payload-size 16777216
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
the PID, Torch/CUDA versions, GPU name, and Linux CPU/NUMA affinity; summaries
are split by counter-derived cache phase and include mapping hit ratio and cold
import latency.

Record topology, driver/toolkit/IMEX versions, CTest XML, Provider metrics, and
the JSON result stream. Hardware results are `PASS`, `FAIL`, or `NOT RUN`; a
skip is not acceptance evidence.

The harness parsers, Master admin mock, and metric-delta assertions can be
tested in a build container without Torch or a GPU:

```bash
python3 -m unittest discover -s scripts/gb200 \
  -p 'test_nvlink_host_numa_harness.py' -v
python3 -m py_compile scripts/gb200/*.py
```
