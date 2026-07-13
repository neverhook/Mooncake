# GB200 NVLink HOST_NUMA validation

This harness validates the default-off NVL72 HOST_NUMA V1 path: Node A exports
NUMA-local DRAM through NVLink Fabric and Node B Consumers transfer between the
pooled DRAM and HBM. The Consumer uses `ctypes` + `libcudart`; Torch is not
required.

All operator-facing configuration is loaded from one cluster config on every
invocation. Do not source the config and do not export Mooncake endpoints or
library paths manually. The wrapper deliberately ignores inherited
`PYTHONPATH`, `LD_LIBRARY_PATH`, endpoint variables, and runtime feature flags.

## One-time configuration

On Node A:

```bash
cd /workspace/Mooncake
cp scripts/gb200/gb200.conf.example gb200.conf
vi gb200.conf
```

Edit these required values:

```bash
NODE_A_IP="10.192.9.60"
NODE_B_IP="<node-b-ip>"
RUN_ID="gb200-20260713-01"
BUILD_DIR="/workspace/Mooncake/build-nvlink-host-numa"
```

Keep the default `METADATA_PORT=8079` unless that port is also occupied. Copy
the completed `gb200.conf` to `/workspace/Mooncake/gb200.conf` on Node B. The
two copies must have the same `NODE_A_IP`, `NODE_B_IP`, `RUN_ID`, ports, and
test parameters.

Before any process is started, verify the resolved configuration on both
nodes:

```bash
scripts/gb200/gb200.sh print-config
```

The output must point `PYTHONPATH` and `LD_LIBRARY_PATH` at the configured
`BUILD_DIR`, and must show:

- metadata: `http://<node-a>:8079/metadata`
- Master RPC: `<node-a>:50051`
- Master admin: `http://<node-a>:9003`
- Provider: `<node-a>:12345`

Use `--config /absolute/path/to/file` before the action when the config is not
`/workspace/Mooncake/gb200.conf`.

## Execution checklist

### 1. Build and validate on both nodes

Run separately on Node A and Node B:

```bash
cd /workspace/Mooncake
scripts/gb200/gb200.sh build
```

The configured default is strict hardware mode (`RUN_HARDWARE_TESTS=1`). It
builds all targets with `WITH_EP=OFF`, `USE_CUDA=ON`, and `USE_MNNVL=ON`, then
runs unit, Fabric, and RDMA suites with zero skips. Set `SKIP_PREFLIGHT=1` only
when a preflight result for the same build and host has already been captured.

### 2. Start the control plane on Node A

```bash
cd /workspace/Mooncake
scripts/gb200/gb200.sh master-start
scripts/gb200/gb200.sh master-status
```

`master-start` fails before launch if RPC, metadata, or admin ports are already
occupied. It waits for all three listeners and performs a PUT/GET/DELETE probe
against the embedded metadata server. Logs and PID files are written beneath
`RESULT_ROOT/RUN_ID` (default `/tmp/mooncake-gb200/<run-id>`).

### 3. Start the HOST_NUMA Provider on Node A

```bash
scripts/gb200/gb200.sh provider-start
scripts/gb200/gb200.sh provider-status
```

The Provider is ready only when its local capacity metrics and the Master's
`/get_all_segments` plus `/query_segment` views agree. If readiness does not
arrive within `PROVIDER_DIAGNOSTIC_DELAY_SEC`, the wrapper automatically saves
live port ownership, metadata health, all segment listings, and both raw and
URL-encoded segment queries. On failure, inspect:

```text
/tmp/mooncake-gb200/<run-id>/provider.log
/tmp/mooncake-gb200/<run-id>/provider-start-diagnose.log
```

Diagnostics can also be rerun without setting any environment variable:

```bash
scripts/gb200/gb200.sh diagnose
```

### 4. Run a single-GPU correctness test on Node B

```bash
cd /workspace/Mooncake
scripts/gb200/gb200.sh consumer 0
```

Repeat for other visible GPUs if desired. Each process allocates HBM with
`cudaMalloc`, verifies HBM-to-pool and pool-to-HBM data byte for byte, and
requires the first transfer to prove a mapping miss/lazy import followed by a
cache hit. `MC_CUDART_LIBRARY` can be set in `gb200.conf` when `libcudart.so` is
outside the standard CUDA paths.

### 5. Run the concurrent benchmark on Node B

```bash
scripts/gb200/gb200.sh bench
```

Devices, payload sizes, and iterations come from `DEVICES`, `PAYLOAD_SIZES`,
and `ITERATIONS` in `gb200.conf`. The benchmark rejects duplicate GPU IDs and
requires at least two iterations so it can distinguish cold and warm mapping
behavior. Output is shown on the terminal and saved to
`RESULT_ROOT/RUN_ID/bench.jsonl`; single-GPU output is similarly saved as
`consumer-gpu-<device>.jsonl`.

### 6. Stop processes on Node A

```bash
scripts/gb200/gb200.sh provider-stop
scripts/gb200/gb200.sh master-stop
```

Or stop both in dependency order:

```bash
scripts/gb200/gb200.sh stop
```

Only PIDs recorded by this wrapper are signalled. It never uses `killall`,
`pkill`, or a forced kill.

## Acceptance evidence

Retain the following for the same `RUN_ID`:

- build/CTest output and JUnit files under `BUILD_DIR` on both nodes;
- `print-config` output from both nodes;
- Node A `master.log`, `provider.log`, ready JSON, and any diagnostic log;
- Node B single-GPU Consumer JSON and concurrent benchmark JSON;
- topology, driver/toolkit versions, and IMEX daemon/channel state.

Required hardware results are `PASS` or `FAIL`; `NOT RUN` and skipped tests are
not acceptance evidence. Leave `MC_USE_NVLINK_IPC` unset. When the IMEX daemon
is intentionally managed outside the container, set
`MC_IMEX_DAEMON_EXTERNAL=1` in the config; do not use it to hide a missing host
daemon.

## Local harness regression tests

These do not require a GPU or Torch:

```bash
python3 -m unittest discover -s scripts/gb200 \
  -p 'test_nvlink_host_numa_harness.py' -v
python3 -m py_compile scripts/gb200/*.py
bash scripts/gb200/test_nvlink_host_numa_scripts.sh
```
