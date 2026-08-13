# GB200/NVL72 Store EGM Provider/Consumer validation

This directory is validation-only. It lives on the dedicated
`codex/egm-store-pool-gb200-validation` branch and is not part of the Store PR.
The branch is based directly on the Store PR head, which in turn contains the
full #2966 HOST_NUMA/NvlinkTransport prerequisite, and also carries the
cross-node direct GPU-read change submitted separately as #3431.

The harness proves these real Store paths on two GB200 nodes in one NVL72
supernode:

1. Consumer HBM -> Provider EGM through `put_from`.
2. Provider EGM -> Consumer HBM through `get_into`.
3. SHA-256 and byte-for-byte correctness for every transfer.
4. Consumer HBM cleanup and Provider pool teardown/unpublication.
5. Per-transfer elapsed time, per-stream GiB/s, and multi-GPU concurrent-window
   GiB/s in machine-readable JSONL.

The Provider uses `enable_egm_store_pool=true`, `protocol=nvlink`, a nonzero
`global_segment_size`, and `local_buffer_size=0`. Consumers contribute no Store
capacity and also use `local_buffer_size=0`. MEMORY reads therefore submit the
Consumer HBM pointer directly to the selected transport instead of allocating a
Consumer host staging buffer. The harness starts Mooncake Master with its
embedded HTTP metadata service, so it does not require etcd.

Consumer HBM buffers are ordinary `cudaMalloc` local endpoints. In cross-node
fabric mode they are not published through `register_buffer()`: the NVLink
transport explicitly accepts such pointers for local copies, while registration
is a no-op that has no matching metadata record to unregister. The Provider EGM
ranges remain exact-range registered and are fully checked during teardown.

## Prerequisites

- Two GB200 nodes in the same NVL72 supernode, with Fabric/IMEX working across
  the nodes.
- The same checkout, branch head, absolute build path, and config file contents
  on both nodes.
- CUDA toolkit, Ninja, CMake, Python 3, curl, and Mooncake build dependencies.
- The configured Master RPC, metadata HTTP, admin HTTP, Provider, and Consumer
  ports must be reachable between the two nodes.

The default Provider pool is 10 GiB. The four simultaneous 128 MiB objects only
require 512 MiB of live capacity; the larger default leaves room for expanded
payload or concurrency experiments. Pool capacity by itself does not increase
transfer bandwidth.

## 1. Check out the validation branch on both nodes

```bash
git fetch neverhook codex/egm-store-pool-gb200-validation
git switch -c codex/egm-store-pool-gb200-validation \
  --track neverhook/codex/egm-store-pool-gb200-validation
git rev-parse HEAD
```

The SHA printed on Node A and Node B must be identical. The harness records this
SHA in Provider and Consumer evidence and refuses to combine mismatched runs.

Create the same config on both nodes:

```bash
cp scripts/gb200/egm_store_gb200.conf.example egm-store-gb200.conf
vim egm-store-gb200.conf
```

Required edits are `NODE_A_IP`, `NODE_B_IP`, `RUN_ID`, and normally
`BUILD_DIR`. Use a new `RUN_ID` for every acceptance run.

Inspect the fully resolved configuration on both nodes:

```bash
scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf print-config
```

## 2. Build and preflight both nodes

Run on Node A and Node B:

```bash
scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf build

scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf preflight
```

By default, `build` sets `BUILD_UNIT_TESTS=0`, compiles the Master and Python
Store binding, and runs the script-level tests without downloading GoogleTest.
Set `BUILD_UNIT_TESTS=1` in the config to also build the Store EGM pool test and
the two #2966 focused tests, then run `nvlink_vmm_unit` and
`egm_store_pool_unit`. `preflight` records visible GPUs, topology, Fabric state,
GPU NUMA locality, IMEX devices/daemon, and conflicting environment overrides.

`MC_IMEX_DAEMON_EXTERNAL=1` is the default for a container using the host IMEX
daemon. Set it to `0` when the daemon is expected to be visible inside the test
environment so preflight can verify the process directly.

## 3. Start Master and Provider on Node A

```bash
scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf master-start

scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf provider-start

scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf provider-status
```

`provider-start` returns only after Store setup succeeds and Master reports the
mounted EGM chunks and effective capacity. With the default `RESULT_ROOT`, the
readiness JSON is stored at:

```text
/tmp/mooncake-egm-gb200/<RUN_ID>/provider.ready.json
```

If setup fails, collect `provider.log` and `provider-start-diagnose.log` from
the same directory. Provider setup itself is the strict CUDA Fabric
allocation/export/registration/publication probe for this Store implementation.

## 4. Run concurrent Consumers on Node B

An optional single-GPU smoke run is useful before the full concurrent gate:

```bash
scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf consumer 0
```

It uses `THRESHOLD_PAYLOAD_SIZE`, writes separate JSONL/native-log files for
that GPU, and removes its objects before the full benchmark starts.

Run the full validation:

```bash
scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf bench
```

Defaults run GPUs `0,1,2,3`, payloads 4 KiB, 1 MiB, 16 MiB, and 128 MiB, with
four iterations per size. Each result includes fields such as:

```json
{
  "event": "transfer_result",
  "put_duration_ns": 1234567,
  "put_bandwidth_gib_s": 101.25,
  "get_duration_ns": 765432,
  "get_bandwidth_gib_s": 163.31,
  "sha256": "...",
  "status": "PASS"
}
```

The reported Get duration is end-to-end Store API latency for the direct
Provider EGM -> Consumer HBM transfer. Put uses the Consumer HBM pointer as the
local NVLink source and writes the remote Provider EGM range.

The benchmark writes pure JSONL to `bench.jsonl` and native Store/CUDA logs to
`bench.stderr.log`. It emits:

- Raw `transfer_result` records for every GPU, size, and iteration.
- `performance_summary` records with p50/p95/p99 time and per-stream bandwidth.
- `aggregate_iteration` records where total bytes are divided by the common
  earliest-start/latest-end operation window across Consumer processes.
- One final `benchmark_gate` record.

`first` and `steady` mean sequence position only. They do not claim a directly
observed transport-cache state.

No arbitrary bandwidth threshold is enabled by default. To turn performance
into a numerical gate, set `MIN_PUT_GIB_S` and `MIN_GET_GIB_S`; they apply to
steady per-stream p50 at `THRESHOLD_PAYLOAD_SIZE`. Correctness and cleanup are
always hard gates.

## 5. Stop Provider, verify teardown, and stop Master on Node A

After Node B reports `benchmark_gate: PASS`:

```bash
scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf provider-stop

scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf master-stop
```

`provider-stop` waits for Store `close()`, requires a PASS cleanup record, then
queries Master and appends a `provider_unpublished: PASS` record only when no
Provider chunks remain.

## 6. Generate the PR-ready report

Copy the Node A evidence to Node B without modifying it:

```bash
RUN_ID=egm-gb200-20260807  # replace with the value from your config
RESULT_DIR=/tmp/mooncake-egm-gb200/${RUN_ID}
scp NODE_A:${RESULT_DIR}/provider.ready.json \
  ${RESULT_DIR}/provider.ready.node-a.json
scp NODE_A:${RESULT_DIR}/provider.log \
  ${RESULT_DIR}/provider.node-a.log
```

Then run on Node B:

```bash
scripts/gb200/egm_store_gb200.sh \
  --config ./egm-store-gb200.conf report \
  ${RESULT_DIR}/provider.ready.node-a.json \
  ${RESULT_DIR}/provider.node-a.log
```

The generated `${RESULT_DIR}/pr-report.md` is ready to paste into the
Store PR. Report generation fails if Provider/Consumer SHA or run ID differs,
the benchmark gate is not PASS, Provider cleanup is missing, or Master still
contained Provider chunks after teardown.

Keep these raw artifacts with the report:

- Node A: `preflight-*.log`, `master.log`, `provider.log`,
  `provider.ready.json`, and `provider-start-diagnose.log` when present.
- Node B: `preflight-*.log`, `bench.jsonl`, `bench.stderr.log`, and
  `pr-report.md`.

Do not paste a terminal transcript in place of `bench.jsonl`; the report parser
requires one pure JSONL stream and exactly one final benchmark gate.
