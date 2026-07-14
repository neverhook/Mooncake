# NVL72 HOST_NUMA NVLink V1 GB200 Test Report

- **Date:** 2026-07-14
- **Run ID:** `gb200-0714-2`
- **Status:** Partial acceptance PASS — dual-node Provider/Consumer data plane
- **Code under test:** expected `6dd6331ea3ced193b05b8aad28f2fe70e19ed6f3`;
  the submitted artifacts do not contain `git rev-parse HEAD`, so commit
  provenance remains operator-reported rather than independently verified
- **Deferred evidence:** build/CTest/JUnit, strict preflight, complete
  environment and topology, RDMA premise, Provider stop cleanup, rollback, and
  injected-failure results

## Scope

This report evaluates the runtime artifacts supplied for one GB200/NVL72
HOST_NUMA validation run. It covers:

- a Provider exporting a requested 600 MiB HOST_NUMA pool over protocol
  `nvlink`;
- a standalone 128 MiB HBM Consumer correctness run on GPU 0;
- concurrent Consumers on GPUs 0-3 at 4 KiB, 1 MiB, 16 MiB, and 128 MiB;
- byte-for-byte/SHA256 correctness, lazy import, mapping reuse, capacity, and
  report-only performance.

It does not convert missing build, topology, or lifecycle artifacts into a
pass. Those items remain explicitly pending in the acceptance matrix below.

## Submitted artifacts

The raw artifacts were supplied in an operator staging directory outside the
feature worktree and are not committed to Git.

| Artifact | SHA-256 | Notes |
|---|---|---|
| `master.log` | `ed77cccf3bec67f1fe65b45489f2582ba6b5fc392de4f266cf408da3cebe166e` | Master startup, mounts, capacity, request rates, and object-count evidence |
| `provider.log` | `1fb5aa796e9fdd3d64b74411419b42246f4a4003799350d0d3dce070b47d9f57` | Provider config, publication, READY, and 118 stable capacity samples |
| `provider.ready.json` | `0596c6f1f47585da365ec1916ef8893902c9de71702d9668242018994225abd0` | Atomic readiness result |
| `consumer-gpu-0.jsonl ` | `74d7036dd57814702443fded17cb0db8652f4822f115f9e4147773d568cf0ec8` | Four standalone 128 MiB iterations; source filename contains a trailing space |
| `bench.jsonl` | `b942302636dd0f15b3e865d20289f853071ba142f96fdfe5a33137a1b15e809e` | Terminal `cat` transcript containing a complete benchmark stream twice plus the standalone Consumer output |

`bench.jsonl` is not a clean raw JSONL artifact: it includes shell prompts and
an exact duplicate of the complete benchmark output. Result counts and the
performance table below use the first complete 64-result benchmark stream
only. The duplicate does not count as a second acceptance run. Replace this
transcript with the original `/tmp/mooncake-gb200/gb200-0714-2/bench.jsonl`
before final PR archival if it is still available.

## Result summary

| Requirement | Result | Evidence |
|---|---|---|
| Provider setup and readiness | PASS | One `event=ready`; no ERROR/FATAL or structured setup/cleanup failure |
| Requested/effective/Master capacity agreement | PASS | Requested `629145600`; Provider effective and Master capacity both `620756992` |
| Forced chunking | PASS | Five published chunks; four `150994944`-byte ranges and one `16777216`-byte range sum to `620756992` |
| Stable Provider capacity | PASS | 118 capacity samples all report requested `629145600`, effective `620756992`, chunks `5` |
| Standalone GPU 0 128 MiB HBM Put/Get | PASS | Four iterations, four distinct SHA256 values, positive Put/Get throughput, no mismatch/failure |
| Cold import followed by mapping hit | PASS | First standalone Put has one miss/import; each Get has one hit and zero miss/import |
| Four-GPU concurrent correctness | PASS | One complete matrix: 4 GPUs × 4 sizes × 4 iterations = 64 valid results |
| Maximum 128 MiB concurrent payload | PASS | GPUs 0-3 each complete four 128 MiB iterations; Master observes 512 MiB and four keys at peak |
| Per-object cleanup during benchmark | PASS | Master returns from 512 MiB/four keys to 0 B/zero keys after the 128 MiB phase |
| NVLink data path selected | PASS | Consumer logs create `NvlinkTransport` CUDA streams and use `cudaMemcpyBatchAsync`; memcpy fallback is disabled |
| Multi-NUMA placement | EVIDENCE PENDING | Chunk count is proven, but per-NUMA labels/topology were not supplied |
| Provider teardown and zero resource leaks | EVIDENCE PENDING | No `provider-stop`/post-stop segment and handle evidence supplied |
| Build, strict preflight, hardware/RDMA JUnit | EVIDENCE PENDING | Intentionally deferred by the operator |
| Full driver/CUDA/IMEX/topology provenance | EVIDENCE PENDING | Intentionally deferred; runtime records only show CUDA runtime 12.8 and driver API 13.0 |
| Rollback/default-off regression | NOT RUN IN THIS EVIDENCE SET | No disable/restart/legacy HBM result supplied |
| Injected mount/allocation failure rollback | NOT RUN IN THIS EVIDENCE SET | No injected-failure artifact supplied |

## Provider capacity and publication

The Provider was configured with:

```text
enable_nvlink_host_numa=true
global_segment_size=600 MB
local_buffer_size=0
nvlink_host_numa_nodes=auto
protocol=nvlink
local_hostname=10.192.9.60:12345
```

The READY record is internally consistent:

```text
requested_capacity_bytes = 629145600
effective_capacity_bytes = 620756992
expected_chunk_count      = 5
master_capacity_bytes     = 620756992
master_chunk_count        = 5
master_used_bytes         = 0
```

Provider publication completed approximately 0.22 seconds after the first
Provider log record. Master subsequently reported 592 MiB of available memory
and one live Provider client.

Five warnings report `cuMemGetAddressRange` result 201 for HOST_NUMA VMM
allocations. In this implementation that condition enters the guarded
exact-owned-range fallback. All five ranges were then published, Provider
setup reached READY, and the complete data-plane test passed. No ERROR or FATAL
records were present.

Master also logged `segment_already_exists` warnings while representing five
chunks under the same Provider segment name. They were non-blocking in this
run: READY saw exactly five chunks and Master capacity matched the Provider.
They are retained as an observation rather than classified as a test failure.

## Standalone 128 MiB correctness

GPU 0 completed iterations 0-3 with `134217728` bytes per iteration. Each
record contains a 64-character SHA256 digest and positive Put/Get throughput.
The first Put records one mapping miss and one lazy import; its following Get
records one hit and no second import. The fourth Put is fully warm.

Observed context in the supplied runtime record:

```text
CUDA runtime library: libcudart.so.12
CUDA runtime version: 12.8
CUDA driver API version: 13.0
Store binding: /workspace/Mooncake/build-nvlink-host-numa/mooncake-integration/store.cpython-312-aarch64-linux-gnu.so
CPU affinity: 0-143
Memory-node affinity: 0-1
```

## Concurrent benchmark

The evaluated complete stream contains exactly:

```text
devices:    0,1,2,3
sizes:      4096,1048576,16777216,134217728 bytes
iterations: 0,1,2,3
results:    64
summaries:  12
```

All 64 result records have valid SHA256 fields, positive Put/Get throughput,
and consistent mapping miss/import counters. Benchmark summary generation also
proves every Consumer subprocess returned success and emitted the expected
iteration set.

The first complete stream reports the following initial characterization.
Performance remains report-only until machine and topology provenance is
attached.

| Payload | Cold/mixed Put mean GiB/s | Warm Put mean GiB/s | Warm Get mean GiB/s | Cold import p50 us | Get latency p50/p95/p99 us |
|---:|---:|---:|---:|---:|---:|
| 4 KiB | 0.0046 | 0.0167 | 0.0260 | 379 | 145 / 180 / 211 |
| 1 MiB | 0.7790 | 3.3498 | 5.7235 | 1064 | 169 / 243 / 256 |
| 16 MiB | 12.7487 | 37.1024 | 63.4150 | 378 | 245 / 284 / 305 |
| 128 MiB | 51.3996 | 90.7113 | 147.1797 | 556 | 836 / 883 / 1310 |

During the four-GPU 128 MiB phase, Master reported:

```text
Mem Storage: 512.00 MB / 592.00 MB (86.5%)
Keys: 4
```

The next Master sample returned to `0 B / 592.00 MB` and zero keys, providing
evidence that benchmark objects were removed after validation.

## PR readiness and remaining evidence

The supplied artifacts are sufficient to claim that the dual-node GB200 data
plane passed for one named run: Provider capacity/readiness, HBM Put/Get,
cold/import-to-hit behavior, four concurrent Consumer GPUs, and 128 MiB
payload correctness all succeeded.

They are not sufficient to claim full hardware acceptance. Before the PR is
described as satisfying every design gate, attach or explicitly waive:

1. Node A and Node B `git rev-parse HEAD` output;
2. clean original `bench.jsonl` without terminal transcript duplication;
3. build logs and unit/hardware/RDMA JUnit with test and skip counts;
4. strict preflight output;
5. complete GPU/NUMA/NVLink/IMEX topology and software provenance;
6. per-NUMA Provider metrics or segment-to-NUMA evidence;
7. Provider stop followed by zero remaining segments/resources;
8. default-off rollback and existing HBM regression;
9. injected mount/allocation failure rollback if required for this PR's
   acceptance claim.

## Suggested PR validation statement

```text
GB200 dual-node HOST_NUMA data-plane validation: PASS (run gb200-0714-2).
The Provider requested 600 MiB and published five chunks with 592 MiB effective
capacity, matching Master exactly. A standalone GPU completed four 128 MiB
HBM Put/Get iterations, and four concurrent GPUs completed 4 KiB, 1 MiB,
16 MiB, and 128 MiB payloads with byte-for-byte/SHA256 correctness and observed
cold-import-to-hit behavior. Build/JUnit, strict preflight, full topology,
RDMA premise, teardown, and rollback evidence are tracked separately and are
not claimed by this runtime report.
```
