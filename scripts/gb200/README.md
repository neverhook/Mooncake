# GB200/NVL72 KVPool and EGM validation

This directory exists only on `codex/egm-store-pool-gb200-validation`. It is
based on the Store EGM pool PR and is not part of a production PR.

## Two-command workflow

Both nodes must have the same checkout. No config file, run ID, Node B IP,
GPU list, NUMA node, HCA, payload matrix, or result copy is required.

Run on Node A and leave it in the foreground:

```bash
scripts/gb200/egm_store_gb200.sh full-provider \
  --listen-ip 10.192.8.81
```

Node A builds the focused targets, runs preflight, selects free ports, starts
Master, Store Provider, and the ordinary-DRAM RDMA target, then prints the exact
Node B command. Run that command on Node B, for example:

```bash
scripts/gb200/egm_store_gb200.sh full-consumer \
  --provider http://10.192.8.81:8079
```

Node B obtains the session manifest from Node A and refuses to run unless both
checkouts have the same SHA. It discovers its routable address and visible
GPUs, builds, runs preflight and the complete matrix, publishes completion to
Node A, then prints one compact evidence block. Node A performs reverse-order
cleanup and prints its evidence block. Paste these two blocks back into Codex:

```text
BEGIN_MOONCAKE_GB200_EVIDENCE_V1 NODE=B
{...}
END_MOONCAKE_GB200_EVIDENCE_V1

BEGIN_MOONCAKE_GB200_EVIDENCE_V1 NODE=A
{...}
END_MOONCAKE_GB200_EVIDENCE_V1
```

Raw artifacts remain under `/tmp/mooncake-egm-gb200/<RUN_ID>/` on each node and
are covered by the SHA-256 artifact manifest in the evidence block. A full
default run is intentionally long because every stable raw-link sample has an
at-least-one-second measurement window and the best RDMA configuration gets a
ten-second sustained run.

If bootstrap port 8079 is occupied, change only Node A's command:

```bash
scripts/gb200/egm_store_gb200.sh full-provider \
  --listen-ip 10.192.8.81 --bootstrap-port 18079
```

The printed Node B command includes the selected port. `--skip-build` is an
advanced rerun option on either full command.

## Report taxonomy

| Report name | Memory path | Submission | Data mover | Acceptance role |
|---|---|---|---|---|
| `EGM_H2D` | remote EGM to local HBM | Host CPU | GPU CE or SM | RFC V1 Get and link ceiling |
| `EGM_D2H` | local HBM to remote EGM | Host CPU | GPU CE or SM | RFC V1 Put and link ceiling |
| `KVPOOL_HOST_H2H_RDMA` | ordinary remote DRAM and local DRAM | Host CPU | RNIC DMA | typical cross-node KVPool baseline |
| `EGM_H2H_CE` | local EGM and remote EGM | Host CPU | GPU CE | supplementary MNNVL evidence |
| `EGM_H2H_SM` | local EGM and remote EGM | Host-launched kernel | GPU SM load/store | supplementary DSA evidence |
| `LOCAL_HOST_MEMCPY` | local DRAM to DRAM | Host CPU | CPU cores | local DRAM baseline |

`EGM_H2H_CE` is host-submitted and GPU-CE-executed. It is not a CPU data
plane. Unsupported EGM H2H is reported as
`UNSUPPORTED_OR_ROUTE_UNVERIFIED` and does not block the Store V1 evidence.

## Measurements and gates

The raw EGM benchmark runs payloads 128 MiB, 512 MiB, 1 GiB, 2 GiB, and 4 GiB
with 1, 2, 4, and 8 streams, first on each GPU independently and then with 2/4
concurrent GPUs. It records one lazy-init probe, three warmups, and ten steady
windows.
Reported ceilings require CV at most 5 percent and no more than 3 percent gain
at the payload/stream matrix edge; otherwise the result is
`UNBOUNDED_BY_MATRIX`. The report shows utilization against the 450 GB/s
single-direction C2C reference without turning that reference into a numerical
code gate.

The Store matrix uses payloads through 2 GiB. A CUDA helper fills and verifies
the entire allocation on device, avoiding multi-GiB Python host copies. Its
first transport initialization sample and two warmups are excluded from the
ten steady samples.

The dependent-load tests cover HBM, local EGM, and remote EGM working sets of
2 MiB, 256 MiB, and 1 GiB. Each of 30 samples performs 262,144 accesses. Raw
cycles/op are retained; ns/op is calculated from the GPU clock queried for that
run. Store latency is named
`store_system_fence_completion_latency` because each 64-bit store is followed
by `__threadfence_system()`.

The ordinary Host H2H baseline reuses `transfer_engine_bench` with NUMA-local
DRAM and RDMA auto-discovery. It scans 64 KiB, 1 MiB, and 8 MiB blocks with
1/4/8 threads, sustains the best configuration for ten seconds, and measures
64 B and 4 KiB QD=1 submit-to-completion latency. That latency is an RNIC
request-completion measurement, not CPU load/store latency.

The final gate requires:

- exact source SHA and manifest digest;
- H2D/D2H raw and Store correctness;
- stable, bounded H2D/D2H CE ceilings;
- Node B NVLink and RDMA byte deltas with no new C2C errors;
- Node A remote-EGM NVLink byte deltas, healthy full-bandwidth C2C/Fabric
  capability, and no new C2C errors; and
- successful Provider, RDMA target, and Master cleanup.

Preflight requires CUDA/IMEX, an active RDMA port, and driver support for
`nvidia-smi nvlink -gt d`, `nvidia-smi c2c -s`, `nvidia-smi c2c -e`, and the
Fabric section of full `nvidia-smi -q`. Container-local `dcgmi`, a reachable
DCGM hostengine, and exporter reconfiguration are not required.

The deployed tools expose cumulative NVLink bytes and C2C capability/error
counters, but not direct C2C traffic bytes. Successful evidence is therefore
explicitly labeled `C2C_ROUTE_INFERRED`: it combines positive NVLink TX/RX
deltas, healthy full-bandwidth C2C/Fabric state, zero C2C error deltas, remote
HOST_NUMA mapping, and end-to-end byte correctness. It must not be described as
direct C2C PMU byte-count verification.

## Advanced and diagnostic actions

The prior config-based commands remain available for focused diagnosis:

```bash
cp scripts/gb200/egm_store_gb200.conf.example egm-store-gb200.conf
scripts/gb200/egm_store_gb200.sh --config ./egm-store-gb200.conf print-config
scripts/gb200/egm_store_gb200.sh --config ./egm-store-gb200.conf build
scripts/gb200/egm_store_gb200.sh --config ./egm-store-gb200.conf preflight
scripts/gb200/egm_store_gb200.sh --config ./egm-store-gb200.conf diagnose
scripts/gb200/egm_store_gb200.sh --config ./egm-store-gb200.conf stop
```

`BUILD_UNIT_TESTS=0` is the default, so the runtime build does not download
GoogleTest. `MC_IMEX_DAEMON_EXTERNAL=1` is the default for containers using the
host-managed IMEX daemon. The EGM pool default is 20 GiB per Provider node.
