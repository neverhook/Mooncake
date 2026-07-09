# GB200 Dual Protocol Warm Read Test Report

Date: 2026-07-09
Branch: `codex/nvlink-host-numa-dual-register`
Commit under test: `de8d16f37f8c05f23ce9c52388efd6eee13a38e3`

## Summary

The dual-node GB200 smoke validated both routing decisions:

- same scale-up-domain selected `nvlink`
- different scale-up-domain selected `rdma`

The warm-read test also confirmed why one-shot `get_into_ms` was misleading.
The first same-domain NVLink read paid lazy fabric import/map/access setup, but
subsequent reads in the same reader process used the cached mapping and were
substantially faster than RDMA for this 128 MiB remote HOST_NUMA DRAM to local
HBM read workload.

No cleanup `cuMemUnmap` or `cuMemAddressFree` failure was observed in this run.

## Environment

- Node A provider IP: `10.192.9.60`
- Node B reader IP: `10.192.9.15`
- Object size: `128 MiB`
- Destination: Node B CUDA device `0`, local HBM pointer
- Provider replica memory kind: `HOST_NUMA`
- Provider scale-up domain: `gb200-nvl`
- Protocol list: `nvlink,rdma`
- RDMA device allowlist: `mlx5_0,mlx5_1,mlx5_4,mlx5_5`
- Metadata server: `etcd://10.192.9.60:2379`
- Master server: `10.192.9.60:50051`

## Scripts

Build:

```bash
cd /workspace/Mooncake
git pull --ff-only neverhook codex/nvlink-host-numa-dual-register
sh scripts/gb200_build.sh
```

Node A control plane:

```bash
STOP_OLD_MASTER=1 sh scripts/gb200_start_control_plane_node_a.sh
```

Node A provider:

```bash
NODE_A_IP=10.192.9.60 SIZE_MB=128 LOCAL_BUFFER_SIZE_MB=256 \
  sh scripts/gb200_provider_dual_node_a.sh
```

Node B same-domain reader:

```bash
GET_WARMUP=1 GET_REPEAT=5 \
NODE_A_IP=10.192.9.60 NODE_B_IP=10.192.9.15 SIZE_MB=128 CUDA_DEVICE=0 \
  sh scripts/gb200_reader_dual_same_domain_node_b.sh
```

Node B different-domain reader:

```bash
GET_WARMUP=1 GET_REPEAT=5 \
NODE_A_IP=10.192.9.60 NODE_B_IP=10.192.9.15 SIZE_MB=128 CUDA_DEVICE=0 \
  sh scripts/gb200_reader_dual_diff_domain_node_b.sh
```

## Test Cases

### Case 1: Same Scale-Up Domain, NVLink Expected

Configuration:

- reader `MC_NVLINK_SCALE_UP_DOMAIN_ID=gb200-nvl`
- expected protocol: `nvlink`
- transfer: remote HOST_NUMA DRAM to local HBM

Selection result:

```text
selected_protocol: nvlink
memory_kind: HOST_NUMA
remote scale_up_domain_id: gb200-nvl
```

Latency result:

| Phase | Iteration | get_into_ms | MiB/s |
|---|---:|---:|---:|
| warmup | 0 | 18.428 | 6,945.84 |
| measured | 0 | 0.790 | 162,021.01 |
| measured | 1 | 0.768 | 166,755.91 |
| measured | 2 | 0.909 | 140,809.28 |
| measured | 3 | 0.758 | 168,761.02 |
| measured | 4 | 0.752 | 170,298.59 |

Summary:

- cold NVLink `get_into_ms`: `18.428`
- warm measured min: `0.752 ms`
- warm measured max: `0.909 ms`
- warm measured avg: `0.795 ms`
- warm measured throughput: `160,936.21 MiB/s`
- warm measured throughput: about `157.16 GiB/s`

### Case 2: Different Scale-Up Domain, RDMA Expected

Configuration:

- reader `MC_NVLINK_SCALE_UP_DOMAIN_ID=gb200-nvl-different`
- provider replica scale-up-domain `gb200-nvl`
- expected protocol: `rdma`
- transfer: remote HOST_NUMA DRAM to local HBM via backend RNICs

Selection result:

```text
selected_protocol: rdma
memory_kind: HOST_NUMA
remote scale_up_domain_id: gb200-nvl
reader scale_up_domain_id: gb200-nvl-different
```

Latency result:

| Phase | Iteration | get_into_ms | MiB/s |
|---|---:|---:|---:|
| warmup | 0 | 10.726 | 11,933.34 |
| measured | 0 | 10.805 | 11,846.36 |
| measured | 1 | 4.270 | 29,979.63 |
| measured | 2 | 4.082 | 31,359.79 |
| measured | 3 | 4.201 | 30,470.63 |
| measured | 4 | 4.177 | 30,644.32 |

Summary:

- cold RDMA `get_into_ms`: `10.726`
- measured min: `4.082 ms`
- measured max: `10.805 ms`
- measured avg: `5.507 ms`
- measured throughput: `23,244.03 MiB/s`
- measured throughput: about `22.70 GiB/s`
- excluding measured iteration 0, steady RDMA avg is about `4.183 ms`
  and about `29.90 GiB/s`

## Interpretation

The same-domain NVLink test now shows the expected warm-path advantage. The
first read is not a pure data-plane measurement because it pays lazy CUDA
fabric import, virtual address reservation, VMM mapping, and access setup. The
subsequent measured reads reuse the process-local mapping cache and are the
right comparison point for steady-state worker behavior.

The different-domain RDMA test also has a warmup effect. The first measured
RDMA iteration is slower than later iterations, likely due to endpoint or
worker-path warmup. The later RDMA iterations are stable around 4.1 to 4.3 ms
for 128 MiB.

For steady-state claims, compare warm NVLink measured iterations against warm
RDMA measured iterations:

- NVLink warm avg: `0.795 ms`
- RDMA measured avg: `5.507 ms`
- RDMA measured avg excluding first measured iteration: about `4.183 ms`

The result supports the intended routing policy:

- same scale-up-domain HOST_NUMA to HBM reads should prefer NVLink/EGM
- cross scale-up-domain reads should select RDMA fallback
- one-shot cold latency should be reported separately from warm read latency

## Validation Gaps

- This report covers one 128 MiB object size. Smaller KV block sizes need their
  own latency distribution because per-transfer fixed cost will dominate.
- Transport-internal timing is not yet split into selector, remote mapping,
  submit, and completion polling phases.
- RDMA endpoint warmup is not separately measured, so iteration 0 is included
  in the reported `measured_get_into_summary`.
- The current production branch still performs NVLink remote fabric import
  lazily; a production prewarm design is documented separately before making
  product-path changes.
