# NVL72 EGM KVPool Architecture on Upstream Mooncake

Date: 2026-07-09
Status: Proposed architecture for re-implementation from `origin/main`
Baseline: `origin/main` (`origin/HEAD -> origin/main`)

## Purpose

This document redesigns the GB200/NVL72 scale-up EGM unified memory pool
support from the upstream Mooncake baseline. It intentionally does not treat
the exploratory `codex/nvlink-host-numa-dual-register` branch as the production
shape. That branch is useful as validation evidence, but several changes are
too invasive for the first production implementation.

The target design has two separable layers:

1. Extend the existing `NvlinkTransport` to support EGM/HOST_NUMA fabric memory
   for same scale-up-domain reads.
2. Add dual-protocol capability for fallback, so the same memory pool can use
   NVLink/EGM within an NVL72 domain and RDMA across domains.

## Independent Architecture Review Input

An independent infrastructure and networking review reached the same main
conclusion: the current branch should be treated as validation evidence and
test harness work, not as the production architecture. The review specifically
called out three production boundaries that this spec adopts:

- keep EGM support as a minimal extension of `NvlinkTransport`
- keep dual-protocol fallback in worker-side candidate selection, not inside
  `NvlinkTransport`
- avoid broad Store read-path renames or persistent `selected_protocol`
  metadata when the upstream `SelectBestReplica` contract can be extended more
  narrowly

## Principles

- Keep the upstream `ReplicaType::MEMORY` contract. EGM does not require a new
  replica type.
- Preserve existing read-selection naming where possible. Upstream
  `SelectBestReplica` can be extended or wrapped; a broad rename such as
  `select for read` is not required to prove the feature.
- Keep master scheduling requester-agnostic in the first version. The worker
  has the local GPU pointer type, scale-up-domain id, transport health, and
  RNIC/GPU locality needed for protocol selection.
- Treat fabric export/import/map as lifecycle work, not as unavoidable
  per-transfer work.
- Keep fallback explicit. `NvlinkTransport` should not silently resubmit a
  failed transfer over RDMA.
- Keep transfer decisions local and ephemeral. `selected_protocol` is a result
  of one worker-side candidate decision; it is not persistent replica metadata.
- Preserve upstream code style and naming unless a semantic boundary genuinely
  changes. Avoid broad renames, comment churn, or validation-only helper names
  in production code.
- Keep the first patch set narrow: transport capability, metadata extension,
  worker candidate selection, and validation.

## Problems Observed During Validation

The validation sequence exposed several separate issue classes:

- Test harness metadata defaults: `nvlink_transport_test` initially tried to
  use an etcd metadata plugin in an environment where the self-contained test
  should use P2P metadata.
- Build environment isolation: Go toolchain download failed behind a network
  firewall; the practical fix was to use the same reachable download sources as
  `dependencies.sh`.
- HOST_NUMA VMM access: allocated HOST_NUMA fabric memory must set CPU and GPU
  access correctly. Without CPU access, even local page touch and RDMA MR
  probing are misleading.
- RDMA registration mode: on this GB200 environment HBM registration uses the
  modern DMA-BUF path; legacy `ibv_reg_mr` on CUDA UVA can fail when
  `nvidia-peermem` is absent.
- RDMA network plane selection: auto-discovery can pick the wrong HCA plane.
  Validation pinned backend RNICs to `mlx5_0,mlx5_1,mlx5_4,mlx5_5`.
- CUDA context lifecycle: reader HBM must use the CUDA primary context expected
  by the store/transport teardown path.
- Lazy fabric import: the first NVLink `get_into` paid
  `cuMemImportFromShareableHandle`, VA reserve, map, and access setup. Warm
  reads showed the real steady-state advantage.
- Cleanup symmetry: imported VMM mappings must be unmapped and address-freed
  with the exact locally reserved base and length.

These are not one bug. They are signals that production support needs clear
lifecycle boundaries.

## Current Upstream Shape

Upstream `NvlinkTransport` already has the basic fabric handle mechanism:

- `registerLocalMemory()` exports `CU_MEM_HANDLE_TYPE_FABRIC` when fabric memory
  is supported.
- `relocateSharedMemoryAddress()` lazily imports the remote fabric handle,
  reserves local VA, maps it, sets access, and caches the mapping in
  `remap_entries_`.
- `submitTransferTask()` copies through CUDA memcpy/batch memcpy after the
  target address is relocated.

Upstream multi-protocol support already routes by buffer protocol for
comma-separated segment protocols. It currently prioritizes protocols such as
`hip`, `cxl`, `rdma`, and `tcp`, and it has cross-host skip logic for HIP.

Upstream Store selection already has a local helper named `SelectBestReplica`
that chooses the best complete replica by locality and replica type. The EGM
feature should extend the candidate model without forcing a broad naming churn.

## Layer 1: EGM/HOST_NUMA in NvlinkTransport

### Allocation and provider registration

Provider memory pool setup should create or accept a HOST_NUMA VMM allocation:

- `CU_MEM_ALLOCATION_TYPE_PINNED`
- `CU_MEM_LOCATION_TYPE_HOST_NUMA`
- configured NUMA node id
- `CU_MEM_HANDLE_TYPE_FABRIC`

The local provider must:

1. map the allocation into UVA
2. set CPU access when the memory is CPU-touched or RDMA-registered
3. set GPU access for GPUs that may serve local copies
4. export a fabric handle
5. publish metadata only after required setup succeeds

The export is provider lifecycle work. It belongs to segment mount or memory
pool initialization, not object read.

### NUMA and export-device sharding

Provider-side EGM pool allocation must preserve both CPU NUMA locality and
fabric-route balance. A production GB200 node should not publish one large
HOST_NUMA allocation exported through one CUDA device if the node has multiple
NUMA domains and multiple GPUs capable of serving the fabric path.

The provider should split the pool in two dimensions:

1. split by CPU NUMA node, so each backing allocation is local to the CPU socket
   and attached Grace memory that owns it
2. split again by export device inside that NUMA domain, so fabric traffic does
   not converge on a single GPU

For example, on a GB200 node with two NUMA domains and two GPUs per NUMA domain,
a 600 GB DRAM contribution should be modeled as four shards:

| Shard | NUMA node | Export device | Size |
|---|---:|---|---:|
| 0 | 0 | GPU local to NUMA 0 | 150 GB |
| 1 | 0 | other GPU local to NUMA 0 | 150 GB |
| 2 | 1 | GPU local to NUMA 1 | 150 GB |
| 3 | 1 | other GPU local to NUMA 1 | 150 GB |

Each shard should have its own buffer descriptor or subsegment descriptor with
`memory_kind=HOST_NUMA`, `scale_up_domain_id`, `numa_node`, export-device
identity, base address, length, and fabric handle. Allocation, CPU page touch,
RDMA registration, and fabric export should stay within the shard's NUMA
domain. Cross-UPI placement is a correctness and performance smell, not merely
a scheduler preference.

Mooncake already has pieces of this model: host buffer allocation can be
NUMA-segmented, topology code prefers same-NUMA HCAs, and the current GB200
validation harness can choose one `MC_NVLINK_HOST_NUMA_NODE`. The missing
production piece is explicit export-device sharding inside each NUMA domain and
metadata that lets readers understand which export device backs a shard.

The allocator should balance new objects across `(numa_node, export_device)`
shards while preserving locality. A reader in the same scale-up domain may also
prefer a shard whose export device is closest to the destination GPU when the
topology layer can answer that question.

### Metadata extensions

Extend `TransferMetadata::BufferDesc` minimally:

- `memory_kind`: empty for legacy, `HOST_NUMA` for EGM pool buffers
- `scale_up_domain_id`: configured NVL72 scale-up domain id
- `numa_node`: provider-local NUMA node for HOST_NUMA shards
- `export_device_id` or `export_device_pci_bus_id`: CUDA device that exported
  the fabric handle. Serialized metadata should prefer a stable identity such
  as PCI bus id; local CUDA ordinal can remain a runtime convenience.
- `shard_id` or descriptor generation when needed for cache invalidation
- existing `shm_name`: serialized fabric handle for `nvlink`

The existing `protocol` field remains transport-specific. A dual-registered
range can produce one `nvlink` buffer descriptor and one `rdma` buffer
descriptor that refer to the same backing range.

Do not store `selected_protocol` in `BufferDesc`, allocated buffer descriptors,
or any serialized replica state. Selection depends on requester-local inputs
such as destination pointer type, local scale-up-domain id, transport health,
and RNIC/GPU locality. Persisting the selected protocol would turn a local
decision into stale global metadata.

### Consumer import and mapping

Remote EGM import should be a named transport operation, not only a side effect
inside `relocateSharedMemoryAddress()`.

Recommended API shape:

```cpp
Status NvlinkTransport::prepareRemoteMemory(SegmentID segment_id,
                                            const BufferDesc &buffer);
Status NvlinkTransport::lookupRemoteMapping(SegmentID segment_id,
                                            uint64_t remote_addr,
                                            size_t length,
                                            uint64_t &local_addr);
```

The initial implementation may keep lazy fallback through
`relocateSharedMemoryAddress()`, but production code should expose explicit
prewarm and report whether a transfer used a cold or warm mapping.

### Cache key and invalidation

The mapping cache key should include enough identity to avoid stale mappings:

- segment id
- remote buffer base address
- transport endpoint or segment owner identity
- descriptor generation/version when available

Invalidate when segment metadata changes, the provider process identity
changes, a remote segment is unmounted, or CUDA import/map fails.

## Layer 2: Dual Protocol Fallback

### Data model

A dual-protocol EGM pool is one memory pool and one object copy, not two
replicas. It is reachable through two transport capabilities:

- `nvlink` for same scale-up-domain HOST_NUMA to local HBM reads
- `rdma` for cross-domain reads or when NVLink is disabled/unhealthy

The metadata model should represent capabilities at the buffer level because
RDMA and NVLink carry different transport metadata. This matches upstream
multi-protocol segment behavior and avoids inventing a new replica type.

### Worker-side candidate selection

Keep the master returning complete replicas. On the worker, build transfer
candidates:

```text
(replica, protocol, reason, readiness)
```

Candidate order:

1. local MEMORY
2. same-domain HOST_NUMA via `nvlink`, when destination is HBM and mapping is
   prepared or can be prepared by policy
3. reachable MEMORY via `rdma`
4. existing NOF, LOCAL_DISK, and DISK fallbacks

This can be implemented as an extension to `SelectBestReplica` plus an optional
selected-protocol output, or as a small new helper with a narrow call-site
surface. Avoid broad renames that imply the whole store read path was
redesigned.

The candidate helper should return an explicit reason for rejecting NVLink, for
example domain mismatch, destination is not HBM, remote memory is not
`HOST_NUMA`, mapping prewarm failed, or transport is disabled. These reasons
are required for debugging and for deciding whether RDMA fallback is expected.

### H2H and host-destination reads

Mooncake has host-destination read paths: `get_into` can target host memory, and
the current store code distinguishes device destinations from host destinations
before selecting an EGM candidate. That boundary should remain explicit in the
production design.

The first EGM fast path is remote HOST_NUMA DRAM to local HBM. For H2H reads
such as remote HOST_NUMA DRAM to local host memory, the selector should not
choose `nvlink` by default. It should choose RDMA when the remote buffer
advertises RDMA, or the existing CPU/host fallback path when RDMA is not
available.

Do not treat CPU visibility of a mapped fabric range as sufficient proof that
H2H over NVLink is a supported fast path. If a future implementation needs
EGM-backed H2H, add an explicit capability such as `cpu_accessible` or
`egm_host_visible`, define the copy primitive, and add separate tests for host
destination behavior. It should not reuse the HBM-only selector rule.

### Transport submission

For a selected dual-protocol candidate, submit through an explicit protocol:

```cpp
mp_submitTransfer(batch_id, requests, selected_protocol)
```

Single-protocol segments keep the existing `submitTransfer()` path.

`NvlinkTransport` handles only NVLink/EGM. If the selector chooses RDMA, the
request is submitted to RDMA directly. If NVLink prewarm fails and strict mode
is disabled, the selector can demote NVLink before submission and choose RDMA.

Do not add a generic default such as "multi-protocol means RDMA if no selected
protocol is present" in the transfer submitter. That masks selector bugs and
can silently route same-domain EGM traffic over RDMA. Missing selected protocol
on a dual-protocol candidate should be treated as an invalid selection state.

## Lifecycle

### Worker startup

- initialize transports
- publish local RPC metadata
- load local scale-up-domain id
- discover RNICs and GPUs

Do not import every remote fabric handle blindly at startup.

Validation scripts may pin RNICs to backend-network devices, but production
startup should use topology, allowlists, health probes, and RNIC/GPU locality
instead of hard-coded device names.

### Segment mount

Provider-side segment mount prepares local backing memory:

- HOST_NUMA allocation and access rights
- NVLink fabric export
- RDMA MR registration
- metadata publication

Strict mode should fail mount if configured EGM support cannot be prepared.
Non-strict mode can publish RDMA-only capability with diagnostics.

### Query and prewarm

After a worker learns a replica or receives scheduler hints, it can prewarm
remote transport state:

- RDMA endpoint setup for expected providers
- NVLink fabric import/map/access for same-domain HOST_NUMA segments

Lazy import remains a correctness fallback, but metrics must identify cold
transfers.

### Transfer

The transfer path should be a cache lookup plus offset calculation:

```text
local_mapped_base_or_remote_rkey + object_offset + slice_offset
```

Then submit batched transfer requests through the selected transport.

### Teardown

- unregister local RDMA MR when local buffer or segment is released
- unmap imported NVLink VMM mappings with the exact local base and length
- remove metadata after transport cleanup succeeds, or mark stale entries for
  cleanup if the process dies

## Failure and Fallback Policy

| Failure | Strict Mode | Non-Strict Mode |
|---|---|---|
| HOST_NUMA allocation/export fails | fail mount | publish RDMA-only |
| same-domain fabric import fails during prewarm | fail prewarm and mark unhealthy | demote NVLink candidate |
| same-domain fabric import fails during lazy transfer | return explicit error | retry at caller/selector layer over RDMA if policy allows |
| scale-up-domain mismatch | do not select NVLink | select RDMA if available |
| RDMA backend RNIC unavailable | fail RDMA candidate | select NVLink only if same-domain and healthy |

Transport code should not silently change protocols after submission. Fallback
belongs to candidate selection or an explicit higher-level retry policy.

## Observability

Required metrics and logs:

- provider RDMA registration time
- provider NVLink fabric export time
- reader local destination registration time
- reader remote NVLink import/map/access time
- RDMA endpoint setup time
- cold vs warm transfer latency
- selected protocol and rejection reason for skipped protocols
- scale-up-domain id and memory kind in debug logs
- cleanup errors with mapping identity and length

Performance reports must distinguish:

- cold end-to-end read
- warm steady-state read
- pure CUDA copy time when CUDA events are available

## Test Plan

### Transfer Engine Unit Tests

- HOST_NUMA fabric allocation grants CPU and GPU access.
- `NvlinkTransport` exports HOST_NUMA fabric metadata.
- imported mapping cleanup uses local mapped base and length.
- provider EGM pool sharding creates one descriptor per
  `(numa_node, export_device)` shard.
- shard metadata carries NUMA node and stable export-device identity.
- scale-up-domain mismatch rejects NVLink before import.
- explicit prewarm populates mapping cache and subsequent relocate is cache-only.

### Store Selection Tests

- same-domain HOST_NUMA + HBM destination selects `nvlink`.
- different-domain HOST_NUMA + HBM destination selects `rdma`.
- non-HBM destination does not incorrectly select NVLink EGM.
- H2H host-destination reads select RDMA or existing host fallback, not the
  HBM-only EGM path.
- RDMA-only compatibility is unchanged.
- existing `SelectBestReplica` behavior for local memory, NOF, LOCAL_DISK, and
  DISK remains unchanged.

### GB200 Integration Tests

- single-node HOST_NUMA fabric read: remote DRAM to local HBM.
- dual-node same-domain: selected `nvlink`, warm reads faster than cold reads.
- dual-node different-domain: selected `rdma`.
- RDMA backend RNIC allowlist avoids cross-plane connection failures.
- NUMA/export-device sharding: a 600 GB GB200 node contribution is published as
  four 150 GB local shards on a two-NUMA, four-GPU node, with no cross-UPI page
  placement in the expected steady state.
- cleanup emits no CUDA unmap/address-free errors.

## Migration From Current Branch

Keep from the validation branch:

- GB200 scripts and test methodology.
- HOST_NUMA access fixes.
- RDMA backend RNIC pinning in validation wrapper.
- CUDA primary context handling in reader harness.
- cleanup symmetry for imported VMM mappings.
- cold/warm latency reporting.

Rework before production:

- avoid broad Store read selector renames
- avoid baking validation-only scripts into production APIs
- move EGM remote import/map to explicit prewarm/lifecycle APIs
- keep dual-protocol fallback at selector or higher retry layer
- replace single-node HOST_NUMA validation knobs with production NUMA and
  export-device shard metadata
- reduce branch delta by starting from `origin/main` and landing the feature in
  small slices

## Recommended Implementation Slices

1. `NvlinkTransport` HOST_NUMA metadata, cleanup hardening, and stable
   export-device identity.
2. Explicit NVLink remote mapping prewarm/cache API.
3. Provider EGM shard allocator: split by NUMA node and export device.
4. Minimal metadata fields: `memory_kind`, `scale_up_domain_id`, `numa_node`,
   and export-device identity.
5. Store candidate selector extension with selected protocol output and H2H
   rejection reasons.
6. Dual-protocol EGM/RDMA mount path.
7. GB200 validation scripts and docs.
8. Production metrics for cold/warm transfer phases.

This ordering proves the EGM transport premise before changing broad Store
selection behavior, and it keeps RDMA fallback isolated from NVLink transport
internals.
