# GB200 Dual Protocol Prewarm Architecture

Date: 2026-07-09
Status: Draft spec for production implementation

## Context

The GB200 dual-protocol validation path proves that one HOST_NUMA memory
replica can be reachable through both `nvlink` and `rdma`. The current harness
also shows an important performance measurement issue: the first same-domain
NVLink `get_into` can include CUDA fabric handle import, virtual address
reservation, VMM mapping, and access setup. RDMA has analogous one-time costs,
such as memory registration and endpoint setup, but those costs are already
mostly paid before the measured read in the current script.

Production behavior should not rely on the first real read to discover and
prepare remote transport state. Worker startup, segment mount, or an explicit
prewarm step should prepare the local fast-path state for both protocols.

## Goals

- Keep the store replica model unchanged: a dual-protocol object is still one
  memory replica with one backing byte range.
- Move local registration and remote access preparation out of the first
  business transfer whenever the remote segment is known in advance.
- Make the transfer fast path use cached transport state: selected protocol,
  local registered buffer metadata, remote base mapping, and offset.
- Preserve RDMA fallback for cross scale-up-domain traffic and for NVLink
  preparation failures when strict mode is disabled.
- Provide measurement categories that separate cold setup, warmup, and steady
  state transfer latency.

## Non-Goals

- Do not introduce a new `ReplicaType`.
- Do not make the master choose requester-specific protocols in the first
  production version.
- Do not hide a failed NVLink submission by internally resubmitting over RDMA
  inside `NvlinkTransport`.
- Do not require all possible remote segments to be imported at process start;
  large deployments need bounded, demand-aware prewarm.

## Current Behavior

Provider-side registration already publishes the information needed by each
transport:

- RDMA registration creates local MRs and publishes lkey/rkey metadata.
- NVLink HOST_NUMA registration exports a CUDA fabric handle and publishes
  memory kind, scale-up domain id, base address, size, and handle data.

Reader-side behavior is asymmetric today:

- The reader's local HBM buffer registration is explicit and measured outside
  `get_into`.
- RDMA can use the provider-published rkey and reader-local MR metadata during
  transfer.
- NVLink fabric import is lazy. `relocateSharedMemoryAddress()` imports the
  fabric handle, reserves VA, maps it, sets access, and caches the mapped base
  in `remap_entries_` during the first transfer that touches the segment.

The lazy NVLink import is correct functionally, but it makes one-shot latency
comparisons misleading.

## Target Architecture

### Provider local prepare

When a worker owns a HOST_NUMA segment, mount/register prepares both transports
before the segment becomes eligible for replica placement.

1. Allocate or accept the HOST_NUMA VMM range.
2. Register the range with RDMA when RDMA is enabled.
3. Export the range as a CUDA fabric handle when NVLink HOST_NUMA is enabled.
4. Publish one segment descriptor containing protocol-specific buffer entries.
5. Mark the segment ready only after required protocol metadata is published.

Strict mode controls failure handling:

- strict enabled: fail the mount if configured NVLink HOST_NUMA registration
  fails
- strict disabled: keep RDMA-only capability and publish diagnostics

### Reader remote prepare

Reader-side prepare is responsible for converting remote metadata into local
cached fast-path state.

For RDMA:

1. Discover reachable local RNICs.
2. Register local destination buffers before transfer.
3. Establish or prewarm endpoints for likely provider endpoints.
4. Cache remote address/rkey metadata with the selected replica descriptor.

For NVLink HOST_NUMA:

1. Filter remote buffers by `memory_kind=HOST_NUMA`.
2. Require matching `scale_up_domain_id`.
3. Import the CUDA fabric handle.
4. Reserve and map local VA for the remote range.
5. Set GPU access for local devices that may copy from the mapping.
6. Cache `(segment_id, remote_base) -> local_mapped_base, length`.

The transfer path should not repeat these operations while the cache entry is
valid.

### Fast path

After metadata resolution and candidate selection, a transfer should need only:

1. identify `(replica, selected_protocol)`
2. find the cached local destination registration
3. find the cached remote mapping or RDMA remote key
4. compute `remote_base + object_offset + slice_offset`
5. submit one batched transfer through the selected transport

Same-domain HOST_NUMA-to-HBM reads use NVLink when a valid cached mapping is
available. Cross-domain reads use RDMA.

## Lifecycle

### Startup

Workers initialize transport plugins and publish local RPC metadata. They do
not need to import every remote fabric handle immediately. They should be able
to prewarm selected peers or segments once those are known.

### Segment mount

Mount creates provider-local registration state and publishes metadata. For
managed memory pools this is the preferred point to pay provider-side RDMA
registration and NVLink fabric export costs.

### Replica discovery

Readers learn replica metadata from the master. The master remains
requester-agnostic and returns complete replicas. The reader constructs
transfer candidates locally because it knows destination pointer type, local
scale-up domain, local transport health, and local RNIC/GPU state.

### Prewarm

Prewarm can be triggered by one or more events:

- after reader startup for configured colocated peers
- after segment discovery for same scale-up-domain HOST_NUMA segments
- after scheduler hint for expected cache reads
- after first query result but before the first latency-sensitive transfer

The first production version can implement explicit prewarm and lazy fallback.
Lazy fallback remains necessary for correctness when a segment was not known in
advance, but steady-state measurements should report whether a transfer used a
warm cache.

### Teardown and invalidation

Cache entries must be invalidated when:

- the remote segment is unmounted
- the segment descriptor version changes
- the provider process or RPC endpoint changes identity
- CUDA import/map fails
- scale-up domain metadata changes

NVLink imported mappings are released with the exact locally reserved base and
length. RDMA endpoints and local MRs follow existing transport teardown rules.

## Candidate Selection

The worker ranks `(replica, protocol)` candidates:

1. local memory copy
2. same-domain HOST_NUMA over NVLink when destination is HBM and mapping is
   prepared or can be prepared under the current policy
3. RDMA when RNIC path and remote rkey are available
4. disk/offload fallbacks

The selected protocol should be explicit in logs and metrics. A failed NVLink
prewarm can demote that candidate before submit. A failed NVLink transfer
should return a clear error to the caller rather than silently changing the
protocol inside the transport.

## Metrics

Metrics should separate these phases:

- provider local prepare: RDMA MR registration, NVLink fabric export
- reader local prepare: local destination buffer registration
- reader remote prepare: RDMA endpoint setup, NVLink fabric import/map/access
- cold read: first read that includes missing lazy preparation
- warm read: read where all required local and remote transport state is cached
- cleanup: unregister, close, unmap, endpoint teardown

Throughput comparisons between NVLink and RDMA should use warm read latency for
steady-state claims. Cold latency remains useful for startup and cache-miss
analysis, but it is not a pure data-plane measurement.

## Validation Plan

The GB200 dual-node scripts should support:

- `GET_WARMUP=0 GET_REPEAT=1` for current one-shot behavior
- `GET_WARMUP=1 GET_REPEAT=N` to pay lazy mapping once and measure warm reads
- same scale-up-domain reader expecting `selected_protocol=nvlink`
- different scale-up-domain reader expecting `selected_protocol=rdma`

Expected evidence:

- same-domain cold NVLink read may be slower because it includes lazy import
- same-domain warm NVLink reads should avoid repeated import/map work
- diff-domain RDMA reads should not select NVLink
- cleanup must not emit CUDA unmap/address-free errors

If warm NVLink remains slower than RDMA, the next investigation should add
transport-internal timings around `relocateSharedMemoryAddress`,
`cudaMemcpyBatchAsync` submission, and completion polling before changing the
selection policy.
