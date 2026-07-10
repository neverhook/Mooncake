# NVL72 HOST_NUMA NVLink KV Pool V1 Design

- **Status:** Approved design, pending implementation plan
- **Date:** 2026-07-10
- **Baseline:** `origin/main` at `98ff4e4787e99265d25938139551841350ca5f4e`
- **Scope:** Mooncake Store and the legacy `NvlinkTransport` on GB200/NVL72-class systems

## Summary

This design adds a Store-owned CUDA VMM `HOST_NUMA` memory pool that is
reachable through the existing multi-node `nvlink` transport. The target data
path is bidirectional KVCache transfer between a Consumer GPU's HBM and a
Provider's HOST_NUMA memory within one scale-up Fabric domain.

V1 deliberately changes the allocation backing without changing the Store or
Transfer Engine wire model:

- the protocol remains `nvlink`
- Store segments and replica descriptors keep their current schemas
- TE `SegmentDesc` and `BufferDesc` keep their current schemas
- the existing serialized Fabric handle remains the remote access descriptor
- Consumer import, map, access, and copy do not branch on memory kind

The feature is configured locally and is disabled by default. RDMA fallback,
Fabric-domain selection, formal HOST_NUMA-to-HOST_NUMA transfer, online atomic
pool addition, and mapping generation/invalidation are separate follow-up
projects.

## Goals

1. Allocate the Provider's Store capacity from CPU DRAM using CUDA VMM
   `CU_MEM_LOCATION_TYPE_HOST_NUMA`.
2. Export each Provider allocation with `CU_MEM_HANDLE_TYPE_FABRIC` and reuse
   the current NVLink registration and Consumer import path.
3. Preserve NUMA locality by creating independent allocations per selected CPU
   NUMA node.
4. Preserve Store ownership of capacity, object placement, and allocation
   lifetime.
5. Preserve existing HBM/NVLink behavior when the feature is disabled.
6. Keep the allocation layout compatible with later RDMA registration without
   implementing RDMA fallback in V1.
7. Make setup failures atomic from the Provider process's perspective and
   diagnosable in production.

## Non-goals

V1 does not implement:

- RDMA registration as a second live transport capability
- automatic NVLink-to-RDMA fallback
- `fabric_domain_id`, `memory_kind`, `numa_node`, generation, or protocol-hint
  fields in Store or TE metadata
- Consumer-side topology or protocol candidate selection
- formal HOST_NUMA-to-HOST_NUMA transfer support or performance guarantees
- online atomic hot-add of multiple Store segments
- per-GPU or export-device sharding of the Provider pool
- NUMA-aware object placement in the Master
- eager import, prewarm, health demotion, or a mapping circuit breaker
- restart-safe invalidation of Consumer NVLink mappings

The restart/mapping ABA risk is tracked separately in
[Mooncake issue #2832](https://github.com/kvcache-ai/Mooncake/issues/2832).

## Current Architecture Facts

### Store and TE metadata have different roles

Mooncake currently has two segment concepts:

- A Store `Segment` is a Master-managed allocatable capacity region. It carries
  the Provider identity, base, size, TE endpoint, protocol, and host identity.
- A TE `SegmentDesc` is the transport registration catalog for one endpoint.
  Its `BufferDesc` entries describe registered address ranges and carry the
  serialized CUDA IPC or Fabric handle in `shm_name`.

An object replica descriptor identifies a Provider endpoint and an object
address. The Consumer opens the endpoint through TE, finds the `BufferDesc`
covering the object address, imports its handle if necessary, and relocates the
remote address into the locally mapped VA.

The Store descriptor is therefore an object locator, while TE metadata is the
transport registration truth. V1 does not add allocation-backend information
to either model because the existing Fabric import sequence is identical for
device-backed and HOST_NUMA-backed VMM allocations.

### Global segment and local buffer are independent

`global_segment_size` is Store capacity. A nonzero value causes RealClient to
allocate memory and mount it into the Master as one or more Store segments.

`local_buffer_size` is a per-RealClient local transfer workspace. It is not
mounted into the Master. High-level byte APIs can allocate temporary Put/Get
buffers from it; direct APIs can instead use caller-owned memory registered by
`register_buffer()`.

There is no explicit Provider or Consumer role in current setup. The following
combinations remain valid:

| `global_segment_size` | `local_buffer_size` | Effective role |
|---:|---:|---|
| `> 0` | `0` | Provider-only |
| `0` | `> 0` | Consumer with a setup-time local workspace |
| `> 0` | `> 0` | Hybrid Provider/Consumer |

V1 does not infer or rewrite these roles.

### Existing NVLink behavior is already memory-kind agnostic on import

In Fabric mode, `NvlinkTransport::registerLocalMemory()` retains the VMM
allocation handle for a registered VA and exports a Fabric handle. On the
Consumer, `relocateSharedMemoryAddress()` imports that handle, reserves a local
VA, maps it, grants GPU access, and caches the mapping.

The import APIs do not take a memory-kind argument. The imported allocation's
properties can be queried from its handle, but the Consumer does not need them
to map or access the range. This is why `memory_kind` is a local allocation
parameter rather than a V1 metadata field.

## Approaches Considered

### 1. Existing `nvlink` protocol with a different backing (selected)

Store setup selects the CUDA VMM allocation location locally. The existing
Store segment, TE BufferDesc, Fabric handle, and Consumer path remain intact.

This keeps the change concentrated in allocation, setup orchestration, and
resource lifecycle. It also leaves a clean seam for later RDMA registration.

### 2. A new `nvlink_host_numa` protocol

This would make the backing explicit in protocol dispatch, but it would
duplicate NVLink registration/import/copy behavior and spread the backing type
through Store and TE. The protocol would express an allocation detail rather
than a distinct transfer mechanism.

### 3. A metadata-rich dual-protocol foundation in V1

This would add memory kind, Fabric domain, topology, candidate selection, and
RDMA fallback together. It may be appropriate for V2, but it combines several
independent failure and lifecycle models before V1 needs them.

## Configuration

The feature is available only through the existing ConfigDict setup path.
Fixed-argument Python setup, the C ABI, Rust, and Go bindings remain unchanged
in V1.

New flat keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enable_nvlink_host_numa` | boolean string | `false` | Use HOST_NUMA VMM for this process's nonzero global and local allocations |
| `nvlink_host_numa_nodes` | `auto` or CSV integers | `auto` | NUMA nodes used by a nonzero global pool |

Existing size keys keep their current defaults and meanings:

- `global_segment_size` is total Provider capacity, not capacity per NUMA node
- `local_buffer_size` remains a single local workspace size
- `local_buffer_size` keeps its current default; Provider deployments that do
  not need a local workspace should explicitly set it to `0`

Example Provider:

```python
{
    "local_hostname": "provider-a:12345",
    "metadata_server": "...",
    "master_server_addr": "...",
    "protocol": "nvlink",
    "global_segment_size": "600 GB",
    "local_buffer_size": "0",
    "enable_nvlink_host_numa": "true",
    "nvlink_host_numa_nodes": "auto",
}
```

Example HBM Consumer:

```python
{
    "local_hostname": "consumer-a:12345",
    "metadata_server": "...",
    "master_server_addr": "...",
    "protocol": "nvlink",
    "global_segment_size": "0",
    "local_buffer_size": "0",
}
```

The Consumer does not need to enable HOST_NUMA allocation to import a remote
HOST_NUMA Fabric handle. It enables the flag only if its own nonzero local
buffer should use HOST_NUMA VMM.

`nvlink_host_numa_nodes` is evaluated only when the feature is enabled and
`global_segment_size > 0`. Local-buffer placement follows the setup CPU and
does not consume this list.

Enabling the feature with a protocol other than `nvlink` is an error. Enabling
it in a build without CUDA/MNNVL support is also an error. V1 does not silently
fall back to HBM or another protocol.

## Component Design

### `NvlinkVmmAllocation`

Introduce a move-only RAII allocation object at the NVLink/TE boundary. It owns:

- mapped base address
- mapped length
- CUDA VMM allocation handle state
- allocation location and local NUMA/device identity for diagnostics
- whether the allocation was created as Fabric-exportable

It is responsible for:

1. querying allocation granularity
2. creating the physical allocation
3. reserving VA
4. mapping the allocation
5. setting CPU and GPU access
6. releasing the original allocation-handle reference after mapping
7. unmapping, freeing VA, and releasing retained resources during destruction

The local allocation parameter distinguishes `DEVICE` from `HOST_NUMA`; this
parameter is never serialized.

The existing HBM VMM allocation path should delegate to the same primitive
without changing its public behavior. Existing pointer-returning
`allocatePinnedLocalMemory()`/`freePinnedLocalMemory()` entry points remain
source-compatible wrappers.

### Access descriptors

Every Provider global HOST_NUMA allocation grants read/write access to:

- the owning CPU NUMA location
- every CUDA device visible in the Provider process

Every local HOST_NUMA buffer grants the same CPU and visible-GPU access. CPU
access is required for same-process Store copies, local validation, and later
host-MR registration. Granting all visible GPUs is required because V1 has no
per-object GPU locality selector.

A Consumer imported mapping grants read/write access to all CUDA devices
visible in that Consumer. V1 does not claim that the Consumer CPU can directly
dereference a remote imported VA.

### Global pool NUMA discovery

With `nvlink_host_numa_nodes=auto`, setup:

1. enumerates visible CUDA devices
2. obtains each device's PCI BDF
3. reads `/sys/bus/pci/devices/<BDF>/numa_node`
4. rejects unknown or negative nodes
5. deduplicates and sorts the result

An explicit CSV list overrides discovery. Every configured node must be an
online valid NUMA node. Empty discovery or an invalid override fails setup.

NUMA remains Provider-local state. Node IDs are not published in Store or TE
metadata.

### Global capacity planning

For every selected NUMA node, setup queries the minimum CUDA allocation
granularity for the HOST_NUMA allocation properties. It computes a common
alignment compatible with all selected nodes and derives:

```text
effective_total = floor(requested_total / common_alignment)
                  * common_alignment
```

Setup fails if the effective total cannot give at least one allocation unit to
every selected NUMA node. Whole alignment units are distributed as evenly as
possible, so node capacities differ by no more than one common unit. The
effective total never exceeds the configured total.

Each node's capacity is then split into chunks no larger than
`globalConfig().max_mr_size`. The maximum chunk size is itself rounded down to
the node's allocation granularity; a zero result is invalid. A logical NUMA
shard can therefore contain multiple Store segments, but no Store segment
crosses NUMA nodes or VMM allocation handles.

Each chunk has:

- one HOST_NUMA VMM allocation
- one Fabric-exportable allocation handle
- one TE BufferDesc after registration
- one Store Segment after mount

An object allocated in a Store segment never crosses a Fabric handle boundary.

### Local buffer placement

When HOST_NUMA allocation is enabled and `local_buffer_size > 0`, setup chooses
the local buffer's NUMA node from the CPU executing setup:

```text
sched_getcpu() -> numa_node_of_cpu() -> CU_MEM_LOCATION_TYPE_HOST_NUMA.id
```

This makes explicit the node on which ordinary first-touch allocation would
have occurred at that moment. The local buffer is still a VMM HOST_NUMA
allocation; it is not reverted to `aligned_alloc`.

The local buffer:

- remains one ClientBufferAllocator arena
- is CPU- and GPU-accessible
- is registered as a local transfer source/destination
- is not mounted into the Master
- is not Fabric-exportable
- is not published in remote TE metadata

If the process can migrate across NUMA nodes, the setup-time CPU determines the
placement, matching existing first-touch behavior. Setup logs the CPU, selected
NUMA node, and current CUDA device for diagnosis, but the CUDA device does not
override CPU locality.

Failure to resolve either the setup CPU or its NUMA node is a setup error; V1
does not silently choose GPU 0 or the first global-pool NUMA node.

### Store ownership

`RealClient` owns the global and local RAII allocations. For global chunks it
also records the returned Store segment IDs. `ClientBufferAllocator` can
suballocate a local VMM range, but it does not own or directly free the VMM
allocation.

This preserves the distinction between:

- Store ownership: capacity, object placement, mount/unmount, and lifetime
- NVLink allocation implementation: CUDA VMM calls and cleanup
- TE ownership: registration metadata, imported mappings, and copy execution

### Registration semantics

`NvlinkTransport::registerLocalMemory()` must honor `remote_accessible`:

- `true`: retain the VMM handle, identify the complete mapped range, export a
  Fabric handle, and publish the existing BufferDesc format
- `false`: validate and prepare the local allocation for execution, but do not
  export a handle and do not add a remote BufferDesc

The transport tracks local-only registrations internally so that unregister is
balanced and idempotent without touching remote metadata. This local registry
contains only process-local address/length ownership state; it is not a new
wire metadata model.

Global Store segments pass `remote_accessible=true`. The setup-time local
buffer and caller-local transfer buffers pass `false`.

The Store still mounts global chunks with `protocol="nvlink"`. It does not
publish memory kind, Fabric domain, NUMA node, export device, or raw CUDA
allocation properties.

### Master allocation behavior

All chunks from one Provider retain the existing Provider segment name and TE
endpoint. The Master's current random or free-ratio allocation strategy treats
them as equivalent allocators and distributes objects without NUMA awareness.
`preferred_segment` continues to select a Provider, not an individual NUMA
chunk.

V1 does not add round-robin, per-NUMA placement, or caller-visible shard
identity. Those policies require Consumer/GPU topology information that V1
deliberately does not publish.

## Provider Setup and Teardown

### Successful setup

HOST_NUMA setup is startup-only in V1:

1. parse and validate configuration
2. validate visible-device Fabric capability
3. discover or validate global NUMA nodes
4. compute the capacity and chunk plan
5. determine the local buffer CPU NUMA when needed
6. create every local and global VMM allocation without publishing it
7. register the local buffer with `remote_accessible=false`
8. register and mount global chunks one at a time
9. record every successful Store segment ID
10. report readiness only after every chunk succeeds

Actual Fabric export during global registration is part of the Provider
capability check. Cross-node import is not a Provider-local preflight.

### Compensating rollback

Current `Client::MountSegmentAndGetId()` registers TE memory before it asks the
Master to mount the Store segment. If the Master mount fails, the function must
unregister that same TE buffer before returning an error. This is a small
generic correctness improvement used by all protocols.

The HOST_NUMA setup orchestrator additionally rolls back earlier successful
chunks in reverse order:

1. unmount Store segments from the Master
2. unregister their TE buffers
3. unregister the local buffer if it was prepared
4. destroy Store allocator views
5. destroy the VMM allocations

Rollback failures are logged and counted, but setup still returns failure.

### Partial visibility boundary

The current Master API has no transaction spanning several segment mounts.
Other clients can therefore observe a subset of the Provider's chunks during
the startup window. V1 does not add a cross-TE/Master commit protocol.

The deployment must keep the Provider out of readiness until setup completes.
Online atomic hot-add is not supported.

This limitation also exists in the current generic path when
`global_segment_size > max_mr_size`; current RDMA NUMA segmentation normally
uses one contiguous VMA with internally `mbind()`-bound regions per chunk, so
NUMA node count alone does not create multiple Store mounts there.

### Normal teardown

Normal teardown reverses publication before releasing memory:

1. stop readiness/new work
2. unmount Store segments
3. unregister global and local TE memory
4. destroy Store allocator views
5. unmap and release VMM allocations

V1 does not change the existing remote Consumer mapping-invalidation model.

## Consumer Data Path

### Put: local HBM to remote HOST_NUMA

1. The caller registers or reuses a registered HBM buffer.
2. Store allocates an object replica in a normal Provider Store segment.
3. The Consumer opens the Provider TE endpoint.
4. MultiTransport selects `NvlinkTransport` from the existing segment protocol.
5. NVLink locates the BufferDesc covering the object address.
6. On a mapping-cache miss, it imports and maps the Fabric handle.
7. CUDA copies the caller's HBM range into the relocated remote HOST_NUMA VA.

### Get: remote HOST_NUMA to local HBM

The same lookup and mapping path is used in reverse. CUDA copies the relocated
remote HOST_NUMA range into the caller's HBM buffer.

The Consumer does not know whether the remote allocation is device-backed or
HOST_NUMA-backed. No Store selector or new transport dispatch is needed in V1.

### Same-process replicas

The current Store can choose a CPU local-copy optimization when the replica
endpoint is the same process. Provider HOST_NUMA allocations explicitly grant
CPU access, so that optimization remains valid.

### Formal support matrix

| Local endpoint | Remote Store allocation | V1 status |
|---|---|---|
| HBM | HOST_NUMA | Supported Put path |
| HBM destination | HOST_NUMA source | Supported Get path |
| HOST_NUMA | HOST_NUMA | Not a V1 contract |
| ordinary pageable host memory | HOST_NUMA | Not a V1 fast-path or acceptance target |

The existence of a HOST_NUMA local workspace does not make H2H a V1 product
capability. A later H2H design must define registration, copy primitives,
selection, and performance acceptance explicitly.

## Consumer Mapping Lifecycle

V1 keeps lazy mapping:

1. import the Fabric handle
2. reserve a local VA
3. map the allocation
4. grant access to all locally visible GPUs
5. release the import-handle reference that is no longer needed after mapping
6. cache mapped base and length

Every failure path releases resources already acquired in that attempt. A
failed attempt does not enter the success cache and does not create a permanent
negative-cache entry. The current request fails; a later request can retry.

V1 does not retry the same import within the current request. It also does not
switch protocols.

The existing mapping key and invalidation behavior remain unchanged. In
particular, Provider restart and address reuse can leave stale Consumer
mappings. The design documents this as a known pre-existing risk and delegates
the lifecycle redesign to issue #2832.

## Failure Handling

| Failure | V1 behavior |
|---|---|
| Feature enabled with non-NVLink protocol | Fail setup |
| CUDA/MNNVL support missing | Fail setup |
| CUDA IPC forced instead of Fabric | Fail setup |
| Invalid or empty NUMA discovery | Fail setup |
| Allocation/granularity/access failure | Roll back local resources and fail setup |
| Fabric export failure | Roll back and fail setup |
| TE registration failure | Roll back and fail setup |
| Master mount failure | Undo the current TE registration, roll back previous chunks, fail setup |
| Consumer import/reserve/map/access failure | Clean the attempt, fail the request, allow later retry |
| CUDA copy failure | Fail the request; no fallback |
| Different Fabric domain or broken IMEX | Lazy import fails; deployment is responsible for prevention |
| Provider restart/address reuse | Existing risk; no V1 invalidation fix |

## Deployment Contract

V1 assumes:

- all Provider and Consumer nodes using the pool are in one scale-up Fabric
  domain
- IMEX is configured correctly on every participating node
- CUDA_VISIBLE_DEVICES exposes the devices that require access
- Provider readiness is held until all global chunks are mounted
- pure Provider deployments set `local_buffer_size=0` when they do not need a
  local staging workspace

The feature performs local Fabric capability and actual export validation. It
does not exchange test handles among nodes during startup and does not publish
a Fabric-domain identifier.

## Observability

Existing Store metrics continue to report segment capacity, allocated bytes,
Put/Get bytes, and end-to-end operation latency.

Add bounded-cardinality Provider metrics for:

- requested and effective HOST_NUMA capacity
- effective bytes and chunk count per NUMA node
- allocation, access, registration, and mount duration
- initialization failures by stage
- rollback attempts and rollback failures

Add bounded-cardinality Consumer metrics for:

- mapping-cache hits and misses
- lazy-import latency
- import, reserve, map, set-access, and copy failures

Allowed labels include `numa_node`, `stage`, `operation`, and `result`. Endpoint,
segment UUID, address, and Fabric handle must not be metric labels.

Structured startup logs include:

- feature state and configuration source
- requested and effective total capacity
- discovered or overridden global NUMA nodes
- setup CPU, local-buffer NUMA, and current CUDA device
- chunk NUMA, size, granularity, and mount outcome
- readiness success or completed rollback

Consumer error logs identify the lazy-import stage and CUDA result without
logging the Fabric handle.

## Testing

### Ordinary CI

Tests that do not require Fabric hardware cover:

- default-off and invalid configuration behavior
- NUMA list parsing, discovery abstraction, deduplication, and override
- capacity alignment, balanced NUMA distribution, and `max_mr_size` splitting
- RAII cleanup for every injected allocation-stage failure
- local-only versus remote-accessible registration semantics
- unchanged TE and Store serialization
- Master mount failure compensation
- reverse-order rollback after a later chunk fails
- existing HBM/NVLink behavior when the feature is disabled

### Fabric-capable single-host tests

Hardware tests cover:

- global and local HOST_NUMA VMM allocation
- local buffer node selection from the setup CPU
- CPU and all-visible-GPU access descriptors
- CPU read/write of Provider memory
- HBM-to-HOST_NUMA and HOST_NUMA-to-HBM byte verification
- Fabric export/import/map/access
- lazy-import miss followed by cache hit
- cleanup and retry after injected import-stage failure

### GB200 dual-node tests

A target NVL72/GB200 environment validates:

- one Provider process creating chunks on multiple CPU NUMA nodes
- Consumer HBM Put/Get through Mooncake Store
- cold import and warm mapping reuse
- requested/effective capacity and Master capacity agreement
- forced per-NUMA chunking with a reduced `max_mr_size`
- startup rollback and readiness failure after an injected mount failure
- concurrent traffic from multiple Consumer GPUs without data corruption

### RDMA compatibility smoke test

An independent hardware smoke test calls `ibv_reg_mr` and deregistration on a
Provider HOST_NUMA VMM allocation. It validates the V2 registration premise but
does not implement or validate dual-protocol selection or fallback.

### Local environment limitation

The design workstation has no usable CUDA/GB200 environment. Local validation
is limited to CPU tests, parsing, planning, serialization, rollback logic, and
available compilation checks. CUDA VMM, Fabric, GB200, and `ibv_reg_mr` results
must be reported as `NOT RUN` until executed on target hardware.

Generic CI may skip hardware tests with an explicit reason. A designated GB200
acceptance job must fail, rather than skip, when its required Fabric, IMEX, GPU,
or NUMA prerequisites are absent.

## Compatibility and Rollout

The feature is disabled by default. With the flag disabled, all existing Store
allocation, local buffer, HBM registration, metadata, and Consumer behavior
remain unchanged.

Because the wire format and protocol remain unchanged:

- a new Consumer can access an existing HBM Fabric Provider
- an existing Fabric-capable Consumer can use the same import sequence for a
  new HOST_NUMA Provider
- mixed deployment does not need Store or TE schema negotiation

The rollout sequence is:

1. merge default-off with ordinary CI and HBM/NVLink regression coverage
2. run Fabric-capable single-host tests
3. run GB200 dual-node Store tests and the RDMA compatibility smoke test
4. enable on a small Provider canary set
5. inspect effective capacity, shard layout, lazy-import, copy-error, and
   rollback metrics
6. expand only after target-hardware validation is recorded

## V2 Direction

V2 can reuse the V1 allocation and segment layout, then add:

- RDMA registration of the same CPU/GPU-accessible HOST_NUMA allocations
- Fabric-domain discovery and metadata
- per-buffer NVLink and RDMA transport descriptors
- Consumer-side candidate selection
- explicit fallback policy and health state
- optional prewarm
- formal H2H design if required

Provider restart safety remains an independent lifecycle project because it
applies to existing HBM MNNVL and TENT mappings as well as HOST_NUMA.

## Expected Code Boundaries

The implementation plan should keep changes near these boundaries:

| Area | Expected responsibility |
|---|---|
| `mooncake-store/include/types.h` | New ConfigDict keys only |
| `mooncake-store/src/real_client.cpp` | Config parsing, NUMA planning, Store-owned allocation records, setup/rollback orchestration |
| `mooncake-store/include/real_client.h` | RAII allocation/segment ownership fields and helpers |
| `mooncake-store/src/client_service.cpp` | Undo TE registration when Master mount fails |
| `mooncake-transfer-engine/.../nvlink_transport.h` | VMM allocation interface and local/remote registration contract |
| `mooncake-transfer-engine/.../nvlink_transport.cpp` | HOST_NUMA allocation, access, export, import cleanup, metrics |
| Store and TE tests | Config, planning, rollback, metadata compatibility, and hardware-gated data paths |

Do not introduce a new replica type, protocol name, Store descriptor field, or
TE metadata field in V1.
