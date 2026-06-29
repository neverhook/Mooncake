# NVLink HOST_NUMA and RDMA Dual Registration Design

Date: 2026-06-29
Status: Approved design, pending implementation plan

## Context

GB200 NVL72 class systems can expose HOST_NUMA DRAM through the CUDA VMM
fabric handle path when the allocation is created as HOST_NUMA, mapped into
UVA, and exported/imported with IMEX fabric handles. This enables a prefix
cache read path where a local GPU HBM destination reads from remote HOST_NUMA
DRAM inside the same scale-up NVLink domain.

Mooncake already has an NVLink transport that supports fabric handle
export/import for GPU memory, and it already has RDMA transport support for
host memory. The missing piece is a store-facing mode where one memory segment
can be registered for both NVLink HOST_NUMA fabric access and RDMA access, then
let the worker choose the best usable path for each read.

The current store read path is worker-side selection. The master returns all
complete replicas in metadata order. `RealClient::SelectBestReplica()` scans
the returned list and chooses a usable replica locally because the master may
return replicas in any order. This design keeps that behavior.

## Goals

- Support an explicitly enabled GB200/NVL72 scale-up mode for HOST_NUMA DRAM in
  the Mooncake store memory pool.
- Register the same memory segment with both `nvlink` and `rdma` when hardware
  capability and configuration allow it.
- Use NVLink fabric access only for same scale-up domain reads from remote
  HOST_NUMA DRAM into local HBM.
- Fall back to RDMA for cross-domain reads or when NVLink HOST_NUMA is not
  usable.
- Preserve the current master contract: the master returns all complete
  replicas, and the worker chooses the replica and protocol.
- Avoid a new `ReplicaType`; this remains a memory replica with multiple
  transport capabilities.

## Non-Goals

- No hidden data-path fallback inside `NvlinkTransport` after a transfer has
  already been submitted.
- No master-side requester-domain scheduling in the first version.
- No TENT migration in this slice.
- No support for unconfigured or silently inferred scale-up operation. The mode
  must be explicitly enabled.

## Architecture

The feature is modeled as a dual-protocol memory segment:

```text
same store MEMORY replica
  -> one HOST_NUMA VMM virtual address range
  -> registered as nvlink fabric memory when enabled and supported
  -> registered as RDMA memory when RDMA is configured
  -> exposed to workers as available_protocols = ["nvlink", "rdma"]
```

The master remains metadata oriented. It stores and returns the replica
descriptors but does not decide which protocol a requester should use. The
worker has the local context needed for that decision: local endpoints, local
GPU/HBM pointer type, local scale-up domain, enabled feature flags, and runtime
transport availability.

## Transfer Engine Changes

### HOST_NUMA VMM allocation and registration

Add an explicit allocation/register path for HOST_NUMA VMM memory. The provider
side should create memory with CUDA driver VMM properties equivalent to:

- `CU_MEM_ALLOCATION_TYPE_PINNED`
- `CU_MEM_LOCATION_TYPE_HOST_NUMA`
- configured NUMA node id
- `CU_MEM_HANDLE_TYPE_FABRIC`

The allocation is mapped into UVA and made accessible to the local GPUs that
may serve reads. Registration with `NvlinkTransport` exports a fabric handle and
stores it in the NVLink buffer descriptor.

### Multi-protocol metadata

Reuse the existing `ENABLE_MULTI_PROTOCOL` shape rather than introducing a new
protocol name. Extend the valid multi-protocol combination set to include
`nvlink,rdma`.

For a dual-registered segment:

- segment protocol is encoded as both `nvlink` and `rdma`
- each buffer descriptor identifies the specific transport metadata it carries
- RDMA buffer entries include address and keys
- NVLink buffer entries include address, length, memory kind, scale-up domain,
  and fabric handle data

This keeps each transport responsible for only its own registration data.

### Consumer-side NVLink import

The current NVLink import/map/cudaMemcpy read flow can be reused, but it needs
HOST_NUMA-aware checks:

- reject or skip NVLink before import when local and remote scale-up domains do
  not match
- verify that the remote buffer is HOST_NUMA fabric memory for this mode
- bind the copy stream/device to the local HBM destination GPU
- clean up imported mappings safely for HOST_NUMA fabric mappings

`NvlinkTransport` should return a clear unsupported or address error for domain
mismatch. It should not internally resubmit the same request over RDMA.

## Store Changes

### Segment mount and client initialization

Add a store-visible mount mode for dual registration. When enabled, the store
client installs both `nvlink` and `rdma` transports and mounts the segment as a
multi-protocol memory segment.

Recommended first-version behavior:

- if dual registration is disabled, existing behavior is unchanged
- if enabled and NVLink HOST_NUMA capability is present, register both
  `nvlink` and `rdma`
- if enabled but NVLink HOST_NUMA capability is absent, fall back to RDMA-only
  unless strict mode is configured
- if strict mode is configured, fail mount when NVLink HOST_NUMA registration
  fails

### Replica descriptors

Keep `ReplicaType::MEMORY`. Extend memory descriptors with enough information
for worker-side protocol selection:

- available protocols, such as `["nvlink", "rdma"]`
- remote scale-up domain id
- memory kind, such as `HOST_NUMA`
- transport endpoint
- existing address and size

For compatibility, the existing `protocol_` field can continue to represent the
registered protocol string. New optional fields should carry selection metadata
without changing the meaning of old single-protocol descriptors.

### Worker-side selection

Update worker selection from "best replica only" to "best transfer candidate".
A transfer candidate is `(replica, protocol)`.

Candidate order for memory replicas:

1. local memory replica through local memcpy, when applicable
2. same scale-up domain HOST_NUMA memory replica through `nvlink`
3. reachable memory replica through `rdma`
4. existing NOF, LOCAL_DISK, and DISK fallback logic

The worker chooses `nvlink` only when all conditions are true:

- the replica advertises `nvlink`
- the local node has this feature enabled
- the remote segment memory kind is HOST_NUMA
- the local destination is GPU HBM for the optimized read path
- local and remote scale-up domain ids match
- the NVLink transport is installed and healthy

Otherwise it chooses RDMA if the replica advertises `rdma`.

### Transfer submission

When the selected protocol is single-protocol legacy behavior, the worker can
keep using `submitTransfer()`.

When the selected replica advertises multiple protocols, the worker submits via
`mp_submitTransfer(..., selected_protocol)`. This makes the selected path
explicit and avoids relying on a comma-separated segment protocol string in
`selectTransport()`.

## Data Flow

### Provider mount

1. Store client allocates or receives the HOST_NUMA VMM memory range.
2. Store client initializes both NVLink and RDMA transports when configured.
3. Transfer engine registers the range with NVLink and RDMA.
4. Transfer metadata publishes one segment with both protocol capabilities.
5. Store master mounts the segment and tracks it as a normal memory segment.

### Put

Put placement remains master-driven. The master allocates memory replicas as it
does today. A dual-registered segment stores only one object copy in one memory
range; it does not create separate NVLink and RDMA replicas.

The provider writes object data once. Both protocols refer to the same backing
bytes.

### Get

1. Worker queries the master.
2. Master returns all complete replicas.
3. Worker builds transfer candidates from the replica list.
4. Worker picks NVLink for same-domain HOST_NUMA-to-HBM reads when possible.
5. Worker picks RDMA for cross-domain reads or when NVLink is not usable.
6. Worker submits the transfer with the selected protocol.

### Data-plane batching semantics

After the control-plane query and candidate selection complete, the data plane
keeps the existing Mooncake transfer shape. Paged-attention block layouts are
represented as a vector of store `Slice` entries. Each slice becomes one
`TransferRequest`, and the whole request vector is submitted in one batch.

The selected protocol changes only the transport used for that batch:

- single-protocol legacy segments can keep using `submitTransfer()`
- dual-protocol segments use `mp_submitTransfer(..., selected_protocol)`
- the same request vector can be submitted through `nvlink` or `rdma`

This means HOST_NUMA NVLink does not need a new scatter/gather protocol for the
first version. It reuses the existing `Slice -> TransferRequest vector ->
TransferEngine batch -> transport batch` pipeline.

The transport-specific batching remains unchanged:

- RDMA may split each `TransferRequest` into transport slices using
  `globalConfig().slice_size`, group them by RDMA context, and post work
  requests in batches.
- NVLink maps each `TransferRequest` to a memcpy slice, collects source
  pointers, destination pointers, and sizes, then submits them with
  `cudaMemcpyBatchAsync` when available, or per-slice `cudaMemcpyAsync`
  otherwise.

The object layout assumption also stays unchanged: the requester may provide
many scattered local slices, while the selected store replica describes one
linear object interval. If a future feature needs a single logical object whose
remote backing storage is itself non-contiguous, that should be modeled as a
separate object-layout descriptor change, not as part of this transport slice.

## Error Handling

- Feature disabled: no behavior change.
- Unsupported hardware: skip NVLink HOST_NUMA capability and use RDMA-only in
  non-strict mode.
- Strict mode with NVLink registration failure: fail the mount.
- Same-replica NVLink domain mismatch: worker should not select NVLink.
- Unexpected NVLink submission failure: fail that transfer attempt clearly. A
  higher-level retry may choose the next candidate, usually RDMA, but the
  transport itself does not hide the fallback.
- Missing RDMA capability for a cross-domain read: return a clear no-usable
  replica error.

## Observability

Add logs and counters for:

- dual-registration enabled or disabled
- HOST_NUMA VMM allocation capability probe result
- successful NVLink HOST_NUMA registrations
- RDMA-only fallback during mount
- worker candidate selection: local, nvlink, rdma, disk
- NVLink skipped due to domain mismatch
- strict-mode mount failures

## Testing

### Unit tests

- Multi-protocol metadata encode/decode accepts `nvlink,rdma`.
- Metadata keeps separate NVLink and RDMA buffer descriptors for the same
  address range.
- Worker candidate selection chooses NVLink for same domain and RDMA for
  cross-domain.
- Worker candidate selection skips NVLink when the destination is not HBM.
- Worker transfer submission preserves scattered local slices as a single
  batched request vector and switches only the selected protocol.
- Strict and non-strict mount behavior handles NVLink capability failure.

### Transfer engine tests

On a GB200/NVL72 or equivalent fabric-capable system:

- allocate HOST_NUMA VMM memory
- export/import NVLink fabric handle
- read remote HOST_NUMA DRAM into local HBM through NVLink
- verify byte-for-byte correctness

The test must skip cleanly when the hardware or build flags are not present.

### Store integration tests

- mount a dual-protocol memory segment
- Put one object into that segment
- Get from a same-domain HBM destination and verify NVLink selection
- simulate cross-domain metadata and verify RDMA selection
- verify old single-protocol RDMA store behavior is unchanged

## Rollout

1. Land metadata and selection changes behind disabled-by-default config.
2. Add HOST_NUMA VMM NVLink registration path and gated tests.
3. Enable dual registration in a controlled GB200/NVL72 environment.
4. Validate same-domain NVLink reads and cross-domain RDMA fallback.
5. Add metrics dashboards or log-based rollout checks before broader enablement.

## Design Decisions

- Keep master returning all replicas to match current Mooncake read behavior.
- Put replica and protocol selection in the worker because it owns local runtime
  context.
- Reuse `nvlink` protocol rather than adding `nvlink_host_numa`.
- Reuse `ReplicaType::MEMORY`; do not add a protocol-specific replica type.
- Model fallback as explicit worker candidate selection, not hidden transport
  fallback.
