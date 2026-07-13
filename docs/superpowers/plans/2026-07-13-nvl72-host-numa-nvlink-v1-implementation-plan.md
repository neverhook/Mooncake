# NVL72 HOST_NUMA NVLink KV Pool V1 Implementation Plan

- **Status:** Code and local/arm64 checks complete; designated GB200 CI integration pending; GB200/Fabric/RDMA acceptance `NOT RUN`
- **Date:** 2026-07-13
- **Design:** [`2026-07-10-nvl72-host-numa-nvlink-v1-design.md`](../specs/2026-07-10-nvl72-host-numa-nvlink-v1-design.md)
- **Code baseline:** `origin/main@98ff4e4787e99265d25938139551841350ca5f4e`
- **Design commit:** `dfbd474423c64e38a49ade5ddb754d016a5e714f`
- **Primary scope:** Mooncake Store, legacy `NvlinkTransport`, ordinary CI, and GB200/NVL72 acceptance

## Implementation status (2026-07-13)

Tasks 1-7 and Task 8's ordinary-CI, local harness, documentation, and rollout
portions are implemented in the named design worktree. The designated
self-hosted GB200 job in Task 8.3 remains pending runner integration, and its
hardware acceptance remains `NOT RUN`. The implementation includes
ConfigDict-only enablement, deterministic NUMA/chunk planning, Store-owned
HOST_NUMA VMM lifetime, strict Fabric NVLink readiness, transactional
registration/import/rollback, bounded Provider and Consumer metrics,
ordinary/arm64 CTest gates, and the GB200 validation harness.

Current-workstation checks passed for clang-format 20.1.8, `git diff --check`,
workflow YAML parsing, GB200 shell syntax, Python AST/Ruff, CLI entry points,
label membership review, and non-strict preflight behavior. After the first
GB200 compile exposed and the implementation fixed a test namespace error, a
disposable Linux/arm64 CUDA 12.8 development container configured with
`USE_CUDA=ON` and `USE_MNNVL=ON`, compiled the Store/Fabric/RDMA/VMM/metrics/HBM
test targets. The final source passed all 11 `nvlink_host_numa_unit` entries
with event completion disabled and all 12 with it enabled, including the
production-reused orchestration failure matrix and the event polling
regression. Both configurations also passed `transfer_task_test` and
`pybind_client_test`. The final ownership audit passed all 14 VMM
fault-injection cases, including staged release retry, persistent cleanup
failure, and pinned-owner retention. The Consumer now uses `libcudart` directly via
`ctypes` for HBM allocation/copy/verification, so the entire Python validation
path is Torch-free. The Python harness parser/admin/fake-CUDA/module-provenance
suite passed 7/7; the arm64 container loaded the current build-tree Store
binding and CUDA 12.8 `libcudart`. This
is compile and fake-driver/CPU evidence only: the container had no real CUDA
driver, Fabric, IMEX, RNIC, or GB200 topology, so Fabric copy, Store hardware,
and verbs smoke results remain `NOT RUN`.

The remaining external gate is deliberately strict:

```bash
BUILD_DIR=$PWD/build-nvlink-host-numa \
RUN_HARDWARE_TESTS=1 \
scripts/gb200/nvlink_host_numa_build.sh
```

This command must run on a prepared GB200/NVL72 host. Required Fabric/RDMA
testcases must be present and report zero skips; until then their status remains
`NOT RUN`, not `PASS`.

For `RUN_HARDWARE_TESTS=1`, the script builds first, defaults the strict Fabric
probe to the resulting `nvlink_host_numa_fabric_test`, and then runs preflight.
Standalone strict preflight rejects an unset or non-executable probe.

## Outcome

After this plan is complete, a Store process can opt into a Store-owned CUDA
VMM `HOST_NUMA` pool through the ConfigDict setup path. Provider capacity is
split into independent NUMA-local VMM chunks, published as ordinary
`protocol="nvlink"` Store segments, and transferred bidirectionally between a
Consumer GPU's HBM and Provider DRAM through the existing Fabric handle path.

“HBM+DRAM unified pooling transfer” in V1 means a common Store/TE data path
between Consumer HBM and Provider DRAM. It does not make one Provider process
contribute a mixed HBM-and-DRAM capacity pool at the same time: enabling the
flag switches that process's nonzero global/local allocation backing to
HOST_NUMA.

The feature remains disabled by default. Existing protocol names and Store/TE
wire schemas do not change.

## V1 guardrails

The implementation must preserve all of the following:

| Invariant | Enforcement |
|---|---|
| Configuration is local and default-off | Only ConfigDict parses `enable_nvlink_host_numa`; absent and explicit `false` use the old setup path |
| Protocol stays `nvlink` | Do not add `nvlink_host_numa` or another transport |
| Wire schemas stay unchanged | Do not add memory kind, NUMA, Fabric domain, generation, or protocol hints to Store or TE descriptors |
| Store owns capacity and lifetime | `RealClient` owns VMM allocations, allocator views, TE registration state, and mounted segment IDs |
| Consumer is backing-agnostic | Import/map/copy does not branch on DEVICE versus HOST_NUMA |
| Startup is all-or-fail for the Provider process | Allocate all resources before publication; reverse all completed work on failure; hold readiness until complete |
| There is no fallback in V1 | Fabric capability, import, map, or copy failure fails setup/request; do not switch to RDMA or HBM |
| H2H is not a V1 contract | Acceptance is HBM-to-HOST_NUMA Put and HOST_NUMA-to-HBM Get |
| Restart ABA is not fixed here | Keep the current mapping key/invalidation behavior and reference issue #2832 |
| No online hot-add or per-GPU export sharding | NUMA chunks are setup-time Store segments; Master placement remains unchanged |
| Legacy NvlinkTransport only | Reject TENT for enabled V1; do not implement the feature in the TENT NVLink transport |

## Baseline gaps that must be closed first

The implementation cannot safely start in `RealClient` alone. At the named
baseline:

1. `NvlinkTransport::registerLocalMemory()` ignores `remote_accessible` in both
   IPC and Fabric modes, so local workspaces are accidentally published.
2. Fabric registration treats `cuMemRetainAllocationHandle()` failure as
   success even when remote access was requested.
3. `Client::MountSegmentAndGetId()` registers with TE before the Master mount,
   but does not undo TE registration if the Master mount fails.
4. lazy Fabric import leaks the imported handle, reserved VA, or mapping on
   several failure paths, and also retains the import handle after success.
5. transfer submission can mutate task state before relocation fails, leaving a
   batch busy or incorrectly complete.
6. `Client::InitTransferEngine()` does not install `nvlink` when
   `MC_MS_AUTO_DISC=0`.
7. the existing `nvlink` Store global pool is ordinary `aligned_alloc` host
   memory, not HBM and not Fabric-exportable VMM. Existing HBM regression is
   therefore the TE/caller-owned GPU-buffer path.

The work packages below close these gaps in dependency order.

## Dependency order

```text
Task 1: pure config/discovery/capacity planning
                  |
Task 2: unified VMM RAII + strict capability
                  |
Task 3: NVLink registration/import/submission correctness
                  |
Task 4: generic Client mount/install compensation
                  |
Task 5: RealClient setup, rollback, teardown
                  |
Task 6: metrics and structured diagnostics
                  |
Task 7: single-host and GB200 Store acceptance
                  |
Task 8: CI, rollout, and operational handoff
```

Tasks 1 and 4 can be implemented in parallel. Task 2 can start in parallel with
Task 1, but Tasks 3 and 5 depend on its final API and ownership semantics.

## Task 1: Add testable ConfigDict parsing, NUMA discovery, and chunk planning

**Files**

- Modify `mooncake-store/include/types.h`
- Add `mooncake-store/src/nvlink_host_numa.h`
- Add `mooncake-store/src/nvlink_host_numa.cpp`
- Modify `mooncake-store/src/CMakeLists.txt`
- Add `mooncake-store/tests/nvlink_host_numa_config_test.cpp`
- Add `mooncake-store/tests/nvlink_host_numa_setup_test.cpp`
- Modify `mooncake-store/tests/CMakeLists.txt`

### 1.1 Define the internal setup model

Add the two ConfigDict constants beside the existing Store setup keys:

- `CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA`
- `CONFIG_KEY_NVLINK_HOST_NUMA_NODES`

Keep implementation-only types in `mooncake-store/src/nvlink_host_numa.h`, not
in the public Store/binding API. The internal model should include:

- `NvlinkHostNumaOptions`: enabled state and `auto`/explicit node selection
- `NvlinkHostNumaNodePlan`: node ID, allocation granularity, effective bytes
- `NvlinkHostNumaChunkPlan`: node ID, chunk bytes, stable plan index
- `NvlinkHostNumaPlan`: requested total, effective total, common alignment, and
  ordered chunks

Parse boolean values strictly (`true`/`false` and `1`/`0`, case-insensitive).
Reject any other explicit value. Parse CSV by trimming tokens, rejecting empty
tokens, negative values, overflow, and trailing junk, then deduplicate and sort.

`nvlink_host_numa_nodes` is ignored unless the feature is enabled and
`global_segment_size > 0`. A pure Consumer with both sizes zero must not require
NUMA discovery.

### 1.2 Isolate environment discovery

Define an internal environment interface used by the planner. The production
implementation performs:

1. enumerate visible CUDA devices;
2. obtain each device's PCI BDF;
3. read `/sys/bus/pci/devices/<BDF>/numa_node`;
4. validate the result against online libnuma nodes;
5. deduplicate and sort nodes;
6. for a nonzero local buffer, resolve `sched_getcpu()` and
   `numa_node_of_cpu()`.

The fake implementation supplies device/BDF/sysfs/online-node results to CPU
tests. Do not add production `ForTesting` methods to `RealClient`.

Automatic discovery fails on no visible device, unreadable sysfs, an unknown
or negative node, or an empty result. Explicit nodes bypass GPU-to-NUMA
discovery but must all be online and valid.

### 1.3 Implement deterministic capacity planning

The planner accepts requested total bytes, the ordered node/granularity pairs,
and `globalConfig().max_mr_size`:

1. compute a checked common alignment compatible with every selected node's
   CUDA granularity and the Store allocator's `facebook::cachelib::Slab::kSize`
   alignment requirement;
2. floor requested capacity to that alignment without exceeding the request;
3. fail if every node cannot receive at least one unit;
4. distribute whole units evenly in sorted node order, with a difference of at
   most one unit;
5. floor `max_mr_size` to the common alignment, so every chunk length remains
   both CUDA- and Store-slab-aligned;
6. fail if the resulting maximum chunk is zero;
7. split each node share into ordered chunks, none of which crosses a node or
   allocation handle.

Use checked arithmetic for alignment, multiplication, addition, and range
boundaries. The plan returned here is immutable; `RealClient` must not mutate
`global_segment_size` during HOST_NUMA setup.

### 1.4 Unit tests

Cover at least:

- absent flag equals explicit false;
- invalid boolean strings;
- `auto`, whitespace, duplicate CSV nodes, invalid tokens, negative/offline
  nodes, and empty discovery;
- one and multiple node granularities;
- requested total flooring;
- insufficient total for all nodes;
- balanced remainder distribution in stable node order;
- `max_mr_size` equal to, below, and above granularity;
- `max_mr_size` rounding against the full common alignment;
- forced multiple chunks per NUMA node;
- arithmetic overflow;
- CUDA granularity versus Store slab alignment combinations;
- local-buffer setup CPU/NUMA success and failure;
- `global_segment_size=0` and `local_buffer_size=0` role combinations.

**Task exit criterion:** all parsing/discovery/planning behavior runs in an
ordinary non-CUDA test job with fake environment/granularity inputs.

## Task 2: Introduce one move-only CUDA VMM allocation owner

**Files**

- Modify `mooncake-transfer-engine/include/transport/nvlink_transport/nvlink_transport.h`
- Modify `mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp`
- Add `mooncake-transfer-engine/tests/nvlink_vmm_allocation_test.cpp`
- Modify `mooncake-transfer-engine/tests/CMakeLists.txt`

### 2.1 Define `NvlinkVmmAllocation`

Add a move-only RAII type at the NVLink/TE boundary. Its creation options must
carry:

- location type: DEVICE or HOST_NUMA;
- location ID: CUDA device or CPU NUMA node;
- requested bytes;
- whether a Fabric-exportable handle is required.

The object exposes read-only base address, actual aligned length, allocation
granularity, location diagnostics, and exportability. It owns the mapped VA and
all CUDA handle state required for cleanup. It never serializes the location.
Global creation also accepts the planner's required VA alignment and passes it
to `cuMemAddressReserve`; that alignment must be a compatible multiple of CUDA
granularity. This guarantees both VMM base and length meet Store slab alignment.

Use a test-injectable CUDA Driver operation table/internal adapter so every
stage can be tested without real Fabric hardware. The production sequence is:

1. construct `CUmemAllocationProp`;
2. query minimum granularity;
3. align the requested length with overflow checks;
4. `cuMemCreate`;
5. `cuMemAddressReserve`;
6. `cuMemMap`;
7. grant access;
8. release the original allocation-handle reference after a successful map.

For HOST_NUMA, grant read/write access to the owning CPU NUMA location and all
visible CUDA devices. For DEVICE, preserve access for all visible GPUs. A
global HOST_NUMA chunk is Fabric-exportable; the local HOST_NUMA workspace is
not.

Destruction is idempotent and performs the valid suffix of:

```text
cuMemUnmap -> cuMemAddressFree -> cuMemRelease(any still-owned handle)
```

### 2.2 Preserve existing allocation entry points

Keep these signatures unchanged:

- `NvlinkTransport::allocatePinnedLocalMemory(size_t)`
- `NvlinkTransport::freePinnedLocalMemory(void*)`

The allocation wrapper delegates to DEVICE-mode `NvlinkVmmAllocation` when
Fabric VMM is used and preserves the current non-Fabric `cudaMalloc/cudaFree`
behavior. Existing Python integration, benchmark macros, and HBM tests must
continue to compile. Because the wrapper returns a raw pointer, retain its RAII
owner in an internal mutex-protected map keyed by base address; the legacy free
wrapper removes the entry and lets the owner destruct. Store-owned allocations
do not enter this compatibility map.

Do not use `mooncake-transfer-engine/nvlink-allocator/` for Store HOST_NUMA: its
fallback/lifetime behavior is different and V1 must not silently fall back.

### 2.3 Add strict capability/granularity queries

Expose narrow, read-only APIs needed by Store setup:

- whether the build/runtime can create and export Fabric VMM allocations;
- allocation granularity for a specified DEVICE/HOST_NUMA location;
- a strict failure reason suitable for structured logging.

The strict HOST_NUMA preflight fails if CUDA/MNNVL support is absent,
`MC_USE_NVLINK_IPC` forces IPC, there are no visible devices, a device lacks
Fabric-handle support, or a CUDA query fails. It must not modify the legacy
default-off wrapper's IPC/Fabric selection.

### 2.4 Unit and compile tests

The fake-driver test must inject failure at create, reserve, map, every access
descriptor, and post-map release, then assert exact reverse cleanup and zero
live resources. Also verify:

- DEVICE versus HOST_NUMA allocation properties;
- Fabric-exportable versus local-only requested handle types;
- actual aligned length;
- CPU plus all-visible-GPU access descriptors;
- move construction and double-destruction safety; move assignment is
  intentionally deleted because a partially released owner must retain its
  exact retryable CUDA VMM state;
- legacy wrapper source compatibility;
- non-CUDA/MNNVL stubs report unsupported rather than silently allocating.

Keep the public allocation options/owner free of raw CUDA types. Under
`!USE_MNNVL`, provide header-inline unsupported factory/capability behavior so
Store's ConfigDict validation can compile and fail cleanly without linking
`nvlink_transport.cpp`; alternatively guard the complete Store owner/reference
behind `USE_MNNVL`. Do not leave an unconditional Store reference to a symbol
that is only built by the MNNVL transport CMake target.

**Task exit criterion:** fake-driver tests pass in the CUDA/MNNVL compile job,
and default-off HBM wrappers retain their public behavior.

## Task 3: Make NVLink registration, lazy mapping, and submission transactional

**Files**

- Modify `mooncake-transfer-engine/include/transport/nvlink_transport/nvlink_transport.h`
- Modify `mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp`
- Modify `mooncake-transfer-engine/src/transfer_metadata.cpp` only if needed for
  transactional local descriptor rollback
- Modify `mooncake-transfer-engine/include/transport/transport.h`
- Modify `mooncake-transfer-engine/include/multi_transport.h`
- Modify `mooncake-transfer-engine/src/multi_transport.cpp` if terminal failure
  cannot be contained inside `NvlinkTransport`
- Modify `mooncake-transfer-engine/tests/nvlink_transport_test.cpp`
- Modify `mooncake-transfer-engine/tests/transfer_metadata_test.cpp`

### 3.1 Honor the existing registration contract

Add a mutex-protected local registration record keyed by the caller's address.
Record requested range, whole VMM range when available, whether it is remote,
the published base, and any retained handle reference.

For `remote_accessible=false`:

- validate that the pointer is usable for local transfer;
- accept an ordinary CUDA HBM pointer, VMM range, or the legacy Store local
  CPU buffer used when the feature is disabled;
- create only the local record;
- do not export a handle;
- do not add or update a remote BufferDesc.

For `remote_accessible=true`:

- require a retainable, Fabric-exportable VMM allocation in Fabric mode;
- query the exact whole mapped range;
- when the legacy `cuMemGetAddressRange()` query rejects a verified HOST_NUMA
  VMM mapping (for example with CUDA error 201) or returns a null base, use the
  exact base and aligned length only when they match a live
  `NvlinkVmmAllocation` ownership record; arbitrary callers and interior
  subranges are rejected;
  DEVICE allocations still fail the query, and the old guessed 2 MiB fallback
  remains forbidden;
- export one `CUmemFabricHandle`;
- publish the existing BufferDesc fields only;
- if metadata publication fails, undo the local descriptor mutation, release
  the retained handle, and remove the registration record.

`unregisterLocalMemory()` looks up the record. Local-only registration removes
only local state. A published registration removes the recorded BufferDesc base
and balances the retained handle. Repeated/unbalanced unregister is safe and
diagnosable without corrupting metadata.

### 3.2 Make lazy import attempt-scoped

Replace the open-coded Fabric branch in `relocateSharedMemoryAddress()` with a
temporary RAII mapping attempt:

```text
import handle -> reserve VA -> map -> set all visible GPU access
              -> release import handle -> insert success cache entry
```

Each failure releases only resources already acquired in that attempt. Do not
insert a cache entry or a negative cache entry. A later request can retry.

Make cached entries own either an IPC mapping or a Fabric mapped VA explicitly.
Fabric teardown unmaps and frees VA directly; it must not reuse the local
allocation wrapper. Keep the cache key `(target_id, BufferDesc.addr)` and do not
add generation metadata.

Also validate a non-null segment descriptor, deserialize length, and address
range without overflow. Logs include stage and CUDA result, never Fabric handle
bytes.

### 3.3 Leave transfer direction backing-agnostic

Do not add HOST_NUMA branches to `submitBatchMemcpy()`. Preserve the existing
READ/WRITE source/destination selection and CUDA copy primitive.

Resolve all remote addresses before allocating slices or incrementing task
counters. If relocation, submit, or stream query fails, mark the affected task
terminally failed and release any staged slices. Both `submitTransfer()` and
`submitTransferTask()` must leave the batch queryable and freeable.

Do not infer this failure from an empty slice list: the baseline would classify
zero slices as COMPLETED. Add a task-local, non-wire terminal submission status
and one shared `markSubmissionFailed(...)` helper. It sets the task error,
`is_finished`, and any event-driven completion notification before staged slices
are released. `getTransferStatus()`, MultiTransport's Future/event path, and
`freeBatchID()` must all honor this explicit FAILED state before consulting
slice counters. Tests cover both the legacy batch API and event-driven Store
submission so neither path hangs or reports false success.

### 3.4 Metadata and HBM regression tests

- Convert `nvlink_transport_test.cpp` to self-contained `P2PHANDSHAKE` metadata
  and use the server's actual TE endpoint.
- Keep its caller-owned HBM Put/Get test as the default-off regression and
  register two explicit CTest cases: default Fabric mode and
  `MC_USE_NVLINK_IPC=1` IPC mode. The IPC case uses `cudaMalloc`; the Fabric HBM
  case uses the retained `allocatePinnedLocalMemory()` DEVICE VMM wrapper,
  because remote Fabric registration must reject non-exportable `cudaMalloc`
  memory. A single invocation cannot prove both paths.
- Add local-only versus remote registration tests: peer descriptor unchanged
  for false, exactly one BufferDesc for true, balanced unregister, and metadata
  write failure compensation.
- Add fake-driver lazy-import failures at import/reserve/map/set-access; assert
  zero leaked resources, no cache entry, then success and warm cache hit.
- Add submission failure tests proving batch status is FAILED and free succeeds.
- Add a golden `SegmentDesc`/`BufferDesc` round trip in
  `transfer_metadata_test.cpp`; assert there are no V1 memory-kind/NUMA/Fabric
  domain/generation fields.

**Task exit criterion:** registration publication and lazy import are
transactional, later retry works, and HBM/IPC/Fabric default-off paths regress
cleanly.

## Task 4: Fix generic Client compensation and manual NVLink installation

**Files**

- Modify `mooncake-store/src/client_service.cpp`
- Modify `mooncake-store/tests/client_integration_test.cpp`

### 4.1 Compensate Master mount failure

In `Client::MountSegmentAndGetId()`, if TE registration succeeds but
`master_client_.MountSegment()` fails:

1. call `transfer_engine_->unregisterLocalMemory(buffer)` before returning;
2. keep the original Master error as the operation result;
3. log/count a secondary unregister failure without masking the original error;
4. do not insert into `mounted_segments_`.

Add an isolated in-process integration test that makes the Master mount fail
after TE registration, then verifies that the Provider's local TE descriptor no
longer contains the buffer and the address can be registered again.

The unregister contract from Task 3 is idempotent. If this first compensation
attempt itself fails, record it and allow the caller's rollback to retry the
same base; `ERR_ADDRESS_NOT_REGISTERED` on that fallback is treated as already
clean.

### 4.2 Install `nvlink` with discovery disabled

In `Client::InitTransferEngine()`, treat `protocol="nvlink"` like other
non-RDMA transports in the `MC_MS_AUTO_DISC=0` branch:

- ignore/warn on `device_names`;
- call `installTransport("nvlink", nullptr)`;
- return an explicit error when the transport is not built/installed.

Test auto-discovery on/off and unsupported-build behavior. This is not a new
configuration mode; it makes an existing protocol usable under the existing
discovery switch.

**Task exit criterion:** current-chunk mount failure leaves neither Master nor
TE publication, and manual transport initialization supports `nvlink`.

## Task 5: Orchestrate Store-owned HOST_NUMA setup, rollback, and teardown

**Files**

- Modify `mooncake-store/include/real_client.h`
- Modify `mooncake-store/src/real_client.cpp`
- Modify `mooncake-store/tests/nvlink_host_numa_setup_test.cpp`
- Add `mooncake-store/tests/nvlink_host_numa_store_test.cpp`

### 5.1 Keep binding/API compatibility

Only `RealClient::setup_internal(const ConfigDict&)` parses the new keys. Pass
an internal `NvlinkHostNumaOptions` object to the fixed-argument internal setup
implementation with a disabled default. Do not change:

- `setup_real()`;
- the fixed Python positional overload;
- C ABI;
- Rust or Go setup signatures;
- DummyClient behavior.

Reject enabled configuration before publication when protocol is not `nvlink`,
the build/runtime strict preflight fails, IPC is forced, or Client/Transfer
Engine is using TENT. The Consumer does not need the flag to import Provider
Fabric handles.

After Transfer Engine initialization, validate the instantiated transport set,
not only the requested protocol string: it must contain exactly one
Fabric-enabled `NvlinkTransport`. This rejects HCA-driven auto-discovery that
installed RDMA, as well as mixed transport publication, instead of reporting a
false-ready `nvlink` Provider.

### 5.2 Add explicit ownership records

Add startup-only records distinct from the existing public online
`MountedSegmentRecord`/`AllocatedSegmentRecord`. A global record owns:

- one `NvlinkVmmAllocation`;
- its NUMA node and planned index for diagnostics;
- TE registration state;
- optional mounted Store `UUID`.

The local record owns:

- one non-exportable HOST_NUMA `NvlinkVmmAllocation`;
- one non-owning
  `ClientBufferAllocator::create(base, configured_local_buffer_size, "nvlink")`
  view, so alignment padding does not increase caller-visible workspace;
- local-only TE registration state.

Do not put HOST_NUMA pointers in the raw `segment_ptrs_` deleter vectors and do
not let `ClientBufferAllocator` call generic `free()` on them.

### 5.3 Implement enabled startup as a staged transaction

The enabled branch follows this order:

1. parse/validate local configuration;
2. create the Client/TE endpoint and run strict Fabric capability checks;
3. discover/validate global nodes and query each granularity;
4. compute the immutable capacity/chunk plan;
5. resolve setup CPU/local NUMA when local buffer is nonzero;
6. allocate the local buffer and every global VMM chunk, publishing nothing;
7. create the non-owning local allocator view;
8. register the local buffer with `remote_accessible=false`;
9. for each global chunk, call
   `MountSegmentAndGetId(base, actual_length, "nvlink", kWildcardLocation)`;
10. record each returned UUID;
11. report readiness only after the final chunk succeeds.

For global chunks, use the plan/allocation's exact aligned length consistently
in TE BufferDesc, Store Segment, effective capacity, and teardown. For the local
workspace, keep the configured `local_buffer_size` as allocator/registration
capacity and use the allocation's actual mapped length only for ownership and
destruction.

All chunks keep the existing Provider segment name and endpoint. Do not expose
node/chunk identity through `preferred_segment` or replica metadata.
Always retain and unmount the returned UUID for each chunk: existing Master
name-to-segment indexing keeps only one UUID for a repeated Provider name, so
rollback must never try to recover chunk identity by name.

### 5.4 Reverse rollback and normal teardown

On setup failure, walk completed global records in reverse order:

1. unmount by UUID (which unregisters TE through Client);
2. for the current failed mount, call idempotent unregister by base as a
   fallback even though Task 4 already attempted compensation;
3. unregister the local buffer;
4. reset allocator views;
5. destroy all VMM allocations.

Attempt all cleanup steps, record the first setup error as the result, and log
each secondary rollback failure. Task 4 owns the first current-chunk cleanup
attempt; RealClient owns the idempotent fallback plus earlier chunks. If both
current-chunk attempts fail, surface a rollback-failure metric/log and keep the
original setup failure as the returned error.

When any cleanup operation fails, retain the VMM ownership records and let an
explicit close retry; do not recycle a virtual address that may still be named
by Master or TE metadata. If object destruction still cannot prove cleanup,
quarantine the VMM owner until process exit and log the intentional leak.

In `tearDownAll_internal()`, perform the same publication-first reversal before
`client_.reset()`. Preserve the existing atomic close guard. Destructors remain
safe after partial setup or an explicit close.

### 5.5 Store orchestration tests

Drive the orchestrator through an internal operations adapter rather than
production test flags. A fake adapter records allocate/register/mount/unmount/
unregister/destroy calls. Cover failure at every stage and assert:

- all VMM allocations are created before the first publication;
- local registration is local-only;
- global chunks mount in plan order;
- later mount failure unmounts earlier chunks in reverse order;
- the current failed mount is compensated by Client and safely retried by
  RealClient; an already-removed registration is harmless;
- injected failure of Client's first compensation is cleaned by the RealClient
  fallback, while failure of both attempts is reported as rollback failure;
- local registration and allocator view are released before VMM destruction;
- no segment IDs or live allocations remain;
- teardown is idempotent;
- disabled configuration follows the unchanged old allocation/mount path;
- Provider-only, Consumer-only, and hybrid size combinations retain meaning.

**Task exit criterion:** with functioning cleanup operations, setup failure
leaves no Provider capacity or TE buffer published by the process; cleanup
failures are explicitly reported/counted and default-off behavior remains
unchanged.

## Task 6: Add bounded Provider and Consumer observability

**Files**

- Modify `mooncake-store/include/client_metric.h`
- Modify `mooncake-store/src/client_metric.cpp`
- Modify `mooncake-store/include/client_service.h`
- Modify `mooncake-store/src/real_client.cpp`
- Modify `mooncake-transfer-engine/include/transport/nvlink_transport/nvlink_transport.h`
- Modify `mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp`
- Modify `mooncake-transfer-engine/include/transport/transport.h`
- Modify `mooncake-transfer-engine/include/transfer_engine.h`
- Modify `mooncake-transfer-engine/include/transfer_engine_impl.h`
- Modify `mooncake-transfer-engine/include/multi_transport.h`
- Modify `mooncake-transfer-engine/src/transfer_engine.cpp`
- Modify `mooncake-transfer-engine/src/transfer_engine_impl.cpp`
- Modify `mooncake-transfer-engine/src/multi_transport.cpp`
- Modify `mooncake-store/tests/client_metrics_test.cpp`
- Add `mooncake-transfer-engine/tests/nvlink_transport_metrics_test.cpp`

### 6.1 Provider metrics and logs

Extend Client metrics with:

- requested and effective HOST_NUMA capacity;
- effective bytes and chunk count by `numa_node`;
- allocation/access/registration/mount duration;
- initialization failures by `stage`;
- rollback attempts and rollback failures.

Add narrow observation methods on `Client`; `RealClient` calls them at each
stage. Existing `Client::SerializeMetrics()` and `/metrics` expose them without
a new endpoint.

Structured startup logs include feature source/state, requested/effective
capacity, node selection, setup CPU/local node/current CUDA device, each chunk's
node/size/granularity/mount outcome, and final readiness or rollback completion.

### 6.2 Consumer metrics and logs

`NvlinkTransport` owns counters/histograms for:

- mapping-cache hit/miss;
- lazy-import latency;
- import/reserve/map/set-access/copy failures;
- READ/WRITE result.

Add the narrowest read-only metrics snapshot/serialization path from the NVLink
transport through TransferEngine into `Client::SerializeMetrics()`; do not add
wire fields or a general transport-control API. If the TE is used standalone,
the same snapshot must remain queryable/loggable without Store.

Concretely, add an optional/default-no-op metrics append operation at the
`Transport` boundary, implement it in `NvlinkTransport`, forward it through
`MultiTransport` when NVLink is nested there, and aggregate it in
`TransferEngineImpl`/`TransferEngine`. `Client::SerializeMetrics()` appends that
serialized transport output after Client metrics. Other transports retain the
default no-op and require no implementation changes.

Allowed labels are `numa_node`, `stage`, `operation`, and `result`. Tests must
reject or demonstrate absence of endpoint, UUID, address, and Fabric handle
labels. Consumer errors log stage and CUDA result only.

**Task exit criterion:** setup, rollback, cold import, warm hit, and failure
stages are distinguishable with bounded cardinality in serialized metrics and
logs.

## Task 7: Add hardware-gated data-path and GB200 acceptance coverage

**Files**

- Add `mooncake-transfer-engine/tests/nvlink_host_numa_fabric_test.cpp`
- Add `mooncake-transfer-engine/tests/nvlink_host_numa_rdma_smoke_test.cpp`
- Modify `mooncake-transfer-engine/tests/CMakeLists.txt`
- Add `mooncake-store/tests/nvlink_host_numa_store_test.cpp`
- Modify `mooncake-store/tests/CMakeLists.txt`
- Add `scripts/gb200/nvlink_host_numa_preflight.sh`
- Add `scripts/gb200/nvlink_host_numa_build.sh`
- Add `scripts/gb200/nvlink_host_numa_provider.py`
- Add `scripts/gb200/nvlink_host_numa_consumer.py`
- Add `scripts/gb200/nvlink_host_numa_bench.py`

### 7.1 Test labels and prerequisite behavior

Register three labels:

- `nvlink_host_numa_unit`: fake-driver and CPU-only tests;
- `nvlink_host_numa_hardware`: Fabric/VMM/Store tests;
- `nvlink_host_numa_rdma`: the independent RDMA premise smoke test.

Apply `set_tests_properties(... PROPERTIES LABELS ...)` explicitly. The unit
label includes the new config/setup/VMM tests plus `client_integration_test`,
`client_metrics_test`, `serializer_test`, and the filtered
`nvlink_host_numa_transfer_metadata_test` CTest wrapper around the
`transfer_metadata_test` binary. The wrapper runs only the self-contained
submission/schema checks; the pre-existing metadata-plugin cases remain on the
original unlabelled CTest entry. The hardware label includes the HOST_NUMA
Fabric/Store tests and both named HBM regression cases. CI must first run
`ctest -N -L <label>` and assert the expected test names/count so an empty label
cannot be reported as success.

Generic environments may skip hardware tests with a precise reason. When
`MC_REQUIRE_MNNVL_FABRIC=1` is set, missing CUDA, Fabric, IMEX, visible GPU, or
NUMA prerequisites fail the test rather than skipping it.

The RDMA smoke has a separate strict switch,
`MC_REQUIRE_NVLINK_HOST_NUMA_RDMA=1`; when set, missing RNIC/verbs/device access
is a failure. The designated job emits JUnit XML and asserts `skipped=0` for
both required hardware labels because CTest otherwise treats a GTest skip as a
successful process.

The preflight script verifies driver/toolkit, Fabric-handle support,
`MC_USE_NVLINK_IPC` absence, IMEX device/configuration, visible GPU PCI BDFs,
PCI-to-NUMA mapping, and online NUMA nodes. Strict mode additionally requires
an executable allocation/export probe and treats an unset probe as failure. The
build script resolves the build/probe dependency by building first and using its
Fabric test as the default probe before strict preflight. It prints no Fabric
handles.

### 7.2 Fabric-capable single-host test

Validate on each selected CPU NUMA node:

- global and local HOST_NUMA allocation;
- setup CPU to local-buffer node selection;
- CPU read/write and all-visible-GPU access;
- HBM-to-HOST_NUMA and HOST_NUMA-to-HBM byte verification;
- Fabric export/import/map/access;
- cold miss followed by warm cache hit;
- injected import-stage failure cleanup followed by successful retry;
- clean destruction with no live VMM resources.

### 7.3 GB200 dual-node Store test

Use the Python ConfigDict overload. A representative provider run is:

```bash
export BUILD_DIR="${BUILD_DIR:-$PWD/build-nvlink-host-numa}"
export PYTHONPATH="${BUILD_DIR}/mooncake-integration${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${BUILD_DIR}/mooncake-common${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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

On a two-NUMA node, the logical plan is approximately 300 GiB per NUMA node;
`MC_MAX_MR_SIZE=150 GiB` forces four Store segments. This is NUMA/max-MR
chunking, not per-GPU export sharding.

From Node B, run HBM-backed Consumers on each visible GPU. For every payload:

The Consumer must remain Torch-free: it loads `libcudart` with `ctypes`, uses
`cudaMalloc` for both registered buffers, copies deterministic host bytes into
source HBM, and copies destination HBM back for SHA256 and byte comparison.
No Python/C++ binding expansion is required.

- Put from registered HBM into remote HOST_NUMA;
- Get from remote HOST_NUMA into registered HBM;
- verify length and SHA256/byte pattern;
- run at least two iterations to observe cold miss then warm hit;
- run GPU 0/1/2/3 Consumers concurrently with zero corruption.

Verify Master total capacity equals the Provider's effective capacity and the
expected chunk count. The dual-node Python data-plane script does not inject
mount failures. Run the hardware `nvlink_host_numa_store_test` with a
test-binary-only `FailNthMountOperations` wrapper around the production setup
adapter: allocation/Fabric/TE are real, while the Nth Master mount returns an
injected error. Assert no Provider segments remain, all TE BufferDesc entries
are gone, local-only registration is gone, readiness never succeeded, and
rollback metrics match. Do not add a production config key or Provider-script
flag for this injection.

### 7.4 RDMA compatibility smoke

On one Provider HOST_NUMA VMM allocation, call `ibv_reg_mr()` and
`ibv_dereg_mr()` with a selected RNIC. Record success/failure and topology. Do
not install an RDMA transport, publish a second descriptor, or claim fallback
support.

### 7.5 Performance characterization

Run single- and multi-GPU cold/warm tests across representative sizes. Record
Put/Get throughput, p50/p95/p99, cold import latency, mapping hit ratio, per-GPU
concurrency, and CPU-NUMA binding. The first hardware run is report-only for
performance; correctness, effective capacity, chunk count, and zero resource
leaks are hard gates immediately.

**Task exit criterion:** named GB200 evidence demonstrates correct multi-NUMA
Provider capacity and bidirectional Store transfers from multiple Consumer
GPUs. Hardware not executed is reported as `NOT RUN`, never as passed.

## Task 8: Integrate CI, documentation, rollout, and rollback

**Files**

- Modify `.github/workflows/ci.yml`
- Add `.github/workflows/ci-gb200.yml` after the self-hosted runner/cluster entry
  is confirmed
- Modify `mooncake-integration/store/store_py.cpp`
- Add/update GB200 test README beside the scripts

### 8.1 Ordinary CI

Add CPU/fake-driver targets to existing jobs. A representative local matrix is:

```bash
cmake -S . -B build-host -G Ninja \
  -DBUILD_UNIT_TESTS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DWITH_STORE_RUST=OFF \
  -DWITH_EP=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_MNNVL=OFF

cmake --build build-host --target \
  nvlink_host_numa_config_test \
  nvlink_host_numa_setup_test \
  client_integration_test \
  client_metrics_test \
  serializer_test \
  transfer_metadata_test -j

ctest --test-dir build-host -N -L nvlink_host_numa_unit
ctest --test-dir build-host -L nvlink_host_numa_unit --output-on-failure
```

The existing arm64 `USE_CUDA=ON, USE_MNNVL=ON` job must compile the real code
and run only fake-driver/unit-labeled tests. It must not call real CUDA driver
initialization through the stub library. Prefer linking the fake-driver target
only to the isolated VMM driver-adapter/object library, not the full Transfer
Engine, so it has no runtime `libcuda.so.1` dependency. If the target cannot be
isolated, the job must discover the architecture-specific stub directory (for
SBSA normally `/usr/local/cuda/targets/sbsa-linux/lib/stubs`), provide a
temporary `libcuda.so.1 -> libcuda.so` symlink, and prepend that directory to
`LD_LIBRARY_PATH` before running the unit label.

Keep the existing HBM `nvlink_transport_test` in the hardware matrix and rerun
it after disabling HOST_NUMA to prove rollback compatibility.

### 8.2 Python documentation and test harness

Update only the ConfigDict overload's supported-key docstring with:

- `enable_nvlink_host_numa`;
- `nvlink_host_numa_nodes`.

Do not add the keys to the fixed positional overload. The GB200 provider script
must construct and print a redacted ConfigDict, report requested/effective
capacity, and expose readiness only after setup returns success.

### 8.3 Designated GB200 job

The baseline does not reveal the required self-hosted runner label. Confirm the
runner or cluster entry before adding `ci-gb200.yml`. Once connected, the job:

1. builds with CUDA/MNNVL;
2. runs the strict preflight;
3. runs hardware and RDMA labels;
4. launches the dual-node Store harness;
5. uploads topology, versions, metrics, and test results;
6. fails, rather than skips, when required prerequisites are absent.

Until that integration exists, record all hardware items as `NOT RUN`.

### 8.4 Rollout and rollback

Roll out in this order:

1. merge default-off with ordinary CI and HBM/NVLink regression;
2. pass single-host Fabric tests;
3. pass GB200 dual-node correctness and the strict-required RDMA premise smoke
   with JUnit `skipped=0`;
4. enable a small Provider canary set;
5. inspect effective capacity, per-NUMA chunks, lazy import, copy errors, and
   rollback metrics;
6. expand only after target-hardware evidence is recorded.

Rollback is configuration plus Provider restart:

```text
enable_nvlink_host_numa=false
```

There is no online backing switch. Stop readiness/new work, restart the
Provider on the old path, and rerun HBM NVLink plus Store Consumer Put/Get
smoke tests. Do not claim that this invalidates stale Consumer mappings; issue
#2832 remains the restart/address-reuse boundary.

**Task exit criterion:** default-off code is mergeable without GB200 runtime,
while enablement is blocked on explicit Fabric/GB200 evidence and has a tested
restart rollback.

## Verification matrix

| Layer | Required evidence | Hard gate |
|---|---|---|
| CPU ordinary CI | config, CSV/auto abstraction, capacity math, rollback ordering, mount compensation, metrics labels, Store/TE schema golden tests | Yes |
| CUDA/MNNVL compile + fake driver | RAII cleanup at every VMM/import failure, registration contract, terminal batch failure | Yes |
| Existing HBM/IPC/Fabric regression | source-compatible allocation wrappers, HBM Put/Get, IPC mode when feature disabled | Yes |
| Fabric single host | CPU/all-GPU access, bidirectional copies, export/import, miss/hit, retry | Yes before canary |
| GB200 dual node | multi-NUMA chunks, Store Put/Get, concurrent Consumer GPUs, readiness/rollback | Yes before canary |
| RDMA premise | MR register/deregister on HOST_NUMA VMM | Required evidence, not fallback support |
| Performance | cold/warm and concurrency report with machine/software provenance | Report-only initially |

## Definition of done

- Both new ConfigDict keys are implemented and documented; all fixed setup APIs
  remain unchanged.
- Feature absent/false uses the pre-existing setup path and passes regression.
- Store requested/effective capacity and Master capacity agree.
- No Store segment crosses NUMA nodes or VMM handles.
- Local buffers are never published remotely.
- Master mount failure compensates current TE registration; later failure rolls
  back earlier chunks in reverse order.
- Every VMM/import failure test reports zero leaked handles, mappings, and VAs.
- Consumer cold miss, warm hit, and later retry are observable.
- Store and TE serialized schemas have no new V1 fields.
- No RDMA fallback, topology selector, H2H contract, online hot-add, per-GPU
  export sharding, or generation/invalidation change has entered the diff.
- Hardware results include exact GB200 topology, CUDA/driver/IMEX versions, and
  are `PASS`, `FAIL`, or `NOT RUN`; a skip is not accepted as GB200 evidence.
