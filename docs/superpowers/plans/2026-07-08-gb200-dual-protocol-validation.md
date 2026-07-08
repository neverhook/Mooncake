# GB200 Dual Protocol Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GB200 dual-protocol validation explicit and reproducible: provider publishes a HOST_NUMA `nvlink,rdma` replica, reader reads remote HOST_NUMA DRAM into local HBM, and scripts verify same-domain NVLink versus different-domain RDMA selection.

**Architecture:** Keep publishable replica memory and local transfer buffers separate. Provider-mounted HOST_NUMA memory remains metadata-visible via `MountDualProtocolSegmentAndGetId`, while setup-time local buffers and Python `register_buffer()` destinations register only with local transports and do not become remotely discoverable replicas. Python descriptor bindings expose memory-kind, domain, and selected protocol so validation scripts can assert the intended path instead of inferring it from logs.

**Tech Stack:** C++17, Mooncake Store/Transfer Engine, pybind11, POSIX shell, Python ctypes, CUDA Driver API.

## Global Constraints

- Work in `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register` on branch `codex/nvlink-host-numa-dual-register`.
- Preserve provider replica publication through `MountDualProtocolSegmentAndGetId(..., "nvlink,rdma", "HOST_NUMA", scale_up_domain_id)`.
- Do not require PyTorch in GB200 scripts; use Python `ctypes` and CUDA Driver API.
- Use `scripts/gb200_env_exec.sh` as the source-free environment wrapper.
- Provider dual path must use `MC_STORE_MEMCPY=1` for local population of the mounted segment.
- Reader must use `get_into(key, hbm_ptr, size)` to validate `remote HOST_NUMA DRAM -> local HBM`.
- Same `MC_NVLINK_SCALE_UP_DOMAIN_ID` is expected to select `nvlink`; different domain is expected to select `rdma`.
- Verification on the local Mac may be limited to static checks if CUDA/RDMA GB200 build artifacts are unavailable.

---

### Task 1: Local Transfer Buffer Registration

**Files:**
- Modify: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/mooncake-store/include/client_service.h`
- Modify: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/mooncake-store/src/client_service.cpp`
- Modify: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/mooncake-store/src/real_client.cpp`

**Interfaces:**
- Consumes: `GetProtocolSpecificRegistrationProtocols(protocol_)`, `RegisterMemoryForProtocols(...)`, `Client::RegisterLocalMemory(...)`.
- Produces:
  - `Client::RegisterLocalTransferBuffer(void* addr, size_t length, const std::string& location, bool remote_accessible) -> tl::expected<void, ErrorCode>`
  - `RealClient::setup()` uses local-only registration for setup-time buffers.
  - `RealClient::register_buffer_internal()` uses local-only registration for `get_into` / `put_from` buffers.

- [ ] **Step 1: Add the Client interface**

Add this method after `RegisterLocalMemory(...)` in `mooncake-store/include/client_service.h`:

```cpp
    /**
     * @brief Registers a local transfer buffer without publishing it as a
     *        remotely discoverable replica segment.
     */
    tl::expected<void, ErrorCode> RegisterLocalTransferBuffer(
        void* addr, size_t length, const std::string& location,
        bool remote_accessible = false);
```

- [ ] **Step 2: Implement local-only protocol registration**

Add this implementation after `Client::RegisterLocalMemory(...)` in `mooncake-store/src/client_service.cpp`:

```cpp
tl::expected<void, ErrorCode> Client::RegisterLocalTransferBuffer(
    void* addr, size_t length, const std::string& location,
    bool remote_accessible) {
    auto check_result = CheckRegisterMemoryParams(addr, length);
    if (!check_result) {
        return tl::unexpected(check_result.error());
    }

#ifdef ENABLE_MULTI_PROTOCOL
    auto protocols = GetProtocolSpecificRegistrationProtocols(protocol_);
    if (!protocols.empty()) {
        if (RegisterMemoryForProtocols(transfer_engine_.get(), protocols, addr,
                                       length, location,
                                       /*update_metadata=*/false,
                                       remote_accessible) != 0) {
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }
        return {};
    }
#endif

    if (transfer_engine_->registerLocalMemory(addr, length, location,
                                              remote_accessible,
                                              /*update_metadata=*/false) != 0) {
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    return {};
}
```

- [ ] **Step 3: Route setup-time local buffer through the new API**

Replace the local-buffer setup registration in `mooncake-store/src/real_client.cpp`:

```cpp
        auto result = client_->RegisterLocalTransferBuffer(
            client_buffer_allocator_->getBase(), local_buffer_size,
            kWildcardLocation, false);
```

Replace the matching teardown unregister call with:

```cpp
        auto unregister_result = client_->unregisterLocalMemory(
            client_buffer_allocator_->getBase(), false);
```

- [ ] **Step 4: Route Python register_buffer through the new API**

Replace `register_buffer_internal()` registration with:

```cpp
    auto result = client_->RegisterLocalTransferBuffer(
        buffer, size, kWildcardLocation, false);
```

Replace `unregister_buffer_internal()` unregister with:

```cpp
    auto unregister_result = client_->unregisterLocalMemory(buffer, false);
```

- [ ] **Step 5: Validate the task**

Run:

```bash
git diff -- mooncake-store/include/client_service.h mooncake-store/src/client_service.cpp mooncake-store/src/real_client.cpp
```

Expected: only setup-time buffer and explicit `register_buffer()` paths move to metadata-free transfer registration; `MountSegmentAndGetId` and `MountDualProtocolSegmentAndGetId` still publish segment metadata.

### Task 2: Python Descriptor Introspection

**Files:**
- Modify: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/mooncake-integration/store/store_py.cpp`

**Interfaces:**
- Consumes: `AllocatedBuffer::Descriptor::{memory_kind_, scale_up_domain_id_, selected_protocol_}`.
- Produces Python fields:
  - `Descriptor.memory_kind`
  - `Descriptor.scale_up_domain_id`
  - `Descriptor.selected_protocol`
  - `MooncakeDistributedStore.get_selected_replica_desc_for_buffer(key, buffer_ptr, size) -> list[ReplicaDescriptor]`

- [ ] **Step 1: Expose descriptor fields in pybind**

Change the descriptor binding in `mooncake-integration/store/store_py.cpp` to:

```cpp
        .def_readwrite("transport_endpoint",
                       &AllocatedBuffer::Descriptor::transport_endpoint_)
        .def_readwrite("memory_kind",
                       &AllocatedBuffer::Descriptor::memory_kind_)
        .def_readwrite("scale_up_domain_id",
                       &AllocatedBuffer::Descriptor::scale_up_domain_id_)
        .def_readwrite("selected_protocol",
                       &AllocatedBuffer::Descriptor::selected_protocol_)
        .def("__repr__", [](const AllocatedBuffer::Descriptor &desc) {
            return "<Descriptor size=" + std::to_string(desc.size_) +
                   " buffer_address=" + std::to_string(desc.buffer_address_) +
                   " transport_endpoint=" + desc.transport_endpoint_ +
                   " memory_kind=" + desc.memory_kind_ +
                   " scale_up_domain_id=" + desc.scale_up_domain_id_ +
                   " selected_protocol=" + desc.selected_protocol_ + ">";
        });
```

- [ ] **Step 2: Validate the task**

Add a narrow debug binding that casts to `RealClient`, calls `get_selected_replica_desc_for_buffer(key, buffer, size)`, and returns one selected descriptor whose `selected_protocol` was computed with the same destination pointer type used by `get_into`.

- [ ] **Step 3: Validate the task**

Run:

```bash
git diff -- mooncake-integration/store/store_py.cpp mooncake-store/include/real_client.h mooncake-store/src/real_client.cpp
```

Expected: Python can print descriptor metadata and assert the selected path for a specific HBM destination pointer.

### Task 3: GB200 Dual-Machine Scripts

**Files:**
- Create: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/scripts/gb200_build.sh`
- Create: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/scripts/gb200_start_control_plane_node_a.sh`
- Create: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/scripts/gb200_provider_dual_node_a.sh`
- Create: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/scripts/gb200_reader_dual_node_b.sh`
- Create: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/scripts/gb200_reader_dual_same_domain_node_b.sh`
- Create: `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register/scripts/gb200_reader_dual_diff_domain_node_b.sh`

**Interfaces:**
- Consumes: `scripts/gb200_env_exec.sh`, Mooncake Python binding, CUDA driver at `libcuda.so.1`.
- Produces:
  - Build entrypoint for GB200.
  - Node A control-plane starter.
  - Node A provider that mounts/publishes a HOST_NUMA dual segment and keeps it alive.
  - Node B reader wrappers for same-domain NVLink and different-domain RDMA.

- [ ] **Step 1: Add GB200 build script**

Create `scripts/gb200_build.sh` with strict POSIX shell behavior, default `BUILD=/workspace/Mooncake/build-gb200`, `USE_CUDA=ON`, `USE_ETCD=ON`, `STORE_USE_ETCD=ON`, `ENABLE_MULTI_PROTOCOL=ON`, and target `nvlink_transport_test mooncake_store`.

- [ ] **Step 2: Add Node A control-plane script**

Create `scripts/gb200_start_control_plane_node_a.sh` that kills stale `mooncake_master` only when `STOP_OLD_MASTER=1`, starts `${BUILD}/mooncake-store/src/mooncake_master --rpc_address=0.0.0.0 --port=${MASTER_PORT:-50051}`, writes `/tmp/mooncake_master.gb200.log`, and prints the PID plus listener status.

- [ ] **Step 3: Add Node A provider script**

Create `scripts/gb200_provider_dual_node_a.sh` that runs through `gb200_env_exec.sh`, sets `MC_STORE_MEMCPY=1`, initializes `MooncakeDistributedStore` with protocol `nvlink,rdma`, mounts a dual HOST_NUMA segment via setup, writes deterministic bytes with `put_from`, and waits until interrupted.

The Python payload must include:

```python
cfg = ReplicateConfig()
cfg.replica_num = 1
cfg.preferred_segment = local_hostname
rc = store.put_from(key, ctypes.addressof(payload), len(payload), cfg)
```

- [ ] **Step 4: Add common Node B reader script**

Create `scripts/gb200_reader_dual_node_b.sh` that runs through `gb200_env_exec.sh`, allocates HBM via CUDA Driver API `cuMemAlloc_v2`, registers the HBM pointer with `store.register_buffer`, calls `store.get_into(key, hbm_ptr, size)`, copies HBM to host with `cuMemcpyDtoH_v2`, verifies deterministic bytes, prints descriptor fields, and exits nonzero if `EXPECTED_PATH` is set and the selected protocol differs.

- [ ] **Step 5: Add same-domain and different-domain wrappers**

Create:

```sh
scripts/gb200_reader_dual_same_domain_node_b.sh
scripts/gb200_reader_dual_diff_domain_node_b.sh
```

The same-domain wrapper sets:

```sh
EXPECTED_PATH=nvlink
MC_NVLINK_SCALE_UP_DOMAIN_ID=${MC_NVLINK_SCALE_UP_DOMAIN_ID:-gb200-nvl}
```

The different-domain wrapper sets:

```sh
EXPECTED_PATH=rdma
MC_NVLINK_SCALE_UP_DOMAIN_ID=${MC_NVLINK_SCALE_UP_DOMAIN_ID:-gb200-nvl-different}
```

- [ ] **Step 6: Validate shell syntax**

Run:

```bash
sh -n scripts/gb200_build.sh
sh -n scripts/gb200_start_control_plane_node_a.sh
sh -n scripts/gb200_provider_dual_node_a.sh
sh -n scripts/gb200_reader_dual_node_b.sh
sh -n scripts/gb200_reader_dual_same_domain_node_b.sh
sh -n scripts/gb200_reader_dual_diff_domain_node_b.sh
```

Expected: all commands exit `0`.

### Task 4: Repository Verification, Commit, and Push

**Files:**
- Verify all files changed by Tasks 1-3.

**Interfaces:**
- Produces one pushed commit on `neverhook/codex/nvlink-host-numa-dual-register`.

- [ ] **Step 1: Static diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only planned files are modified or created.

- [ ] **Step 2: Build/test what is available locally**

Run the most specific available local verification command:

```bash
cmake --build build --target mooncake_store -j2
```

If local build artifacts do not contain the target or CUDA/RDMA dependencies are absent, record the exact failure and rely on the GB200 scripts for runtime validation.

- [ ] **Step 3: Stage, commit, and push**

Run:

```bash
git add mooncake-store/include/client_service.h \
        mooncake-store/src/client_service.cpp \
        mooncake-store/src/real_client.cpp \
        mooncake-integration/store/store_py.cpp \
        scripts/gb200_build.sh \
        scripts/gb200_start_control_plane_node_a.sh \
        scripts/gb200_provider_dual_node_a.sh \
        scripts/gb200_reader_dual_node_b.sh \
        scripts/gb200_reader_dual_same_domain_node_b.sh \
        scripts/gb200_reader_dual_diff_domain_node_b.sh \
        docs/superpowers/plans/2026-07-08-gb200-dual-protocol-validation.md
git commit -m "test: add gb200 dual protocol validation scripts"
git push neverhook codex/nvlink-host-numa-dual-register
```

Expected: push succeeds.
