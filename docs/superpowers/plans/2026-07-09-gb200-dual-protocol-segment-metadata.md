# GB200 Dual Protocol Segment Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix same-scale-up-domain GB200 reader failures by preserving RDMA local segment metadata when NVLink is installed after RDMA auto-discovery.

**Architecture:** Treat transport installation as incremental local segment construction under `ENABLE_MULTI_PROTOCOL`. RDMA already appends its protocol to the existing local segment; NVLink must follow the same pattern instead of replacing the segment with a single-protocol descriptor.

**Tech Stack:** C++17, GoogleTest, Mooncake Transfer Engine metadata, POSIX shell verification.

## Global Constraints

- Work in `/Users/wugou.cyf/workspace/Mooncake/.worktrees/nvlink-host-numa-dual-register` on branch `codex/nvlink-host-numa-dual-register`.
- Preserve unrelated local changes.
- Keep the fix scoped to metadata construction; do not change path-selection policy.
- Same-domain reader is expected to select `nvlink`; different-domain reader is expected to select `rdma`.
- Local Mac verification may be limited by unavailable CUDA/RDMA GB200 build dependencies.

---

### Task 1: Reproduce Metadata Overwrite With a Unit Test

**Files:**
- Modify: `mooncake-transfer-engine/tests/nvlink_transport_test.cpp`

**Interfaces:**
- Consumes: `TransferMetadata::addLocalSegment(...)`, `NvlinkTransport::install(...)`.
- Produces: a regression test proving an existing RDMA local segment survives NVLink install.

- [x] **Step 1: Write the failing test**

Add this test before the HOST_NUMA fabric metadata test:

```cpp
TEST(NvlinkTransportTest, InstallPreservesExistingMultiProtocolLocalSegment) {
#ifndef ENABLE_MULTI_PROTOCOL
    GTEST_SKIP() << "ENABLE_MULTI_PROTOCOL is not compiled in";
#else
    auto metadata = std::make_shared<TransferMetadata>(P2PHANDSHAKE);
    const std::string local_server_name = "dual-provider:12355";

    auto rdma_desc = std::make_shared<TransferMetadata::SegmentDesc>();
    ASSERT_NE(rdma_desc, nullptr);
    rdma_desc->name = local_server_name;
    rdma_desc->protocol = "rdma";
    rdma_desc->rdma_server_name = "10.192.9.60:12355";

    TransferMetadata::DeviceDesc device;
    device.name = "mlx5_0";
    device.lid = 1;
    device.gid = "0000:0000:0000:0000:0000:ffff:0a00:0001";
    rdma_desc->devices.push_back(device);

    ASSERT_EQ(metadata->addLocalSegment(LOCAL_SEGMENT_ID, local_server_name,
                                        std::move(rdma_desc)),
              0);

    NvlinkTransport transport;
    std::string install_name = local_server_name;
    ASSERT_EQ(transport.install(install_name, metadata, nullptr), 0);

    auto merged_desc =
        metadata->getSegmentDescByID(LOCAL_SEGMENT_ID, false);
    ASSERT_NE(merged_desc, nullptr);
    EXPECT_EQ(merged_desc->name, local_server_name);
    EXPECT_EQ(merged_desc->protocol, "rdma,nvlink");
    EXPECT_EQ(merged_desc->rdma_server_name, "10.192.9.60:12355");
    ASSERT_EQ(merged_desc->devices.size(), 1u);
    EXPECT_EQ(merged_desc->devices[0].name, "mlx5_0");
#endif
}
```

- [x] **Step 2: Attempt red verification locally**

Run:

```bash
cmake --build build-gb200 --target nvlink_transport_test
```

Expected before the fix: the new test fails because `NvlinkTransport::install()` replaces protocol `rdma` with `nvlink` and drops RDMA device metadata.

Actual local result: Mac `build` has no `nvlink_transport_test` target, so red execution must be done in the GB200 `build-gb200` tree.

### Task 2: Preserve Existing Local Segment on NVLink Install

**Files:**
- Modify: `mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp`

**Interfaces:**
- Consumes: `TransferMetadata::getSegmentDescByID(LOCAL_SEGMENT_ID, false)`.
- Produces: `NvlinkTransport::install()` appends `nvlink` under `ENABLE_MULTI_PROTOCOL` without losing existing fields.

- [x] **Step 1: Add a protocol append helper**

Add a local helper in the anonymous namespace:

```cpp
bool hasProtocol(const std::string &protocols, const std::string &protocol) {
    size_t start = 0;
    while (start <= protocols.size()) {
        const size_t end = protocols.find(',', start);
        const std::string token =
            protocols.substr(start, end == std::string::npos
                                        ? std::string::npos
                                        : end - start);
        if (token == protocol) return true;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}

void appendProtocolIfMissing(std::string &protocols,
                             const std::string &protocol) {
    if (hasProtocol(protocols, protocol)) return;
    if (!protocols.empty()) protocols += ",";
    protocols += protocol;
}
```

- [x] **Step 2: Merge instead of replace**

Change `NvlinkTransport::install()` so `ENABLE_MULTI_PROTOCOL` reuses the existing local descriptor when present, then appends `nvlink`:

```cpp
#ifdef ENABLE_MULTI_PROTOCOL
    auto desc = metadata_->getSegmentDescByID(LOCAL_SEGMENT_ID, false);
    if (!desc) desc = std::make_shared<SegmentDesc>();
#else
    auto desc = std::make_shared<SegmentDesc>();
#endif
    if (!desc) return ERR_MEMORY;
    desc->name = local_server_name_;
#ifdef ENABLE_MULTI_PROTOCOL
    appendProtocolIfMissing(desc->protocol, "nvlink");
#else
    desc->protocol = "nvlink";
#endif
```

- [x] **Step 3: Attempt green verification locally**

Run:

```bash
cmake --build build-gb200 --target nvlink_transport_test
```

Expected after the fix: the new test passes, and GB200-specific HOST_NUMA tests continue to build.

Actual local result: Mac `build` still has no `nvlink_transport_test` target; use the GB200 verification command below after pulling the pushed branch.

### Task 3: Static Verification, Commit, and Push

**Files:**
- Validate all changed files.

**Interfaces:**
- Consumes: git worktree state.
- Produces: pushed branch on `neverhook/codex/nvlink-host-numa-dual-register`.

- [x] **Step 1: Run static checks**

Run:

```bash
git diff --check
git status --short
```

- [x] **Step 2: Commit the fix**

Run:

```bash
git add docs/superpowers/plans/2026-07-09-gb200-dual-protocol-segment-metadata.md \
        mooncake-transfer-engine/tests/nvlink_transport_test.cpp \
        mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp
git commit -m "fix: preserve rdma metadata when installing nvlink"
```

- [x] **Step 3: Push to neverhook**

Run:

```bash
git push neverhook codex/nvlink-host-numa-dual-register
```

Expected: remote branch contains the metadata fix and regression test.
