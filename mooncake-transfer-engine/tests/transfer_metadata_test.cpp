// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#if __has_include(<jsoncpp/json/json.h>)
#include <jsoncpp/json/json.h>
#else
#include <json/json.h>
#endif
#include <netdb.h>

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "common.h"
#include "topology.h"

#define private public
#include "transfer_metadata.h"
#undef private

#include "transfer_metadata_plugin.h"
#include "transport/transport.h"

using namespace mooncake;

namespace mooncake {

namespace {

class InMemoryMetadataStoragePlugin : public MetadataStoragePlugin {
   public:
    bool get(const std::string &key, Json::Value &value) override {
        auto it = values_.find(key);
        if (it == values_.end()) return false;
        value = it->second;
        return true;
    }

    bool set(const std::string &key, const Json::Value &value) override {
        values_[key] = value;
        return true;
    }

    bool remove(const std::string &key) override {
        return values_.erase(key) != 0;
    }

   private:
    std::unordered_map<std::string, Json::Value> values_;
};

}  // namespace

class TransferMetadataTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // initialize glog
        google::InitGoogleLogging("TransferMetadataTest");
        FLAGS_logtostderr = 1;  // output to stdout

        const char* env = std::getenv("MC_METADATA_SERVER");
        if (env)
            metadata_server = env;
        else
            metadata_server = P2PHANDSHAKE;
        LOG(INFO) << "metadata_server: " << metadata_server;

        env = std::getenv("MC_LOCAL_SERVER_NAME");
        if (env)
            local_server_name = env;
        else
            local_server_name = "127.0.0.2:12345";
        LOG(INFO) << "local_server_name: " << local_server_name;

        metadata_client = std::make_unique<TransferMetadata>(metadata_server);
        if (metadata_server == P2PHANDSHAKE) {
            metadata_client->p2p_handshake_mode_ = false;
            metadata_client->storage_plugin_ =
                std::make_shared<InMemoryMetadataStoragePlugin>();
        }
    }
    void TearDown() override {
        // clean up glog
        google::ShutdownGoogleLogging();
    }
    std::unique_ptr<TransferMetadata> metadata_client;
    std::string metadata_server;
    std::string local_server_name;
};

TEST_F(TransferMetadataTest, EncodeDecodeNvlinkRdmaSegment) {
#ifndef ENABLE_MULTI_PROTOCOL
    GTEST_SKIP() << "ENABLE_MULTI_PROTOCOL is not compiled in";
#else
    TransferMetadata::SegmentDesc desc;
    desc.name = "dual-segment";
    desc.protocol = "nvlink,rdma";
    desc.rdma_server_name = "10.0.0.1:12345";
    desc.tcp_data_port = 0;

    TransferMetadata::DeviceDesc device;
    device.name = "mlx5_0";
    device.lid = 1;
    device.gid = "0000:0000:0000:0000:0000:ffff:0a00:0001";
    desc.devices.push_back(device);

    TransferMetadata::BufferDesc nvlink_buffer;
    nvlink_buffer.name = "host_numa:0";
    nvlink_buffer.addr = 0x100000;
    nvlink_buffer.length = 4096;
    nvlink_buffer.protocol = "nvlink";
    nvlink_buffer.shm_name = "fabric-handle-bytes";
    nvlink_buffer.memory_kind = "HOST_NUMA";
    nvlink_buffer.scale_up_domain_id = "domain-a";
    desc.buffers.push_back(nvlink_buffer);

    TransferMetadata::BufferDesc rdma_buffer;
    rdma_buffer.name = "host_numa:0";
    rdma_buffer.addr = 0x100000;
    rdma_buffer.length = 4096;
    rdma_buffer.protocol = "rdma";
    rdma_buffer.lkey.push_back(11);
    rdma_buffer.rkey.push_back(22);
    desc.buffers.push_back(rdma_buffer);

    Json::Value encoded;
    ASSERT_EQ(metadata_client->encodeSegmentDesc(desc, encoded), 0);
    ASSERT_TRUE(encoded["protocol"].isArray());
    ASSERT_EQ(encoded["protocol"][0].asString(), "nvlink");
    ASSERT_EQ(encoded["protocol"][1].asString(), "rdma");

    auto decoded =
        metadata_client->decodeSegmentDesc(encoded, "dual-segment");
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(decoded->protocol, "nvlink,rdma");
    ASSERT_EQ(decoded->buffers.size(), 2u);
    EXPECT_EQ(decoded->buffers[0].protocol, "nvlink");
    EXPECT_EQ(decoded->buffers[0].memory_kind, "HOST_NUMA");
    EXPECT_EQ(decoded->buffers[0].scale_up_domain_id, "domain-a");
    EXPECT_EQ(decoded->buffers[0].shm_name, "fabric-handle-bytes");
    EXPECT_EQ(decoded->buffers[1].protocol, "rdma");
    EXPECT_EQ(decoded->buffers[1].rkey[0], 22u);
#endif
}

TEST_F(TransferMetadataTest, RejectsUnsupportedMultiProtocolTriple) {
#ifndef ENABLE_MULTI_PROTOCOL
    GTEST_SKIP() << "ENABLE_MULTI_PROTOCOL is not compiled in";
#else
    TransferMetadata::SegmentDesc desc;
    desc.name = "bad-segment";
    desc.protocol = "nvlink,rdma,tcp";
    Json::Value encoded;
    EXPECT_EQ(metadata_client->encodeSegmentDesc(desc, encoded),
              ERR_INVALID_ARGUMENT);
#endif
}

TEST_F(TransferMetadataTest, RejectsMultiProtocolBufferMissingProtocol) {
#ifndef ENABLE_MULTI_PROTOCOL
    GTEST_SKIP() << "ENABLE_MULTI_PROTOCOL is not compiled in";
#else
    TransferMetadata::SegmentDesc desc;
    desc.name = "bad-buffer-segment";
    desc.protocol = "nvlink,rdma";

    TransferMetadata::BufferDesc buffer;
    buffer.name = "host_numa:0";
    buffer.addr = 0x100000;
    buffer.length = 4096;
    desc.buffers.push_back(buffer);

    Json::Value encoded;
    EXPECT_EQ(metadata_client->encodeSegmentDesc(desc, encoded),
              ERR_INVALID_ARGUMENT);
#endif
}

TEST_F(TransferMetadataTest, RejectsMultiProtocolDecodeBufferOutsidePair) {
#ifndef ENABLE_MULTI_PROTOCOL
    GTEST_SKIP() << "ENABLE_MULTI_PROTOCOL is not compiled in";
#else
    TransferMetadata::SegmentDesc desc;
    desc.name = "dual-segment";
    desc.protocol = "nvlink,rdma";
    desc.tcp_data_port = 0;

    TransferMetadata::DeviceDesc device;
    device.name = "mlx5_0";
    device.lid = 1;
    device.gid = "0000:0000:0000:0000:0000:ffff:0a00:0001";
    desc.devices.push_back(device);

    TransferMetadata::BufferDesc rdma_buffer;
    rdma_buffer.name = "host_numa:0";
    rdma_buffer.addr = 0x100000;
    rdma_buffer.length = 4096;
    rdma_buffer.protocol = "rdma";
    rdma_buffer.lkey.push_back(11);
    rdma_buffer.rkey.push_back(22);
    desc.buffers.push_back(rdma_buffer);

    Json::Value encoded;
    ASSERT_EQ(metadata_client->encodeSegmentDesc(desc, encoded), 0);

    Json::Value tcp_buffer = encoded;
    tcp_buffer["buffers"][0]["protocol"] = "tcp";
    EXPECT_EQ(metadata_client->decodeSegmentDesc(tcp_buffer, "dual-segment"),
              nullptr);

    Json::Value typo_buffer = encoded;
    typo_buffer["buffers"][0]["protocol"] = "typo";
    EXPECT_EQ(metadata_client->decodeSegmentDesc(typo_buffer, "dual-segment"),
              nullptr);
#endif
}

// add and search LocalSegmentMeta
TEST_F(TransferMetadataTest, LocalSegmentTest) {
    auto segment_des = std::make_shared<TransferMetadata::SegmentDesc>();
    segment_des->name = "test_server";
    segment_des->protocol = "rdma";
    TransferMetadata::SegmentID segment_id = 1111111;
    std::string segment_name = "test_segment";
    int re = metadata_client->addLocalSegment(segment_id, segment_name,
                                              std::move(segment_des));
    ASSERT_EQ(re, 0);
    auto des = metadata_client->getSegmentDescByName(segment_name);
    ASSERT_EQ(des, segment_des);
    des = metadata_client->getSegmentDescByID(segment_id, false);
    ASSERT_EQ(des, segment_des);
    auto id = metadata_client->getSegmentID(segment_name);
    ASSERT_EQ(id, segment_id);
    re = metadata_client->removeLocalSegment(segment_name);
    ASSERT_EQ(re, 0);
}

// add and remove LocalMemoryBufferMeta
TEST_F(TransferMetadataTest, LocalMemoryBufferTest) {
    auto segment_des = std::make_shared<TransferMetadata::SegmentDesc>();
    segment_des->name = "test_localMemory";
    segment_des->protocol = "rdma";
    int re = metadata_client->addLocalSegment(
        LOCAL_SEGMENT_ID, "test_local_segment", std::move(segment_des));
    ASSERT_EQ(re, 0);
    uint64_t addr = 0;
    for (int i = 0; i < 10; ++i) {
        TransferMetadata::BufferDesc buffer_des;
        buffer_des.addr = addr + i * 2048;
        buffer_des.length = 1024;
        re = metadata_client->addLocalMemoryBuffer(buffer_des, false);
        ASSERT_EQ(re, 0);
    }
    addr = 1000;
    re = metadata_client->removeLocalMemoryBuffer((void*)addr, false);
    ASSERT_EQ(re, ERR_ADDRESS_NOT_REGISTERED);
    for (int i = 9; i > 0; --i) {
        addr = i * 2048;
        re = metadata_client->removeLocalMemoryBuffer((void*)addr, false);
        ASSERT_EQ(re, 0);
    }
    re = metadata_client->removeLocalSegment("test_local_segment");
    ASSERT_EQ(re, 0);
}

TEST_F(TransferMetadataTest, RemoveLocalMemoryBufferHonorsProtocolFilter) {
#ifndef ENABLE_MULTI_PROTOCOL
    GTEST_SKIP() << "ENABLE_MULTI_PROTOCOL is not compiled in";
#else
    auto segment_des = std::make_shared<TransferMetadata::SegmentDesc>();
    segment_des->name = "test_protocol_filtered_localMemory";
    segment_des->protocol = "nvlink,rdma";
    int re = metadata_client->addLocalSegment(
        LOCAL_SEGMENT_ID, "test_protocol_filtered_local_segment",
        std::move(segment_des));
    ASSERT_EQ(re, 0);

    constexpr uint64_t addr = 0x100000;
    TransferMetadata::BufferDesc rdma_buffer;
    rdma_buffer.addr = addr;
    rdma_buffer.length = 4096;
    rdma_buffer.protocol = "rdma";
    ASSERT_EQ(metadata_client->addLocalMemoryBuffer(rdma_buffer, false), 0);

    TransferMetadata::BufferDesc nvlink_buffer;
    nvlink_buffer.addr = addr;
    nvlink_buffer.length = 4096;
    nvlink_buffer.protocol = "nvlink";
    nvlink_buffer.memory_kind = "HOST_NUMA";
    nvlink_buffer.scale_up_domain_id = "domain-a";
    ASSERT_EQ(metadata_client->addLocalMemoryBuffer(nvlink_buffer, false), 0);

    ASSERT_EQ(metadata_client->removeLocalMemoryBuffer((void*)addr, false,
                                                       "nvlink"),
              0);
    auto desc = metadata_client->getSegmentDescByID(LOCAL_SEGMENT_ID, false);
    ASSERT_NE(desc, nullptr);
    ASSERT_EQ(desc->buffers.size(), 1u);
    EXPECT_EQ(desc->buffers[0].protocol, "rdma");

    ASSERT_EQ(metadata_client->removeLocalMemoryBuffer((void*)addr, false,
                                                       "nvlink"),
              ERR_ADDRESS_NOT_REGISTERED);
    desc = metadata_client->getSegmentDescByID(LOCAL_SEGMENT_ID, false);
    ASSERT_NE(desc, nullptr);
    ASSERT_EQ(desc->buffers.size(), 1u);
    EXPECT_EQ(desc->buffers[0].protocol, "rdma");

    ASSERT_EQ(metadata_client->removeLocalMemoryBuffer((void*)addr, false,
                                                       "rdma"),
              0);
    desc = metadata_client->getSegmentDescByID(LOCAL_SEGMENT_ID, false);
    ASSERT_NE(desc, nullptr);
    EXPECT_TRUE(desc->buffers.empty());

    re = metadata_client->removeLocalSegment(
        "test_protocol_filtered_local_segment");
    ASSERT_EQ(re, 0);
#endif
}

// add, get and remove RPCMetaEntryMeta
TEST_F(TransferMetadataTest, RpcMetaEntryTest) {
    auto hostname_port = parseHostNameWithPort(local_server_name);
    TransferMetadata::RpcMetaDesc desc;
    desc.ip_or_host_name = hostname_port.first.c_str();
    desc.rpc_port = hostname_port.second;
    int re = metadata_client->addRpcMetaEntry("test_server", desc);
    ASSERT_EQ(re, 0);
    TransferMetadata::RpcMetaDesc desc1;
    re = metadata_client->getRpcMetaEntry("test_server", desc1);
    ASSERT_EQ(desc.ip_or_host_name, desc1.ip_or_host_name);
    ASSERT_EQ(desc.rpc_port, desc1.rpc_port);
    re = metadata_client->removeRpcMetaEntry("test_server");
    ASSERT_EQ(re, 0);
}

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
