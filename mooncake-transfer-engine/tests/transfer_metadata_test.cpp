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

#include "transfer_metadata.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <cstdlib>

#include "transport/transport.h"

using namespace mooncake;

namespace mooncake {

class TransferMetadataTestPeer {
   public:
    static int EncodeSegmentDesc(TransferMetadata& metadata,
                                 const TransferMetadata::SegmentDesc& desc,
                                 Json::Value& encoded) {
        return metadata.encodeSegmentDesc(desc, encoded);
    }

    static std::shared_ptr<TransferMetadata::SegmentDesc> DecodeSegmentDesc(
        TransferMetadata& metadata, Json::Value& encoded,
        const std::string& segment_name) {
        return metadata.decodeSegmentDesc(encoded, segment_name);
    }
};

template <typename T>
concept HasMemoryKind = requires(T value) { value.memory_kind; };
template <typename T>
concept HasNumaNode = requires(T value) { value.numa_node; };
template <typename T>
concept HasFabricDomain = requires(T value) { value.fabric_domain_id; };
template <typename T>
concept HasGeneration = requires(T value) { value.generation; };

static_assert(!HasMemoryKind<TransferMetadata::SegmentDesc>);
static_assert(!HasNumaNode<TransferMetadata::SegmentDesc>);
static_assert(!HasFabricDomain<TransferMetadata::SegmentDesc>);
static_assert(!HasGeneration<TransferMetadata::SegmentDesc>);
static_assert(!HasMemoryKind<TransferMetadata::BufferDesc>);
static_assert(!HasNumaNode<TransferMetadata::BufferDesc>);
static_assert(!HasFabricDomain<TransferMetadata::BufferDesc>);
static_assert(!HasGeneration<TransferMetadata::BufferDesc>);

TEST(TransferMetadataSchemaTest, NvlinkDescriptorMatchesV1GoldenJson) {
    TransferMetadata metadata(P2PHANDSHAKE);
    // Freeze the V1 JSON contract from origin/main@98ff4e47. Compare parsed
    // values so key order and whitespace do not become accidental ABI.
    TransferMetadata::SegmentDesc descriptor{};
    descriptor.name = "provider:12345";
    descriptor.protocol = "nvlink";
    descriptor.tcp_data_port = 0;

    TransferMetadata::BufferDesc buffer{};
    buffer.name = descriptor.name;
    buffer.addr = 0x100000000ULL;
    buffer.length = 0x20000ULL;
    buffer.shm_name = "fabric-handle-v1";
    descriptor.buffers.push_back(buffer);

    Json::Value encoded;
    ASSERT_EQ(TransferMetadataTestPeer::EncodeSegmentDesc(metadata, descriptor,
                                                          encoded),
              0);
    ASSERT_TRUE(encoded.isMember("timestamp"));
    ASSERT_TRUE(encoded["timestamp"].isString());
    ASSERT_FALSE(encoded["timestamp"].asString().empty());
    encoded["timestamp"] = "<timestamp>";

    Json::Value expected;
    expected["name"] = descriptor.name;
    expected["protocol"] = descriptor.protocol;
    expected["tcp_data_port"] = descriptor.tcp_data_port;
    expected["timestamp"] = "<timestamp>";
    Json::Value expected_buffers(Json::arrayValue);
    Json::Value expected_buffer;
    expected_buffer["name"] = buffer.name;
    expected_buffer["addr"] = static_cast<Json::UInt64>(buffer.addr);
    expected_buffer["length"] = static_cast<Json::UInt64>(buffer.length);
    expected_buffer["shm_name"] = buffer.shm_name;
    expected_buffers.append(expected_buffer);
    expected["buffers"] = expected_buffers;

    EXPECT_EQ(encoded, expected)
        << "NVLink SegmentDesc wire fields changed from the V1 golden schema";
    for (const char* forbidden :
         {"memory_kind", "numa_node", "fabric_domain_id", "generation"}) {
        EXPECT_FALSE(encoded.isMember(forbidden));
        EXPECT_FALSE(encoded["buffers"][0].isMember(forbidden));
    }

    Json::Value baseline_golden = expected;
    auto decoded = TransferMetadataTestPeer::DecodeSegmentDesc(
        metadata, baseline_golden, descriptor.name);
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(decoded->name, descriptor.name);
    EXPECT_EQ(decoded->protocol, descriptor.protocol);
    EXPECT_EQ(decoded->tcp_data_port, descriptor.tcp_data_port);
    ASSERT_EQ(decoded->buffers.size(), 1U);
    EXPECT_EQ(decoded->buffers[0].name, buffer.name);
    EXPECT_EQ(decoded->buffers[0].addr, buffer.addr);
    EXPECT_EQ(decoded->buffers[0].length, buffer.length);
    EXPECT_EQ(decoded->buffers[0].shm_name, buffer.shm_name);

    Json::Value reencoded;
    ASSERT_EQ(TransferMetadataTestPeer::EncodeSegmentDesc(metadata, *decoded,
                                                          reencoded),
              0);
    ASSERT_TRUE(reencoded.isMember("timestamp"));
    ASSERT_TRUE(reencoded["timestamp"].isString());
    ASSERT_FALSE(reencoded["timestamp"].asString().empty());
    reencoded["timestamp"] = "<timestamp>";
    EXPECT_EQ(reencoded, expected);
}

TEST(TransferTaskSubmissionFailureTest, ZeroSliceFailureIsExplicitlyTerminal) {
    Transport::BatchDesc batch;
    batch.batch_size = 1;
    batch.id = reinterpret_cast<Transport::BatchID>(&batch);
    batch.task_list.resize(1);
    auto& task = batch.task_list.front();
    task.batch_id = batch.id;

    Transport::markSubmissionFailed(task);
    Transport::markSubmissionFailed(task);

    EXPECT_TRUE(task.submission_failed);
    EXPECT_TRUE(task.is_finished);
    EXPECT_EQ(task.slice_count, 0);
    EXPECT_TRUE(batch.has_failure.load());
#ifdef USE_EVENT_DRIVEN_COMPLETION
    EXPECT_TRUE(batch.is_finished.load());
    EXPECT_EQ(batch.finished_task_count.load(), 1);
#endif
}

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
            metadata_server = metadata_server;
        LOG(INFO) << "metadata_server: " << metadata_server;

        env = std::getenv("MC_LOCAL_SERVER_NAME");
        if (env)
            local_server_name = env;
        else
            local_server_name = "127.0.0.2:12345";
        LOG(INFO) << "local_server_name: " << local_server_name;

        metadata_client = std::make_unique<TransferMetadata>(metadata_server);
    }
    void TearDown() override {
        // clean up glog
        google::ShutdownGoogleLogging();
    }
    std::unique_ptr<TransferMetadata> metadata_client;
    std::string metadata_server;
    std::string local_server_name;
};

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
