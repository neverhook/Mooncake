#include "real_client.h"

#include <gtest/gtest.h>

namespace mooncake {

class DirectGpuReadTest : public ::testing::Test {
   protected:
    void SetProtocol(const std::string &protocol) {
        client_.protocol = protocol;
    }

    void RegisterRange(void *buffer, size_t size) {
        client_.registered_buffer_sizes_[buffer] = size;
    }

    bool CanUseDirectRead(void *buffer, size_t size) const {
        return client_.can_use_direct_memory_read(buffer, size);
    }

    RealClient client_;
};

TEST_F(DirectGpuReadTest, NvlinkAcceptsUnregisteredLocalEndpoint) {
    SetProtocol("nvlink");
    char buffer[16];

    EXPECT_TRUE(CanUseDirectRead(buffer, sizeof(buffer)));
}

TEST_F(DirectGpuReadTest, RdmaRequiresRegisteredDestinationRange) {
    SetProtocol("rdma");
    char buffer[16];

    EXPECT_FALSE(CanUseDirectRead(buffer, sizeof(buffer)));
    RegisterRange(buffer, sizeof(buffer));
    EXPECT_TRUE(CanUseDirectRead(buffer + 4, 12));
    EXPECT_FALSE(CanUseDirectRead(buffer + 4, 13));
}

TEST_F(DirectGpuReadTest, OtherProtocolsKeepStagingFallback) {
    SetProtocol("tcp");
    char buffer[16];
    RegisterRange(buffer, sizeof(buffer));

    EXPECT_FALSE(CanUseDirectRead(buffer, sizeof(buffer)));
}

}  // namespace mooncake
